"""Player tracking using Segment Anything 2 (SAM 2) with MPS acceleration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# Enable MPS fallback for unsupported ops and set memory limits
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _bottom_center(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return bottom-center pixel of a binary mask."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    max_y = ys.max()
    xs_at_max = xs[ys == max_y]
    cx = int(xs_at_max.mean())
    cy = int(max_y)
    return (cx, cy)


@dataclass
class TrackingFrame:
    frame_idx: int
    centroid_px: Optional[Tuple[int, int]]
    mask: Optional[np.ndarray]


class PlayerTracker:
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-small",
        device: Optional[torch.device] = None,
        force_cpu: bool = False,
    ):
        if force_cpu:
            self.device = torch.device("cpu")
            print("[Tracker] Forced CPU mode (slower but more stable)")
        else:
            self.device = device or _get_device()
        self.model_id = model_id
        self.max_frames_limit: Optional[int] = None
        self.predictor = self._load_predictor()
        try:
            torch.set_default_device(self.device)
        except Exception:
            # Older torch builds may not support set_default_device on MPS.
            pass

    def _load_predictor(self):
        """Load SAM 2 video predictor and move to the desired device."""
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            import sam2.utils.misc as misc
        except ImportError as exc:
            raise ImportError(
                "sam2 not installed. Install from https://github.com/facebookresearch/sam2"
            ) from exc

        # Patch video loader to avoid decord dependency by using OpenCV.
        # Memory-optimized version that processes on CPU first, then moves to GPU.
        def _cv2_video_loader(
            video_path,
            image_size,
            offload_video_to_cpu,
            img_mean=(0.485, 0.456, 0.406),
            img_std=(0.229, 0.224, 0.225),
            compute_device=torch.device("cpu"),
        ):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video for SAM2: {video_path}")
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Pre-compute normalization constants on CPU
            img_mean_t = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
            img_std_t = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

            frames = []
            max_frames = getattr(self, "max_frames_limit", None)

            # Limit frames for MPS memory management (default 200 if not set)
            if max_frames is None:
                # Auto-limit based on device to prevent OOM
                if compute_device.type == "mps":
                    max_frames = 200  # Safe default for MPS
                    print(f"[Tracker] Auto-limiting to {max_frames} frames for MPS memory management")

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if max_frames is not None and len(frames) >= max_frames:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (image_size, image_size))
                # Convert to tensor and normalize ON CPU first
                t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                t = (t - img_mean_t) / img_std_t  # Normalize each frame individually on CPU
                frames.append(t)

            cap.release()
            if len(frames) == 0:
                raise RuntimeError("No frames decoded by OpenCV loader.")

            # Stack frames (still on CPU)
            images = torch.stack(frames, dim=0)
            del frames  # Free memory immediately

            # Move to GPU only if requested and not offloading to CPU
            if not offload_video_to_cpu and compute_device.type != "cpu":
                # For MPS: move in smaller chunks to avoid memory spikes
                if compute_device.type == "mps":
                    # Keep on CPU and let SAM2 handle per-frame transfers
                    # This is slower but prevents OOM
                    print(f"[Tracker] Loaded {len(images)} frames (keeping on CPU for MPS stability)")
                else:
                    images = images.to(compute_device)

            return images, video_height, video_width

        misc.load_video_frames_from_video_file = _cv2_video_loader

        predictor = SAM2VideoPredictor.from_pretrained(self.model_id, device=self.device.type)
        # SAM2VideoPredictor exposes the underlying model via `.model`
        if hasattr(predictor, "model"):
            predictor.model.to(self.device)
        elif hasattr(predictor, "to"):
            predictor.to(self.device)
        return predictor

    def _select_box(
        self,
        frame0: np.ndarray,
        preset: Optional[Tuple[int, int, int, int]] = None,
        allow_fallback: bool = True,
    ) -> Tuple[int, int, int, int]:
        """Pick the player bounding box. Falls back to center box if none selected."""
        if preset is not None:
            return preset
        window = "Select player to track (hit ENTER)"
        box = cv2.selectROI(window, frame0, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window)
        if box is None or box[2] == 0 or box[3] == 0:
            if allow_fallback:
                h, w = frame0.shape[:2]
                bw = int(w * 0.3)
                bh = int(h * 0.3)
                bx = int((w - bw) / 2)
                by = int((h - bh) / 2)
                print("[Tracker] No ROI selected; using center box fallback.")
                return (bx, by, bw, bh)
            raise RuntimeError("No ROI selected for tracking.")
        return box  # (x, y, w, h)

    def track(
        self,
        video_path: Path,
        display: bool = False,
        store_masks: bool = False,
        homography: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        allow_bbox_fallback: bool = True,
        max_frames: Optional[int] = None,
    ) -> Tuple[List[TrackingFrame], float]:
        """Run SAM2 propagation and return per-frame masks and centroids."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            cap.release()
            raise ValueError(f"Unable to read video: {video_path}")

        box = self._select_box(frame0, preset=bbox, allow_fallback=allow_bbox_fallback)
        box_xyxy = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)

        # Initialize SAM2 video predictor
        self.max_frames_limit = max_frames
        inference_state = self.predictor.init_state(video_path=str(video_path))
        self.predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=0, obj_id=1, box=box_xyxy
        )

        # Restart video reader to align frames with propagation output
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if display:
            cv2.namedWindow("CourtSense - Tracking", cv2.WINDOW_NORMAL)

        results: List[TrackingFrame] = []
        overlay_color = (0, 255, 0)

        cast_device = "mps" if self.device.type == "mps" else self.device.type
        if self.device.type in ("mps", "cuda"):
            cast_dtype = torch.float16
        else:
            cast_dtype = torch.float32

        with torch.inference_mode(), torch.autocast(device_type=cast_device, dtype=cast_dtype):
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(inference_state):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                # Select mask for our object id
                if 1 in object_ids:
                    obj_index = list(object_ids).index(1)
                    mask_t = masks[obj_index]
                    if hasattr(mask_t, "shape"):
                        mask_np = mask_t.squeeze().detach().to("cpu").numpy()
                    else:
                        mask_np = np.array(mask_t)
                    mask_bin = mask_np > 0
                    centroid = _bottom_center(mask_bin)
                else:
                    mask_bin = np.zeros(frame.shape[:2], dtype=bool)
                    centroid = None

                results.append(
                    TrackingFrame(
                        frame_idx=frame_idx,
                        centroid_px=centroid,
                        mask=mask_bin if store_masks else None,
                    )
                )

                if display:
                    overlay = frame.copy()
                    colored = np.zeros_like(frame)
                    colored[mask_bin] = overlay_color
                    overlay = cv2.addWeighted(overlay, 0.65, colored, 0.35, 0)
                    metric_text = ""
                    if centroid:
                        if homography is not None:
                            try:
                                from modules.calibration import pixel_to_meter

                                metric_xy = pixel_to_meter([centroid], homography)[0]
                                metric_text = f"{metric_xy[0]:.2f} m, {metric_xy[1]:.2f} m"
                            except Exception:
                                metric_text = ""
                        cv2.circle(overlay, centroid, 5, (255, 255, 255), -1)
                        cv2.putText(
                            overlay,
                            f"Frame {frame_idx}",
                            (centroid[0] + 8, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    if metric_text:
                        cv2.putText(
                            overlay,
                            metric_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    cv2.imshow("CourtSense - Tracking", overlay)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        cap.release()
        if display:
            cv2.destroyWindow("CourtSense - Tracking")
        return results, fps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM2 player tracker demo")
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument("--no-display", action="store_true", help="Disable live overlay window")
    args = parser.parse_args()

    tracker = PlayerTracker()
    tracking, fps = tracker.track(args.video, display=not args.no_display)
    print(f"Tracked {len(tracking)} frames at {fps:.2f} fps metadata.")
