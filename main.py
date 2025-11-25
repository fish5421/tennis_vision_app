from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from modules import coach, geometry
from modules.calibration import (
    CalibrationResult,
    COURT_DST_METERS,
    HALF_COURT_DST_METERS,
    auto_homography,
    compute_homography,
    compute_homography_interactive,
    get_first_frame,
    pixel_to_meter,
)
from modules.auto_calibrate import auto_calibrate, detect_court_corners
from modules.tracker import PlayerTracker


def parse_args():
    parser = argparse.ArgumentParser(description="CourtSense MVP - Recovery Speed Analyzer")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/input_video.mp4"),
        help="Path to input video.",
    )
    parser.add_argument("--model", default="facebook/sam2-hiera-small", help="SAM2 model id.")
    parser.add_argument("--no-display", action="store_true", help="Disable live tracking overlay.")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama call and use heuristic advice.")
    parser.add_argument("--auto-court", action="store_true", help="Use frame bounds for homography (headless).")
    parser.add_argument("--auto-court-detector", action="store_true", help="Detect court lines automatically (Hough).")
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Manual bbox x,y,w,h to skip ROI UI (e.g., 100,200,150,300).",
    )
    parser.add_argument("--auto-bbox", action="store_true", help="Use center 30%% box as ROI (headless).")
    parser.add_argument("--max-frames", type=int, default=200, help="Limit frames processed (default 200 for MPS stability).")
    parser.add_argument(
        "--half-court",
        action="store_true",
        help="Calibrate only near half-court (baseline to net). Use when far baseline isn't visible.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (slower but avoids MPS memory issues).",
    )
    return parser.parse_args()


def fill_coords(tracking_frames, h_matrix: np.ndarray):
    """Convert pixel centroids to meters; forward-fill missing detections."""
    coords_m = []
    last_valid = None
    for tf in tracking_frames:
        if tf.centroid_px is not None:
            metr = pixel_to_meter([tf.centroid_px], h_matrix)[0]
            last_valid = metr
        if last_valid is None:
            coords_m.append(np.array([0.0, 0.0]))
        else:
            coords_m.append(last_valid)
    return coords_m


def main():
    args = parse_args()
    video_path: Path = args.video
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    half_court = args.half_court
    mode_str = "HALF-COURT" if half_court else "FULL COURT"

    print("Grabbing first frame for calibration...")
    frame0 = get_first_frame(video_path)
    if args.auto_court:
        calib: CalibrationResult = auto_homography(frame0)
        print("Homography computed with auto-court (frame bounds).")
    elif args.auto_court_detector:
        det = detect_court_corners(frame0)
        if det is None:
            print("Auto court detector failed; falling back to manual clicks.")
            calib = compute_homography_interactive(frame0, half_court=half_court)
        else:
            h_mat, src_pts, dbg = det
            dst_pts = HALF_COURT_DST_METERS if half_court else COURT_DST_METERS
            calib = CalibrationResult(
                h_matrix=h_mat, src_points=src_pts.astype(np.float32),
                dst_points=dst_pts, is_half_court=half_court
            )
            out_path = Path("data/auto_court_debug.png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), dbg)
            print(f"Homography computed with auto court detector. Debug saved to {out_path}")
    else:
        calib: CalibrationResult = compute_homography_interactive(frame0, half_court=half_court)
        print(f"Homography computed via manual clicks ({mode_str} mode).")

    tracker = PlayerTracker(model_id=args.model, force_cpu=args.cpu)
    device_info = "CPU" if args.cpu else "MPS/GPU"
    print(f"Running SAM2 tracking on {device_info} (press ESC to stop)...")

    bbox_tuple = None
    if args.bbox:
        try:
            parts = [int(p.strip()) for p in args.bbox.split(",")]
            if len(parts) == 4:
                bbox_tuple = tuple(parts)  # type: ignore
        except Exception:
            bbox_tuple = None
    if bbox_tuple is None and args.auto_bbox:
        h, w = frame0.shape[:2]
        bw = int(w * 0.3)
        bh = int(h * 0.3)
        bx = int((w - bw) / 2)
        by = int((h - bh) / 2)
        bbox_tuple = (bx, by, bw, bh)
    tracking_frames, fps = tracker.track(
        video_path=video_path,
        display=not args.no_display,
        store_masks=False,
        homography=calib.h_matrix,
        bbox=bbox_tuple,
        allow_bbox_fallback=True,
        max_frames=args.max_frames,
    )
    print(f"Tracking complete on {len(tracking_frames)} frames (video FPS metadata: {fps:.2f}).")

    coords_m = fill_coords(tracking_frames, calib.h_matrix)
    valid_centroids = sum(1 for tf in tracking_frames if tf.centroid_px is not None)
    if len(tracking_frames) > 0:
        print(f"Valid centroids: {valid_centroids}/{len(tracking_frames)}")

    # Validate coordinates and warn about potential calibration issues
    validation = geometry.validate_coordinates(coords_m, half_court=half_court)
    print(f"Coordinate ranges: X={validation['x_range'][0]:.2f} to {validation['x_range'][1]:.2f}m, "
          f"Y={validation['y_range'][0]:.2f} to {validation['y_range'][1]:.2f}m")
    print(f"Player on near side: {validation['near_side_pct']:.1f}% of frames")
    if validation["warnings"]:
        print("\n⚠️  CALIBRATION WARNINGS:")
        for w in validation["warnings"]:
            print(f"   - {w}")
        print("   Consider re-running with correct corner click order:")
        print("   1. Near-Left (your baseline left)")
        print("   2. Near-Right (your baseline right)")
        print("   3. Far-Right (opponent's baseline right)")
        print("   4. Far-Left (opponent's baseline left)")
        print()

    if len(coords_m) >= 2:
        dists = np.linalg.norm(np.diff(np.array(coords_m), axis=0), axis=1)
        if len(dists) > 0:
            print(
                f"Per-frame displacement (m): min {dists.min():.3f}, median {np.median(dists):.3f}, max {dists.max():.3f}"
            )
            np.save("data/coords_m.npy", np.array(coords_m))
            np.save("data/coords_px.npy", np.array([tf.centroid_px for tf in tracking_frames]))
            with open("data/coords_stats.txt", "w") as f:
                f.write(f"Valid: {valid_centroids}/{len(tracking_frames)}\n")
                f.write(f"Displacement min {dists.min():.4f} med {np.median(dists):.4f} max {dists.max():.4f}\n")
                f.write(f"Half-court mode: {half_court}\n")
                f.write(f"Coordinate validation: {validation}\n")
                f.write(f"Homography:\n{calib.h_matrix}\n")
    if len(coords_m) < 2:
        print("Not enough tracking points to analyze.")
        return

    shot_frames = geometry.detect_shots(coords_m, fps)
    print(f"Detected {len(shot_frames)} shots")

    # Compute dynamic ideal position based on where the player actually moves
    # This handles cases where player is behind baseline (negative Y coords)
    ideal_pos = geometry.compute_dynamic_ideal_position(coords_m)
    print(f"Using dynamic ideal recovery position: ({ideal_pos[0]:.2f}m, {ideal_pos[1]:.2f}m)")

    recovery_stats = geometry.compute_recovery_times(coords_m, shot_frames, fps, ideal_position=ideal_pos)
    print(f"Computed recovery stats for {len(recovery_stats)} shots")

    for stat in recovery_stats:
        print(f"Shot Detected at Frame {stat['frame_hit']}, Recovery Time: {stat['time_to_recover']:.2f} seconds")

    topdown = geometry.render_topdown_path(coords_m, shot_frames, half_court=half_court)
    if topdown is not None:
        out_path = Path("data/topdown_path.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), topdown)
        print(f"Top-down path saved to {out_path}")

    coords_array = np.array(coords_m)
    feedback = coach.generate_feedback(
        recovery_stats,
        use_ollama=not args.no_ollama,
        coords_m=coords_array
    )
    print("\nCoach Feedback:\n", feedback)


if __name__ == "__main__":
    main()
