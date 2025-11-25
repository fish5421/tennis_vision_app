"""Calibration utilities for CourtSense.

This module lets the user click the four singles-court corners in the first
frame of the video and computes a homography that maps pixel coordinates to
real-world meters using standard tennis court dimensions.

COORDINATE SYSTEM:
- X-axis: Across the court (0 = left sideline, 8.23m = right sideline)
- Y-axis: Along the court length (0 = near baseline where tracked player is,
          23.77m = far baseline)
- The tracked player should be on the NEAR side (Y close to 0)
- Origin (0,0) is at the near-left corner of the singles court

CALIBRATION MODES:
1. Full-court: Click all 4 corners of the full singles court
   - Near-left (BL in image), Near-right (BR), Far-right (TR), Far-left (TL)
2. Near-side: Click 4 corners of just the near half (service box to baseline)
   - Useful when far baseline isn't fully visible
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Standard singles court dimensions (meters): width 8.23m, length 23.77m.
# Half-court length (baseline to net) = 11.885m
COURT_WIDTH = 8.23
COURT_LENGTH = 23.77
HALF_COURT_LENGTH = COURT_LENGTH / 2.0  # 11.885m

# Full court destination points: origin at NEAR baseline (where tracked player is)
# Order: Near-Left, Near-Right, Far-Right, Far-Left (counter-clockwise from origin)
COURT_DST_METERS = np.array(
    [[0.0, 0.0], [COURT_WIDTH, 0.0], [COURT_WIDTH, COURT_LENGTH], [0.0, COURT_LENGTH]],
    dtype=np.float32
)

# Half-court (near side only): from baseline to net
HALF_COURT_DST_METERS = np.array(
    [[0.0, 0.0], [COURT_WIDTH, 0.0], [COURT_WIDTH, HALF_COURT_LENGTH], [0.0, HALF_COURT_LENGTH]],
    dtype=np.float32
)


@dataclass
class CalibrationResult:
    h_matrix: np.ndarray
    src_points: np.ndarray
    dst_points: np.ndarray
    is_half_court: bool = False  # True if only near half-court was calibrated


def get_first_frame(video_path: Path) -> np.ndarray:
    """Grab the first frame of the video."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read first frame from {video_path}")
    return frame


def _collect_user_points(frame: np.ndarray, half_court: bool = False) -> np.ndarray:
    """Open an interactive window for the user to click four court corners.

    IMPORTANT: Click order is NEAR to FAR (bottom to top of image typically):
    1. Near-Left (bottom-left of court, where tracked player's baseline is)
    2. Near-Right (bottom-right of court)
    3. Far-Right (top-right, opponent's side or net line if half-court)
    4. Far-Left (top-left)

    This ensures Y=0 is at the NEAR baseline where the tracked player is.
    """
    mode_str = "HALF-COURT (baseline to net)" if half_court else "FULL COURT"
    window = f"Calibration ({mode_str}): click corners"
    points: List[Tuple[int, int]] = []

    # Labels for each point in click order - plain English
    point_labels = [
        "1: Bottom-Left corner (closest to camera)",
        "2: Bottom-Right corner (closest to camera)",
        "3: Top-Right corner (far end of court)",
        "4: Top-Left corner (far end of court)"
    ]

    def _on_mouse(event, x, y, _flags, _param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_mouse)

    while True:
        vis = frame.copy()
        # Show which point to click next
        next_idx = len(points)
        if next_idx < 4:
            next_label = point_labels[next_idx]
            # Black text with white outline for visibility
            cv2.putText(
                vis, f"Click: {next_label}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA  # White outline
            )
            cv2.putText(
                vis, f"Click: {next_label}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA  # Black text
            )

        for idx, (px, py) in enumerate(points):
            cv2.circle(vis, (px, py), 10, (0, 0, 0), -1)  # Black circle
            cv2.circle(vis, (px, py), 8, (0, 255, 0), -1)  # Green inner
            label = point_labels[idx].split(":")[0]  # Just "1", "2", etc.
            # Black text with white outline
            cv2.putText(
                vis, label, (px + 12, py - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA
            )
            cv2.putText(
                vis, label, (px + 12, py - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA
            )
        # draw provisional polygon
        if len(points) >= 2:
            cv2.polylines(vis, [np.array(points, dtype=np.int32)], False, (0, 0, 255), 2)
        if len(points) == 4:
            cv2.polylines(vis, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)

        instructions = f"{mode_str} | r:reset | c/SPACE:confirm | ESC:cancel"
        # Black text with white outline for visibility
        cv2.putText(vis, instructions, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, instructions, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("Calibration cancelled by user.")
        if key == ord("r"):
            points = []
        if len(points) == 4 and key in (ord("c"), 32, 13):  # c, space, enter
            break

    cv2.destroyWindow(window)
    return np.array(points, dtype=np.float32)


def compute_homography(frame: np.ndarray, half_court: bool = False) -> CalibrationResult:
    """Run interactive calibration and return the homography matrix.

    Args:
        frame: First frame of the video
        half_court: If True, calibrate only the near half (baseline to net)
    """
    dst_points = HALF_COURT_DST_METERS if half_court else COURT_DST_METERS
    src_points = _collect_user_points(frame, half_court=half_court)
    h_matrix, _ = cv2.findHomography(src_points, dst_points, method=0)
    if h_matrix is None:
        raise RuntimeError("cv2.findHomography failed to compute a matrix.")
    _log_homography_quality(src_points, h_matrix, half_court=half_court)
    return CalibrationResult(
        h_matrix=h_matrix, src_points=src_points, dst_points=dst_points, is_half_court=half_court
    )


def auto_homography(frame: np.ndarray, padding: int = 0) -> CalibrationResult:
    """Compute homography using frame bounds as corners (non-interactive fallback)."""
    h, w = frame.shape[:2]
    src_points = np.array(
        [
            [padding, padding],
            [w - 1 - padding, padding],
            [w - 1 - padding, h - 1 - padding],
            [padding, h - 1 - padding],
        ],
        dtype=np.float32,
    )
    h_matrix, _ = cv2.findHomography(src_points, COURT_DST_METERS, method=0)
    if h_matrix is None:
        raise RuntimeError("cv2.findHomography failed to compute a matrix.")
    _log_homography_quality(src_points, h_matrix)
    return CalibrationResult(h_matrix=h_matrix, src_points=src_points, dst_points=COURT_DST_METERS)


def _log_homography_quality(src_points: np.ndarray, h_matrix: np.ndarray, half_court: bool = False) -> None:
    """Print a simple sanity check comparing mapped court width/length to expected."""
    try:
        mapped = pixel_to_meter(src_points, h_matrix)
        widths = np.linalg.norm(mapped[1] - mapped[0])
        lengths = np.linalg.norm(mapped[3] - mapped[0])

        expected_length = HALF_COURT_LENGTH if half_court else COURT_LENGTH
        w_err = abs(widths - COURT_WIDTH) / COURT_WIDTH
        l_err = abs(lengths - expected_length) / expected_length

        mode_str = "half-court" if half_court else "full court"
        if w_err > 0.2 or l_err > 0.2:
            print(
                f"[Calibration] Warning: {mode_str} size off by {w_err*100:.1f}% width, {l_err*100:.1f}% length. "
                "Re-click corners for better accuracy."
            )
        else:
            print(f"[Calibration] {mode_str.title()} width {widths:.2f} m, length {lengths:.2f} m (within tolerance).")
    except Exception:
        pass


def _overlay_expected_court(frame: np.ndarray, h_matrix: np.ndarray, half_court: bool = False) -> np.ndarray:
    """Draw the projected standard singles rectangle onto the frame for visual validation."""
    dst_points = HALF_COURT_DST_METERS if half_court else COURT_DST_METERS
    dst = dst_points.astype(np.float32).reshape(-1, 1, 2)
    h_inv = np.linalg.inv(h_matrix)
    src_proj = cv2.perspectiveTransform(dst, h_inv).reshape(-1, 2).astype(int)
    vis = frame.copy()
    cv2.polylines(vis, [src_proj], True, (0, 255, 255), 2)
    corner_labels = ["Bottom-L", "Bottom-R", "Top-R", "Top-L"]
    for i, p in enumerate(src_proj):
        cv2.circle(vis, tuple(p), 6, (0, 255, 255), -1)
        cv2.putText(
            vis, corner_labels[i], (p[0] + 6, p[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )
    return vis


def _preview_and_confirm(frame: np.ndarray, src_points: np.ndarray, h_matrix: np.ndarray, half_court: bool = False) -> bool:
    """Show clicked corners and projected court overlay; return True if user accepts."""
    vis = frame.copy()
    pts_int = src_points.astype(int)
    cv2.polylines(vis, [pts_int], isClosed=True, color=(0, 0, 255), thickness=3)
    corner_labels = ["1:Bottom-L", "2:Bottom-R", "3:Top-R", "4:Top-L"]
    for i, (x, y) in enumerate(pts_int):
        cv2.circle(vis, (x, y), 10, (0, 0, 0), -1)  # Black outline
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)  # Green inner
        # Black text with white outline
        cv2.putText(vis, corner_labels[i], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(vis, corner_labels[i], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    vis_overlay = _overlay_expected_court(vis, h_matrix, half_court=half_court)
    mode_str = "HALF-COURT" if half_court else "FULL COURT"
    # Black text with white outline
    cv2.putText(
        vis_overlay, f"{mode_str} | y:accept | r:redo | esc:cancel",
        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4
    )
    cv2.putText(
        vis_overlay, f"{mode_str} | y:accept | r:redo | esc:cancel",
        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
    )
    # Show coordinate system reminder - black with white outline
    cv2.putText(
        vis_overlay, "Start clicking at BOTTOM of court (closest to camera)",
        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3
    )
    cv2.putText(
        vis_overlay, "Start clicking at BOTTOM of court (closest to camera)",
        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
    )
    window = "Confirm court corners"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, vis_overlay)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("y"), ord("Y")):
            out_path = Path("data/manual_court_debug.png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), vis_overlay)
            cv2.destroyWindow(window)
            return True
        if key in (ord("r"), ord("R")):
            cv2.destroyWindow(window)
            return False
        if key == 27:  # ESC
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("Calibration cancelled by user.")


def compute_homography_interactive(frame: np.ndarray, half_court: bool = False) -> CalibrationResult:
    """Loop until user accepts clicked corners; returns CalibrationResult.

    Args:
        frame: First frame of the video
        half_court: If True, calibrate only the near half (baseline to net)
    """
    dst_points = HALF_COURT_DST_METERS if half_court else COURT_DST_METERS
    expected_length = HALF_COURT_LENGTH if half_court else COURT_LENGTH
    min_length = expected_length * 0.8
    max_length = expected_length * 1.2

    while True:
        src_points = _collect_user_points(frame, half_court=half_court)
        h_matrix, _ = cv2.findHomography(src_points, dst_points, method=0)
        if h_matrix is None:
            print("[Calibration] Homography failed, please reselect points.")
            continue
        mapped = pixel_to_meter(src_points, h_matrix)
        width = np.linalg.norm(mapped[1] - mapped[0])
        length = np.linalg.norm(mapped[3] - mapped[0])
        if width < 7.0 or width > 9.5 or length < min_length or length > max_length:
            print(f"[Calibration] Rejected: computed court {width:.2f} x {length:.2f} m (expected ~{COURT_WIDTH:.2f} x {expected_length:.2f}).")
            continue
        if _preview_and_confirm(frame, src_points, h_matrix, half_court=half_court):
            _log_homography_quality(src_points, h_matrix, half_court=half_court)
            return CalibrationResult(
                h_matrix=h_matrix, src_points=src_points, dst_points=dst_points, is_half_court=half_court
            )


def pixel_to_meter(points_px: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    """Project pixel points (Nx2) to meters using the homography."""
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    meters = cv2.perspectiveTransform(pts, h_matrix).reshape(-1, 2)
    return meters


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual court calibration demo.")
    parser.add_argument("video", type=Path, help="Path to input video")
    args = parser.parse_args()

    frame0 = get_first_frame(args.video)
    calib = compute_homography(frame0)
    print("Homography:\n", calib.h_matrix)
    print("Pixels:", calib.src_points)
    print("Meters:", calib.dst_points)
