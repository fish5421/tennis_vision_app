"""Geometry utilities: pixel->meter mapping, kinematics, and shot analysis.

COORDINATE SYSTEM (matches calibration.py):
- X-axis: Across the court (0 = left sideline, 8.23m = right sideline)
- Y-axis: Along the court length (0 = near baseline, 23.77m = far baseline)
- The tracked player should be on the NEAR side (Y close to 0)
- Ideal recovery position is center of NEAR baseline: (4.115m, 0m)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Court dimensions
COURT_WIDTH = 8.23
COURT_LENGTH = 23.77
HALF_COURT_LENGTH = COURT_LENGTH / 2.0

# Ideal recovery position: center of the NEAR baseline (where tracked player is)
# X = 4.115m (center of 8.23m width), Y = 0m (near baseline)
IDEAL_POSITION_NEAR = np.array([COURT_WIDTH / 2, 0.0], dtype=np.float32)

# Legacy: ideal position if tracking far-side player (Y = 23.77m)
IDEAL_POSITION_FAR = np.array([COURT_WIDTH / 2, COURT_LENGTH], dtype=np.float32)

# Default ideal position (near baseline - standard case)
IDEAL_POSITION = IDEAL_POSITION_NEAR


def compute_dynamic_ideal_position(coords_m: Sequence[Sequence[float]]) -> np.ndarray:
    """Compute ideal recovery position based on actual player position data.

    For players behind the baseline (negative Y), the ideal recovery position
    is center-court at the player's median Y position (they stay behind baseline).
    """
    coords = np.asarray(coords_m, dtype=np.float32)
    if len(coords) == 0:
        return IDEAL_POSITION_NEAR.copy()

    # Use median Y as the baseline for this player's style
    median_y = np.median(coords[:, 1])
    # Ideal X is always court center
    ideal_x = COURT_WIDTH / 2

    return np.array([ideal_x, median_y], dtype=np.float32)


def validate_coordinates(coords_m: Sequence[Sequence[float]], half_court: bool = False) -> dict:
    """Check if coordinates fall within expected court bounds.

    Returns dict with validation results and warnings.
    """
    coords = np.asarray(coords_m, dtype=np.float32)
    if len(coords) == 0:
        return {"valid": False, "message": "No coordinates to validate"}

    max_y = HALF_COURT_LENGTH if half_court else COURT_LENGTH
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    warnings = []

    # Check X bounds (should be 0 to 8.23m)
    if x_min < -1.0 or x_max > COURT_WIDTH + 1.0:
        warnings.append(f"X coordinates ({x_min:.2f} to {x_max:.2f}m) outside court width (0-{COURT_WIDTH}m)")

    # Check Y bounds
    if y_min < -2.0:
        warnings.append(f"Y min ({y_min:.2f}m) is negative - player behind baseline (may be OK)")
    if y_max > max_y + 2.0:
        expected = "half-court" if half_court else "full court"
        warnings.append(
            f"Y max ({y_max:.2f}m) exceeds {expected} length ({max_y}m) - "
            "calibration may be incorrect or player is on wrong side"
        )

    # Check if player is on near side (Y should be mostly < half court for near player)
    near_side_pct = np.mean(coords[:, 1] < HALF_COURT_LENGTH) * 100
    if near_side_pct < 50:
        warnings.append(
            f"Player spends {100-near_side_pct:.0f}% of time on far side - "
            "verify calibration click order (near baseline should be Y=0)"
        )

    return {
        "valid": len(warnings) == 0,
        "x_range": (float(x_min), float(x_max)),
        "y_range": (float(y_min), float(y_max)),
        "near_side_pct": float(near_side_pct),
        "warnings": warnings,
    }


def compute_velocities(coords_m: Sequence[Sequence[float]], fps: float) -> np.ndarray:
    """Return per-frame velocity vectors (m/s) from metric coordinates."""
    coords = np.asarray(coords_m, dtype=np.float32)
    if len(coords) < 2:
        return np.empty((0, 2), dtype=np.float32)
    diffs = np.diff(coords, axis=0)
    return diffs * fps


def detect_shots(
    coords_m: Sequence[Sequence[float]],
    fps: float,
    speed_thresh: float = 2.0,
    min_frames_between_shots: int = 15,
    min_direction_change_magnitude: float = 0.3,
) -> List[int]:
    """Detect shot frames using direction changes and speed patterns.

    A shot is detected when:
    1. Horizontal velocity (vx) changes sign significantly
    2. Speed drops below threshold (player stops/slows to hit)
    3. Sufficient frames have passed since last shot

    Args:
        coords_m: Player coordinates in meters
        fps: Video frame rate
        speed_thresh: Max speed (m/s) to consider as "stopped" for a shot (default 2.0)
        min_frames_between_shots: Minimum gap between detected shots (default 15 = 0.5s at 30fps)
        min_direction_change_magnitude: Min |vx| before sign flip to count (filters noise)
    """
    velocities = compute_velocities(coords_m, fps)
    if len(velocities) < 2:
        return []

    vx = velocities[:, 0]
    speed = np.linalg.norm(velocities, axis=1)
    shot_frames: List[int] = []
    last_shot_frame = -min_frames_between_shots  # Allow first shot immediately

    for i in range(1, len(vx)):
        # Check for direction change with sufficient magnitude
        direction_changed = (
            np.sign(vx[i]) != np.sign(vx[i - 1])
            and abs(vx[i - 1]) > min_direction_change_magnitude
        )
        speed_low = speed[i] < speed_thresh
        enough_time_passed = (i + 1 - last_shot_frame) >= min_frames_between_shots

        if direction_changed and speed_low and enough_time_passed:
            shot_frame = i + 1  # velocity i refers to movement into frame i+1
            shot_frames.append(shot_frame)
            last_shot_frame = shot_frame

    return shot_frames


def compute_recovery_times(
    coords_m: Sequence[Sequence[float]],
    shot_frames: Iterable[int],
    fps: float,
    ideal_position: np.ndarray = IDEAL_POSITION,
    recover_radius: float = 1.5,
) -> List[Dict]:
    """For each shot frame, find when the player returns near the ideal spot."""
    coords = np.asarray(coords_m, dtype=np.float32)
    results: List[Dict] = []
    for sid, sf in enumerate(shot_frames, start=1):
        recover_idx = None
        for idx in range(sf, len(coords)):
            dist = np.linalg.norm(coords[idx] - ideal_position)
            if dist < recover_radius:
                recover_idx = idx
                break
        if recover_idx is None:
            continue
        time_delta = (recover_idx - sf) / fps
        results.append(
            {"shot_id": sid, "frame_hit": sf, "frame_recover": recover_idx, "time_to_recover": float(time_delta)}
        )
    return results


def render_topdown_path(
    coords_m: Sequence[Sequence[float]],
    shot_frames: Iterable[int],
    image_width: int = 500,
    margin: int = 24,
    half_court: bool = False,
) -> np.ndarray:
    """Render a simple top-down court view with the path overlay.

    Coordinate system: Y=0 is at the BOTTOM of the rendered image (near baseline).
    This matches the court coordinate system where Y=0 is the tracked player's baseline.
    """
    coords = np.asarray(coords_m, dtype=np.float32)
    if len(coords) == 0:
        return None

    court_width = COURT_WIDTH
    court_length = HALF_COURT_LENGTH if half_court else COURT_LENGTH
    scale = image_width / court_width
    image_height = int(court_length * scale)

    canvas = np.full((image_height + 2 * margin, image_width + 2 * margin, 3), 20, dtype=np.uint8)

    # Draw court outline
    cv2.rectangle(
        canvas,
        (margin, margin),
        (margin + image_width, margin + image_height),
        (80, 180, 255),
        2,
    )

    # Draw net line (at Y = half court length if full court, or at top if half court)
    net_y = HALF_COURT_LENGTH if not half_court else court_length
    net_y_px = int(margin + (court_length - net_y) * scale)  # Flip Y for rendering
    cv2.line(canvas, (margin, net_y_px), (margin + image_width, net_y_px), (255, 255, 255), 2)

    # Draw ideal position marker (center of near baseline)
    ideal_pos = IDEAL_POSITION_NEAR
    ideal_y_px = int(margin + (court_length - ideal_pos[1]) * scale)  # Flip Y
    ideal_x_px = int(margin + ideal_pos[0] * scale)
    cv2.circle(canvas, (ideal_x_px, ideal_y_px), 8, (255, 200, 0), 2)  # Orange circle for ideal spot

    def _to_px(x_m, y_m):
        # Flip Y so Y=0 (near baseline) is at the BOTTOM of the image
        x = int(margin + x_m * scale)
        y = int(margin + (court_length - y_m) * scale)
        return (x, y)

    path_pts = [_to_px(x, y) for x, y in coords]
    for i in range(1, len(path_pts)):
        cv2.line(canvas, path_pts[i - 1], path_pts[i], (0, 255, 0), 2)

    for sf in shot_frames:
        if sf < len(path_pts):
            cv2.circle(canvas, path_pts[sf], 6, (0, 0, 255), -1)

    if len(path_pts) > 0:
        cv2.circle(canvas, path_pts[-1], 5, (255, 255, 255), -1)

    mode_str = "Half-court" if half_court else "Full court"
    cv2.putText(
        canvas,
        f"{mode_str} path (Y=0 at bottom/near baseline)",
        (margin, margin - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    # Add legend
    legend_y = image_height + margin + 15
    cv2.circle(canvas, (margin + 10, legend_y), 5, (0, 255, 0), -1)
    cv2.putText(canvas, "Path", (margin + 20, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(canvas, (margin + 70, legend_y), 5, (0, 0, 255), -1)
    cv2.putText(canvas, "Shots", (margin + 80, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(canvas, (margin + 140, legend_y), 5, (255, 200, 0), 2)
    cv2.putText(canvas, "Ideal", (margin + 150, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return canvas
