"""Automatic court corner detection from a single frame using classical CV.

Two strategies:
1) Color-based court extraction (preferred): segment the colored court,
   fit the outer doubles rectangle, then shrink to singles by known offsets.
2) Line-based (fallback): white-line edge detection + Hough/LSD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

from modules.calibration import COURT_DST_METERS, CalibrationResult

# Singles vs doubles dimensions (meters)
COURT_WIDTH_DOUBLES = 10.97
COURT_WIDTH_SINGLES = 8.23
SINGLES_OFFSET = (COURT_WIDTH_DOUBLES - COURT_WIDTH_SINGLES) / 2.0  # 1.37m each side
OFFSET_RATIO = SINGLES_OFFSET / COURT_WIDTH_DOUBLES  # ~0.1248


def _prep_line_mask(frame: np.ndarray) -> np.ndarray:
    """Return an edge map highlighting bright court lines."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    # Combine global and adaptive thresholding
    _, mask1 = cv2.threshold(L, 200, 255, cv2.THRESH_BINARY)
    mask2 = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.Canny(mask, 40, 120, apertureSize=3)
    return edges


def _avg_xy(line: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = line
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _intersect(l1: Tuple[int, int, int, int], l2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def _lsd_lines(edges: np.ndarray, min_len: int) -> List[Tuple[int, int, int, int]]:
    """Use LSD as a fallback to get line segments."""
    lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_ADV)
    lines, _, _, _ = lsd.detect(edges)
    results = []
    if lines is None:
        return results
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if np.hypot(x2 - x1, y2 - y1) >= min_len:
            results.append((int(x1), int(y1), int(x2), int(y2)))
    return results


def detect_court_corners(frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (h_matrix, src_pts, debug_img) or None if detection fails."""
    color_res = _detect_court_by_color(frame)
    if color_res is not None:
        return color_res
    h, w = frame.shape[:2]
    scale = 960.0 / w if w > 960 else 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame.copy()
    edges = _prep_line_mask(small)

    lines: List[Tuple[int, int, int, int]] = []
    min_len = int(small.shape[1] * 0.08)  # court lines can be short in perspective

    # Try HoughLinesP first
    hp = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=25)
    if hp is not None:
        lines.extend([tuple(l[0]) for l in hp])

    # Fallback to LSD if too few lines
    if len(lines) < 6:
        lines.extend(_lsd_lines(edges, min_len))

    if len(lines) < 4:
        return None

    vertical, horizontal = [], []
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(dx) < abs(dy) * 0.6:  # near vertical
            vertical.append((x1, y1, x2, y2))
        elif abs(dy) < abs(dx) * 0.6:  # near horizontal
            horizontal.append((x1, y1, x2, y2))

    if len(vertical) < 2 or len(horizontal) < 2:
        return None

    # pick outermost lines by avg x/y
    v_sorted = sorted(vertical, key=lambda l: _avg_xy(l)[0])
    h_sorted = sorted(horizontal, key=lambda l: _avg_xy(l)[1])
    left, right = v_sorted[0], v_sorted[-1]
    top, bottom = h_sorted[0], h_sorted[-1]

    tl = _intersect(top, left)
    tr = _intersect(top, right)
    br = _intersect(bottom, right)
    bl = _intersect(bottom, left)
    pts = [tl, tr, br, bl]
    if any(p is None for p in pts):
        return None
    src = np.array(pts, dtype=np.float32)
    # Rescale back to original frame coords
    if scale != 1.0:
        src /= scale

    h_matrix, _ = cv2.findHomography(src, COURT_DST_METERS, method=0)
    if h_matrix is None:
        return None

    # Quality check: aspect ratio
    mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h_matrix).reshape(-1, 2)
    width = np.linalg.norm(mapped[1] - mapped[0])
    length = np.linalg.norm(mapped[3] - mapped[0])
    ratio_err = abs((width / length) - (8.23 / 23.77))
    if ratio_err > 0.35:
        return None

    debug = small.copy()
    for l in [left, right, top, bottom]:
        cv2.line(debug, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2)
    for i, p in enumerate(src if scale == 1.0 else src * scale):
        cv2.circle(debug, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
        cv2.putText(debug, str(i + 1), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return h_matrix, src, debug


# --- Color-based court detection ------------------------------------------------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _detect_court_by_color(frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Detect outer doubles rectangle by color mask and derive singles corners."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Broad blue range; adjust if courts differ.
    lower = np.array([70, 30, 30])
    upper = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cont = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cont)
    if area < frame.shape[0] * frame.shape[1] * 0.1:
        return None

    epsilon = 0.02 * cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, epsilon, True)
    if len(approx) == 4:
        quad = approx[:, 0, :]
    else:
        rect = cv2.minAreaRect(cont)
        quad = cv2.boxPoints(rect)
    quad = _order_quad(quad.astype(np.float32))

    # Derive singles by moving inset along top/bottom edges by OFFSET_RATIO
    tl, tr, br, bl = quad
    singles_tl = tl + OFFSET_RATIO * (tr - tl)
    singles_tr = tr - OFFSET_RATIO * (tr - tl)
    singles_br = br - OFFSET_RATIO * (br - bl)
    singles_bl = bl + OFFSET_RATIO * (br - bl)
    singles = np.array([singles_tl, singles_tr, singles_br, singles_bl], dtype=np.float32)

    h_matrix, _ = cv2.findHomography(singles, COURT_DST_METERS, method=0)
    if h_matrix is None:
        return None

    debug = frame.copy()
    cv2.polylines(debug, [quad.astype(int)], True, (0, 255, 0), 2)
    cv2.polylines(debug, [singles.astype(int)], True, (0, 0, 255), 2)
    for i, p in enumerate(singles):
        cv2.circle(debug, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
        cv2.putText(debug, str(i + 1), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return h_matrix, singles, debug


def auto_calibrate(frame: np.ndarray) -> Optional[CalibrationResult]:
    res = detect_court_corners(frame)
    if res is None:
        return None
    h_matrix, src_pts, _debug = res
    return CalibrationResult(h_matrix=h_matrix, src_points=src_pts.astype(np.float32), dst_points=COURT_DST_METERS)
