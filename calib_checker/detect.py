"""Frame-based checkerboard detection on raw sensor images (8-bit, 16-bit, color)."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def to_gray8(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Any image -> uint8 gray. 16-bit/float data is stretched between percentiles
    (raw Boson Y16 has a narrow, drifting range, so a fixed scale does not work)."""
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    if img.dtype == np.uint8:
        return img
    f = img.astype(np.float32)
    sub = f[::4, ::4]
    lo, hi = np.percentile(sub, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        return np.zeros(img.shape[:2], np.uint8)
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def detect_checkerboard(gray: np.ndarray, pattern: Tuple[int, int],
                        max_search_width: int = 1000) -> Optional[np.ndarray]:
    """Return Nx2 float32 corners (row-major, OpenCV order) or None.
    Searches a downscaled copy of large images, then refines at full resolution."""
    scale = 1.0
    search = gray
    if gray.shape[1] > max_search_width:
        scale = max_search_width / gray.shape[1]
        search = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(search, pattern, flags=flags)
    if not ok:
        # Fallback: classic detector is more tolerant to low-contrast thermal boards.
        ok, corners = cv2.findChessboardCorners(
            search, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    if scale != 1.0:
        corners = (corners + 0.5) / scale - 0.5
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
    return canonical_order(corners.reshape(-1, 2), pattern)


def canonical_order(corners: np.ndarray, pattern: Tuple[int, int]) -> np.ndarray:
    """Boards whose square counts have equal parity look identical after a 180 deg in-plane
    rotation, and OpenCV may return either order from frame to frame. Pick the order whose
    first corner is nearer the image top-left so consecutive frames stay comparable.
    (Only a display/stillness convention: true corner identity is resolved downstream.)"""
    cols, rows = pattern
    if (cols + 1) % 2 != (rows + 1) % 2:
        return corners
    if corners[-1].sum() < corners[0].sum():
        return corners[::-1].copy()
    return corners
