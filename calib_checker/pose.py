"""Board geometry, PnP and the pose features used for coverage (depth, tilt, image cells)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .camera import Camera, make_T, transform
from .session import TargetSpec


def board_points(t: TargetSpec) -> np.ndarray:
    """Inner-corner object points (z=0) in OpenCV row-major order."""
    xs, ys = np.meshgrid(np.arange(t.inner_cols), np.arange(t.inner_rows))
    return np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)] * t.square_m


def board_center(t: TargetSpec) -> np.ndarray:
    return np.array([(t.inner_cols - 1) / 2, (t.inner_rows - 1) / 2, 0.0]) * t.square_m


def board_outline(t: TargetSpec) -> np.ndarray:
    """Physical outline (outer squares + border) as a closed 5-point loop, z=0."""
    m = t.square_m + t.border_m
    x0, y0 = -m, -m
    x1 = (t.inner_cols - 1) * t.square_m + m
    y1 = (t.inner_rows - 1) * t.square_m + m
    return np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, 0]], float)


def solve_pose(cam: Camera, corners_px: np.ndarray, t: TargetSpec) -> Optional[np.ndarray]:
    """T_cam_board from detected corners (undistorted with the camera's own model)."""
    obj = board_points(t)
    n = cam.normalize(corners_px)
    I = np.eye(3)
    ok, rvec, tvec = cv2.solvePnP(obj, n, I, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, n, I, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec)


@dataclass
class PoseFeatures:
    depth_m: float            # z of the board center in the camera frame
    tilt_deg: float           # angle between board normal and viewing ray
    tilt_dir: str             # "frontal" or which board edge is farther (left/right/top/bottom_far)
    center_cam: np.ndarray
    normal_cam: np.ndarray    # unit board normal, oriented toward the camera


def pose_features(T_cam_board: np.ndarray, t: TargetSpec, frontal_max_deg: float) -> PoseFeatures:
    c = transform(T_cam_board, board_center(t))[0]
    v = c / np.linalg.norm(c)
    n = T_cam_board[:3, 2].copy()
    if np.dot(n, v) > 0:          # orient normal toward the camera
        n = -n
    tilt = float(np.degrees(np.arccos(np.clip(-np.dot(n, v), -1, 1))))
    if tilt < frontal_max_deg:
        d = "frontal"
    else:
        # component of the normal perpendicular to the viewing ray, in camera x/y.
        # plane z = z0 - a*x (left edge farther) has camera-facing normal (-a, 0, -1),
        # so a negative x component means the left edge is farther away (same for y/top).
        p = n - np.dot(n, v) * v
        if abs(p[0]) >= abs(p[1]):
            d = "left_far" if p[0] < 0 else "right_far"
        else:
            d = "top_far" if p[1] < 0 else "bottom_far"
    return PoseFeatures(float(c[2]), tilt, d, c, n)


def project_board(cam: Camera, T_cam_board: np.ndarray, t: TargetSpec) -> np.ndarray:
    return cam.project(transform(T_cam_board, board_points(t)))


def min_square_px(corners_px: np.ndarray, t: TargetSpec) -> float:
    g = np.asarray(corners_px, float).reshape(t.inner_rows, t.inner_cols, 2)
    dx = np.linalg.norm(np.diff(g, axis=1), axis=2)
    dy = np.linalg.norm(np.diff(g, axis=0), axis=2)
    return float(min(dx.min(), dy.min()))
