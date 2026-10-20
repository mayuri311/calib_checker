"""Camera model with the distortion model kept consistent end-to-end.

radtan -> cv2 pinhole calls, equidistant -> cv2.fisheye calls, none -> zero distortion.
Points are always undistorted with the camera's own model; downstream math works in
normalized pinhole coordinates with zero distortion.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class Camera:
    MODELS = ("radtan", "equidistant", "none")

    def __init__(self, K, D, model: str, resolution: Tuple[int, int]):
        if model not in self.MODELS:
            raise ValueError(f"unknown distortion model {model}")
        self.K = np.asarray(K, float).reshape(3, 3)
        self.D = np.asarray(D, float).ravel()
        self.model = model
        self.width, self.height = int(resolution[0]), int(resolution[1])
        if model == "equidistant" and self.D.size != 4:
            raise ValueError("equidistant needs 4 coefficients")

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def scaled(self, resolution: Tuple[int, int]) -> "Camera":
        """Same camera at a uniformly rescaled resolution (e.g. ZED pub_downscale_factor)."""
        sx, sy = resolution[0] / self.width, resolution[1] / self.height
        if abs(sx - sy) > 1e-3:
            raise ValueError(f"non-uniform rescale {self.resolution} -> {resolution}")
        S = np.diag([sx, sy, 1.0])
        # pixel centers: x' + 0.5 = s (x + 0.5)
        S[0, 2], S[1, 2] = 0.5 * (sx - 1), 0.5 * (sy - 1)
        return Camera(S @ self.K, self.D, self.model, resolution)

    # ------------------------------------------------------------------ projection
    def project(self, P_cam: np.ndarray) -> np.ndarray:
        """Nx3 points in camera frame -> Nx2 pixels (NaN for points behind the camera)."""
        P = np.asarray(P_cam, float).reshape(-1, 3)
        out = np.full((len(P), 2), np.nan)
        front = P[:, 2] > 1e-6
        if not front.any():
            return out
        Pf = P[front].reshape(-1, 1, 3)
        z = np.zeros(3)
        if self.model == "equidistant":
            px, _ = cv2.fisheye.projectPoints(Pf, z, z, self.K, self.D)
        else:
            D = self.D if self.model == "radtan" else np.zeros(4)
            px, _ = cv2.projectPoints(Pf, z, z, self.K, D)
        out[front] = px.reshape(-1, 2)
        return out

    def normalize(self, px: np.ndarray) -> np.ndarray:
        """Nx2 pixels -> Nx2 undistorted normalized coordinates (z = 1)."""
        p = np.asarray(px, float).reshape(-1, 1, 2)
        if self.model == "equidistant":
            n = cv2.fisheye.undistortPoints(p, self.K, self.D)
        elif self.model == "radtan":
            n = cv2.undistortPoints(p, self.K, self.D)
        else:
            q = p.reshape(-1, 2)
            return np.linalg.solve(self.K, np.c_[q, np.ones(len(q))].T).T[:, :2]
        return n.reshape(-1, 2)

    def rays(self, px: np.ndarray) -> np.ndarray:
        n = self.normalize(px)
        return np.hstack([n, np.ones((len(n), 1))])

    def in_image(self, px: np.ndarray, margin: float = 0.0) -> np.ndarray:
        px = np.asarray(px, float).reshape(-1, 2)
        ok = np.isfinite(px).all(axis=1)
        x, y = px[:, 0], px[:, 1]
        with np.errstate(invalid="ignore"):
            ok &= (x >= margin) & (y >= margin) & (x <= self.width - 1 - margin) & (y <= self.height - 1 - margin)
        return ok

    def boundary_pixels(self, n_per_side: int = 16) -> np.ndarray:
        w, h = self.width - 1, self.height - 1
        t = np.linspace(0, 1, n_per_side, endpoint=False)
        top = np.c_[t * w, np.zeros_like(t)]
        right = np.c_[np.full_like(t, w), t * h]
        bottom = np.c_[(1 - t) * w, np.full_like(t, h)]
        left = np.c_[np.zeros_like(t), (1 - t) * h]
        return np.vstack([top, right, bottom, left])

    def solid_angle_proxy(self) -> float:
        """Area of the image boundary on the z=1 plane; used to rank FOV size."""
        n = self.normalize(self.boundary_pixels())
        x, y = n[:, 0], n[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def transform(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float).reshape(-1, 3)
    return P @ T[:3, :3].T + T[:3, 3]


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).ravel()
    return T


def rot_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))
