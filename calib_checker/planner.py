"""Next-hold planner: greedy set cover over a candidate pose library, ordered for short moves.

Candidates are placed in each edge's restrictive sensor (cell centre x depth bin x tilt),
checked for full-board visibility and detectable square size in both sensors of every
edge, and scored by how many still-missing bins they fill across all edges at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .camera import make_T, transform
from .coverage import EdgeCoverage, depth_bin_centers, observe
from .pose import board_center, board_points, min_square_px
from .session import Session, TargetSpec

Item = Tuple[str, str, object]      # (edge, "pose"|"cell", bin)


@dataclass
class Candidate:
    idx: int
    sensor: str                      # restrictive sensor it was generated in
    T_ref_board: np.ndarray
    items: Dict[str, Set[Item]] = field(default_factory=dict)   # edge -> items it can fill


def _outer_corners(t: TargetSpec) -> np.ndarray:
    s = t.square_m
    x1, y1 = t.inner_cols * s, t.inner_rows * s
    return np.array([[-s, -s, 0], [x1, -s, 0], [x1, y1, 0], [-s, y1, 0]], float)


def _frontal_R(v: np.ndarray) -> np.ndarray:
    z = v / np.linalg.norm(v)                 # board z points away from the camera
    x = np.array([1.0, 0, 0]) - z[0] * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.c_[x, y, z]


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(axis, float) * np.radians(deg))
    return R


class Planner:
    def __init__(self, session: Session, cams, T_ref: Dict[str, np.ndarray],
                 coverage: Dict[str, EdgeCoverage]):
        self.session, self.cams, self.T_ref, self.coverage = session, cams, T_ref, coverage
        self.skipped: Set[int] = set()
        self.candidates = self._generate()
        self._mark_unreachable()

    # ------------------------------------------------------------------ library
    def _generate(self) -> List[Candidate]:
        spec, t = self.session.coverage, self.session.target
        cands: List[Candidate] = []
        sensors = sorted({c.geom.restrictive for c in self.coverage.values()})
        for r in sensors:
            if r not in self.T_ref:
                continue
            cam = self.cams[r]
            for gy in range(spec.grid_rows):
                for gx in range(spec.grid_cols):
                    px = np.array([[(gx + 0.5) * cam.width / spec.grid_cols,
                                    (gy + 0.5) * cam.height / spec.grid_rows]])
                    ray = cam.rays(px)[0]
                    for d in depth_bin_centers(spec):
                        c = ray * d / ray[2]
                        R0 = _frontal_R(c)
                        rots = [R0]
                        for L in spec.tilt_levels_deg:
                            for axis in (R0[:, 0], R0[:, 1]):
                                rots += [_rot(axis, L) @ R0, _rot(axis, -L) @ R0]
                        for R in rots:
                            T_r_board = make_T(R, c - R @ board_center(t))
                            T_ref_board = self.T_ref[r] @ T_r_board
                            cand = Candidate(len(cands), r, T_ref_board)
                            self._evaluate(cand)
                            if cand.items:
                                cands.append(cand)
        return cands

    def _T_s_board(self, s: str, T_ref_board: np.ndarray) -> Optional[np.ndarray]:
        if s not in self.T_ref:
            return None
        return np.linalg.inv(self.T_ref[s]) @ T_ref_board

    def _evaluate(self, cand: Candidate) -> None:
        spec, t = self.session.coverage, self.session.target
        outer, inner = _outer_corners(t), board_points(t)
        for name, cov in self.coverage.items():
            g = cov.geom
            obs = {}
            for s in (g.restrictive, g.other):
                T = self._T_s_board(s, cand.T_ref_board)
                if T is None:
                    break
                cam = self.cams[s]
                if not cam.in_image(cam.project(transform(T, outer)), spec.edge_margin_px).all():
                    break
                px = cam.project(transform(T, inner))
                if not np.isfinite(px).all() or min_square_px(px, t) < spec.min_square_px:
                    break
                obs[s] = (T, px)
            else:
                T_r, px_r = obs[g.restrictive]
                bins, _ = observe(self.cams[g.restrictive], px_r, T_r, t, spec)
                items = {(name, "cell", c) for c in bins.cells}
                if bins.pose is not None:
                    items.add((name, "pose", bins.pose))
                cand.items[name] = items

    def _mark_unreachable(self) -> None:
        for name, cov in self.coverage.items():
            reachable = {i[2] for c in self.candidates for i in c.items.get(name, ()) if i[1] == "pose"}
            cov.unreachable = {b for b in cov.all_pose_bins() if b not in reachable}

    # ------------------------------------------------------------------ planning
    def _missing(self) -> Set[Item]:
        m: Set[Item] = set()
        for name, cov in self.coverage.items():
            m |= {(name, "pose", b) for b in cov.missing_pose_bins()}
            m |= {(name, "cell", c) for c in cov.missing_cells()}
        return m

    @staticmethod
    def _gain(items: Set[Item]) -> float:
        return sum(1.0 if i[1] == "pose" else 0.5 for i in items)

    @staticmethod
    def _motion(Ta: np.ndarray, Tb: np.ndarray) -> float:
        dt = np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])
        dz = np.degrees(np.arccos(np.clip(abs(np.dot(Ta[:3, 2], Tb[:3, 2])), -1, 1)))
        return dt + dz / 30.0          # 30 deg of re-orientation ~ 1 m of travel

    def plan(self, current_T_ref_board: Optional[np.ndarray] = None, max_len: int = 64) -> List[Candidate]:
        missing = self._missing()
        pool = [c for c in self.candidates if c.idx not in self.skipped]
        order: List[Candidate] = []
        here = current_T_ref_board
        while missing and pool and len(order) < max_len:
            best, best_score = None, 0.0
            for c in pool:
                gain = self._gain(set().union(*c.items.values()) & missing)
                if gain <= 0:
                    continue
                score = gain / (1.0 + (self._motion(here, c.T_ref_board) if here is not None else 0.0))
                if score > best_score:
                    best, best_score = c, score
            if best is None:
                break
            order.append(best)
            missing -= set().union(*best.items.values())
            pool.remove(best)
            here = best.T_ref_board
        return order

    def skip(self, cand: Candidate) -> None:
        self.skipped.add(cand.idx)


def suggest_depth_range(session: Session, cams, T_ref: Dict[str, np.ndarray],
                        lo: float = 0.3, hi: float = 6.0, step: float = 0.05) -> Optional[Tuple[float, float, str]]:
    """Depth range (in the restrictive sensors) where a centred, face-on board is fully visible
    in both sensors of every enabled edge and its squares are big enough to detect everywhere.
    Returns (min_m, max_m, limiting sensor at max) or None."""
    from .coverage import edge_geoms
    spec, t = session.coverage, session.target
    outer, inner = _outer_corners(t), board_points(t)
    ok_depths, fail_by = [], {}
    for d in np.arange(lo, hi + 1e-9, step):
        good = True
        for g in edge_geoms(session, cams, T_ref):
            if g.restrictive not in T_ref:
                continue
            cam_r = cams[g.restrictive]
            c = cam_r.rays(np.array([[cam_r.width / 2, cam_r.height / 2]]))[0] * d
            R = _frontal_R(c)
            T_ref_board = T_ref[g.restrictive] @ make_T(R, c - R @ board_center(t))
            for s in (g.restrictive, g.other):
                if s not in T_ref:
                    continue
                T = np.linalg.inv(T_ref[s]) @ T_ref_board
                cam = cams[s]
                if not cam.in_image(cam.project(transform(T, outer)), spec.edge_margin_px).all():
                    good = False
                    break
                if min_square_px(cam.project(transform(T, inner)), t) < spec.min_square_px:
                    good = False
                    fail_by[round(float(d), 3)] = s
                    break
            if not good:
                break
        if good:
            ok_depths.append(float(d))
    if not ok_depths:
        return None
    beyond = [v for k, v in sorted(fail_by.items()) if k > ok_depths[-1]]
    return round(ok_depths[0], 2), round(ok_depths[-1], 2), (beyond[0] if beyond else "")
