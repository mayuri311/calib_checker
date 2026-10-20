"""Coverage bookkeeping in the most restrictive sensor of each edge.

For frame-based target calibration a view only counts for an edge if the full board is
seen by both sensors, so the narrower sensor bounds what is achievable. All bins
(depth x tilt pose bins, image grid cells) are therefore defined in that sensor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .camera import Camera
from .pose import PoseFeatures, pose_features
from .session import CoverageSpec, EdgeSpec, Session, TargetSpec

PoseBin = Tuple[int, str]          # (depth bin index, tilt class)


@dataclass
class EdgeGeom:
    edge: EdgeSpec
    restrictive: str
    other: str
    T_r_o: Optional[np.ndarray]    # prior extrinsic restrictive <- other (None if unknown)

    @property
    def name(self) -> str:
        return self.edge.name


def edge_geoms(session: Session, cams: Dict[str, Camera], T_ref: Dict[str, np.ndarray]) -> List[EdgeGeom]:
    out = []
    for e in session.enabled_edges:
        a, b = e.a, e.b
        r, o = (a, b) if cams[a].solid_angle_proxy() <= cams[b].solid_angle_proxy() else (b, a)
        T = None
        if r in T_ref and o in T_ref:
            T = np.linalg.inv(T_ref[r]) @ T_ref[o]
        out.append(EdgeGeom(e, r, o, T))
    return out


def tilt_classes(spec: CoverageSpec) -> List[str]:
    dirs = ["left_far", "right_far", "top_far", "bottom_far"]
    return ["frontal"] + [f"{d}@{int(l)}" for l in spec.tilt_levels_deg for d in dirs]


def tilt_class(f: PoseFeatures, spec: CoverageSpec) -> Optional[str]:
    if f.tilt_dir == "frontal":
        return "frontal"
    levels = sorted(spec.tilt_levels_deg)
    # snap to the highest level the tilt reaches, allowing 5 deg slack
    reached = [l for l in levels if f.tilt_deg >= l - 5.0]
    if not reached:
        return None
    return f"{f.tilt_dir}@{int(reached[-1])}"


def depth_bin(depth: float, spec: CoverageSpec) -> Optional[int]:
    if not (spec.depth_min_m <= depth <= spec.depth_max_m):
        return None
    w = (spec.depth_max_m - spec.depth_min_m) / spec.depth_bins
    return min(int((depth - spec.depth_min_m) / w), spec.depth_bins - 1)


def depth_bin_centers(spec: CoverageSpec) -> List[float]:
    w = (spec.depth_max_m - spec.depth_min_m) / spec.depth_bins
    return [spec.depth_min_m + (i + 0.5) * w for i in range(spec.depth_bins)]


def cells_hit(cam: Camera, corners_px: np.ndarray, spec: CoverageSpec) -> FrozenSet[int]:
    px = np.asarray(corners_px, float).reshape(-1, 2)
    cx = np.clip((px[:, 0] / cam.width * spec.grid_cols).astype(int), 0, spec.grid_cols - 1)
    cy = np.clip((px[:, 1] / cam.height * spec.grid_rows).astype(int), 0, spec.grid_rows - 1)
    return frozenset((cy * spec.grid_cols + cx).tolist())


@dataclass(frozen=True)
class Bins:
    pose: Optional[PoseBin]
    cells: FrozenSet[int]


def observe(cam_r: Camera, corners_r: np.ndarray, T_r_board: np.ndarray,
            target: TargetSpec, spec: CoverageSpec) -> Tuple[Bins, PoseFeatures]:
    f = pose_features(T_r_board, target, spec.frontal_max_deg)
    db = depth_bin(f.depth_m, spec)
    tc = tilt_class(f, spec)
    pose = (db, tc) if db is not None and tc is not None else None
    return Bins(pose, cells_hit(cam_r, corners_r, spec)), f


@dataclass
class EdgeCoverage:
    geom: EdgeGeom
    spec: CoverageSpec
    pose_counts: Dict[PoseBin, int] = field(default_factory=dict)
    cell_counts: Dict[int, int] = field(default_factory=dict)
    accepted: List[PoseFeatures] = field(default_factory=list)   # restrictive-frame poses of accepted holds
    accepted_T: List[np.ndarray] = field(default_factory=list)   # T_restrictive_board of accepted holds (display)
    unreachable: set = field(default_factory=set)                 # pose bins the planner cannot place

    def all_pose_bins(self) -> List[PoseBin]:
        return [(d, t) for d in range(self.spec.depth_bins) for t in tilt_classes(self.spec)]

    def all_cells(self) -> List[int]:
        return list(range(self.spec.grid_cols * self.spec.grid_rows))

    def missing_pose_bins(self) -> List[PoseBin]:
        return [b for b in self.all_pose_bins()
                if b not in self.unreachable and self.pose_counts.get(b, 0) < self.spec.min_per_bin]

    def missing_cells(self) -> List[int]:
        return [c for c in self.all_cells() if self.cell_counts.get(c, 0) < self.spec.min_per_cell]

    def new_bins(self, bins: Bins) -> Tuple[List[PoseBin], List[int]]:
        p = [bins.pose] if bins.pose is not None and bins.pose in self.missing_pose_bins() else []
        missing_cells = set(self.missing_cells())
        return p, sorted(c for c in bins.cells if c in missing_cells)

    def duplicate_of(self, f: PoseFeatures) -> Optional[int]:
        # Compare board centre and plane normal only: invariant to the in-plane 180 deg
        # corner-order flip of a symmetric board, and in-plane rotation adds no diversity here.
        for i, g in enumerate(self.accepted):
            dt = np.linalg.norm(g.center_cam - f.center_cam)
            dr = np.degrees(np.arccos(np.clip(np.dot(g.normal_cam, f.normal_cam), -1, 1)))
            if dt < self.spec.dup_trans_m and dr < self.spec.dup_rot_deg:
                return i
        return None

    def add(self, bins: Bins, f: PoseFeatures, T_r_board: Optional[np.ndarray] = None) -> None:
        if bins.pose is not None:
            self.pose_counts[bins.pose] = self.pose_counts.get(bins.pose, 0) + 1
        for c in bins.cells:
            self.cell_counts[c] = self.cell_counts.get(c, 0) + 1
        self.accepted.append(f)
        if T_r_board is not None:
            self.accepted_T.append(T_r_board)

    def fraction(self) -> float:
        total = len([b for b in self.all_pose_bins() if b not in self.unreachable]) + len(self.all_cells())
        missing = len(self.missing_pose_bins()) + len(self.missing_cells())
        return 1.0 - missing / max(total, 1)

    def complete(self) -> bool:
        return not self.missing_pose_bins() and not self.missing_cells()
