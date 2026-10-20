"""Static-hold detection and online hold selection.

Each sensor is judged on its own frames (no cross-sensor time sync): a sensor is "still"
while its board corners stay within `still_px` RMS of the latest detection. An edge is
ready once both of its sensors have been still for `hold_s`; the hold then becomes a
record with a stamp window per sensor. Downstream tools pair frames inside those windows.
A hold is accepted only if it fills coverage the session is still missing.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .camera import Camera
from .coverage import Bins, EdgeCoverage, EdgeGeom, edge_geoms, observe
from .pose import PoseFeatures, solve_pose
from .session import Session

FFC_PAD_S = 0.5
MAX_GAP_S = 0.5
BUFFER_S = 10.0


def corner_rms(a: np.ndarray, b: np.ndarray) -> float:
    """RMS corner displacement, invariant to a reversed (180 deg) corner order."""
    if a.shape != b.shape:
        return float("inf")
    d = np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))
    r = np.sqrt(np.mean(np.sum((a[::-1] - b) ** 2, axis=1)))
    return float(min(d, r))


@dataclass
class Frame:
    t: float                     # arrival time (monotonic, seconds)
    stamp_ns: int                # sensor header stamp
    corners: Optional[np.ndarray]
    frozen: bool = False


class SensorTrack:
    def __init__(self, name: str, cam: Camera, still_px: float):
        self.name, self.cam, self.still_px = name, cam, still_px
        self.frames: Deque[Frame] = deque()
        self.ffc_on: Deque[float] = deque()
        self.T_cam_board: Optional[np.ndarray] = None
        self.motion_px: float = float("nan")

    def add(self, f: Frame) -> None:
        self.frames.append(f)
        while self.frames and f.t - self.frames[0].t > BUFFER_S:
            self.frames.popleft()
        ref = self.frames[-1].corners
        if len(self.frames) > 1 and ref is not None and self.frames[-2].corners is not None:
            self.motion_px = corner_rms(self.frames[-2].corners, ref)
        else:
            self.motion_px = float("nan")

    def add_ffc(self, t: float, active: bool) -> None:
        if active:
            self.ffc_on.append(t)
        while self.ffc_on and t - self.ffc_on[0] > BUFFER_S:
            self.ffc_on.popleft()

    def still_since(self, now: float) -> Optional[float]:
        """Earliest arrival time from which every frame up to now matches the latest one."""
        if not self.frames:
            return None
        last = self.frames[-1]
        if last.corners is None or last.frozen or now - last.t > MAX_GAP_S:
            return None
        since = last.t
        prev_t = last.t
        for f in reversed(self.frames):
            if f.corners is None or f.frozen or prev_t - f.t > MAX_GAP_S:
                break
            if corner_rms(f.corners, last.corners) > self.still_px:
                break
            since, prev_t = f.t, f.t
        return since

    def stamp_window(self, t0: float, t1: float) -> Optional[Tuple[int, int]]:
        st = [f.stamp_ns for f in self.frames if t0 <= f.t <= t1 and f.corners is not None]
        return (min(st), max(st)) if st else None

    def ffc_between(self, t0: float, t1: float) -> bool:
        return any(t0 - FFC_PAD_S <= t <= t1 + FFC_PAD_S for t in self.ffc_on)


@dataclass
class EdgeResult:
    accepted: bool
    reason: str
    pose_bin: Optional[Tuple[int, str]] = None
    new_pose_bins: List = field(default_factory=list)
    new_cells: List[int] = field(default_factory=list)
    depth_m: float = float("nan")
    tilt_deg: float = float("nan")
    tilt_dir: str = ""
    duplicate_of: Optional[int] = None


@dataclass
class HoldRecord:
    id: int
    wall_time: float
    accepted: bool
    edges: Dict[str, EdgeResult]
    windows: Dict[str, Tuple[int, int]]        # sensor -> (stamp_start_ns, stamp_end_ns)
    forced: bool = False
    # internal: kept for re-building coverage, not serialized
    _obs: Dict[str, Tuple[Bins, PoseFeatures, np.ndarray]] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_obs", None)
        return d


class Engine:
    def __init__(self, session: Session, cams: Dict[str, Camera], T_ref: Dict[str, np.ndarray]):
        self.session, self.cams, self.T_ref = session, cams, T_ref
        self.tracks = {s.name: SensorTrack(s.name, cams[s.name], s.still_px) for s in session.sensors}
        self.geoms: List[EdgeGeom] = edge_geoms(session, cams, T_ref)
        self.coverage: Dict[str, EdgeCoverage] = {g.name: EdgeCoverage(g, session.coverage) for g in self.geoms}
        self.records: List[HoldRecord] = []
        self._captured: Dict[str, Tuple[float, float]] = {}   # edge -> still periods already used

    # ------------------------------------------------------------------ inputs
    def add_frame(self, sensor: str, t: float, stamp_ns: int, corners: Optional[np.ndarray],
                  frozen: bool = False) -> None:
        tr = self.tracks[sensor]
        tr.add(Frame(t, stamp_ns, corners, frozen))
        tr.T_cam_board = solve_pose(tr.cam, corners, self.session.target) if corners is not None else None

    def add_ffc(self, sensor: str, t: float, active: bool) -> None:
        self.tracks[sensor].add_ffc(t, active)

    # ------------------------------------------------------------------ hold logic
    def edge_progress(self, now: float) -> Dict[str, float]:
        """Seconds each edge has been jointly still (0 if not)."""
        out = {}
        for g in self.geoms:
            sa = self.tracks[g.restrictive].still_since(now)
            sb = self.tracks[g.other].still_since(now)
            out[g.name] = 0.0 if sa is None or sb is None else now - max(sa, sb)
        return out

    def step(self, now: Optional[float] = None) -> Optional[HoldRecord]:
        now = time.monotonic() if now is None else now
        hold_s = self.session.coverage.hold_s
        ready = []
        for g in self.geoms:
            sa = self.tracks[g.restrictive].still_since(now)
            sb = self.tracks[g.other].still_since(now)
            if sa is None or sb is None or now - max(sa, sb) < hold_s:
                continue
            if self._captured.get(g.name) == (sa, sb):
                continue
            ready.append((g, sa, sb))
        if not ready:
            return None
        rec = HoldRecord(len(self.records), time.time(), False, {}, {})
        for g, sa, sb in ready:
            self._captured[g.name] = (sa, sb)
            t0 = max(sa, sb)
            for s in (g.restrictive, g.other):
                w = self.tracks[s].stamp_window(t0, now)
                if w:
                    rec.windows[s] = w
            rec.edges[g.name] = self._evaluate(g, t0, now, rec)
        rec.accepted = any(r.accepted for r in rec.edges.values())
        self.records.append(rec)
        return rec

    def _evaluate(self, g: EdgeGeom, t0: float, t1: float, rec: HoldRecord) -> EdgeResult:
        tr = self.tracks[g.restrictive]
        for s in (g.restrictive, g.other):
            if self.tracks[s].ffc_between(t0, t1):
                return EdgeResult(False, f"FFC on {s} during hold")
        T = tr.T_cam_board
        corners = tr.frames[-1].corners
        if T is None or corners is None:
            return EdgeResult(False, f"no pose in {g.restrictive}")
        bins, f = observe(tr.cam, corners, T, self.session.target, self.session.coverage)
        rec._obs[g.name] = (bins, f, T)
        cov = self.coverage[g.name]
        new_p, new_c = cov.new_bins(bins)
        res = EdgeResult(False, "", bins.pose, new_p, new_c, f.depth_m, f.tilt_deg, f.tilt_dir)
        res.duplicate_of = cov.duplicate_of(f)
        if res.duplicate_of is not None:
            res.reason = f"duplicate of an earlier hold (pose #{res.duplicate_of})"
        elif not new_p and not new_c:
            res.reason = "no new coverage"
        else:
            res.accepted, res.reason = True, "new coverage"
            cov.add(bins, f, T)
        return res

    # ------------------------------------------------------------------ manual controls
    def _rebuild(self) -> None:
        for g in self.geoms:
            cov = self.coverage[g.name]
            unreachable = cov.unreachable
            self.coverage[g.name] = EdgeCoverage(g, self.session.coverage, unreachable=unreachable)
        for rec in self.records:
            for name, res in rec.edges.items():
                if res.accepted and name in rec._obs:
                    self.coverage[name].add(*rec._obs[name])

    def reject_last(self) -> Optional[int]:
        for rec in reversed(self.records):
            if rec.accepted:
                rec.accepted = False
                for res in rec.edges.values():
                    if res.accepted:
                        res.accepted, res.reason = False, "rejected by operator"
                self._rebuild()
                return rec.id
        return None

    def force_accept_last(self) -> Optional[int]:
        for rec in reversed(self.records):
            if not rec.accepted and rec._obs:
                rec.accepted = rec.forced = True
                for name in rec._obs:
                    rec.edges[name].accepted, rec.edges[name].reason = True, "accepted by operator"
                self._rebuild()
                return rec.id
        return None

    @property
    def accepted_records(self) -> List[HoldRecord]:
        return [r for r in self.records if r.accepted]

    def complete(self) -> bool:
        return all(c.complete() for c in self.coverage.values())
