"""ROS-free runtime shared by the live node and bag replay.

Feeds frames into the engine, re-plans after each hold, picks the focus (most restrictive
sensor of the least-covered edge), produces operator hints, drives the viz and writes the
session output directory:
    <output_dir>/<name>_<YYYYmmdd_HHMMSS>/session.yaml, holds.yaml, session.rrd
"""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import yaml

from .camera import Camera
from .coverage import EdgeGeom
from .detect import detect_checkerboard, to_gray8
from .engine import Engine, HoldRecord
from .hints import Hint, make_hint
from .planner import Candidate, Planner, suggest_depth_range
from .session import Session, load_kalibr, resolve_cameras, resolve_extrinsics, save_session

STATUS_WAITING, STATUS_RUNNING, STATUS_DONE = "waiting_for_topics", "running", "finished"


class Runtime:
    def __init__(self, session: Session, viz_factory: Optional[Callable] = None, out_root: Optional[str] = None,
                 on_hold: Optional[Callable[[HoldRecord], None]] = None):
        self.session = session
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(out_root or session.output_dir).expanduser() / f"{session.name}_{stamp}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        save_session(session, self.out_dir / "session.yaml")
        kal = load_kalibr(session.kalibr_yaml)
        self._cams_kalibr = resolve_cameras(session, kal)
        self.T_ref = resolve_extrinsics(session, kal)
        self.cams: Dict[str, Camera] = {}
        self.viz_factory, self.on_hold = viz_factory, on_hold
        self.viz = None
        self.engine: Optional[Engine] = None
        self.planner: Optional[Planner] = None
        self.plan: List[Candidate] = []
        self.target: Optional[Candidate] = None
        self.focus_edge: Optional[str] = None
        self.pinned_edge: Optional[str] = None
        self.hint: Optional[Hint] = None
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.status = STATUS_WAITING
        self.last_event = ""
        self.lock = threading.RLock()
        self._last_hash: Dict[str, bytes] = {}
        self._last_viz: Dict[str, float] = {}
        self.thermal_like = {s.name for s in session.sensors if s.ffc_topic}
        if session.target.rotationally_ambiguous:
            self.warnings.append(f"{session.target.inner_cols}x{session.target.inner_rows} board is 180° "
                                 "ambiguous: corner order must be disambiguated downstream")

    # ------------------------------------------------------------------ startup
    def _maybe_start(self) -> None:
        if self.engine is not None or len(self.cams) < len(self.session.sensors):
            return
        self.engine = Engine(self.session, self.cams, self.T_ref)
        missing = [s.name for s in self.session.sensors if s.name not in self.T_ref]
        if missing:
            self.warnings.append(f"no prior extrinsic for {missing}: planner/ghosts disabled for their edges")
        self.planner = Planner(self.session, self.cams, self.T_ref, self.engine.coverage)
        rng = suggest_depth_range(self.session, self.cams, self.T_ref)
        c = self.session.coverage
        if rng and (c.depth_min_m < rng[0] or c.depth_max_m > rng[1]):
            self.warnings.append(f"configured depth {c.depth_min_m}-{c.depth_max_m} m; detectable face-on range is "
                                 f"{rng[0]}-{rng[1]} m (far limit set by {rng[2] or 'FOV'})")
        if self.viz_factory:
            self.viz = self.viz_factory(self.cams, self.T_ref, self.out_dir, self.thermal_like)
        self._replan(None)
        self.status = STATUS_RUNNING

    def _register_camera(self, sensor: str, w: int, h: int) -> bool:
        cam = self._cams_kalibr[sensor]
        if (w, h) == cam.resolution:
            self.cams[sensor] = cam
            return True
        try:
            self.cams[sensor] = cam.scaled((w, h))
            self.warnings.append(f"{sensor}: image {w}x{h} vs Kalibr {cam.width}x{cam.height}; intrinsics rescaled")
            return True
        except ValueError as e:
            self.errors.append(f"{sensor}: {e} — fix Kalibr YAML or topic")
            return False

    # ------------------------------------------------------------------ inputs
    def on_frame(self, sensor: str, img: np.ndarray, stamp_ns: int, t: Optional[float] = None) -> None:
        t = time.monotonic() if t is None else t
        with self.lock:
            if sensor not in self.cams:
                if not self._register_camera(sensor, img.shape[1], img.shape[0]):
                    return
                self._maybe_start()
        h = hashlib.blake2b(np.ascontiguousarray(img[::7, ::7]).tobytes(), digest_size=8).digest()
        frozen = self._last_hash.get(sensor) == h
        self._last_hash[sensor] = h
        corners = detect_checkerboard(to_gray8(img), self.session.target.pattern)
        with self.lock:
            if self.engine is None:
                return
            self.engine.add_frame(sensor, t, stamp_ns, corners, frozen)
            tr = self.engine.tracks[sensor]
            still = tr.still_since(t) is not None
            motion = tr.motion_px
        if self.viz is not None:
            rate = 10.0 if sensor == self.viz.focus else 5.0
            if t - self._last_viz.get(sensor, -1e9) >= 1.0 / rate:
                self._last_viz[sensor] = t
                self.viz.image(sensor, img, stamp_ns)
                self.viz.corners(sensor, corners, still, motion)

    def on_ffc(self, sensor: str, active: bool, t: Optional[float] = None) -> None:
        with self.lock:
            if self.engine is not None:
                self.engine.add_ffc(sensor, time.monotonic() if t is None else t, active)

    # ------------------------------------------------------------------ periodic
    def tick(self, now: Optional[float] = None) -> Optional[HoldRecord]:
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.engine is None:
                return None
            rec = self.engine.step(now)
            if rec is not None:
                self._on_record(rec)
            self._update_focus_and_hint()
            if self.engine.complete() and self.status == STATUS_RUNNING:
                self.status = STATUS_DONE
                self.last_event = "coverage complete — you can stop"
            progress = self.engine.edge_progress(now)
        if self.viz is not None:
            self._draw(progress)
        return rec

    def _on_record(self, rec: HoldRecord) -> None:
        parts = [f"{k}: {'✓' if v.accepted else '✗'} {v.reason}" for k, v in rec.edges.items()]
        self.last_event = f"hold #{rec.id} {'ACCEPTED' if rec.accepted else 'rejected'} — " + "; ".join(parts)
        self._write_holds()
        if rec.accepted:
            self._replan(self._live_T_ref_board())
        if self.viz is not None:
            self.viz.event(self.last_event, rec.accepted, rec.id)
        if self.on_hold:
            self.on_hold(rec)

    def _replan(self, here) -> None:
        if self.planner is not None:
            self.plan = self.planner.plan(here)

    def _live_T_ref_board(self, sensor: Optional[str] = None) -> Optional[np.ndarray]:
        sensors = [sensor] if sensor else [s.name for s in self.session.sensors]
        for s in sensors:
            T = self.engine.tracks[s].T_cam_board
            if T is not None and s in self.T_ref:
                return self.T_ref[s] @ T
        return None

    def _geom(self, edge: str) -> EdgeGeom:
        return self.engine.coverage[edge].geom

    def _update_focus_and_hint(self) -> None:
        cov = self.engine.coverage
        if self.pinned_edge in cov:
            focus = self.pinned_edge
        else:
            open_edges = [e for e, c in cov.items() if not c.complete()] or list(cov)
            focus = min(open_edges, key=lambda e: cov[e].fraction())
        self.focus_edge = focus
        self.target = next((c for c in self.plan if focus in c.items), self.plan[0] if self.plan else None)
        if self.target is not None:
            # focus the view on the edge the target actually serves
            if focus not in self.target.items:
                self.focus_edge = focus = next(iter(self.target.items))
        r = self._geom(focus).restrictive
        if self.target is not None and r in self.T_ref:
            T_live = self.engine.tracks[r].T_cam_board
            T_tgt = np.linalg.inv(self.T_ref[r]) @ self.target.T_ref_board
            self.hint = make_hint(self.session, T_live, T_tgt)
        else:
            self.hint = None

    # ------------------------------------------------------------------ drawing
    def _draw(self, progress: Dict[str, float]) -> None:
        with self.lock:
            cov = self.engine.coverage
            focus = self.focus_edge
            r = self._geom(focus).restrictive
            self.viz.set_focus(r)
            on = bool(self.hint and self.hint.on_target)
            self.viz.ghost(self.target.T_ref_board if self.target else None, self.session.target, on)
            T_live = self.engine.tracks[r].T_cam_board
            depth = float(T_live[2, 3]) if T_live is not None else float(np.mean(
                [self.session.coverage.depth_min_m, self.session.coverage.depth_max_m]))
            restrictive = sorted({c.geom.restrictive for c in cov.values()})
            self.viz.footprints(restrictive, depth)
            sp = self.session.coverage
            for c in cov.values():
                if c.geom.restrictive == r and c.geom.name == focus:
                    self.viz.cells(r, c.cell_counts, sp.grid_cols, sp.grid_rows, sp.min_per_cell)
            accepted = [self.T_ref[c.geom.restrictive] @ T for c in cov.values()
                        if c.geom.restrictive in self.T_ref for T in c.accepted_T]
            self.viz.boards_3d(self._live_T_ref_board(), self.target.T_ref_board if self.target else None,
                               accepted, self.session.target)
            self.viz.coverage({e: c.fraction() for e, c in cov.items()})
            self.viz.hud(self.hud_markdown(progress))

    def hud_markdown(self, progress: Optional[Dict[str, float]] = None) -> str:
        with self.lock:
            if self.engine is None:
                return f"### Waiting for topics\nseen: {sorted(self.cams)}"
            hold_s = self.session.coverage.hold_s
            lines = []
            if self.status == STATUS_DONE:
                lines.append("## ✅ Coverage complete")
            if self.hint is not None:
                lines.append("## " + self.hint.lines[0])
                lines += [f"- {l}" for l in self.hint.lines[1:]]
            elif not self.plan:
                lines.append("## no planned pose (all reachable bins covered or no prior extrinsic)")
            if progress and self.focus_edge in progress:
                p = min(progress[self.focus_edge] / hold_s, 1.0)
                lines.append(f"\n**hold** {'■' * int(p * 10)}{'□' * (10 - int(p * 10))} "
                             f"{progress[self.focus_edge]:.1f}/{hold_s:.1f} s")
            acc = len(self.engine.accepted_records)
            lines.append(f"\n**holds** {acc} accepted / {len(self.engine.records)} total · "
                         f"**planned remaining** {len(self.plan)}")
            lines.append("\n| edge (restrictive) | done | missing pose bins | unreachable |\n|---|---|---|---|")
            for e, c in self.engine.coverage.items():
                miss = ", ".join(f"d{d}:{t}" for d, t in c.missing_pose_bins()[:4])
                more = len(c.missing_pose_bins()) - 4
                lines.append(f"| {e} ({c.geom.restrictive}) | {100 * c.fraction():.0f}% | "
                             f"{miss}{f' +{more}' if more > 0 else ''} | {len(c.unreachable)} |")
            if self.last_event:
                lines.append(f"\n**last:** {self.last_event}")
            for w in self.errors + self.warnings:
                lines.append(f"\n⚠ {w}")
            return "\n".join(lines)

    # ------------------------------------------------------------------ controls
    def reject_last(self) -> str:
        with self.lock:
            i = self.engine.reject_last() if self.engine else None
            self._after_control()
            return f"rejected hold #{i}" if i is not None else "nothing to reject"

    def accept_last(self) -> str:
        with self.lock:
            i = self.engine.force_accept_last() if self.engine else None
            self._after_control()
            return f"force-accepted hold #{i}" if i is not None else "no rejected hold with a pose"

    def skip_target(self) -> str:
        with self.lock:
            if self.planner is None or self.target is None:
                return "no target"
            self.planner.skip(self.target)
            self._replan(self._live_T_ref_board())
            return "skipped target"

    def pin_edge(self, edge: Optional[str]) -> str:
        with self.lock:
            self.pinned_edge = edge or None
            return f"focus pinned to {edge}" if edge else "focus: automatic"

    def finish(self) -> str:
        with self.lock:
            self.status = STATUS_DONE
            self._write_holds()
            return f"wrote {self.out_dir / 'holds.yaml'}"

    def _after_control(self) -> None:
        self._write_holds()
        if self.engine:
            if not self.engine.complete() and self.status == STATUS_DONE:
                self.status = STATUS_RUNNING
            self._replan(self._live_T_ref_board())

    def _write_holds(self) -> None:
        if self.engine is None:
            return
        data = {
            "session": self.session.name,
            "restrictive": {e: c.geom.restrictive for e, c in self.engine.coverage.items()},
            "coverage": {e: round(c.fraction(), 3) for e, c in self.engine.coverage.items()},
            "holds": [r.to_dict() for r in self.engine.records],
        }
        tmp = self.out_dir / "holds.yaml.tmp"
        with open(tmp, "w") as f:
            yaml.safe_dump(_plain(data), f, sort_keys=False)
        tmp.replace(self.out_dir / "holds.yaml")

    def status_dict(self) -> dict:
        with self.lock:
            d = {"status": self.status, "out_dir": str(self.out_dir), "warnings": self.warnings + self.errors,
                 "last_event": self.last_event, "focus_edge": self.focus_edge,
                 "hint": self.hint.lines if self.hint else []}
            if self.engine:
                d["coverage"] = {e: round(c.fraction(), 3) for e, c in self.engine.coverage.items()}
                d["holds_accepted"] = len(self.engine.accepted_records)
                d["holds_total"] = len(self.engine.records)
                d["planned_remaining"] = len(self.plan)
            return d


def _plain(x):
    """numpy/tuple -> yaml-safe python."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x
