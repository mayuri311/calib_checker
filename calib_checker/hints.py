"""Turn live-vs-target board pose into short operator instructions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .pose import PoseFeatures, pose_features
from .session import Session


@dataclass
class Hint:
    on_target: bool
    lines: List[str]
    d_center_m: float
    d_normal_deg: float


def _dir_text(tilt_dir: str) -> str:
    return {"left_far": "LEFT edge away", "right_far": "RIGHT edge away",
            "top_far": "TOP edge away", "bottom_far": "BOTTOM edge away",
            "frontal": "face-on"}.get(tilt_dir, tilt_dir)


def make_hint(session: Session, T_cam_live: Optional[np.ndarray], T_cam_target: np.ndarray) -> Hint:
    """Both poses in the same (restrictive) camera frame. x right, y down, z forward."""
    spec, t = session.coverage, session.target
    tgt = pose_features(T_cam_target, t, spec.frontal_max_deg)
    want = f"target: {tgt.depth_m:.2f} m, " + (
        "face-on" if tgt.tilt_dir == "frontal" else f"tilt {tgt.tilt_deg:.0f}° {_dir_text(tgt.tilt_dir)}")
    if T_cam_live is None:
        return Hint(False, ["board not detected in this view", want], float("nan"), float("nan"))
    live: PoseFeatures = pose_features(T_cam_live, t, spec.frontal_max_deg)
    d = tgt.center_cam - live.center_cam
    if session.move_mode == "rig":
        d = -d                      # moving the payload moves the board the opposite way in the image
        who = "move rig"
    else:
        who = "move board"
    words = []
    for val, pos, neg in ((d[0], "right", "left"), (d[1], "down", "up"), (d[2], "away", "closer")):
        if abs(val) >= 0.03:
            words.append(f"{pos if val > 0 else neg} {abs(val) * 100:.0f} cm")
    dn = float(np.degrees(np.arccos(np.clip(np.dot(live.normal_cam, tgt.normal_cam), -1, 1))))
    lines = []
    if words:
        lines.append(f"{who}: " + ", ".join(words))
    if dn >= 8.0:
        lines.append(f"rotate board: now {live.tilt_deg:.0f}° {_dir_text(live.tilt_dir)} -> "
                     + ("face-on" if tgt.tilt_dir == "frontal" else f"{tgt.tilt_deg:.0f}° {_dir_text(tgt.tilt_dir)}"))
    dc = float(np.linalg.norm(tgt.center_cam - live.center_cam))
    on = dc < max(0.05, 0.08 * tgt.depth_m) and dn < 8.0
    if on:
        lines = ["ON TARGET — hold still"]
    lines.append(want)
    return Hint(on, lines, dc, dn)
