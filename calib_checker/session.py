"""Session configuration: target, sensors, edges, coverage spec, Kalibr intrinsics."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from .camera import Camera


@dataclass
class TargetSpec:
    inner_cols: int = 6          # inner corners along board x
    inner_rows: int = 8          # inner corners along board y
    square_m: float = 0.05
    border_m: float = 0.05       # extra margin beyond the outer squares (for ghost outline)

    @property
    def pattern(self) -> Tuple[int, int]:
        return (self.inner_cols, self.inner_rows)

    @property
    def rotationally_ambiguous(self) -> bool:
        # Square counts are inner+1; a board whose square counts are both odd or both
        # even maps onto itself under a 180 deg in-plane rotation.
        return (self.inner_cols + 1) % 2 == (self.inner_rows + 1) % 2


@dataclass
class SensorSpec:
    name: str
    topic: str
    kalibr_cam: str = ""         # key in the Kalibr camchain (e.g. "cam1"); matched by rostopic if empty
    still_px: float = 0.3        # corner RMS jitter threshold for "static"
    ffc_topic: str = ""          # Bool topic; hold is rejected if FFC is active inside it


@dataclass
class EdgeSpec:
    a: str
    b: str
    enabled: bool = True

    @property
    def name(self) -> str:
        return f"{self.a}--{self.b}"


@dataclass
class CoverageSpec:
    depth_min_m: float = 0.8
    depth_max_m: float = 3.0
    depth_bins: int = 3
    frontal_max_deg: float = 10.0
    tilt_levels_deg: List[float] = field(default_factory=lambda: [25.0])
    grid_cols: int = 3
    grid_rows: int = 3
    min_per_bin: int = 1
    min_per_cell: int = 1
    hold_s: float = 1.5
    dup_trans_m: float = 0.05
    dup_rot_deg: float = 5.0
    min_square_px: float = 8.0   # planner: smallest apparent square for reliable detection
    edge_margin_px: float = 10.0


@dataclass
class Session:
    name: str = "session"
    target: TargetSpec = field(default_factory=TargetSpec)
    kalibr_yaml: str = ""
    reference: str = "zed_left"
    sensors: List[SensorSpec] = field(default_factory=list)
    edges: List[EdgeSpec] = field(default_factory=list)
    coverage: CoverageSpec = field(default_factory=CoverageSpec)
    # Prior extrinsics T_ref_sensor (4x4) used only for drawing ghosts/footprints and planning.
    # If empty they are composed from the Kalibr camchain T_cn_cnm1 entries.
    extrinsics_prior: Dict[str, List[List[float]]] = field(default_factory=dict)
    move_mode: str = "board"     # "board": hints say how to move the board; "rig": how to move the payload
    output_dir: str = "~/calib_sessions"

    def sensor(self, name: str) -> SensorSpec:
        for s in self.sensors:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def enabled_edges(self) -> List[EdgeSpec]:
        return [e for e in self.edges if e.enabled]


def _build(cls, d):
    if d is None:
        return cls()
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def session_from_dict(d: dict) -> Session:
    s = _build(Session, {k: v for k, v in d.items()
                         if k not in ("target", "sensors", "edges", "coverage")})
    s.target = _build(TargetSpec, d.get("target"))
    s.coverage = _build(CoverageSpec, d.get("coverage"))
    s.sensors = [_build(SensorSpec, x) for x in d.get("sensors", [])]
    s.edges = [_build(EdgeSpec, x) for x in d.get("edges", [])]
    return s


def coerce_types(s: Session) -> Session:
    """UI number widgets yield floats; restore the integer fields."""
    t, c = s.target, s.coverage
    t.inner_cols, t.inner_rows = int(t.inner_cols), int(t.inner_rows)
    for k in ("depth_bins", "grid_cols", "grid_rows", "min_per_bin", "min_per_cell"):
        setattr(c, k, int(getattr(c, k)))
    c.tilt_levels_deg = [float(x) for x in c.tilt_levels_deg]
    return s


def load_session(path: str | Path) -> Session:
    with open(Path(path).expanduser()) as f:
        return coerce_types(session_from_dict(yaml.safe_load(f) or {}))


def save_session(session: Session, path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(session), f, sort_keys=False)


# ---------------------------------------------------------------- Kalibr camchain

@dataclass
class KalibrCam:
    key: str
    camera: Camera
    rostopic: str
    T_cn_cnm1: Optional[np.ndarray]


def load_kalibr(path: str | Path) -> List[KalibrCam]:
    """Parse a Kalibr camchain-style YAML (cam0, cam1, ... entries)."""
    with open(Path(path).expanduser()) as f:
        d = yaml.safe_load(f) or {}
    cams = []
    for key in sorted((k for k in d if k.startswith("cam")), key=lambda k: int(k[3:] or 0)):
        c = d[key]
        if c.get("camera_model", "pinhole") != "pinhole":
            raise ValueError(f"{key}: only pinhole camera_model is supported, got {c['camera_model']}")
        fu, fv, pu, pv = c["intrinsics"]
        K = np.array([[fu, 0, pu], [0, fv, pv], [0, 0, 1]], float)
        model = {"radtan": "radtan", "equidistant": "equidistant", "none": "none"}[c.get("distortion_model", "none")]
        cam = Camera(K, np.asarray(c.get("distortion_coeffs", []), float), model, tuple(c["resolution"]))
        T = np.asarray(c["T_cn_cnm1"], float) if "T_cn_cnm1" in c else None
        cams.append(KalibrCam(key, cam, c.get("rostopic", ""), T))
    return cams


def resolve_cameras(session: Session, kalibr: List[KalibrCam]) -> Dict[str, Camera]:
    """Map each session sensor to its Kalibr camera (by kalibr_cam key, else by rostopic)."""
    by_key = {k.key: k for k in kalibr}
    by_topic = {k.rostopic: k for k in kalibr if k.rostopic}
    out = {}
    for s in session.sensors:
        kc = by_key.get(s.kalibr_cam) or by_topic.get(s.topic)
        if kc is None:
            raise KeyError(f"no Kalibr camera for sensor {s.name} (kalibr_cam={s.kalibr_cam!r}, topic={s.topic})")
        out[s.name] = kc.camera
    return out


def chain_extrinsics(kalibr: List[KalibrCam]) -> Dict[str, np.ndarray]:
    """Compose T_cn_cnm1 into T_cam0_camN for every camera in the chain."""
    out = {}
    T_cn_c0 = np.eye(4)
    for i, kc in enumerate(kalibr):
        if i > 0:
            if kc.T_cn_cnm1 is None:
                break
            T_cn_c0 = kc.T_cn_cnm1 @ T_cn_c0
        out[kc.key] = np.linalg.inv(T_cn_c0)
    return out


def resolve_extrinsics(session: Session, kalibr: List[KalibrCam]) -> Dict[str, np.ndarray]:
    """T_ref_sensor for every sensor; explicit session priors win over the Kalibr chain."""
    chain = chain_extrinsics(kalibr)
    by_topic = {k.rostopic: k.key for k in kalibr if k.rostopic}
    T_c0 = {}
    for s in session.sensors:
        key = s.kalibr_cam or by_topic.get(s.topic, "")
        if key in chain:
            T_c0[s.name] = chain[key]
    out = {}
    for s in session.sensors:
        if s.name in session.extrinsics_prior:
            out[s.name] = np.asarray(session.extrinsics_prior[s.name], float)
        elif s.name in T_c0 and session.reference in T_c0:
            out[s.name] = np.linalg.inv(T_c0[session.reference]) @ T_c0[s.name]
    if session.reference not in out:
        out[session.reference] = np.eye(4)
    return out
