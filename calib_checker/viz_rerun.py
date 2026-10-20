"""Rerun visualization, centred on the most restrictive sensor of the focused edge.

Data sent to the viewer (all display-only; detection runs on raw frames before this):
  /cams/<sensor>/image      EncodedImage JPEG (16-bit thermal -> percentile 8-bit -> colormap)
  /cams/<sensor>/corners    Points2D        /cams/<sensor>/ghost   LineStrips2D (target board)
  /cams/<sensor>/fov/<s>    LineStrips2D    footprint of restrictive sensor s at board depth
  /cams/<sensor>/cells      Boxes2D         coverage grid (restrictive sensors only)
  /world/...                Pinhole + Transform3D frustums, live/target/accepted boards
  /hud (markdown), /events (TextLog), /coverage (BarChart), /stillness/<sensor> (Scalars)
Overlays for a downscaled image are scaled by the same factor so they stay aligned.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .camera import Camera, transform
from .detect import to_gray8
from .pose import board_outline, board_points

GREEN, AMBER, RED, CYAN, GREY = [60, 220, 90], [255, 180, 0], [255, 60, 60], [0, 200, 255], [150, 150, 150]


def densify(poly: np.ndarray, n: int = 12) -> np.ndarray:
    """Insert points along each edge so straight board edges bend correctly under fisheye."""
    out = []
    for a, b in zip(poly[:-1], poly[1:]):
        for s in np.linspace(0, 1, n, endpoint=False):
            out.append(a + s * (b - a))
    out.append(poly[-1])
    return np.array(out)


class RerunViz:
    def __init__(self, cams: Dict[str, Camera], T_ref: Dict[str, np.ndarray], session_name: str,
                 rrd_path: Optional[str], grpc_port: Optional[int] = 9876, web_port: int = 9090,
                 host: str = "localhost", thermal_like: Optional[set] = None):
        self.cams, self.T_ref = cams, T_ref
        self.thermal_like = thermal_like or set()
        self.scale: Dict[str, float] = {s: 1.0 for s in cams}
        self.focus: Optional[str] = None
        rr.init("calib_checker", recording_id=session_name)
        sinks = []
        if grpc_port:
            sinks.append(rr.GrpcServerSink(port=grpc_port, server_memory_limit="512MiB"))
        if rrd_path:
            sinks.append(rr.FileSink(rrd_path))
        rr.set_sinks(*sinks)
        self.grpc_url = f"rerun+http://{host}:{grpc_port}/proxy" if grpc_port else None
        self.web_url = None
        if web_port and grpc_port:
            rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=self.grpc_url)
            from urllib.parse import quote
            self.web_url = f"http://{host}:{web_port}/?url={quote(self.grpc_url, safe='')}"
        self._log_static()

    # ------------------------------------------------------------------ layout
    def set_focus(self, sensor: str) -> None:
        if sensor == self.focus:
            return
        self.focus = sensor
        others = [s for s in self.cams if s != sensor]
        side = [rrb.Spatial2DView(origin=f"/cams/{s}", name=s) for s in others]
        bp = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"/cams/{sensor}", name=f"{sensor} — restrictive view"),
                    rrb.Vertical(*side),
                    column_shares=[3, 1]),
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/hud", name="instructions"),
                    rrb.Spatial3DView(origin="/world", name="rig + boards"),
                    rrb.Vertical(
                        rrb.BarChartView(origin="/coverage", name="coverage per edge"),
                        rrb.TimeSeriesView(origin="/stillness", name="corner motion [px]"),
                        rrb.TextLogView(origin="/events", name="holds")),
                    column_shares=[2, 2, 1.5]),
                row_shares=[3, 2]),
            collapse_panels=True)
        rr.send_blueprint(bp)

    def _log_static(self) -> None:
        rr.log("/world", rr.ViewCoordinates.RDF, static=True)
        for s, cam in self.cams.items():
            if s not in self.T_ref:
                continue
            T = self.T_ref[s]
            rr.log(f"/world/{s}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]), static=True)
            rr.log(f"/world/{s}", rr.Pinhole(image_from_camera=cam.K, resolution=list(cam.resolution),
                                             image_plane_distance=0.25), static=True)

    # ------------------------------------------------------------------ per-sensor streams
    def image(self, sensor: str, raw: np.ndarray, stamp_ns: int) -> None:
        scale = 1.0 if sensor == self.focus else 0.5
        self.scale[sensor] = scale
        g = raw
        if raw.dtype != np.uint8 or raw.ndim == 2:
            g = to_gray8(raw)
            if sensor in self.thermal_like:
                g = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
        elif raw.shape[2] == 4:
            g = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        if scale != 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 80])   # g is BGR or gray
        if not ok:
            return
        rr.set_time("sensor_time", timestamp=stamp_ns * 1e-9)
        rr.log(f"/cams/{sensor}/image", rr.EncodedImage(contents=jpg.tobytes(), media_type="image/jpeg"))

    def corners(self, sensor: str, corners: Optional[np.ndarray], still: bool, motion_px: float) -> None:
        k = self.scale.get(sensor, 1.0)
        if corners is None:
            rr.log(f"/cams/{sensor}/corners", rr.Clear(recursive=False))
        else:
            rr.log(f"/cams/{sensor}/corners", rr.Points2D(corners * k, radii=2.0 if k == 1 else 1.0,
                                                          colors=GREEN if still else AMBER))
        if np.isfinite(motion_px):
            rr.log(f"/stillness/{sensor}", rr.Scalars([motion_px]))

    # ------------------------------------------------------------------ overlays
    def ghost(self, T_ref_board: Optional[np.ndarray], target, on_target: bool) -> None:
        for s, cam in self.cams.items():
            path = f"/cams/{s}/ghost"
            if T_ref_board is None or s not in self.T_ref:
                rr.log(path, rr.Clear(recursive=False))
                continue
            T = np.linalg.inv(self.T_ref[s]) @ T_ref_board
            px = cam.project(transform(T, densify(board_outline(target))))
            if not np.isfinite(px).all():
                rr.log(path, rr.Clear(recursive=False))
                continue
            rr.log(path, rr.LineStrips2D([px * self.scale.get(s, 1.0)], radii=3.0 if on_target else 1.5,
                                         colors=GREEN if on_target else CYAN))

    def footprints(self, restrictive: List[str], depth_m: float) -> None:
        """Outline of each restrictive sensor's FOV, at the given depth, drawn in every other view."""
        for r in restrictive:
            if r not in self.T_ref:
                continue
            rays = self.cams[r].rays(self.cams[r].boundary_pixels(24))
            P_r = rays * (depth_m / rays[:, 2:3])
            P_r = np.vstack([P_r, P_r[:1]])
            for s, cam in self.cams.items():
                if s == r or s not in self.T_ref:
                    continue
                T_s_r = np.linalg.inv(self.T_ref[s]) @ self.T_ref[r]
                px = cam.project(transform(T_s_r, P_r))
                px = px[np.isfinite(px).all(axis=1)]
                rr.log(f"/cams/{s}/fov/{r}", rr.LineStrips2D([px * self.scale.get(s, 1.0)], colors=GREY,
                                                             radii=1.0, labels=f"{r} FOV @ {depth_m:.1f} m"))

    def cells(self, sensor: str, counts: Dict[int, int], cols: int, rows: int, min_count: int) -> None:
        cam, k = self.cams[sensor], self.scale.get(sensor, 1.0)
        w, h = cam.width / cols * k, cam.height / rows * k
        mins, colors, labels = [], [], []
        for i in range(cols * rows):
            mins.append([(i % cols) * w, (i // cols) * h])
            n = counts.get(i, 0)
            colors.append(GREEN if n >= min_count else RED)
            labels.append(str(n))
        rr.log(f"/cams/{sensor}/cells", rr.Boxes2D(mins=mins, sizes=[[w, h]] * len(mins), colors=colors,
                                                    labels=labels, radii=0.5))

    def boards_3d(self, live: Optional[np.ndarray], target: Optional[np.ndarray], accepted: List[np.ndarray],
                  t) -> None:
        outline = densify(board_outline(t), 2)
        for name, T, color in (("live", live, GREEN), ("target", target, CYAN)):
            if T is None:
                rr.log(f"/world/board_{name}", rr.Clear(recursive=False))
            else:
                rr.log(f"/world/board_{name}", rr.LineStrips3D([transform(T, outline)], colors=color, radii=0.004))
        if accepted:
            rr.log("/world/accepted", rr.LineStrips3D([transform(T, outline) for T in accepted],
                                                      colors=[120, 120, 255], radii=0.002))

    # ------------------------------------------------------------------ HUD
    def hud(self, markdown: str) -> None:
        rr.log("/hud", rr.TextDocument(markdown, media_type=rr.MediaType.MARKDOWN))

    def coverage(self, fractions: Dict[str, float]) -> None:
        rr.log("/coverage", rr.BarChart(np.array([100 * f for f in fractions.values()])))

    def event(self, text: str, ok: bool, hold_id: int) -> None:
        rr.set_time("hold", sequence=hold_id)
        rr.log("/events", rr.TextLog(text, level=rr.TextLogLevel.INFO if ok else rr.TextLogLevel.WARN))
