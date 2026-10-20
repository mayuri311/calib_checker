"""Browser config + control page (NiceGUI), served from the Orin.

Before start: session form (target, Kalibr YAML, sensors, edges, coverage spec) with a
Kalibr camera table and a detectable-depth-range suggestion.
After start: operator controls + status on the left, embedded Rerun web viewer on the right.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

from nicegui import app, ui

from .session import (EdgeSpec, SensorSpec, Session, coerce_types, load_kalibr, load_session, resolve_cameras,
                      resolve_extrinsics, save_session)

LAST = Path("~/calib_sessions/last_session.yaml").expanduser()


def _default_session_path() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("calib_checker")) / "config" / "tartan_rgbt_session.yaml"
    except Exception:
        return Path(__file__).resolve().parent.parent / "config" / "tartan_rgbt_session.yaml"


DEFAULT = _default_session_path()


def _initial_session() -> Session:
    for p in (LAST, DEFAULT):
        if p.exists():
            try:
                return load_session(p)
            except Exception:
                pass
    return Session()


def run_ui(node, port: int = 8080) -> None:
    state = {"session": _initial_session()}

    @ui.page("/")
    def index():
        if node.runtime is None:
            config_page(node, state)
        else:
            control_page(node)

    ui.run(host="0.0.0.0", port=port, reload=False, show=False, title="calib_checker")


# ---------------------------------------------------------------------- config
def _num(label, obj, attr, **kw):
    return ui.number(label, value=getattr(obj, attr), **kw).bind_value(obj, attr).classes("w-36")


def config_page(node, state):
    s: Session = state["session"]
    ui.markdown("## calib_checker — session setup")

    with ui.row().classes("w-full items-start gap-8"):
        with ui.card():
            ui.label("Session").classes("text-lg")
            ui.input("name").bind_value(s, "name")
            ui.input("output dir").bind_value(s, "output_dir")
            ui.select(["board", "rig"], label="hints describe moving the").bind_value(s, "move_mode")
            ui.label("Checkerboard (frame-based target)").classes("text-lg mt-4")
            _num("inner corners x", s.target, "inner_cols", min=2, precision=0, format="%d")
            _num("inner corners y", s.target, "inner_rows", min=2, precision=0, format="%d")
            _num("square [m]", s.target, "square_m", step=0.005, format="%.3f")
            _num("border beyond squares [m]", s.target, "border_m", step=0.005, format="%.3f")

        with ui.card():
            ui.label("Coverage").classes("text-lg")
            c = s.coverage
            for attr, label in (("depth_min_m", "depth min [m]"), ("depth_max_m", "depth max [m]"),
                                ("depth_bins", "depth bins"), ("frontal_max_deg", "face-on below [°]"),
                                ("grid_cols", "grid cols"), ("grid_rows", "grid rows"),
                                ("min_per_bin", "holds per pose bin"), ("hold_s", "hold duration [s]"),
                                ("min_square_px", "min square [px]")):
                _num(label, c, attr)
            tilt = ui.input("tilt levels [°] (comma separated)",
                            value=", ".join(str(x) for x in c.tilt_levels_deg))
            tilt.on_value_change(lambda e: setattr(c, "tilt_levels_deg",
                                                   [float(x) for x in e.value.split(",") if x.strip()]))
            depth_msg = ui.label("")

            def suggest():
                try:
                    from .planner import suggest_depth_range
                    kal = load_kalibr(s.kalibr_yaml)
                    rng = suggest_depth_range(s, resolve_cameras(s, kal), resolve_extrinsics(s, kal))
                    if rng is None:
                        depth_msg.set_text("no depth where the board fits all enabled edges")
                        return
                    c.depth_min_m, c.depth_max_m = rng[0], rng[1]
                    depth_msg.set_text(f"set {rng[0]}–{rng[1]} m (far limit: {rng[2] or 'FOV'}; "
                                       "assumes Kalibr resolution = topic resolution)")
                except Exception as e:
                    depth_msg.set_text(f"error: {e}")
            ui.button("suggest depth range", on_click=suggest)

    with ui.card().classes("w-full"):
        ui.label("Intrinsics (consolidated Kalibr camchain YAML)").classes("text-lg")
        ui.input("path").bind_value(s, "kalibr_yaml").classes("w-full")
        kal_table = ui.markdown("")

        def show_kalibr():
            try:
                rows = ["| key | topic | model | resolution | fx fy cx cy |", "|---|---|---|---|---|"]
                for k in load_kalibr(s.kalibr_yaml):
                    K = k.camera.K
                    rows.append(f"| {k.key} | {k.rostopic} | {k.camera.model} | {k.camera.width}x{k.camera.height} | "
                                f"{K[0,0]:.1f} {K[1,1]:.1f} {K[0,2]:.1f} {K[1,2]:.1f} |")
                kal_table.set_content("\n".join(rows))
            except Exception as e:
                kal_table.set_content(f"**error:** {e}")
        ui.button("load", on_click=show_kalibr)

    @ui.refreshable
    def sensors_and_edges():
        with ui.card().classes("w-full"):
            ui.label("Sensors").classes("text-lg")
            ui.select([x.name for x in s.sensors], label="reference frame").bind_value(s, "reference")
            for sen in list(s.sensors):
                with ui.row().classes("items-center"):
                    ui.input("name").bind_value(sen, "name").classes("w-36")
                    ui.input("image topic").bind_value(sen, "topic").classes("w-80")
                    ui.input("kalibr cam (blank = match topic)").bind_value(sen, "kalibr_cam").classes("w-48")
                    ui.input("FFC Bool topic (thermal)").bind_value(sen, "ffc_topic").classes("w-64")
                    _num("still [px]", sen, "still_px", step=0.05)
                    ui.button(icon="delete", on_click=lambda x=sen: (s.sensors.remove(x), sensors_and_edges.refresh()))
            ui.button("add sensor", on_click=lambda: (s.sensors.append(SensorSpec("new", "/topic")),
                                                      sensors_and_edges.refresh()))
            ui.label("Edges to cover (spanning tree)").classes("text-lg mt-4")
            existing = {(e.a, e.b) for e in s.edges} | {(e.b, e.a) for e in s.edges}
            for a, b in itertools.combinations([x.name for x in s.sensors], 2):
                if (a, b) not in existing:
                    s.edges.append(EdgeSpec(a, b, enabled=False))
            names = {x.name for x in s.sensors}
            s.edges[:] = [e for e in s.edges if e.a in names and e.b in names]
            with ui.row():
                for e in s.edges:
                    ui.checkbox(e.name).bind_value(e, "enabled")
    sensors_and_edges()

    msg = ui.label("")

    def start():
        try:
            coerce_types(s)
            save_session(s, LAST)
            node.start(s)
            ui.navigate.reload()
        except Exception as e:
            msg.set_text(f"cannot start: {e}")
    with ui.row():
        ui.button("save", on_click=lambda: (save_session(s, LAST), msg.set_text(f"saved {LAST}")))
        ui.button("START", on_click=start).props("color=positive")


# ---------------------------------------------------------------------- control
def control_page(node):
    rt = node.runtime
    with ui.row().classes("w-full no-wrap items-start"):
        with ui.column().classes("w-96"):
            ui.markdown("## calib_checker")
            out = ui.label("")
            with ui.row():
                ui.button("reject last", on_click=lambda: out.set_text(rt.reject_last())).props("color=negative")
                ui.button("accept last", on_click=lambda: out.set_text(rt.accept_last()))
            with ui.row():
                ui.button("skip target", on_click=lambda: out.set_text(rt.skip_target()))
                ui.button("finish", on_click=lambda: out.set_text(rt.finish())).props("color=positive")
            edges = ["auto"] + [e.name for e in rt.session.enabled_edges]
            ui.select(edges, value="auto", label="focus edge",
                      on_change=lambda e: out.set_text(rt.pin_edge(None if e.value == "auto" else e.value)))
            hud = ui.markdown("")
            ui.timer(0.5, lambda: hud.set_content(rt.hud_markdown(
                rt.engine.edge_progress(time.monotonic()) if rt.engine else None)))
        url = rt.viz.web_url if rt.viz else None
        if url:
            ui.element("iframe").props(f'src="{url}"').style("flex:1; height:92vh; border:0")
        else:
            ui.label("Rerun viewer starts once every sensor topic has delivered a frame — reload then.")
