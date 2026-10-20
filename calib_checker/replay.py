"""Run the same online engine over a recorded MCAP/db3 bag (no ROS install needed).

    calib_checker_replay --session session.yaml --bag /logging/<bag_dir> [--rrd out.rrd] [--rate 15]

Bag log time stands in for arrival time. Frames are decimated to --rate per sensor to mimic
the live node's processing rate. Writes the usual session dir (session.yaml, holds.yaml).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .ros_io import image_to_numpy, stamp_ns
from .runtime import Runtime
from .session import load_session


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", default=None, help="output root (default: session.output_dir)")
    ap.add_argument("--rrd", action="store_true", help="also write session.rrd for the Rerun viewer")
    ap.add_argument("--rate", type=float, default=15.0, help="max processed frames/s per sensor")
    a = ap.parse_args(argv)

    from rosbags.highlevel import AnyReader

    session = load_session(a.session)
    topic_sensor = {s.topic: s.name for s in session.sensors}
    ffc_sensor = {s.ffc_topic: s.name for s in session.sensors if s.ffc_topic}

    viz_factory = None
    if a.rrd:
        from .viz_rerun import RerunViz

        def viz_factory(cams, T_ref, out_dir, thermal_like):
            return RerunViz(cams, T_ref, session.name, str(out_dir / "session.rrd"),
                            grpc_port=None, web_port=0, thermal_like=thermal_like)

    rt = Runtime(session, viz_factory, out_root=a.out)
    last_proc, next_tick = {}, None
    with AnyReader([Path(a.bag)]) as reader:
        conns = [c for c in reader.connections if c.topic in topic_sensor or c.topic in ffc_sensor]
        missing = set(topic_sensor) - {c.topic for c in conns}
        if missing:
            print(f"warning: topics not in bag: {sorted(missing)}")
        for conn, t_ns, raw in reader.messages(connections=conns):
            t = t_ns * 1e-9
            if conn.topic in ffc_sensor:
                rt.on_ffc(ffc_sensor[conn.topic], bool(reader.deserialize(raw, conn.msgtype).data), t)
            else:
                s = topic_sensor[conn.topic]
                if t - last_proc.get(s, -1e9) >= 1.0 / a.rate:
                    last_proc[s] = t
                    msg = reader.deserialize(raw, conn.msgtype)
                    rt.on_frame(s, image_to_numpy(msg), stamp_ns(msg.header), t)
            if next_tick is None:
                next_tick = t
            while t >= next_tick:
                rec = rt.tick(next_tick)
                if rec is not None:
                    print(rt.last_event)
                next_tick += 0.1
    print(rt.finish())
    st = rt.status_dict()
    print(f"accepted {st.get('holds_accepted', 0)} / {st.get('holds_total', 0)} holds; coverage {st.get('coverage')}")


if __name__ == "__main__":
    main()
