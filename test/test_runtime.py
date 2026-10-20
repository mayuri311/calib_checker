"""End-to-end without ROS: rendered images (pinhole ZED + fisheye Bosons, 16-bit thermal)
-> real detection -> runtime holds -> holds.yaml + .rrd."""
import numpy as np
import yaml

from calib_checker.runtime import Runtime
from calib_checker.session import save_session

from synthetic import make_session, render, write_kalibr


def test_runtime_end_to_end(tmp_path):
    kp = write_kalibr(tmp_path / "camchain.yaml")
    s = make_session(kp, depth_min_m=0.6, depth_max_m=2.0, hold_s=1.0)
    save_session(s, tmp_path / "session.yaml")
    from calib_checker.viz_rerun import RerunViz

    def viz_factory(cams, T_ref, out_dir, thermal_like):
        return RerunViz(cams, T_ref, "test", str(out_dir / "session.rrd"), grpc_port=None, web_port=0,
                        thermal_like=thermal_like)

    rt = Runtime(s, viz_factory, out_root=str(tmp_path / "out"))
    rng = np.random.default_rng(0)
    clock = 0.0

    def run(T_ref_board, dur):
        nonlocal clock
        recs = []
        base = {}
        for sen in s.sensors:
            cam = rt.cams.get(sen.name, rt._cams_kalibr[sen.name])
            T = np.linalg.inv(rt.T_ref[sen.name]) @ T_ref_board
            base[sen.name] = render(cam, T, s.target, np.uint16 if sen.ffc_topic else np.uint8)
        for _ in range(int(dur * 10)):
            for sen in s.sensors:
                b = base[sen.name]
                noise = rng.normal(0, 8.0 if b.dtype == np.uint16 else 1.0, b.shape)   # sensor noise -> fresh hash
                img = np.clip(b + noise, 0, np.iinfo(b.dtype).max).astype(b.dtype)
                rt.on_frame(sen.name, img, int(clock * 1e9), clock)
            r = rt.tick(clock)
            if r:
                recs.append(r)
            clock += 0.1
        return recs

    away = np.eye(4); away[2, 3] = 20.0
    # follow the planner for a few holds
    for _ in range(4):
        run(away, 0.3)
        assert rt.target is not None
        recs = run(rt.target.T_ref_board, 1.6)
        assert recs and recs[0].accepted, recs and rt.last_event
    st = rt.status_dict()
    assert st["holds_accepted"] == 4
    assert rt.hint is not None
    md = rt.hud_markdown({})
    assert "holds" in md
    rt.finish()
    data = yaml.safe_load(open(rt.out_dir / "holds.yaml"))
    assert len(data["holds"]) == 4
    assert set(data["holds"][0]["windows"]) == {"zed_left", "thermal_left", "thermal_right"}
    assert (rt.out_dir / "session.rrd").exists()
