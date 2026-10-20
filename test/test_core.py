import cv2
import numpy as np
import pytest

from calib_checker.camera import Camera, make_T, transform
from calib_checker.detect import detect_checkerboard, to_gray8
from calib_checker.engine import Engine
from calib_checker.hints import make_hint
from calib_checker.planner import Planner, _frontal_R, _rot
from calib_checker.pose import board_center, board_points, pose_features, project_board, solve_pose
from calib_checker.session import (chain_extrinsics, load_kalibr, load_session, resolve_cameras,
                                   resolve_extrinsics, save_session)

from synthetic import make_session, write_kalibr


@pytest.fixture
def rig(tmp_path):
    kp = write_kalibr(tmp_path / "camchain.yaml")
    s = make_session(kp)
    kal = load_kalibr(kp)
    return s, resolve_cameras(s, kal), resolve_extrinsics(s, kal)


def board_pose(cam_dist=1.5, tilt_axis=None, tilt_deg=0.0, offset=(0, 0)):
    from calib_checker.session import TargetSpec
    t = TargetSpec(6, 8, 0.05, 0.03)
    c = np.array([offset[0], offset[1], cam_dist])
    R = _frontal_R(c)
    if tilt_axis is not None:
        R = _rot(R[:, tilt_axis], tilt_deg) @ R
    return make_T(R, c - R @ board_center(t)), t


# ---------------------------------------------------------------- camera / kalibr
@pytest.mark.parametrize("model,D", [("radtan", [0.1, -0.05, 0.001, 0.002]),
                                     ("equidistant", [-0.11, 0.2, -0.57, 0.45]),
                                     ("none", [])])
def test_project_normalize_roundtrip(model, D):
    cam = Camera([[420, 0, 320], [0, 420, 256], [0, 0, 1]], D, model, (640, 512))
    P = np.c_[np.random.uniform(-0.6, 0.6, (50, 2)), np.ones(50)] * 2.0
    px = cam.project(P)
    n = cam.normalize(px)
    assert np.allclose(n, P[:, :2] / P[:, 2:], atol=1e-5)


def test_scaled_camera_matches_downscale():
    cam = Camera([[672, 0, 959.5], [0, 672, 599.5], [0, 0, 1]], [], "none", (1920, 1200))
    half = cam.scaled((960, 600))
    # pixel-centre convention: the image centre stays the image centre
    assert np.isclose(half.K[0, 2], 479.5) and np.isclose(half.K[1, 2], 299.5) and np.isclose(half.K[0, 0], 336)


def test_kalibr_chain(rig, tmp_path):
    s, cams, T_ref = rig
    assert cams["thermal_left"].model == "equidistant"
    assert np.allclose(T_ref["thermal_left"][:3, 3], [0.04, 0.03, 0.0])
    assert np.allclose(T_ref["thermal_right"][:3, 3], [0.16, 0.03, 0.0])
    save_session(s, tmp_path / "s.yaml")
    s2 = load_session(tmp_path / "s.yaml")
    assert s2.target.inner_cols == 6 and len(s2.edges) == 3


# ---------------------------------------------------------------- pose / detection
def test_pose_roundtrip_and_tilt_labels(rig):
    s, cams, _ = rig
    cam = cams["thermal_left"]
    T, t = board_pose(1.5, tilt_axis=1, tilt_deg=25)
    px = project_board(cam, T, t)
    T2 = solve_pose(cam, px, t)
    assert np.allclose(T2[:3, 3], T[:3, 3], atol=1e-4)
    f = pose_features(T, t, 10)
    assert abs(f.tilt_deg - 25) < 1.0
    # verify the label by geometry: which physical edge is farther
    P = transform(T, board_points(t))
    left = P[P[:, 0] < np.median(P[:, 0])][:, 2].mean()
    right = P[P[:, 0] > np.median(P[:, 0])][:, 2].mean()
    assert f.tilt_dir == ("left_far" if left > right else "right_far")


def render_board(cam, T, t, dtype=np.uint8):
    """Pinhole-only renderer via homography (cam model must be 'none')."""
    sq = 40
    cols, rows = t.inner_cols + 1, t.inner_rows + 1
    tex = np.zeros(((rows + 2) * sq, (cols + 2) * sq), np.uint8) + 255
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                tex[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
    # first inner corner is the pixel boundary at index 2*sq, i.e. continuous coord 2*sq - 0.5
    o = -(2 * sq - 0.5) * t.square_m / sq
    A = np.array([[t.square_m / sq, 0, o], [0, t.square_m / sq, o], [0, 0, 1]])
    # render 4x supersampled then area-downsample, otherwise minification aliasing biases corners
    ss = 4
    big = cam.scaled((cam.width * ss, cam.height * ss))
    H = big.K @ np.c_[T[:3, 0], T[:3, 1], T[:3, 3]] @ A
    img = cv2.warpPerspective(tex, H, big.resolution, borderValue=128)
    img = cv2.resize(img, cam.resolution, interpolation=cv2.INTER_AREA)
    if dtype == np.uint16:
        return (img.astype(np.uint16) * 20 + 22000)
    return img


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_detection_accuracy(rig, dtype):
    s, cams, _ = rig
    cam = cams["zed_left"]
    T, t = board_pose(1.2, tilt_axis=0, tilt_deg=20, offset=(0.1, -0.05))
    img = render_board(cam, T, t, dtype)
    corners = detect_checkerboard(to_gray8(img), t.pattern)
    assert corners is not None
    gt = project_board(cam, T, t)
    # allow for the 180-degree ordering ambiguity of a 6x8 board
    err = min(np.abs(corners - gt).max(), np.abs(corners[::-1] - gt).max())
    assert err < 0.3


# ---------------------------------------------------------------- engine
def feed(engine, cams, T_ref, T_ref_board, t0, dur, t, sensors=None, jitter=0.0, dt=1 / 15):
    rec = None
    for k in range(int(dur / dt)):
        now = t0 + k * dt
        for name in sensors or cams:
            T = np.linalg.inv(T_ref[name]) @ T_ref_board
            px = project_board(cams[name], T, t)
            if not cams[name].in_image(px).all():
                engine.add_frame(name, now, int(now * 1e9), None)
                continue
            px = px + np.random.normal(0, jitter, px.shape)
            engine.add_frame(name, now, int(now * 1e9), px)
        r = engine.step(now)
        rec = rec or r
    return rec


def test_hold_accept_duplicate_ffc_and_motion(rig):
    s, cams, T_ref = rig
    e = Engine(s, cams, T_ref)
    T, t = board_pose(1.4)
    rec = feed(e, cams, T_ref, T, 0.0, 2.0, t, jitter=0.05)
    assert rec is not None and rec.accepted
    assert set(rec.windows) == {"zed_left", "thermal_left", "thermal_right"}
    # same pose again after moving away -> duplicate
    feed(e, cams, T_ref, board_pose(2.5)[0], 3.0, 0.3, t)
    rec2 = feed(e, cams, T_ref, T, 4.0, 2.0, t, jitter=0.05)
    assert rec2 is not None and not rec2.accepted
    assert all("duplicate" in r.reason for r in rec2.edges.values())
    # FFC during hold -> rejected
    T3, _ = board_pose(2.2, tilt_axis=1, tilt_deg=25)
    e.add_ffc("thermal_left", 7.0, True)
    rec3 = feed(e, cams, T_ref, T3, 6.5, 2.0, t)
    assert rec3 is not None
    assert "FFC" in rec3.edges["zed_left--thermal_left"].reason
    # moving board -> never a hold
    moving = None
    for k in range(40):
        Tm, _ = board_pose(1.0 + 0.02 * k)
        moving = moving or feed(e, cams, T_ref, Tm, 20 + k / 15, 1 / 15, t)
    assert moving is None
    # operator undo
    n_before = len(e.accepted_records)
    assert e.reject_last() is not None and len(e.accepted_records) == n_before - 1


def test_frozen_frames_are_not_still(rig):
    s, cams, T_ref = rig
    e = Engine(s, cams, T_ref)
    T, t = board_pose(1.4)
    for k in range(40):
        now = k / 15
        for name in cams:
            px = project_board(cams[name], np.linalg.inv(T_ref[name]) @ T, t)
            e.add_frame(name, now, int(now * 1e9), px, frozen=(name == "thermal_left"))
        rec = e.step(now)
        if rec:
            assert "zed_left--thermal_left" not in rec.edges


# ---------------------------------------------------------------- planner end-to-end
def test_plan_execution_reaches_full_coverage(rig):
    s, cams, T_ref = rig
    e = Engine(s, cams, T_ref)
    p = Planner(s, cams, T_ref, e.coverage)
    assert p.candidates
    plan = p.plan()
    assert 5 <= len(plan) <= 40
    t = s.target
    clock, holds = 0.0, 0
    for _ in range(80):
        plan = p.plan()
        if not plan:
            break
        cand = plan[0]
        feed(e, cams, T_ref, np.eye(4) @ board_pose(5.0)[0], clock, 0.2, t)   # break stillness
        clock += 0.5
        rec = feed(e, cams, T_ref, cand.T_ref_board, clock, s.coverage.hold_s + 0.3, t, jitter=0.03)
        clock += 3
        assert rec is not None and rec.accepted, rec and {k: v.reason for k, v in rec.edges.items()}
        holds += 1
    assert e.complete(), {k: (c.missing_pose_bins(), c.missing_cells()) for k, c in e.coverage.items()}
    print("holds to completion:", holds,
          {k: len(c.unreachable) for k, c in e.coverage.items()})


def test_hint_directions(rig):
    s, cams, _ = rig
    live, t = board_pose(1.5, offset=(0.0, 0.0))
    target, _ = board_pose(2.0, offset=(0.3, 0.0))
    h = make_hint(s, live, target)
    assert not h.on_target and "right" in h.lines[0] and "away" in h.lines[0]
    assert make_hint(s, target, target).on_target
