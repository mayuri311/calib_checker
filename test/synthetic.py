"""Synthetic TartanRGBT-like rig for tests: ZED left (rectified pinhole) + 2 Bosons (equidistant)."""
import numpy as np
import yaml

from calib_checker.camera import make_T
from calib_checker.session import CoverageSpec, EdgeSpec, SensorSpec, Session, TargetSpec

ZED_K = [336.0, 336.0, 479.5, 299.5]
BOSON_L = dict(K=[421.2, 420.8, 317.6, 255.5], D=[-0.1105, 0.2048, -0.5774, 0.4512])
BOSON_R = dict(K=[417.4, 417.0, 314.3, 255.9], D=[-0.10, 0.18, -0.50, 0.40])


def T_from_xyz(x, y, z):
    return make_T(np.eye(3), [x, y, z])


def write_kalibr(path):
    # T_cn_cnm1 maps cam(n-1) coordinates into cam n. Thermal left sits 4 cm right / 3 cm below ZED.
    T_zed_tl = T_from_xyz(0.04, 0.03, 0.0)
    T_tr_tl = T_from_xyz(-0.12, 0.0, 0.0)          # thermal right sits 12 cm right of thermal left
    d = {
        "cam0": dict(camera_model="pinhole", intrinsics=ZED_K, distortion_model="none",
                     distortion_coeffs=[], resolution=[960, 600], rostopic="/zed/zed_node/left/image_rect_color"),
        "cam1": dict(camera_model="pinhole", intrinsics=BOSON_L["K"], distortion_model="equidistant",
                     distortion_coeffs=BOSON_L["D"], resolution=[640, 512], rostopic="/thermal_left/image",
                     T_cn_cnm1=np.linalg.inv(T_zed_tl).tolist()),
        "cam2": dict(camera_model="pinhole", intrinsics=BOSON_R["K"], distortion_model="equidistant",
                     distortion_coeffs=BOSON_R["D"], resolution=[640, 512], rostopic="/thermal_right/image",
                     T_cn_cnm1=T_tr_tl.tolist()),
    }
    with open(path, "w") as f:
        yaml.safe_dump(d, f)
    return path


def make_session(kalibr_path, **cov):
    return Session(
        name="test",
        target=TargetSpec(6, 8, 0.05, 0.03),
        kalibr_yaml=str(kalibr_path),
        reference="zed_left",
        sensors=[SensorSpec("zed_left", "/zed/zed_node/left/image_rect_color"),
                 SensorSpec("thermal_left", "/thermal_left/image", ffc_topic="/thermal_left/ffc_status"),
                 SensorSpec("thermal_right", "/thermal_right/image", ffc_topic="/thermal_right/ffc_status")],
        edges=[EdgeSpec("zed_left", "thermal_left"), EdgeSpec("zed_left", "thermal_right"),
               EdgeSpec("thermal_left", "thermal_right")],
        coverage=CoverageSpec(**cov),
    )


def render(cam, T_cam_board, target, dtype=np.uint8, ss=3, rng=None):
    """Ray-cast renderer for any camera model (fisheye included), supersampled ss x ss."""
    import cv2
    h, w = cam.height, cam.width
    ys, xs = np.mgrid[0:h * ss, 0:w * ss]
    px = np.c_[(xs.ravel() + 0.5) / ss - 0.5, (ys.ravel() + 0.5) / ss - 0.5]
    rays = cam.rays(px)
    R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
    n = R[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (n @ t) / (rays @ n)
    P = rays * lam[:, None]
    B = (P - t) @ R                                # board coordinates
    u = np.floor(B[:, 0] / target.square_m + 1).astype(int)   # square index, first inner corner at 1
    v = np.floor(B[:, 1] / target.square_m + 1).astype(int)
    inside = (lam > 0) & (u >= 0) & (v >= 0) & (u <= target.inner_cols) & (v <= target.inner_rows)
    border = (lam > 0) & ~inside & (B[:, 0] > -target.square_m - target.border_m) & \
        (B[:, 1] > -target.square_m - target.border_m) & \
        (B[:, 0] < target.inner_cols * target.square_m + target.border_m) & \
        (B[:, 1] < target.inner_rows * target.square_m + target.border_m)
    img = np.full(len(px), 128.0)
    img[border] = 235
    img[inside] = np.where((u[inside] + v[inside]) % 2 == 0, 20, 235)
    img = img.reshape(h * ss, w * ss)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if rng is not None:
        img = img + rng.normal(0, 1.0, img.shape)
    if dtype == np.uint16:
        return np.clip(img * 8 + 21000, 0, 65535).astype(np.uint16)   # Y16-like narrow range
    return np.clip(img, 0, 255).astype(np.uint8)
