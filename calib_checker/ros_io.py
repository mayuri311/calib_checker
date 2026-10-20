"""sensor_msgs/Image -> numpy without cv_bridge (works for rclpy and rosbags messages)."""
from __future__ import annotations

import numpy as np

_ENC = {
    "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1),
    "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1),
    "bgr8": (np.uint8, 3), "rgb8": (np.uint8, 3), "8UC3": (np.uint8, 3),
    "bgra8": (np.uint8, 4), "rgba8": (np.uint8, 4), "8UC4": (np.uint8, 4),
}


def image_to_numpy(msg) -> np.ndarray:
    enc = msg.encoding
    if enc not in _ENC:
        raise ValueError(f"unsupported encoding {enc}")
    dtype, ch = _ENC[enc]
    dt = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    buf = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, np.ndarray) else msg.data.tobytes(), np.uint8)
    rows = buf.reshape(msg.height, msg.step)[:, : msg.width * ch * dt.itemsize]
    img = rows.view(dt).reshape(msg.height, msg.width, ch) if ch > 1 else rows.view(dt).reshape(msg.height, msg.width)
    img = img.astype(dtype, copy=False)
    if enc in ("rgb8", "rgba8"):
        img = img[..., [2, 1, 0] + ([3] if ch == 4 else [])]
    return np.ascontiguousarray(img)


def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
