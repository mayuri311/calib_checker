# calib_checker

Live guidance for collecting **static holds** for frame-based, target-based extrinsic calibration (thermal–RGB, thermal–thermal, event-frame–RGB). The tool tells you where to put the checkerboard next. It accepts a hold only when the hold adds pose diversity, and it tells you when coverage is complete. The accepted holds are the final view set, so no offline culling step is needed.

- **Static holds:** a still board looks the same at *t* and *t+δ*, so leftover timestamp jitter between sensors can't bias the result.
- **No time sync in this tool:** each sensor is judged on its own frames. The output is a stamp window per sensor per hold, and frames are paired downstream.
- **Method-agnostic:** the objective is coverage (depth × tilt pose bins + image-grid cells), measured in the **most restrictive sensor** of each edge. That sensor is the narrower FOV, because a view only counts if the full board is in both sensors.
- **Intrinsics:** taken from one consolidated Kalibr camchain YAML.

See [PLAN.md](PLAN.md) for the design rationale.

## How it fits into `tartan_rgbt_ws`

The checker is an extra, read-only ROS 2 node. It never reconfigures drivers, never opens the Boson serial port, and never touches `bag_recorder_pid`.

| | Topic / service | Notes |
|---|---|---|
| sub | `/zed/zed_node/left/image_rect_color` | BGRA, best-effort depth-1 QoS (never back-pressures the driver) |
| sub | `/thermal_{left,right}/image` | raw 16UC1. Subscribing makes the Boson driver publish raw frames (`publish_if_subscribed`); recording is unaffected |
| sub | `/thermal_{left,right}/ffc_status` | holds overlapping an FFC are rejected (the driver runs FFC every `ffc_interval: 1` min) |
| sub | event-frame topic (from the metavision container) | any rendered `sensor_msgs/Image`; same `ROS_DOMAIN_ID=70` |
| pub | `/calib_checker/holds` | `std_msgs/String`, one JSON per hold (accepted or not, reasons, per-sensor stamp windows) |
| pub | `/diagnostics` | status + coverage, already in the recorder's topic list |
| srv | `/calib_checker/{reject_last,accept_last,skip_target,finish}` | `std_srvs/Trigger` |

Integration steps on the Orin:

1. Clone into the workspace and build:
   ```bash
   cd /home/airlab/code/tartan_rgbt_ws/src && git clone https://github.com/mayuri311/calib_checker.git
   docker exec -it tartan_rgbt bash
   pip install "rerun-sdk==0.39.0" "nicegui==3.18.0"     # or add to docker/Dockerfile.l4t-humble
   cd /root/tartan_rgbt_ws && colcon build --packages-select calib_checker && source install/setup.bash
   ```
2. Optional: to keep hold windows inside the same MCAP, add `/calib_checker/holds` to `src/bag_recorder_pid/config/tartan_rgbt.yaml`.
3. Optional: add a tmux window to `docker/tmux_config.yaml` that does **not** auto-start the checker:
   ```yaml
     - window_name: Calib Checker
       panes:
         - shell_command:
           - source /root/tartan_rgbt_ws/install/setup.bash
           - echo "ros2 launch calib_checker checker.launch.py viz_host:=192.168.50.202"
   ```

## Use

```bash
ros2 launch calib_checker checker.launch.py viz_host:=192.168.50.202
```

Open `http://192.168.50.202:8080` on a laptop or tablet.

1. **Setup page:**
   - Enter the checkerboard: inner corners (6×8), square size (0.05 m), border.
   - Enter the Kalibr YAML path and press **load**. This shows a table of camera models and resolutions.
   - Fill in sensors and topics, and tick the edges to cover.
   - **Suggest depth range** finds where the board is fully visible and detectable in every enabled edge.
   - Press **START**.
2. **Control page:**
   - Instructions and coverage appear on the left, with the embedded Rerun viewer on the right.
   - The big panel is the **restrictive view** of the least-covered edge. The cyan outline is where the board should go and turns green when you're on target.
   - Hold still for `hold_s`. The hold is captured automatically and accepted only if it fills missing bins.
   - Buttons: **reject last** / **accept last** (operator override), **skip target**, **finish**.

Headless mode (control via services only): `ros2 launch calib_checker checker.launch.py session:=<session.yaml> headless:=true`.

Offline, run the same engine on an existing bag (no ROS needed, only `rosbags`):

```bash
calib_checker_replay --session config/tartan_rgbt_session.yaml --bag /logging/<bag_dir> --rrd
```

### Output

Each session writes `~/calib_sessions/<name>_<timestamp>/` containing:
- `session.yaml`
- `holds.yaml`: each hold's `accepted` flag, per-edge reason, pose bin, depth and tilt, and `windows: {sensor: [stamp_start_ns, stamp_end_ns]}`
- `session.rrd`: open it with `rerun session.rrd`

To build the solver input, take frames from each sensor inside the windows of accepted holds.

## What goes to the Rerun viewer

Detection, PnP and stillness all run on the raw ROS images on the Orin. The viewer only receives display copies.

| Entity | Archetype | Source / size |
|---|---|---|
| `/cams/<s>/image` | `EncodedImage` JPEG q80 | 16-bit thermal → p1–p99 stretch → inferno colormap. Full resolution for the focus sensor at 10 Hz (~50 KB), ½ resolution at 5 Hz for the others |
| `/cams/<s>/corners` | `Points2D` | green = still, amber = moving |
| `/cams/<s>/ghost` | `LineStrips2D` | target board outline projected through Kalibr intrinsics + prior extrinsic (edges densified so they bend correctly under fisheye) |
| `/cams/<s>/fov/<r>` | `LineStrips2D` | FOV boundary of restrictive sensor `r` at the live board depth |
| `/cams/<r>/cells` | `Boxes2D` | coverage grid, count per cell |
| `/world/*` | `Pinhole`+`Transform3D` (static), `LineStrips3D` | frustums; live / target / accepted boards |
| `/hud`, `/events`, `/coverage`, `/stillness/<s>` | `TextDocument`, `TextLog`, `BarChart`, `Scalars` | instructions, hold log, % per edge, corner motion |

Total is roughly 5–9 Mbps. The node runs a gRPC server on `:9876` and the web viewer on `:9090` (both from the `rerun-sdk` package), and tees everything to `session.rrd`. Timelines: `sensor_time` (each sensor's own header stamp), `log_time`, and `hold` (index).

## Notes

- **6×8 inner corners is 180°-ambiguous.** It's 7×9 squares, and both counts are odd. Corners are put into a canonical order for stillness and display only, so a downstream solver must resolve the true corner order.
- **ZED topic resolution:** the ZED topic is the rectified, 2×-downscaled stream. Give its Kalibr entry as rectified pinhole (`distortion_model: none`); a uniform resolution difference is rescaled automatically. At 960×600 a 50 mm square drops below 8 px beyond ≈2 m, which caps the usable depth.
- **Python version:** Rerun 0.39 warns that Python 3.10 (the Humble/Jetson default) support ends with 0.39. Pin `rerun-sdk==0.39.0` and use the same viewer version.

## Tests

```bash
pip install numpy opencv-python-headless pyyaml pytest rerun-sdk==0.39.0
python -m pytest -q --import-mode=importlib -o pythonpath=". test" test
```

The tests use a synthetic rig: a ZED-like pinhole camera plus two Boson-like equidistant cameras. They render 16-bit thermal frames and cover:
- intrinsics/Kalibr parsing
- pose and tilt labels
- detection accuracy
- hold acceptance, duplicate and FFC rejection
- planner-driven completion (12 holds for 3 edges)
- a full runtime run that writes `holds.yaml` and `.rrd`
