# Plan: `calib_checker` — live static-hold guidance & online view selection (TartanRGBT)

## Context
Cross-modal extrinsics (ZED X ↔ 2× Boson 640+, ZED ↔ Prophesee) need static holds that cover the narrow thermal/event FOVs across depth and tilt — enough diversity, minimal recording time, no redundant holds. Holds are static so that the remaining, not-yet-resolved temporal-offset jitter between sensors cannot become a confounding variable (a static board looks identical at t and t+δ). Today there is no live feedback on whether a hold is good/new or where to go next, the Foxglove path is laggy (driver → render topics → bridge), and view selection happens offline with cell+depth binning.

Outcome: one ROS2 node on the Orin that reads the existing driver topics read-only, shows per-sensor views with a ghost "put the board here" quad, detects static holds, accepts only holds that add pose diversity (online selection — the session ends with the selected set already decided), and stops you when coverage is complete. Scope boundaries:
- Frame-based, target-based only: every sensor is an `sensor_msgs/Image` topic with a checkerboard in it (event frames arrive already rendered).
- Method-agnostic: coverage/diversity is the objective, not a specific solver's uncertainty. Solver (OpenCV / simple-evrgb-cal) stays downstream.
- No time sync / frame correspondence logic: the tool emits hold windows; pairing frames inside a window happens downstream.
- Intrinsics come from one consolidated Kalibr YAML (user provides later).

## Repo facts this relies on (pinned submodule commits)
- ZED (`src/zed-ros2-wrapper` @24d219b): `/zed/zed_node/left/image_rect_color` (+`/camera_info`), BGRA ≈960×600 (`pub_downscale_factor: 2.0`), 30 Hz, stamp = `TIME_REFERENCE::IMAGE`; `auto_exposure_gain: true` in `zed_wrapper/config/common_stereo.yaml:42`. Rect images → use pinhole K with zero distortion for the ZED (Kalibr entry for the ZED must match this rectified stream; flag mismatch).
- Boson (`src/flir_thermal_driver` @3e1b1b8, `flir_ros_sync`): `/thermal_{left,right}/image` = `16UC1` raw 640×512; `/thermal_X/ffc_status` (`std_msgs/Bool`, no header); driver runs FFC every `ffc_interval: 1` min (`config/tartan_rgbt_camera_params.yaml:16,32`); images publish **only when subscribed** (`publish_if_subscribed`, `src/flir_ros_sync.cpp:707`). Driver owns the serial port → checker never touches Boson serial.
- Container: `network_mode/ipc: host`, `ROS_DOMAIN_ID=70`, FastDDS with SHM profile (`docker/docker-compose.yaml`, `bag_recorder_pid/config/fastdds_optimized_qos.xml`). Tmux layout in `docker/tmux_config.yaml`.
- Recorder (`bag_recorder_pid`): records topics in `config/tartan_rgbt.yaml` incl. `/diagnostics`; started/stopped by GPIO button (`trigger_using_gpio.py` republishes switch state at 1 Hz). Checker never touches recording.
- Event camera: metavision_driver in another container on the same Orin; must share domain 70 / host network so its rendered frame topic is discoverable.

## ROS integration (what changes in the current stack)
New ament_python package `src/calib_checker/` in tartan_rgbt_ws, built with the rest (`colcon build --packages-select calib_checker`).

| Direction | Topic / service | Type | Notes |
|---|---|---|---|
| sub | `/zed/zed_node/left/image_rect_color` | Image (bgra8) | QoS best-effort, depth 1 (compatible with reliable pubs, never back-pressures driver) |
| sub | `/thermal_left/image`, `/thermal_right/image` | Image (16UC1) | Subscribing turns on raw publishing in the driver (extra SHM traffic only; recorder unaffected) |
| sub | `/thermal_left/ffc_status`, `/thermal_right/ffc_status` | Bool | Arrival-time-stamped in node; hold rejected if FFC true inside window |
| sub | event frame topic (configurable, from metavision container) | Image | Only needs to be a rendered frame |
| sub | `/tf_static` (optional) | TF | Prior extrinsic for ghost/footprint projection; else from session config |
| pub | `/calib_checker/holds` | `std_msgs/String` (JSON) v1 → custom `calib_checker_msgs/Hold` later | One msg per accepted/rejected hold: id, per-sensor `[t_start,t_end]` in each sensor's header-stamp domain, board pose per sensor, bins filled, reason |
| pub | `/diagnostics` | DiagnosticArray | Checker status (topic rates, detections, coverage %) — already in the recorder's topic list |
| srv | `/calib_checker/{accept,reject_last,skip_target,finish}` | std_srvs/Trigger | Called by the control page or `ros2 service call` |

Integration steps:
1. Add `/calib_checker/holds` to `bag_recorder_pid/config/tartan_rgbt.yaml` so hold windows live inside the same MCAP → downstream extraction cuts frames per hold without any culling.
2. Add a tmux window "Calib Checker" to `docker/tmux_config.yaml` that is **not** auto-run (commented command / `ros2 launch calib_checker checker.launch.py`), so normal data collection is untouched.
3. Pip deps in Dockerfile: `rerun-sdk` (pinned, aarch64 wheel), `nicegui` (config/control page). OpenCV already present via ROS.
4. Processing at 5–10 Hz on latest frame per sensor (keep-last 1), detection in a worker thread per sensor; no message_filters sync (by design).
5. Offline replay: same engine fed by a `rosbags` MCAP reader using the same topic names → produces the same holds output for already-recorded calib bags.

## Start-up config screen
NiceGUI page served by the node on the Orin (`http://192.168.50.202:8080`, open on laptop/tablet). Saves `~/calib_sessions/<name>/session.yaml`; "load last session" button.
- **Target**: checkerboard inner cols × rows, square size (m), board outer dims (for ghost quad), thermal-visible yes/no per sensor. (6×8 heated board default; note: a board with both square counts the same parity is 180°-ambiguous — irrelevant to coverage, flagged for downstream.)
- **Intrinsics**: path to consolidated Kalibr YAML; table maps each Kalibr `camN` (by `rostopic` field, editable) to a sensor; shows model (pinhole-radtan / pinhole-equi), resolution, and a red mark if incoming image size ≠ Kalibr resolution.
- **Sensors**: name, image topic, encoding (auto from first msg), role (wide reference / narrow).
- **Pairs to cover**: spanning-tree edges with checkboxes (zed_left–thermal_left, zed_left–thermal_right, thermal_left–thermal_right, zed_left–event).
- **Coverage spec**: working depth range (m) and # depth bins (default 3), tilt set (0°, 25°, 45° × 4 directions), image grid per narrow cam (default 3×3), min holds per bin (1), hold duration (default 1.5 s), stillness threshold (px).
- **Prior extrinsic**: from `/tf_static` or manual CAD values (used only to draw ghosts/footprints).
- "Start" → node begins; page switches to control view (see Viz).
Preflight on that page is deliberately small: topics alive + rate, intrinsics loaded and resolution-matched, ZED auto-exposure on → warning. Nothing else.

## Core engine (method-agnostic)
```
calib_checker/
  session.py      load/save session.yaml; Kalibr camchain parser → Camera(K, D, model)
  camera.py       per-model undistort/project (radtan, equidistant, pinhole-zero-D)
  detect.py       findChessboardCornersSB; 16UC1 → robust percentile normalize before detect
  pose.py         PnP per sensor → board pose, depth, tilt magnitude/direction, image-cell occupancy
  stillness.py    per-sensor corner-motion RMS over sliding window (each sensor judged on its own frames, no cross-sensor timing)
  holds.py        hold = all sensors of ≥1 enabled pair still & detected for ≥ hold_duration (coarse arrival-time overlap is enough because board is static); gates: completeness, FFC, frozen thermal frame, novelty
  coverage.py     bins per narrow sensor: image cell × depth bin × tilt bin; per-pair co-visibility counts
  planner.py      target pose generation + ordering (below)
  hints.py        live vs target pose → "back 20 cm, tilt top away 15°, shift left 8 cm" (payload frame)
  output.py       publish /calib_checker/holds, write session/holds.yaml, save .rrd
  ros_node.py, ui_config.py (NiceGUI), viz_rerun.py, replay.py
```
**Online selection (replaces offline culling):** a still window becomes an accepted hold only if it fills at least one unfilled coverage bin (or bin below min count) for some enabled pair; otherwise rejected live as "duplicate of hold #k". Accepted holds are the final set.

**Planner (diversity per unit time):**
1. At start, build a candidate pose library in the reference (ZED) frame: board placed so it lands in each narrow-cam image cell at each depth bin, at each tilt; prune poses not fully visible in both sensors of an edge, or outside depth range.
2. Greedy set cover: each candidate covers a set of bins across all enabled pairs (a pose seen by both Bosons covers bins for 3 edges) → minimal pose list (expect ~15–25).
3. Order by nearest-neighbour on a motion cost (translation + λ·rotation) so consecutive holds need small moves; re-plan whenever a hold fills bins (including ones you did "off-plan").
4. Done when every bin of every enabled pair meets min count → HUD says FINISHED.
Optional plug-in hook `Evaluator` (off by default) for method-specific indicators (e.g. quick stereoCalibrate reprojection) — kept outside the planner objective.

## Visualization principle: show the most restrictive modality first
For frame-based target calibration, a hold only counts if the full board is visible in **both** sensors of an edge. The narrower sensor (Boson 4.9 mm lens, 95° HFOV, narrower than the ZED X; the Prophesee when relevant) is therefore the constraint, so the viz is centred on it:
- For each edge, the "restrictive view" is the sensor whose FOV, projected at the current board depth, is the smaller one. For thermal_left–thermal_right it is their **overlap region**. This is computed automatically from the Kalibr intrinsics and the prior extrinsic, not hand-picked.
- Coverage bins, the ghost quad, the hints and the progress all live in that restrictive sensor's image. The ZED view is a small inset whose only job is to show where the restrictive FOV lands, so you can find the board.
- The large panel auto-switches to the edge with the lowest coverage, or you can pin one on the control page.

## Data format: what goes from the Orin to the Rerun viewer
Detection, PnP and stillness all run on the Orin on the **raw** ROS data. The viewer only gets display copies, so lossy compression never affects any calibration decision.

| Stream | Arrives in node (ROS) | Sent to Rerun | Approx. size/rate |
|---|---|---|---|
| Restrictive sensor (e.g. `/thermal_left/image`) | `Image` 16UC1 640×512 (≈655 KB) | 16-bit → percentile normalize (p1–p99) → uint8 → colormap → `rr.EncodedImage` JPEG (q≈80), full resolution because it is the main view | ≈40–60 KB × 10 Hz |
| Other narrow sensors (e.g. `thermal_right`, event) | 16UC1 / mono8 | same pipeline, downscaled ½ | ≈15 KB × 5 Hz each |
| ZED inset | `Image` bgra8 ≈960×600 (≈2.3 MB) | BGRA → BGR → resize to ~480 px wide → `rr.EncodedImage` JPEG | ≈20 KB × 5 Hz |
| Corners | float32 Nx2 | `rr.Points2D` (+ per-point colors) | <1 KB |
| Ghost quad, FOV footprint, coverage cells | computed | `rr.LineStrips2D`, `rr.Boxes2D` | <1 KB |
| Frustums / camera models | Kalibr + prior extrinsic | `rr.Pinhole` + `rr.Transform3D`, logged once with `static=True` | once |
| Board poses (live, target, accepted) | PnP | `rr.Transform3D` + `rr.LineStrips3D` | <1 KB |
| HUD, coverage charts, stillness | computed | `rr.TextDocument` (markdown), `rr.BarChart`, `rr.Scalars` | <1 KB |

- **Total** ≈ 0.6–1 MB/s (≈5–9 Mbps), which is fine over the Orin's Wi-Fi or the reverse tunnel. Raw 16-bit is opt-in for bench debugging only: `rr.Image` uint16 with the viewer's colormap, ≈6.5 MB/s at 10 Hz.
- **Wire format:** the SDK turns each log call into Arrow record batches and streams them over Rerun's gRPC log protocol to the viewer. The `.rrd` session file is the same stream written to disk.
- **Timelines:** each image is logged with its own sensor's header stamp on `sensor_time`, plus `wall_time` and a `hold` index. Cross-sensor alignment is never implied, which matches the "no time sync in this tool" rule.

## Visualization — Rerun, in depth
**Process layout:** node logs with `rerun-sdk` and serves a gRPC stream from the Orin (`rr.serve_grpc()`; laptop connects with matching-version `rerun --connect`, or over the existing reverse tunnel). Pin SDK and viewer to the same version and verify exact API names at implementation time. Also tee to `session/<name>.rrd` for review.

**Single browser page (preferred):** NiceGUI control page embeds the Rerun **web viewer** (`@rerun-io/web-viewer` / `rr.serve_web_viewer`) pointed at the Orin's stream, so config + controls + viz are one tab on laptop or tablet. Rerun is view-only (no input widgets), which is why controls live in NiceGUI: Accept / Reject last / Skip target / Finish buttons → the Trigger services; keyboard shortcuts mirrored.

**Blueprint (sent from code with `rr.send_blueprint`):**
- Row 1 — a large `Spatial2DView` for the current restrictive sensor, with a small ZED inset and a thumbnail strip of the other narrow sensors (`/cams/thermal_left`, `/cams/thermal_right`, `/cams/event`, `/cams/zed_left`). Each entity subtree contains:
  - `image`: `EncodedImage` JPEG (downscaled to ~480 px wide, 10 Hz; thermal 16-bit → percentile-normalized → colormap before encode). Keeps link at a few Mbps vs raw BGRA/16-bit.
  - `corners`: `Points2D` (green = gate ok, amber = moving, red = incomplete).
  - `ghost`: `LineStrips2D` target board outline projected through Kalibr intrinsics + prior extrinsic; thick green when live pose within tolerance.
  - `coverage`: `Boxes2D` for the image cells, fill alpha ∝ count; or semi-transparent `Image` heat overlay.
  - On `zed_left` only, `fov/<narrow_cam>`: `LineStrips2D` boundary of each narrow camera's FOV projected at the live board depth (moves with parallax) — tells you where in the RGB the thermal/event can see.
- Row 2 left — `Spatial3DView` `/world`: `Transform3D` + `Pinhole` per sensor (frustums, image_plane_distance ≈ working depth); accepted hold boards as faded `LineStrips3D`; next target board highlighted; live board solid.
- Row 2 right — `TextDocumentView` `/hud` (markdown: current instruction, hold progress "■■■□ 1.1/1.5 s", last reject reason, holds n / planned N, coverage %), `BarChartView` `/coverage/depth` and `/coverage/tilt`, `TimeSeriesView` `/stillness/<sensor>` (corner RMS motion with threshold line).
- Timelines: `wall_time` for live, `hold` sequence index so you can scrub to any accepted hold afterwards.

**Why Rerun over alternatives:** bypasses ROS render topics entirely (overlays are vectors, images are pre-compressed), native 2D overlays + 3D frustums, `.rrd` replay for free. Alternatives considered: Foxglove via its standalone SDK websocket (skips foxglove_bridge, has interactive panels — viable fallback if Rerun embedding proves fiddly); custom canvas-only web page (most control, most work). Current foxglove_bridge setup stays as-is for everything else.

## Build order
1. `session.py` (Kalibr parser), `camera.py`, `detect.py`, `pose.py`, `coverage.py` + `replay.py` on an existing thermal-RGB calib MCAP → coverage report + hold windows offline.
2. `stillness.py`, `holds.py`, online novelty selection; compare selected holds vs current binning script output.
3. ROS node + topics/services + Rerun viz (no planner).
4. NiceGUI config/control page with embedded web viewer.
5. Planner + hints; event sensor via config only.

## Verification
- Unit: synthetic board poses through known K/D (radtan + equidistant) → PnP depth/tilt/cell bins correct; Kalibr YAML parse round-trip; stillness flags injected 0.5 px jitter; novelty rejects a repeated pose.
- Replay: run on a recorded calib MCAP; check hold windows match visible pauses, FFC windows rejected, selected set covers bins with fewer holds than existing culling.
- Live on Orin: `ros2 launch calib_checker checker.launch.py`; open page, configure, collect; confirm `/health/*/frequency` unchanged, recorder unaffected, `/calib_checker/holds` present in the MCAP when recording, planner reaches FINISHED in ~15–25 holds.

## Resolved / found during implementation
- Target: 6×8 inner corners, 50 mm squares (7×9 squares, so it's 180°-ambiguous). Corners are canonicalized for stillness; downstream resolves the true corner order.
- At the ZED's published 960×600 a 50 mm square falls under 8 px beyond ≈2 m, so the ZED (not the Boson) caps far depth. The config page suggests a feasible depth range.
- Still open: consolidated Kalibr YAML (expected camchain format; the ZED entry must describe the rectified stream), event-frame topic name.
