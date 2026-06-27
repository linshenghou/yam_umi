# sim_teleop

Clean YAM teleoperation package extracted from the older HuMI
`humi_data_collection/packages/htc_interface` scripts.

## Current Scope

`sim_teleop` currently supports live teleoperation and YAM episode recording:

1. Read one or more Vive Trackers from OpenVR.
2. Convert tracker motion into end-effector motion.
3. Solve YAM arm IK with J-PARSE or mink.
4. Drive the YAM arm and LINEAR_4310 gripper in a MuJoCo viewer.
5. Optionally read a BRT encoder for gripper open/close.
6. Save tracker-first YAM teleop episodes with `--record-dir`.
7. Record raw Vive Tracker trajectories without MuJoCo via
   `python -m sim_teleop.record_tracker`.
8. Stream raw Vive Tracker poses over ZMQ to a LAN receiver (e.g. Ubuntu)
   via `python -m sim_teleop.stream_pose` — see Live Pose Streaming below.

It does not yet implement MuJoCo replay.

## Model Assets

The vendored i2rt model is modular: YAM arm XML, LINEAR_4310 gripper XML, and
gripper mount config are stored separately and combined at runtime. For data
collection and replay, `sim_teleop` materializes the specific model we use:

```text
sim_teleop/models/
  yam_linear_4310_tracker.xml
  yam_linear_4310_tracker.frames.urdf
  yam_linear_4310_tracker.meta.json
```

`yam_linear_4310_tracker.xml` is the authoritative MuJoCo replay model. It
contains the arm, LINEAR_4310 gripper, `grasp_site`, `ee_site`, and
`tracker_site`.

`yam_linear_4310_tracker.frames.urdf` is a lightweight kinematic reference
URDF. It starts from the YAM arm URDF and adds fixed `grasp_site`, `ee_site`,
and `tracker_site` frames. `ee_site` is colocated with `grasp_site` but uses
the tracker orientation, so `ee_site -> tracker_site` is pure translation.
MuJoCo remains the source for replay and gripper geometry.

The metadata records `T_grasp_tracker`, `T_link6_grasp`, source asset paths,
and the relative tracker replay rule:

```text
T_tracker_delta = inv(T_world_tracker_0) @ T_world_tracker_t
T_grasp_delta = T_grasp_tracker @ T_tracker_delta @ inv(T_grasp_tracker)
T_world_grasp_target = T_world_grasp_0 @ T_grasp_delta
```

Regenerate the assets after changing tracker mount geometry:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.export_model
```

## Main Entry

Run from the repository root:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop --ik-method jparse --resolution 1024
```

Useful variants:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop --ik-method mink --resolution 1024
& ".venv\Scripts\python.exe" -m sim_teleop --control-site ee_site --ik-method jparse --resolution 1024
& ".venv\Scripts\python.exe" -m sim_teleop --control-site ee_site --ik-method mink --joint6-axis positive --resolution 1024
& ".venv\Scripts\python.exe" -m sim_teleop --port COM5 --resolution 1024
& ".venv\Scripts\python.exe" -m sim_teleop --record-dir data/yam_teleop
```

`--control-site ee_site` controls the tracker-aligned `ee_site` instead of
`grasp_site`. This is useful for isolating whether the tracker/grasp relative
rotation is causing a teleoperation issue.

`--joint6-axis positive` is an experimental A/B test that overrides MuJoCo
`joint6` from the i2rt config value `0 0 -1` to `0 0 1`.

Check the exported MuJoCo model before starting OpenVR:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop --check-model-only
```

By default, teleop loads:

```text
sim_teleop/models/yam_linear_4310_tracker.xml
```

You can override it explicitly:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop --model-xml sim_teleop/models/yam_linear_4310_tracker.xml
```

Minimal MuJoCo model visualization, with mesh hidden and only `grasp_site`,
`ee_site`, and `tracker_site` highlighted:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_model
```

For a terminal-only check:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_model --check-only
& ".venv\Scripts\python.exe" -m sim_teleop.check_frames
```

Tracker-pose-only recording:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.record_tracker -o data/tracker_poses
```

Tracker link / mount validation:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.validate_tracker_link --urdf-only
& ".venv\Scripts\python.exe" -m sim_teleop.validate_tracker_link
```

Stable model asset export:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.export_model
```

Viewer keys:

```text
R  reset tracker reference and enter CONTROL mode
S  start recording an episode, if --record-dir is set
T  stop and save the active episode
O  record current encoder value as gripper open
C  record current encoder value as gripper closed
Q  save active recording, quit the loop, and close OpenVR
```

## Environment

The active Windows environment should live at the repository root:

```text
.venv
```

It was copied from the older HTC interface venv and verified with:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop --help
```

Key dependencies (versions from the verified `.venv`):

```text
# Teleoperation / hardware
openvr==2.12.1401            Vive Tracker / OpenVR pose polling
pyserial==3.5                COM port for the BRT gripper encoder
minimalmodbus==2.1.1         Modbus for the gripper controller
python-can==4.5.0            CAN bus support

# Simulation / IK
mujoco==3.9.0                YAM arm + LINEAR_4310 viewer / replay
mink==1.1.1                  whole-body IK solver
pyroki                       (local install) IK solver
jax==0.10.1, jaxlib==0.10.1  J-PARSE IK backend
jaxlie==1.5.0                SE(3) / SO(3) Lie algebra
yourdfpy==0.0.60             URDF kinematic reference

# Network / streaming
pyzmq==27.1.0                LAN pose streaming (PUB/SUB)

# Math / data
numpy==2.4.6
scipy==1.17.1
pandas==3.0.3
pyarrow==24.0.0

# Visualization (episode review)
evo==1.36.5                  trajectory metrics / plotting (TUM)
rerun-sdk==0.33.0            interactive 3D pose viewer
matplotlib==3.11.0
trimesh==4.12.2

# Config / CLI
tyro==1.0.13                 CLI argument parsing
pydantic==2.13.4
rich==14.3.4
loguru==0.7.3
```

Dump the full frozen environment anytime with:

```powershell
& ".venv\Scripts\python.exe" -m pip freeze
```

The package also needs source dependencies from the vendored HuMI tree:

```text
third_party/HuMI-main/third_party/i2rt-main
third_party/HuMI-main/humi_data_collection/packages/htc_interface/yam_ik_controller
```

`sim_teleop.robot` resolves those paths automatically for the current repo
layout and for the old `packages/htc_interface` layout.

## Live Pose Streaming (Windows → Ubuntu)

Stream live Vive Tracker poses from the Windows machine (SteamVR) to another
machine on the LAN (e.g. Ubuntu) over ZeroMQ. The receiver needs no SteamVR
install and can drive IK, logging, or teleop with the live pose.

```text
Windows (SteamVR + tracker)                Ubuntu (LAN)
┌────────────────────────────┐           ┌─────────────────────────┐
│ stream_pose.py             │   ZMQ     │ receive_pose.py         │
│  openvr → 4x4 pose         │ ────────► │  zmq.SUB connect        │
│  zmq.PUB bind tcp://*:1234 │   LAN     │   tcp://<WIN_IP>:1234   │
└────────────────────────────┘           └─────────────────────────┘
```

Each frame is JSON, matching `record_tracker` on-disk multi-tracker schema:

```json
{
  "timestamp": 1780000000.123,
  "trackers": [
    {"serial": "3B-A33M02233", "role": "left_eef", "tracker_pose": [[...4x4...]]},
    {"serial": "3B-A33M01660", "role": "right_eef", "tracker_pose": [[...4x4...]]}
  ]
}
```

`tracker_pose` is a 4x4 homogeneous matrix in the SteamVR
`TrackingUniverseStanding` frame (raw, no robot-frame transform). The sender
reuses `sim_teleop.tracker.read_tracker_poses`, so streamed poses are directly
comparable to recorded ones.

### Windows side (sender)

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.stream_pose --port 1234
```

Options:

```text
--port PORT          ZMQ publisher port (default 1234)
--host HOST          bind interface; '*' = all / LAN (default '*')
--frequency HZ       publish rate (default 120)
--serial-prefix PFX  only stream trackers whose serial starts with PFX
                     (default TRACKER_SERIAL_PREFIX from tracker.py)
```

Find the Windows LAN IP with `ipconfig` (e.g. `192.168.1.10`).

### Ubuntu side (receiver)

Install the single dependency (no openvr/SteamVR needed):

```bash
pip install pyzmq
```

Run from the repo root. The receiver has no `sim_teleop` imports, so it also
works as a standalone file:

```bash
python -m sim_teleop.receive_pose --host 192.168.1.10 --port 1234
# or directly without the package:
python sim_teleop/receive_pose.py --host 192.168.1.10 --port 1234
```

Options:

```text
--host WIN_IP   IP of the Windows sender (default 127.0.0.1)
--port PORT     port the sender binds (default 1234)
--print-rate N  print one frame every N seconds; 0 = every frame (default 0.5)
```

Use the pose in your own code by polling the newest frame:

```python
import zmq, numpy as np
sock = zmq.Context().socket(zmq.SUB)
sock.setsockopt(zmq.CONFLATE, 1); sock.setsockopt(zmq.RCVHWM, 1)
sock.setsockopt_string(zmq.SUBSCRIBE, "")
sock.connect("tcp://192.168.1.10:1234")
msg = sock.recv_json()
pose = np.array(msg["trackers"][0]["tracker_pose"])  # 4x4
pos, rot = pose[:3, 3], pose[:3, :3]
```

### Streaming Checklist

1. Both machines on the same subnet; `ping <WIN_IP>` works from Ubuntu.
2. Allow the publisher port through the Windows firewall. On the first-run
   popup check "Private network", or open TCP 1234 manually. This is the most
   common reason Ubuntu sees no data.
3. SteamVR detects the tracker before starting `stream_pose`.
4. PUB/SUB drops the first few frames after a subscriber connects ("slow
   joiner"). The receiver keeps only the newest frame via `CONFLATE`, so this
   is fine for real-time use.

## Relation To Old HuMI Code

Old Windows-side tracker recording:

```powershell
& ".venv\Scripts\python.exe" -m htc_scripts.record_pose --rpc.serve --rec.output-dir data/my-tracker-recordings
```

Old full-body IK and data processing:

```bash
uv run online-ik <task_name> --rpc-address tcp://<windows_ip>:4242
uv run offline-ik <task_name> -i data/my-raw-recordings
uv run replay -i data/my-raw-recordings_ik_recomputed
uv run run-pipeline data/my-recording-session --gopro_timezone +08:00
```

That pipeline targets HuMI/G1 full-body data. For YAM MuJoCo teleop, reuse the
episode idea, but keep a separate YAM-specific schema.

## YAM Recording Schema

Tracker-only file layout:

```text
data/tracker_poses/session_YYYYmmdd_HHMMSS/
  metadata.json
  tracker_recording_YYYY.mm.dd_HH.MM.SS.ffffff.json
```

Tracker-only episode frame:

```json
{
  "timestamp": 1780000000.123,
  "serial": "3B-A33M02233",
  "tracker_pose": [[...], [...], [...], [...]],
  "trackers": [
    {"serial": "3B-A33M02233", "role": "left_eef", "tracker_pose": [[...]]},
    {"serial": "3B-A33M01660", "role": "right_eef", "tracker_pose": [[...]]}
  ],
  "poses_by_role": {
    "left_eef": [[...], [...], [...], [...]],
    "right_eef": [[...], [...], [...], [...]]
  }
}
```

`serial` and top-level `tracker_pose` are kept for compatibility with the
single-tracker replay path and prefer `left_eef` when a mapping is configured.
All poses in one frame are returned by the same OpenVR
`TrackingUniverseStanding` query, so the tracker matrices share one SteamVR
standing coordinate system.

Teleop file layout:

```text
data/yam_teleop/session_YYYYmmdd_HHMMSS/
  metadata.json
  recording_YYYY.mm.dd_HH.MM.SS.ffffff.json
```

Recommended episode frame fields:

```json
{
  "timestamp": 1780000000.123,
  "tracker_pose": [[...], [...], [...], [...]],
  "target_ee_pose": [[...], [...], [...], [...]],
  "realized_ee_pose": [[...], [...], [...], [...]],
  "arm_q": [0, 0, 0, 0, 0, 0],
  "ik_ok": true,
  "ik_error_m": 0.001,
  "mode": "CONTROL"
}
```

Minimum fields for tracker-trajectory replay:

```text
timestamp
tracker_pose
```

Useful fields for debugging and future training:

```text
tracker_pose
target_ee_pose
realized_ee_pose
ik_ok
ik_error_m
```

## Suggested Next Entries

Replay:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.replay_tracker data/tracker_poses
& ".venv\Scripts\python.exe" -m sim_teleop.replay_tracker data/tracker_poses --hide-mesh --loop
& ".venv\Scripts\python.exe" -m sim_teleop.replay_tracker data/tracker_poses --precompute-only
& ".venv\Scripts\python.exe" -m sim_teleop.replay_tracker data/tracker_poses --ik-mode window --initial-pose ready --stride 2 --action-frames 12 --exec-frames 4 --hide-mesh
```

Replay defaults to the validated setup:

```text
control_site = ee_site
joint6_axis  = positive
IK           = mink
initial_pose = ready
```

The MuJoCo replay viewer uses yellow for the target control pose, cyan for the
realized control site, and magenta for `tracker_site`.

Note: the vendored YAM URDF currently does not contain a tracker link. The
current validation path injects a `tracker_site` into the MuJoCo XML and checks
that the FK-derived `T_EE_TRACK` matches the configured mount transform.

## Episode Visualization (Rerun + evo)

Review recorded tracker episodes in an interactive 3D Rerun viewer (each tracker
shown as a moving triad plus its smoothed trajectory) and/or export TUM
trajectories for evo metrics.

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker                          # auto-pick the latest episode under --root
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker <episode.json>           # specify a file
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker <f> --axis-length 0.08   # triad axis length in meters
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker --all-trackers           # show every tracker at once
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker --tracker-role right_eef # pick one by role
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker --no-rerun --tum-out data/traj.tum  # TUM export only
```

Options:

```text
input                 Episode JSON; omit to auto-pick the latest under --root
--root ROOT           Search root for auto-picking the latest episode
--tracker-role ROLE   Tracker role or serial to visualize (left_eef, right_eef, ...)
--all-trackers        Visualize all trackers in the recording at once
--axis-length M       Triad axis length in meters
--tum-out PATH        Write a TUM trajectory file; with multiple trackers use a dir/prefix
--no-rerun            Only export TUM files; skip the Rerun viewer
```

## Raw Sensor Data Collection (cameras + encoders + trackers)

`sim_teleop.data_collection` records synchronized raw sensor streams —
RealSense RGB video (PyAV H.264), BRT gripper encoders, and Vive Trackers —
for later conversion to a LeRobot dataset. Two collectors share the same
sensor process classes:

- **`collect_smoke`** — one fixed-duration episode; a pipeline smoke test.
  Writes a `raw_smoke_session_v1` layout (single episode, no hotkeys).
- **`collect_session`** — hot-start sensors once, then record many episodes
  interactively with `c` / `q` hotkeys. Use this for real collection.
  Writes a `hotkey_session_v1` layout (see below).

### Architecture

One parent process orchestrates; each sensor device owns a dedicated worker.
Low-dimensional streams (encoders, tracker) run as `multiprocessing.Process`
subclasses that poll their device and push samples into a shared-memory ring
buffer. The camera rig is heavier (RealSense SDK + PyAV), so it runs as a
separate `subprocess.Popen` interpreter driven by stdin commands and file
markers, and writes video straight to disk instead of through the ring.

```text
                 PARENT (collect_session.main, sole consumer)
                 • creates Timebase, ring buffers, worker procs
                 • hotkey loop: stamps t_start/t_stop, slices rings, saves
   ┌──────────────────────┬───────────────────────┬──────────────────────┐
   │ mp.Process (spawn)   │ mp.Process (spawn)    │ subprocess.Popen      │
   │ EncoderProcess × N   │ TrackerProcess × 1    │ realsense_rgb_record  │
   │  30 Hz serial poll   │  120 Hz OpenVR poll   │   --serve             │
   │  ring.put(sample)    │  ring.put(sample)     │  stdin: START/STOP/   │
   │  shared: ring +      │  shared: ring +       │         QUIT          │
   │   timebase + 2×Event │   timebase + 2×Event  │  file: realsense_     │
   │                      │                       │        ready/done.json│
   ▼                      ▼                       ▼
 SharedMemoryRingBuffer (mp.RawArray per column + mp.Value count + mp.Lock)
   │                                                 │
   ▼                                                 ▼
 parent reads only at episode stop:                  cameras/cam*/color.mp4
 get_last_k → mask by [t_start, t_stop]              (+ per-frame .npy timestamps)
   ▼
 lowdim/encoder_*.npz, tracker.npz
```

Design points:

- **One producer per device.** `EncoderProcess` / `TrackerProcess` subclass
  `mp.Process` and override `run()`. `__init__` runs in the parent and stashes
  picklable shared resources; `start()` spawns the child, which calls `run()`.
- **Two `mp.Event`s per worker.** `ready_event` (child→parent: "device open,
  first sample OK") gates startup; `stop_event` (parent→child: "exit loop")
  drives clean shutdown, with `terminate()` as a fallback.
- **Always-on ring + windowed slice.** Workers write to the ring continuously
  whether or not an episode is recording. "Recording" only means the parent
  stamps `t_start`/`t_stop` and a per-stream `count_start` at hotkey moments,
  then at stop pulls the last K samples and masks by `[t_start, t_stop]`. So
  there is no startup latency and no pre-buffer loss.
- **Shared host timebase.** A frozen `Timebase(wall0, perf0)` is created once
  in the parent and inherited by every worker. All `timestamp` values are
  `wall0 + (perf_counter() - perf0)`, so camera/tracker/encoder clocks are
  directly comparable. Blocking reads are timestamped at the midpoint of their
  `[read_start, read_end]` host interval.
- **Capacity guard.** `--max-episode-s` sizes each ring (`rate × seconds + 256`).
  If an episode runs longer and `produced > capacity`, older samples are
  overwritten and the collector prints a `ring truncated` warning.

### Running collect_session

Requires `.venv` with `pyrealsense2` + `av` (PyAV) in addition to the
teleop stack, plus an encoder mapping config (default
`sim_teleop/configs/encoder_mapping.json`, produced by `calibrate_encoders`).

Full real collection (all sensors):

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.data_collection.collect_session -o data/sessions
```

Encoder raw values only (skip calibration; `normalized`/`metric` saved as NaN).
This is the typical choice when calibrating later from the raw stream:

```powershell
& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.collect_session `
  -o data\pokeumi_202606241148 --encoder-raw-only
```

The collector first hot-starts every sensor (waits on each `ready_event`),
prints `all sensors hot`, then loops on hotkeys:

```text
c   start a new episode  (cameras receive START, t_start stamped)
q   at the menu      → quit the session
q   during recording → stop, save the episode, return to the menu
```

Start/stop are also cued by a system beep (disable with `--no-cue-sound`).

Options:

```text
-o, --output-dir DIR            session root (default data/sessions)
--encoder-frequency FLOAT       encoder polling rate in Hz (default 30)
--tracker-frequency FLOAT       tracker polling rate in Hz (default 120)
--camera-width/--height/--fps   RealSense stream config (640/480/30)
--max-cameras N                 cap number of cameras (default 3)
--max-episode-s FLOAT           ring-buffer capacity guard in seconds (default 180)
--encoder-mapping PATH          encoder role→port+calibration config
--encoder-raw-only              record raw encoder only; normalized/metric are NaN
--no-camera                     run without RealSense (encoders + trackers only)
--no-cue-sound                  disable start/stop beeps
--realsense-python PATH         interpreter for the camera subprocess (default: this venv)
```

### Recorded data layout

Every session writes a `README_metadata.md` (human-readable field guide) next
to its `metadata.json`; each episode's `cameras/` folder writes its own
camera-side `README_metadata.md`. The on-disk tree for a `hotkey_session_v1`
session:

```text
data/sessions/session_YYYYmmdd_HHMMSS/
  metadata.json                 session: schema, timebase, sensor config, episodes[]
  README_metadata.md            session field guide (auto-generated)
  cameras_rig_ready.json        marker: camera serve warmed up
  realsense_serve.log           camera subprocess stdout/stderr
  episode_000/
    metadata.json               episode: t_start/t_stop/duration/counts + field_descriptions
    cameras/
      metadata.json             camera episode: schema=realsense_rgb_raw_v2, intrinsics, camera list
      README_metadata.md        camera field guide (auto-generated)
      realsense_ready.json      marker: camera confirmed START
      realsense_done.json       marker + frame count: camera confirmed STOP
      cam0/
        color.mp4               H.264 / yuv420p / CRF 18 RGB video
        color_timestamps.npy    PRIMARY per-frame host timestamp (seconds)
        device_timestamps.npy   RealSense frame.get_timestamp() (seconds)
        timestamp_domain.npy    RealSense timestamp-domain label per frame
        frame_counter.npy       raw RealSense frame counter
        receive_durations_ms.npy  host wait_for_frames duration (ms, diagnostics)
        metadata_frame_timestamp.npy    raw SDK frame_timestamp
        metadata_sensor_timestamp.npy   raw SDK sensor_timestamp
        metadata_time_of_arrival.npy    raw SDK time_of_arrival
        metadata_backend_timestamp.npy  raw SDK backend_timestamp
        sample.png              last-frame preview (quick inspection only)
      cam1/...  cam2/...        one folder per camera
    lowdim/
      encoder_left.npz          sliced by [t_start, t_stop] (see schema below)
      encoder_right.npz
      tracker.npz
  episode_001/ ...
```

With `--encoder-raw-only` there is no calibration yet, so `encoder_*` still
contains all columns but `normalized` and `metric` are NaN. File names follow
`encoder_{side}.npz` where side is the role with the `_encoder` suffix dropped
(`left_encoder` → `encoder_left.npz`).

### Low-dimensional .npz schemas

Each `.npz` is a compressed zip of equal-length 1-D arrays (poses are per-row
4×4). Every row shares the column order below. `timestamp` is on the shared
host timebase.

`encoder_*.npz` (one row per encoder poll, ~30 Hz):

```text
timestamp            float64   host midpoint of the read
read_start_timestamp float64   host time just before the Modbus read
read_end_timestamp   float64   host time just after the Modbus read
raw                  int32     raw BRT count (-1 on read failure)
normalized           float32   0 = fully closed, 1 = fully open (NaN if raw-only)
metric               float32   displacement in mm (NaN if raw-only)
valid                int8      1 if the read succeeded, else 0
```

`tracker.npz` (one row per OpenVR poll, ~120 Hz; poses are SteamVR
`TrackingUniverseStanding` world frame `T_world_tracker`, identity when the
tracker pose is unavailable):

```text
timestamp            float64   host midpoint of the read
read_start_timestamp float64   host time just before the OpenVR query
read_end_timestamp   float64   host time just after the OpenVR query
left_eef_pose        (N,4,4) float64   left tracker world pose
right_eef_pose       (N,4,4) float64   right tracker world pose
left_eef_valid       int8      1 if left pose was available this frame
right_eef_valid      int8      1 if right pose was available this frame
num_trackers         int16     total trackers seen this frame
```

### Camera metadata

`episode_NNN/cameras/metadata.json` (`schema: realsense_rgb_raw_v2`) records
the stream config (`width/height/fps/duration`), the shared `wall0/perf0`
anchors, the resolved `camera_roles` (serial → `left_cam`/`right_cam`/`egocam`)
plus full per-camera intrinsics and firmware info, a `timestamp_files` dict
describing every `.npy`, and a `cameras[]` list mapping each
`camera_index`/`role`/`serial_number` to its `video_path`.

### Time synchronization

All streams share one host timebase, so alignment is a plain timestamp
intersection/interpolation — no per-device clock conversion needed:

1. Load `camX/color_timestamps.npy` as the master video timeline.
2. Interpolate `lowdim/tracker.npz` and `lowdim/encoder_*.npz` (keyed on their
   `timestamp` column) onto those camera frame timestamps.

Use `color_timestamps.npy` (host midpoint), **not** `device_timestamps.npy`,
as the camera timeline: the device timestamp is per-camera uptime and is not
comparable across cameras or with the host clock.

### Downstream

Raw `.mp4` + `.npz` episodes feed `sim_teleop.data_collection.convert_to_lerobot`,
which builds a LeRobot v2.1 dataset (videos re-encoded to AV1 on import).
Recording uses H.264 specifically because PyAV/decord read it reliably and the
conversion re-encodes, so the recording codec and the final dataset codec are
decoupled. See [README.md](README.md#convert-a-raw-session-to-lerobot-v21) for
the conversion command and the 20-D state/action convention.

## Pre-collection Checklist

1. SteamVR detects the tracker.
2. `python -m sim_teleop --help` works in the selected venv.
3. `TRACKER_SERIAL_PREFIX` in `tracker.py` matches the mounted tracker.
4. Pressing `R` in the MuJoCo viewer enters CONTROL mode.
5. Console error stays small during slow 6-DOF tracker motion.
6. If using the gripper encoder, press `O` and `C` once and confirm
   `encoder_calibration.json` is saved.
7. Run one short recording and immediately replay it before collecting a
   full session.
