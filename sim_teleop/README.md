# sim_teleop

Clean YAM teleoperation package extracted from the older HuMI
`humi_data_collection/packages/htc_interface` scripts.

## Current Scope

`sim_teleop` currently supports live teleoperation and YAM episode recording:

1. Read one Vive Tracker from OpenVR.
2. Convert tracker motion into end-effector motion.
3. Solve YAM arm IK with J-PARSE or mink.
4. Drive the YAM arm and LINEAR_4310 gripper in a MuJoCo viewer.
5. Optionally read a BRT encoder for gripper open/close.
6. Save tracker-first YAM teleop episodes with `--record-dir`.
7. Record raw Vive Tracker trajectories without MuJoCo via
   `python -m sim_teleop.record_tracker`.

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

Important dependencies:

```text
openvr
mujoco
numpy
mink
pyroki
jax
jaxlie
yourdfpy
minimalmodbus
pyserial
```

The package also needs source dependencies from the vendored HuMI tree:

```text
third_party/HuMI-main/third_party/i2rt-main
third_party/HuMI-main/humi_data_collection/packages/htc_interface/yam_ik_controller
```

`sim_teleop.robot` resolves those paths automatically for the current repo
layout and for the old `packages/htc_interface` layout.

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
  "serial": "3B-A33M...",
  "tracker_pose": [[...], [...], [...], [...]]
}
```

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
```

Replay defaults to the validated setup:

```text
control_site = ee_site
joint6_axis  = positive
IK           = mink
```

The MuJoCo replay viewer uses yellow for the target control pose, cyan for the
realized control site, and magenta for `tracker_site`.

Note: the vendored YAM URDF currently does not contain a tracker link. The
current validation path injects a `tracker_site` into the MuJoCo XML and checks
that the FK-derived `T_EE_TRACK` matches the configured mount transform.

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
