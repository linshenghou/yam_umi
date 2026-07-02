# Vive Tracker + camera hand-eye calibration

Each handheld data-collection rig rigidly bolts a RealSense to a Vive-tracked
frame. To turn tracker poses into camera poses we need the fixed transform of the
camera optical frame in the tracker frame:

```
X = T_tracker_camera
T_world_camera = T_world_tracker @ X
```

This is the classic eye-in-hand `AX = XB` problem, solved with
`cv2.calibrateHandEye`:

- `A = T_world_tracker`  (gripper2base) — from OpenVR
- `B = T_camera_target`  (target2cam)   — from `solvePnP` on a checkerboard
- `X = T_tracker_camera` (cam2gripper)  — solved for

## How to calibrate

Run on the capture host **locally** (not over SSH — OpenVR needs a local
session), with SteamVR up, the tracker tracked, and the RealSense connected:

```
python -m sim_teleop.calibrate_handeye_vive --side left  --tracker-serial <LEFT_SERIAL>
python -m sim_teleop.calibrate_handeye_vive --side right --tracker-serial <RIGHT_SERIAL>
```

- The camera serial per side comes from `configs/realsense_cameras.json`
  (`left_cam` / `right_cam`); override with `--camera-serial`. Only one RealSense
  needs to be connected at a time.
- Hold the rig still in a varied pose, `SPACE` to capture one pose + one image at
  once (static capture avoids any timestamp sync), `u` to undo, `q`/`Esc` to
  solve. Collect ≥15 poses, each a **big rotation change** — pure translation
  does not constrain the rotation of `X`.
- Output: `data/handeye_calibration/T_tracker_camera_{side}.npy` and a
  paste-ready `configs/camera_mounts.json` block.

Intrinsics are read live from the RealSense color stream at the recorded
resolution (640×480).

## Result quality

Cross-checked across PARK/TSAI/HORAUD/ANDREFF/DANIILIDIS (all agree). Residual =
spread of the fixed board pose in the world frame across poses:

| arm   | translation std | rotation spread | camera↔tracker |
|-------|-----------------|-----------------|----------------|
| right | ~2–3 mm         | ~0.9°           | ~88 mm         |
| left  | ~7 mm           | ~2.7°           | ~92 mm         |

The two arms share one bracket and the independent calibrations agree to
**7.6 mm / 0.88°**, which cross-validates both. The measured mounts replaced the
earlier CAD-guess values in `configs/camera_mounts.json`.

## Visual verification (optional dev tools)

Both need `rerun-sdk` (already in `requirements-data-collection.txt`);
`visualize_urdf_camera` also needs `trimesh`.

- `python -m sim_teleop.visualize_handeye --side both`
  Camera optical frame at the origin, tracker at `inv(X)`, per side. With
  `--side both` the two tracker triads nearly coincide (the bracket check).

- `python -m sim_teleop.visualize_urdf_camera --urdf /path/to/dual_yam.urdf --side both`
  Renders the dual-YAM robot with the camera at the CAD `camera_link` and the
  tracker placed via `tracker = camera @ inv(X)`, so you can see the
  link6 / camera / tracker geometry on the arm. `--pose zero|reach`, `--joints`,
  `--no-mesh`, `--no-gripper` control the view.

Note: the URDF overlay places the tracker treating the CAD `camera_link` frame as
the camera optical frame — the camera↔tracker **distance** is exact; the tracker
**orientation** carries any CAD-body-vs-optical rotation offset.

For an assumption-free check, record a short session and replay it with
`sim_teleop.visualize_replay`, which projects the live camera image through the
mount frustum onto the 3D scene.
