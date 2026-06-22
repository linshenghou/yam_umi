# Windows setup — data collection only

Bring a fresh Windows 10/11 PC from zero to recording Vive Tracker + BRT
gripper encoder data (and LAN-streaming it). This is the **data-collection
subset** of the `yam_umi` environment: MuJoCo simulation / IK (`mujoco`,
`jax`, `pyroki`, `mink`, …) are intentionally **not** installed — none of the
recording code imports them.

Everything below is PowerShell, run from the repo root after cloning.

---

## 0. What you need up front

- Windows 10 or 11, with **admin** rights.
- A Vive Tracker + its USB dongle (or a base-station setup) for the tracker.
- A **BRT encoder** over USB-serial for the gripper.
- (Optional) An Intel RealSense camera for RGB capture.
- The two **machine-specific config files** from an already-working machine
  (they are git-ignored, so `git clone` will NOT bring them):
  - `encoder_calibration.json`            (gripper open/closed raw values)
  - `sim_teleop/configs/tracker_mapping.json`   (tracker serial → role)

## 1. Install Python 3.12

Install **Python 3.12.x** (3.12.10 was used on the reference machine) from
<https://www.python.org/downloads/windows/>. In the installer, tick
**"Add python.exe to PATH"**.

Verify:

```powershell
py -3.12 --version
```

## 2. Get the code

```powershell
git clone <repo-url> yam_umi
cd yam_umi
```

(Or copy the folder over — but do **not** copy the old machine's `.venv`,
paths inside it are absolute and will break. Rebuild the venv below.)

## 3. Create the virtual environment

```powershell
py -3.12 -m venv .venv
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
```

## 4. Install Python dependencies

```powershell
& ".venv\Scripts\python.exe" -m pip install -r requirements-data-collection.txt
```

Sanity check that the key imports resolve:

```powershell
& ".venv\Scripts\python.exe" -c "import openvr, serial, minimalmodbus, zmq, numpy, scipy, rerun, evo; print('core OK')"
& ".venv\Scripts\python.exe" -c "import pyrealsense2, cv2; print('realsense OK')"
```

## 5. Install SteamVR (system, manual — required for the tracker)

`openvr` is just the Python binding; the **runtime** must come from SteamVR.

1. Install the **Steam** client and sign in.
2. In Steam: **Library → Tools → SteamVR → Install**, then launch it once.
3. Plug in the tracker / dongle, make sure SteamVR shows the tracker as
   tracked and green before proceeding.

> This step cannot be done over SSH — it needs the Steam GUI and may prompt
> for UAC. Do it at the target machine directly.

## 6. Install the BRT encoder USB-serial driver

Plug in the encoder. Windows usually loads a USB-to-serial driver
automatically; confirm a COM port appears in **Device Manager → Ports
(COM & LPT)** (e.g. `COM6`). If a driver is needed, install the one from the
encoder vendor. Note the COM number for calibration.

## 7. (Optional) Intel RealSense

If you will use `realsense_rgb_check.py`:

1. Install the **Intel RealSense SDK 2.0** (core runtime + viewer) from
   <https://github.com/IntelRealSense/librealsense/releases>.
2. The Python binding (`pyrealsense2`) was already installed in step 4.

## 8. Copy the machine-specific configs

These are git-ignored, so copy them from a working machine into the repo root:

```
encoder_calibration.json
sim_teleop/configs/tracker_mapping.json
```

If `encoder_calibration.json` is not available, recalibrate on this machine
(no mujoco/openvr needed):

```powershell
& ".\.venv\Scripts\python.exe" -m gripper.calibrate            # interactive
& ".\.venv\Scripts\python.exe" -m gripper.calibrate --show     # stream raw values
```

## 9. Open the firewall for LAN pose streaming (optional)

If this PC will **stream** poses to another machine over the LAN
(`python -m sim_teleop.stream_pose`), open the port once from an **admin**
PowerShell (the default port is 5557, see `stream_pose.py`):

```powershell
New-NetFirewallRule -DisplayName "yam_umi pose stream" -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 5557
```

## 10. Verify end-to-end

```powershell
# Tracker visible to openvr?
& ".\.venv\Scripts\python.exe" -m sim_teleop.tracker --help

# Record a short clip (tracker + encoder), then visualize it:
& ".\.venv\Scripts\python.exe" -m sim_teleop.record_tracker -o data/tracker_poses
& ".\.venv\Scripts\python.exe" -m sim_teleop.visualize_tracker
```

You should get a `[TRACKER] SAVED data\tracker_poses\session_...\*.json` line,
and the Rerun viewer should show the cyan trajectory + start/end frames. If the
trajectory is hidden in the viewer, use **Blueprint → reset to default** in the
Rerun UI to pick up the layout.

---

## What does NOT work in this slim env

These scripts import `mujoco` / `robot` / `transform` and will fail until the
simulation stack is added (that's fine — they are not needed for recording):

- `python -m sim_teleop` (live teleop)
- `sim_teleop.replay_tracker`, `sim_teleop.check_frames`, `sim_teleop.check_mount`
- `sim_teleop.analyze_tracker_recording`, `sim_teleop.test_tracker_rotation`
- `sim_teleop.visualize_model`, `sim_teleop.export_model`
- `sim_teleop.validate_tracker_link`

If you later need them on this machine, install the full stack from a
`pip freeze` of the reference machine (it includes mujoco / jax / pyroki).
