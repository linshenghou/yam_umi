#!/usr/bin/env python3
"""Eye-in-hand calibration: Vive Tracker + checkerboard -> X = T_tracker_camera.

A RealSense is rigidly bolted to a Vive-tracked rig. This solves the fixed rigid
transform of the camera optical frame in the tracker frame via the classic
AX = XB hand-eye problem (``cv2.calibrateHandEye``, eye-in-hand convention):

    A = T_world_tracker   (gripper2base) -- from OpenVR
    B = T_camera_target   (target2cam)   -- from solvePnP on the checkerboard
    X = T_tracker_camera  (cam2gripper)  -- what we solve for

Workflow: hold the rig still in a varied pose, press SPACE to grab one Vive pose
and one image *at the same instant*. Staying still is what avoids any timestamp
sync -- this script does no temporal alignment (that lives in the capture
pipeline). Collect >= 15 poses, each with a big rotation change (pitch/roll/yaw);
pure translation does not constrain the rotation of X.

The result X is exactly the per-role mount transform stored in
``configs/camera_mounts.json`` (``translation_m`` + ``rotation_wxyz``), so on
``q`` the script saves ``T_tracker_camera_{side}.npy`` and also prints a
paste-ready mount block.

Run on pokeumi -- NOT over SSH (OpenVR/SteamVR needs a local session) -- with
SteamVR up, the tracker green, and the RealSense plugged in. Run once per arm::

    & ".venv\\Scripts\\python.exe" -m sim_teleop.calibrate_handeye_vive --side left
    & ".venv\\Scripts\\python.exe" -m sim_teleop.calibrate_handeye_vive --side right
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .tracker import (
    DEFAULT_TRACKER_MAPPING_PATH,
    TRACKER_SERIAL_PREFIX,
    load_tracker_mapping,
    read_tracker_poses,
)
from .data_collection.realsense_rgb_record import (
    DEFAULT_CAMERA_CONFIG_PATH,
    _device_info,
    _devices,
    _import_cv2,
    _import_realsense,
)

# ===== CONFIG: measure / override on site ==================================
SQUARE_SIZE = 0.04            # checkerboard square edge in metres -- MEASURE with calipers
CHECKERBOARD = (11, 8)        # inner corners (cols, rows) = (12x9 squares - 1).
                              # Auto-retries the swapped (8, 11) if detection fails.
MIN_POSES = 15               # recommended minimum; 3 is the hard floor cv2 needs
COLOR_WIDTH, COLOR_HEIGHT, COLOR_FPS = 640, 480, 30  # match the recorded color stream (D405)

# side -> role in the existing config files (see tracker_mapping.json /
# realsense_cameras.json / camera_mounts.json).
SIDE_TO_TRACKER_ROLE = {"left": "left_eef", "right": "right_eef"}
SIDE_TO_CAMERA_ROLE = {"left": "left_cam", "right": "right_cam"}

# Hand-eye solvers to cross-check. The first one is the value we save.
HANDEYE_METHODS = ("PARK", "TSAI", "HORAUD", "ANDREFF", "DANIILIDIS")


def _resolve_tracker_serial(side: str, override: str, mapping_path: Path) -> str | None:
    """Tracker serial for this arm: explicit override, else the role mapping."""
    if override:
        return override
    role = SIDE_TO_TRACKER_ROLE[side]
    serial = load_tracker_mapping(mapping_path).get(role)
    if not serial:
        print(
            f"[CAL] WARNING: no '{role}' in {mapping_path} (and no --tracker-serial); "
            "falling back to the first tracked device. Pass --tracker-serial to be safe.",
            flush=True,
        )
    return serial


def _resolve_camera_serial(side: str, override: str, camera_config_path: Path) -> str | None:
    """RealSense serial for this arm: explicit override, else the role config."""
    if override:
        return override
    role = SIDE_TO_CAMERA_ROLE[side]
    if camera_config_path.exists():
        data = json.loads(camera_config_path.read_text(encoding="utf-8"))
        cfg = data.get("roles", {}).get(role, {})
        serial = cfg.get("serial_number") if isinstance(cfg, dict) else cfg
        if serial:
            return str(serial)
    print(
        f"[CAL] WARNING: no '{role}' serial in {camera_config_path} (and no "
        "--camera-serial); using the first RealSense device.",
        flush=True,
    )
    return None


def get_vive_pose(vr_system, target_serial: str | None, serial_prefix: str | None):
    """Return T_world_tracker (4x4) for this arm's tracker, or None if not tracked.

    Reads with no prefix filter so an explicit ``target_serial`` always matches;
    when no serial is known we fall back to the first prefix-matching tracker.
    """
    poses = read_tracker_poses(vr_system, None)
    if not poses:
        return None
    if target_serial:
        for serial, mat in poses:
            if serial == target_serial:
                return mat
        return None  # our tracker is not currently visible -> skip this capture
    if serial_prefix:
        for serial, mat in poses:
            if serial.startswith(serial_prefix):
                return mat
    return poses[0][1]


def _object_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    return objp * square


def _find_corners(cv2, gray, patterns, *, fast: bool):
    """Try each (cols, rows) pattern; return (found, corners, pattern) for the first hit."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    for pattern in patterns:
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if found:
            return True, corners, pattern
    return False, None, patterns[0]


def _open_camera(rs, camera_serial: str | None):
    """Start a color pipeline (explicit serial if given) and return (pipeline, K, dist)."""
    if camera_serial:
        # Validate the serial up front so we can list devices on a typo.
        available = [_device_info(rs, d).get("serial_number") for d in _devices(rs)]
        if camera_serial not in available:
            raise SystemExit(
                f"ERROR: RealSense {camera_serial} not found. Connected: {available}"
            )
    cfg = rs.config()
    if camera_serial:
        cfg.enable_device(camera_serial)
    cfg.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, COLOR_FPS)
    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)
    # Factory color intrinsics from the active stream profile (same path the
    # recording pipeline reports), matching the resolution we actually record.
    intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    K = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]]
    )
    dist = np.array(intr.coeffs, dtype=np.float64)
    print(
        f"[CAL] intrinsics fx={intr.fx:.2f} fy={intr.fy:.2f} "
        f"ppx={intr.ppx:.2f} ppy={intr.ppy:.2f} "
        f"({intr.width}x{intr.height}) dist={dist.round(5).tolist()}",
        flush=True,
    )
    return pipeline, K, dist


def _solve_and_report(cv2, R_hand, t_hand, R_targ, t_targ):
    """Run every hand-eye method, print a cross-check, return saved X (first method)."""
    saved = None
    for name in HANDEYE_METHODS:
        method = getattr(cv2, f"CALIB_HAND_EYE_{name}")
        try:
            Rx, tx = cv2.calibrateHandEye(R_hand, t_hand, R_targ, t_targ, method=method)
        except Exception as exc:  # noqa: BLE001 - some solvers are picky on few poses
            print(f"[CAL] {name:10s} failed: {exc}", flush=True)
            continue
        ang = np.degrees(np.linalg.norm(cv2.Rodrigues(Rx)[0]))
        print(
            f"[CAL] {name:10s} t(mm)={(tx.ravel() * 1000).round(1)} "
            f"rot(deg)={ang:.1f}",
            flush=True,
        )
        if saved is None:
            saved = np.eye(4)
            saved[:3, :3] = Rx
            saved[:3, 3] = tx.reshape(3)
    return saved


def _report_residual(cv2, X, R_hand, t_hand, R_targ, t_targ) -> None:
    """The checkerboard is fixed, so T_world_target = A @ X @ B should be constant."""
    world_targets = []
    for Rh, th, Rt, tt in zip(R_hand, t_hand, R_targ, t_targ):
        A = np.eye(4); A[:3, :3] = Rh; A[:3, 3] = th.reshape(3)
        B = np.eye(4); B[:3, :3] = Rt; B[:3, 3] = tt.reshape(3)
        world_targets.append(A @ X @ B)
    trans = np.array([T[:3, 3] for T in world_targets])
    R_ref = world_targets[0][:3, :3]
    angs = [
        np.degrees(np.linalg.norm(cv2.Rodrigues(R_ref.T @ T[:3, :3])[0]))
        for T in world_targets
    ]
    print(
        "[CAL] residual: board-in-world translation std(mm)="
        f"{(trans.std(axis=0) * 1000).round(1)} "
        f"range(mm)={((trans.max(0) - trans.min(0)) * 1000).round(1)} | "
        f"rotation spread(deg) max={max(angs):.2f}",
        flush=True,
    )
    print(
        "[CAL] good calibration ~ a few mm / <1-2 deg; large values mean too few "
        "poses, too little rotation, or a wrong SQUARE_SIZE.",
        flush=True,
    )


def _mount_block(side: str, X: np.ndarray) -> dict:
    """Format X as a configs/camera_mounts.json entry (translation_m + rotation_wxyz)."""
    from scipy.spatial.transform import Rotation

    quat_wxyz = Rotation.from_matrix(X[:3, :3]).as_quat(scalar_first=True)
    return {
        SIDE_TO_CAMERA_ROLE[side]: {
            "parent_tracker_role": SIDE_TO_TRACKER_ROLE[side],
            "translation_m": [round(float(v), 5) for v in X[:3, 3]],
            "rotation_wxyz": [round(float(v), 5) for v in quat_wxyz],
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--tracker-serial", default="", help="Override tracker serial (else from mapping)."
    )
    parser.add_argument(
        "--camera-serial", default="", help="Override RealSense serial (else from config)."
    )
    parser.add_argument(
        "--tracker-mapping", type=Path, default=DEFAULT_TRACKER_MAPPING_PATH
    )
    parser.add_argument(
        "--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG_PATH
    )
    parser.add_argument(
        "--serial-prefix",
        default=TRACKER_SERIAL_PREFIX,
        help="Tracker serial prefix for fallback selection (empty disables).",
    )
    parser.add_argument("--square-size", type=float, default=SQUARE_SIZE)
    parser.add_argument("--min-poses", type=int, default=MIN_POSES)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/handeye_calibration")
    )
    args = parser.parse_args()

    rs = _import_realsense()
    cv2 = _import_cv2()
    import openvr

    side = args.side
    serial_prefix = args.serial_prefix or None
    tracker_serial = _resolve_tracker_serial(side, args.tracker_serial, args.tracker_mapping)
    camera_serial = _resolve_camera_serial(side, args.camera_serial, args.camera_config)
    patterns = (CHECKERBOARD, (CHECKERBOARD[1], CHECKERBOARD[0]))
    print(
        f"[CAL] side={side} tracker={tracker_serial or 'auto'} "
        f"camera={camera_serial or 'auto'} square={args.square_size}m "
        f"board(inner)={CHECKERBOARD}",
        flush=True,
    )

    pipeline, K, dist = _open_camera(rs, camera_serial)
    openvr.init(openvr.VRApplication_Other)
    vr_system = openvr.VRSystem()
    time.sleep(1.0)  # let OpenVR warm up

    objp = _object_points(CHECKERBOARD, args.square_size)
    R_hand, t_hand, R_targ, t_targ = [], [], [], []
    subpix_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    print(
        f"[CAL] [{side}] SPACE=capture  u=undo last  q/ESC=solve & save. "
        "Each capture: big rotation change, hold still, keep the tracker unobscured.",
        flush=True,
    )
    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            found, corners, pattern = _find_corners(cv2, gray, patterns, fast=True)
            tracked = get_vive_pose(vr_system, tracker_serial, serial_prefix) is not None

            vis = color.copy()
            if found:
                cv2.drawChessboardCorners(vis, pattern, corners, found)
            cv2.putText(
                vis,
                f"{side} captured:{len(R_hand)}/{args.min_poses} "
                f"board:{'OK' if found else 'no'} tracker:{'OK' if tracked else 'LOST'}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if (found and tracked) else (0, 0, 255),
                2,
            )
            cv2.imshow("handeye-calib", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord("u") and R_hand:
                for buf in (R_hand, t_hand, R_targ, t_targ):
                    buf.pop()
                print(f"[CAL] undo -> {len(R_hand)} captures", flush=True)
                continue
            if key != ord(" "):
                continue

            # --- SPACE: grab one synchronized (pose, image) pair ---
            if not found:
                print("[CAL] no checkerboard, skip", flush=True)
                continue
            T_world_tracker = get_vive_pose(vr_system, tracker_serial, serial_prefix)
            if T_world_tracker is None:
                print("[CAL] tracker not tracked, skip", flush=True)
                continue
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_crit)
            ok, rvec, tvec = cv2.solvePnP(
                objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                print("[CAL] solvePnP failed, skip", flush=True)
                continue
            R_targ.append(cv2.Rodrigues(rvec)[0])
            t_targ.append(tvec.reshape(3, 1))
            R_hand.append(T_world_tracker[:3, :3].copy())
            t_hand.append(T_world_tracker[:3, 3].reshape(3, 1))
            print(f"[CAL] captured {len(R_hand)}", flush=True)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        try:
            openvr.shutdown()
        except Exception:
            pass

    n = len(R_hand)
    if n < 3:
        raise SystemExit(f"ERROR: only {n} captures; need >=3 (>={args.min_poses} recommended).")
    if n < args.min_poses:
        print(f"[CAL] WARNING: only {n} captures (< {args.min_poses}); result may be poor.", flush=True)

    X = _solve_and_report(cv2, R_hand, t_hand, R_targ, t_targ)
    if X is None:
        raise SystemExit("ERROR: every hand-eye solver failed.")
    _report_residual(cv2, X, R_hand, t_hand, R_targ, t_targ)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"T_tracker_camera_{side}.npy"
    np.save(out_path, X)
    print(f"\n[CAL] X = T_tracker_camera_{side} (saved to {out_path}):\n{X.round(5)}", flush=True)
    print(
        "\n[CAL] paste into configs/camera_mounts.json under \"cameras\" "
        "(then verify the frustum in the Rerun viewer):\n"
        + json.dumps(_mount_block(side, X), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
