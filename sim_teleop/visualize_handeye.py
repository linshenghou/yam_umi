"""Visualize the hand-eye result X = T_tracker_camera in Rerun.

Puts the RealSense **optical frame at the origin** (Rerun world = RDF: +X right,
+Y down, +Z forward -- the RealSense/OpenCV optical convention, the same frame
``solvePnP`` and ``calibrate_handeye_vive`` work in) and draws, relative to it:

  - the camera optical triad + a Pinhole frustum (real intrinsics, so the FOV is
    physically correct) pointing along +Z,
  - the Vive tracker triad at T_camera_tracker = inv(X),
  - a baseline arrow from the camera to the tracker, labelled with the distance.

So you can eyeball whether the calibrated camera<->tracker geometry is physically
sane (offset magnitude, which way the camera looks relative to the tracker). The
residual printed by the calibration only proves self-consistency; this shows the
actual frame relationship.

``--side both`` overlays both trackers on one optical origin: because the two
rigs share the same bracket, the two tracker triads should nearly coincide -- a
visual version of the left/right cross-check.

Reads the mount from configs/camera_mounts.json by default (populated by
calibrate_handeye_vive), or a raw --npy T_tracker_camera_{side}.npy.

Run from the repo root::

    & ".venv\\Scripts\\python.exe" -m sim_teleop.visualize_handeye --side right
    & ".venv\\Scripts\\python.exe" -m sim_teleop.visualize_handeye --side both
    & ".venv\\Scripts\\python.exe" -m sim_teleop.visualize_handeye --npy data/handeye_calibration/T_tracker_camera_left.npy --side left
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_MOUNTS_PATH = Path(__file__).resolve().parent / "configs" / "camera_mounts.json"
DEFAULT_CAMERA_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "realsense_cameras.json"
)
SIDE_TO_CAMERA_ROLE = {"left": "left_cam", "right": "right_cam"}

# Distinct triad-tint per side so overlaid trackers are tellable apart.
SIDE_TINT = {"left": (60, 160, 255), "right": (255, 140, 60)}


def _quat_wxyz_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = float(np.linalg.norm([w, x, y, z])) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_quat_xyzw(m: np.ndarray) -> list[float]:
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def _X_from_mounts(mounts_path: Path, side: str) -> np.ndarray:
    """T_tracker_camera (4x4) from the camera_mounts.json entry for this side."""
    role = SIDE_TO_CAMERA_ROLE[side]
    cams = json.loads(mounts_path.read_text(encoding="utf-8")).get("cameras", {})
    if role not in cams or "translation_m" not in cams[role]:
        raise SystemExit(f"No mount for '{role}' in {mounts_path} (run the calibration first).")
    w, x, y, z = cams[role]["rotation_wxyz"]
    X = np.eye(4)
    X[:3, :3] = _quat_wxyz_to_R(w, x, y, z)
    X[:3, 3] = cams[role]["translation_m"]
    return X


def _intrinsics(camera_config_path: Path, side: str) -> dict | None:
    role = SIDE_TO_CAMERA_ROLE[side]
    if not camera_config_path.exists():
        return None
    data = json.loads(camera_config_path.read_text(encoding="utf-8"))
    return data.get("roles", {}).get(role, {}).get("intrinsics")


def _log_camera(rr, side: str, intr: dict | None, axis_length: float, plane_dist: float) -> None:
    """Camera optical frame at the world origin: triad + physically-correct frustum."""
    rr.log(
        "world/camera_optical",
        rr.Transform3D(translation=[0.0, 0.0, 0.0]),
        rr.TransformAxes3D(axis_length=axis_length),
        static=True,
    )
    rr.log("world/camera_optical/label", rr.Points3D([[0, 0, 0]], labels=[f"{side}_cam optical"]), static=True)
    if intr:
        k = [
            [intr["fx"], 0.0, intr["ppx"]],
            [0.0, intr["fy"], intr["ppy"]],
            [0.0, 0.0, 1.0],
        ]
        rr.log(
            "world/camera_optical",
            rr.Pinhole(
                image_from_camera=k,
                width=intr["width"],
                height=intr["height"],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=plane_dist,
            ),
            static=True,
        )


def _log_tracker(rr, side: str, X: np.ndarray, axis_length: float) -> None:
    """Tracker triad at T_camera_tracker = inv(X), plus a baseline arrow from the camera."""
    R = X[:3, :3]
    t = X[:3, 3]
    R_ct = R.T
    t_ct = -R_ct @ t  # tracker origin expressed in the camera optical frame
    quat_xyzw = _R_to_quat_xyzw(R_ct)
    dist_mm = float(np.linalg.norm(t_ct) * 1000.0)
    tint = SIDE_TINT[side]

    rr.log(
        f"world/tracker_{side}",
        rr.Transform3D(translation=t_ct.tolist(), quaternion=rr.Quaternion(xyzw=quat_xyzw)),
        rr.TransformAxes3D(axis_length=axis_length),
        static=True,
    )
    rr.log(
        f"world/tracker_{side}/label",
        rr.Points3D([[0, 0, 0]], labels=[f"{side}_eef tracker"], colors=[tint]),
        static=True,
    )
    rr.log(
        f"world/baseline_{side}",
        rr.Arrows3D(origins=[[0.0, 0.0, 0.0]], vectors=[t_ct.tolist()], colors=[tint],
                    labels=[f"{side}: {dist_mm:.0f} mm"]),
        static=True,
    )
    print(
        f"[HANDEYE] {side}: camera->tracker offset = {(t_ct * 1000).round(1)} mm "
        f"(|d|={dist_mm:.1f} mm)",
        flush=True,
    )


def run(*, sides: list[str], mounts_path: Path, camera_config_path: Path, npy: Path | None,
        axis_length: float, plane_dist: float, save: Path | None) -> None:
    try:
        import rerun as rr
        from rerun import blueprint as rrb
    except ImportError as exc:  # optional dev dependency
        raise SystemExit("rerun-sdk is required for this viz: pip install rerun-sdk") from exc

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(origin="/", contents="world/**", name="Hand-eye: camera <-> tracker")
    )
    rr.init("yam_umi/handeye", spawn=(save is None), default_blueprint=blueprint)
    # World == the RealSense optical frame: +X right, +Y down, +Z forward (RDF).
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # Frustum uses the first side's intrinsics (one optical origin in the scene).
    _log_camera(rr, sides[0], _intrinsics(camera_config_path, sides[0]), axis_length, plane_dist)
    for side in sides:
        X = np.load(npy) if npy is not None else _X_from_mounts(mounts_path, side)
        _log_tracker(rr, side, X, axis_length)

    print(
        "[HANDEYE] world = optical frame (RDF): X=red=right, Y=green=down, "
        "Z=blue=forward (camera looks along +Z).",
        flush=True,
    )
    if len(sides) > 1:
        print("[HANDEYE] both trackers share one bracket -> triads should nearly coincide.", flush=True)
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save))
        print(f"[HANDEYE] saved -> {save} (open with: rerun {save})", flush=True)
    else:
        print("[HANDEYE] viewer launched; orbit the 3D view to inspect the geometry.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=("left", "right", "both"), default="right")
    parser.add_argument(
        "--npy", type=Path, default=None,
        help="Load X from a raw T_tracker_camera_{side}.npy instead of camera_mounts.json "
        "(single side only).",
    )
    parser.add_argument("--mounts-config", type=Path, default=DEFAULT_MOUNTS_PATH)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG_PATH)
    parser.add_argument("--axis-length", type=float, default=0.05, help="Triad axis length (m).")
    parser.add_argument(
        "--image-plane-distance", type=float, default=0.1,
        help="Pinhole image-plane distance (m); visualization-only, larger = bigger frustum.",
    )
    parser.add_argument("--save", type=Path, default=None, help="Save a .rrd instead of spawning the viewer.")
    args = parser.parse_args()

    sides = ["left", "right"] if args.side == "both" else [args.side]
    if args.npy is not None and args.side == "both":
        raise SystemExit("--npy loads a single side; use --side left/right with --npy.")
    run(
        sides=sides,
        mounts_path=args.mounts_config,
        camera_config_path=args.camera_config,
        npy=args.npy,
        axis_length=args.axis_length,
        plane_dist=args.image_plane_distance,
        save=args.save,
    )


if __name__ == "__main__":
    main()
