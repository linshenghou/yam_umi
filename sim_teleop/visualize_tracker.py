"""Interactive visualization of a recorded Vive Tracker pose episode.

Reads a `vive_tracker_pose_episode_v1` JSON (as written by
``sim_teleop.record_tracker``), exports a TUM trajectory file so you can feed
it to `evo <https://github.com/michaelgrupp/evo>`_, and launches a Rerun
viewer showing:

  * the full position trajectory as a continuous bold line,
  * **start** and **end** coordinate frames (triads), drawn larger,
  * a time-scrubbable "current" tracker pose you can play/scrub along the
    timeline.

Usage::

    python -m sim_teleop.visualize_tracker <episode.json>
    python -m sim_teleop.visualize_tracker            # auto-picks latest
    python -m sim_teleop.visualize_tracker <f> --no-rerun --tum-out x.tum

With the TUM file exported, ``evo`` can also visualize / evaluate it::

    evo_traj tum <episode>.tum --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_ROOT = Path("data/tracker_poses")


def latest_episode(root: Path) -> Path:
    files = sorted(root.rglob("tracker_recording_*.json"))
    if not files:
        raise SystemExit(f"No tracker_recording_*.json under {root}")
    return files[-1]


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps[N], poses[N,4,4]) from an episode JSON."""
    data = json.loads(path.read_text())
    frames = data["episode"]
    if not frames:
        raise SystemExit(f"Empty episode in {path}")
    ts = np.array([f["timestamp"] for f in frames], dtype=float)
    poses = np.array([f["tracker_pose"] for f in frames], dtype=float)  # (N,4,4)
    return ts, poses


def poses_to_tum_rows(ts: np.ndarray, poses: np.ndarray) -> str:
    """Convert 4x4 poses to TUM rows: ``timestamp tx ty tz qx qy qz qw``."""
    xyz = poses[:, :3, 3]
    # scipy as_quat is scalar-last (x, y, z, w) — matches TUM's qx qy qz qw.
    quat_xyzw = Rotation.from_matrix(poses[:, :3, :3]).as_quat(scalar_first=False)
    lines = [
        f"{ts[i]:.6f} {xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} "
        f"{quat_xyzw[i, 0]:.6f} {quat_xyzw[i, 1]:.6f} {quat_xyzw[i, 2]:.6f} {quat_xyzw[i, 3]:.6f}"
        for i in range(len(ts))
    ]
    return "\n".join(lines) + "\n"


_AXIS_COLORS = ([220, 40, 40], [40, 200, 60], [60, 120, 255])  # X Y Z


def _log_triad(
    rr, path: str, pos: np.ndarray, quat_xyzw: np.ndarray, length: float, radius: float
) -> None:
    """Log an RGB coordinate frame (X=red, Y=green, Z=blue) as 3 short segments."""
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    world_axes = (R @ (np.eye(3) * length).T).T  # row i = axis i vector in world
    tips = pos + world_axes
    strips = [[pos.tolist(), tips[i].tolist()] for i in range(3)]
    rr.log(path, rr.LineStrips3D(strips, radii=radius, colors=_AXIS_COLORS))


def run_rerun(
    ts: np.ndarray,
    poses: np.ndarray,
    axis_length: float,
) -> None:
    import rerun as rr
    from rerun import blueprint as rrb

    xyz = poses[:, :3, 3]
    quat_xyzw = Rotation.from_matrix(poses[:, :3, :3]).as_quat(scalar_first=False)
    span = float((xyz.max(axis=0) - xyz.min(axis=0)).max())  # largest axial extent
    triad_r = max(span * 0.01, 0.003)   # frame axis radius (m)
    line_r = max(span * 0.014, 0.006)   # trajectory tube radius (m) — bold

    # Draw the trajectory as short 2-point segments — the exact same structure
    # the triad uses (and the triad renders). One long N-point strip did not
    # show up reliably in this Rerun build, but this segmented form does.
    step = max(1, len(xyz) // 80)
    path = xyz[::step]
    segs = [[path[j].tolist(), path[j + 1].tolist()] for j in range(len(path) - 1)]
    seg_colors = [[80, 200, 255]] * len(segs)  # bright cyan

    # Force a single 3D view containing every entity, so the trajectory is
    # never dropped from the auto-generated blueprint (which is what hid it).
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(origin="/", contents="$origin/**", name="Tracker"),
    )
    rr.init("yam_umi/tracker_pose", spawn=True, default_blueprint=blueprint)

    # NOTE: in this Rerun build, data logged before the first set_time
    # (timeless) does not render in the viewer — only time-based data does
    # (the scrubbable tracker triad proved that). So the trajectory and the
    # persistent start/end frames are re-logged at every timestamp with the
    # same data, reading as "always on".
    for i in range(len(ts)):
        rr.set_time("time", timestamp=float(ts[i]))
        rr.set_time("frame", sequence=i)

        # Whole trajectory as a continuous bold cyan line.
        rr.log("trajectory/full", rr.LineStrips3D(segs, radii=line_r, colors=seg_colors))
        # Persistent start (green) / end (red) frames.
        _log_triad(rr, "frames/start", xyz[0], quat_xyzw[0], axis_length * 2.0, triad_r * 1.4)
        _log_triad(rr, "frames/end", xyz[-1], quat_xyzw[-1], axis_length * 2.0, triad_r * 1.4)
        # Current tracker pose — the only thing that changes frame to frame.
        _log_triad(rr, "tracker/current", xyz[i], quat_xyzw[i], axis_length * 1.5, triad_r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Episode JSON. If omitted, auto-pick the latest under --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Search root for auto-picking the latest episode.",
    )
    parser.add_argument(
        "--tum-out",
        type=Path,
        default=None,
        help="Write a TUM trajectory file here. Default: <episode>.tum.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.06,
        help="Triad axis length (m) for the current frame; start/end use 2x.",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Only export the TUM file; skip the Rerun viewer.",
    )
    args = parser.parse_args()

    episode_path = args.input or latest_episode(args.root)
    ts, poses = load_episode(episode_path)
    print(
        f"[VIZ] Loaded {len(ts)} frames ({ts[-1] - ts[0]:.2f} s) from\n      {episode_path}"
    )

    tum_path = args.tum_out or episode_path.with_suffix(".tum")
    tum_path.write_text(poses_to_tum_rows(ts, poses))
    print(f"[VIZ] Wrote TUM trajectory -> {tum_path}")

    if args.no_rerun:
        print("[VIZ] --no-rerun set; skipping viewer.")
        print(f"      Try: evo_traj tum {tum_path} --plot")
        return

    run_rerun(ts, poses, args.axis_length)
    print("[VIZ] Rerun viewer launched — scrub the bottom timeline to replay.")
    print("      Persistent: cyan trajectory path + start(green)/end(red) frames.")
    print("      Dynamic (scrubs with time): 'tracker/current' frame.")
    print(f"      evo alt: evo_traj tum {tum_path} --plot")


if __name__ == "__main__":
    main()
