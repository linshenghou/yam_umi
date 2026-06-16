"""Analyze recorded tracker deltas and their mapped grasp-site deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .transform import T_EE_TRACK, T_EE_TRACK_INV


def _resolve_recording(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("tracker_recording_*.json"))
    if not candidates:
        candidates = sorted(path.glob("session_*/tracker_recording_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No tracker_recording_*.json found under {path}")
    return candidates[-1]


def _rotvec_deg(rotation: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).ravel())
    if quat[0] < 0:
        quat = -quat
    w = float(np.clip(quat[0], -1.0, 1.0))
    s = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if s < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arccos(w)
    axis = quat[1:] / s
    return np.degrees(axis * angle)


def _axis_label(vec: np.ndarray) -> str:
    idx = int(np.argmax(np.abs(vec)))
    sign = "+" if vec[idx] >= 0.0 else "-"
    return f"{sign}{'XYZ'[idx]}"


def _print_axis_map() -> None:
    print("Configured tracker axes expressed in grasp frame:")
    for i, axis in enumerate(("+X", "+Y", "+Z")):
        vec = T_EE_TRACK[:3, :3][:, i]
        print(f"  tracker {axis} -> grasp {_axis_label(vec)} {np.round(vec, 6)}")


def _sample_indices(count: int) -> list[int]:
    if count <= 1:
        return [0]
    raw = [0, count // 4, count // 2, (3 * count) // 4, count - 1]
    return sorted(set(raw))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze tracker recording rotation through T_grasp_tracker."
    )
    parser.add_argument(
        "recording",
        type=Path,
        help="Recording JSON, session directory, or data/tracker_poses root.",
    )
    args = parser.parse_args()

    recording = _resolve_recording(args.recording)
    payload = json.loads(recording.read_text(encoding="utf-8"))
    frames = payload.get("episode", [])
    if not frames:
        raise ValueError(f"No frames in {recording}")

    poses = [np.asarray(frame["tracker_pose"], dtype=float) for frame in frames]
    timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=float)
    serials = sorted({str(frame.get("serial", "")) for frame in frames})

    tracker_init_inv = np.linalg.inv(poses[0])
    rows = []
    for pose, timestamp in zip(poses, timestamps):
        tracker_delta = tracker_init_inv @ pose
        grasp_delta = T_EE_TRACK @ tracker_delta @ T_EE_TRACK_INV
        rows.append(
            {
                "t": float(timestamp - timestamps[0]),
                "tracker_rv": _rotvec_deg(tracker_delta[:3, :3]),
                "grasp_rv": _rotvec_deg(grasp_delta[:3, :3]),
                "tracker_pos": tracker_delta[:3, 3].copy(),
                "grasp_pos": grasp_delta[:3, 3].copy(),
            }
        )

    tracker_rv = np.vstack([row["tracker_rv"] for row in rows])
    grasp_rv = np.vstack([row["grasp_rv"] for row in rows])
    tracker_pos = np.vstack([row["tracker_pos"] for row in rows])
    grasp_pos = np.vstack([row["grasp_pos"] for row in rows])
    rot_norm = np.linalg.norm(tracker_rv, axis=1)
    max_rot_idx = int(np.argmax(rot_norm))

    print(f"Recording: {recording}")
    print(f"Frames: {len(frames)}  duration: {timestamps[-1] - timestamps[0]:.3f}s")
    print(f"Serials: {', '.join(serials)}")
    _print_axis_map()
    print()

    print("Rotation samples, degrees:")
    for idx in _sample_indices(len(rows)):
        row = rows[idx]
        print(
            f"  t={row['t']:.3f}s  "
            f"tracker={np.round(row['tracker_rv'], 2)}  "
            f"grasp={np.round(row['grasp_rv'], 2)}"
        )
    print()

    print("Largest rotation:")
    print(
        f"  t={rows[max_rot_idx]['t']:.3f}s  "
        f"tracker={np.round(tracker_rv[max_rot_idx], 2)} deg  "
        f"grasp={np.round(grasp_rv[max_rot_idx], 2)} deg"
    )

    for axis_idx, axis in enumerate("XYZ"):
        idx = int(np.argmax(np.abs(tracker_rv[:, axis_idx])))
        print(
            f"Max |tracker {axis}|: "
            f"t={rows[idx]['t']:.3f}s  "
            f"tracker={np.round(tracker_rv[idx], 2)} deg  "
            f"grasp={np.round(grasp_rv[idx], 2)} deg"
        )
    print()

    print("Translation ranges from first frame:")
    print(f"  tracker min={np.round(tracker_pos.min(axis=0), 4)} m")
    print(f"  tracker max={np.round(tracker_pos.max(axis=0), 4)} m")
    print(f"  grasp   min={np.round(grasp_pos.min(axis=0), 4)} m")
    print(f"  grasp   max={np.round(grasp_pos.max(axis=0), 4)} m")


if __name__ == "__main__":
    main()
