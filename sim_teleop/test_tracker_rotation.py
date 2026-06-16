"""Live diagnostic for tracker-local rotation and mapped grasp-site rotation."""

from __future__ import annotations

import time

import mujoco
import numpy as np
import openvr

from .tracker import read_pose
from .transform import T_EE_TRACK, T_EE_TRACK_INV


def _rotvec_deg(rotation: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).ravel())
    if quat[0] < 0:
        quat = -quat
    w = float(np.clip(quat[0], -1.0, 1.0))
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arccos(w)
    axis = quat[1:] / s
    return np.degrees(axis * angle)


def main() -> None:
    print("Initializing OpenVR...", flush=True)
    openvr.init(openvr.VRApplication_Other)
    vr_system = openvr.VRSystem()
    time.sleep(2.0)

    try:
        pose = read_pose(vr_system)
        if pose is None:
            print("ERROR: No tracker found.")
            return

        input("Hold tracker at reference pose, then press Enter...")
        pose0 = read_pose(vr_system)
        if pose0 is None:
            print("ERROR: No tracker at reference.")
            return
        pose0_inv = np.linalg.inv(pose0)
        print("Rotate one physical axis. Ctrl+C to quit.")
        print("Columns are rotation-vector components in degrees.")
        print("tracker=[x y z] is OpenVR tracker-local delta.")
        print("grasp=[x y z] is after T_grasp_tracker conjugation.\n")

        while True:
            pose = read_pose(vr_system)
            if pose is None:
                time.sleep(0.02)
                continue
            tracker_delta = pose0_inv @ pose
            grasp_delta = T_EE_TRACK @ tracker_delta @ T_EE_TRACK_INV
            tracker_rv = _rotvec_deg(tracker_delta[:3, :3])
            grasp_rv = _rotvec_deg(grasp_delta[:3, :3])
            print(
                "\r"
                f"tracker={tracker_rv.round(1)} deg  "
                f"grasp={grasp_rv.round(1)} deg  ",
                end="",
                flush=True,
            )
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
