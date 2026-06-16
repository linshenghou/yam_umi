"""OpenVR Vive Tracker reading."""
import numpy as np
import openvr

TRACKER_SERIAL_PREFIX = "3B-A33M"


def read_tracker_poses(
    vr_system: openvr.IVRSystem,
    serial_prefix: str | None = TRACKER_SERIAL_PREFIX,
) -> list[tuple[str, np.ndarray]]:
    """Return valid Vive Tracker poses as (serial, 4x4 matrix) pairs."""
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount,
    )
    out: list[tuple[str, np.ndarray]] = []
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if not poses[i].bPoseIsValid:
            continue
        if vr_system.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_GenericTracker:
            continue
        serial = vr_system.getStringTrackedDeviceProperty(
            i, openvr.Prop_SerialNumber_String
        )
        if serial_prefix is not None and not serial.startswith(serial_prefix):
            continue
        m34 = poses[i].mDeviceToAbsoluteTracking
        mat = np.eye(4)
        for r in range(3):
            for c in range(4):
                mat[r, c] = m34[r][c]
        out.append((serial, mat))
    return out


def read_pose(vr_system: openvr.IVRSystem) -> "np.ndarray | None":
    """Return the first valid Vive Tracker pose as a 4x4 matrix, or None."""
    tracker_poses = read_tracker_poses(vr_system)
    return tracker_poses[0][1] if tracker_poses else None
