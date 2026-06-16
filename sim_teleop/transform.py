"""Tracker → EE coordinate transform.

The Vive Tracker is physically mounted on the gripper.  At runtime we get
T_world_tracker (from OpenVR) and want T_world_ee(t), the EE pose.

We use a *similarity transform* (conjugation) so that tracker deltas are
re-expressed in the EE local frame:

    T_ee(0)_ee(t) = T_ee_track @ T_track(0)_track(t) @ T_ee_track⁻¹

This module holds the constants and helper functions for this mapping.
"""
import mujoco
import numpy as np

# ── T_ee_track: tracker local frame expressed in EE (grasp_site) local frame ─
#
# Measured axis correspondences (tracker glued to gripper):
#   tracker -Y (forward)  → EE +Z (forward)
#   tracker +X (right)    → EE -Y (right)
#   tracker -Z (up)       → EE -X (up)
R_EE_TRACK = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=float)

# Tracker origin in the EE (grasp_site) frame: the tracker sits 0.1495 m
# along EE -Z from the grasp_site (mount geometry from CAD).
T_EE_TRACK_POS = np.array([0.0, 0.0, -0.1495])

T_EE_TRACK = np.eye(4)
T_EE_TRACK[:3, :3] = R_EE_TRACK
T_EE_TRACK[:3, 3] = T_EE_TRACK_POS
T_EE_TRACK_INV = np.linalg.inv(T_EE_TRACK)

# grasp_site position in the gripper body frame (tip of the LINEAR_4310).
EE_TIP_OFFSET = np.array([0.0, 0.0, -0.1347])
# grasp_site orientation in the gripper body frame (quaternion wxyz).
GRASP_SITE_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])


def tracker_pose_in_gripper_frame() -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, quat_wxyz) of the tracker site in the gripper body frame.

    Used to inject a 'tracker_site' into the MuJoCo XML so that the mount
    transform can be read back from FK instead of being hard-coded.
    """
    rg = np.zeros(9)
    mujoco.mju_quat2Mat(rg, GRASP_SITE_QUAT_WXYZ)
    t_gripper_grasp = np.eye(4)
    t_gripper_grasp[:3, :3] = rg.reshape(3, 3)
    t_gripper_grasp[:3, 3] = EE_TIP_OFFSET

    t_gripper_tracker = t_gripper_grasp @ T_EE_TRACK
    pos = t_gripper_tracker[:3, 3]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, t_gripper_tracker[:3, :3].ravel())
    return pos, quat


def ee_delta(
    tracker_init_inv: np.ndarray,
    T_curr: np.ndarray,
    t_ee_track: np.ndarray,
    t_ee_track_inv: np.ndarray,
) -> np.ndarray:
    """Compute T_ee(0)_ee(t): how the EE should move from its initial pose.

    Args:
        tracker_init_inv: Inverse of the tracker pose at reset time.
        T_curr: Current tracker pose from OpenVR.
        t_ee_track: T_ee_track derived from the model's FK (injected site).
        t_ee_track_inv: Inverse of t_ee_track.

    Returns:
        4x4 delta transform in the EE local frame.
    """
    T_delta = tracker_init_inv @ T_curr
    return t_ee_track @ T_delta @ t_ee_track_inv


def delta_rotvec_deg(ref_R: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation of R relative to ref_R (in ref frame), returned in degrees.

    The z-component is roll about the EE pointing axis — useful for
    diagnosing whether z-roll is correctly tracked.
    """
    delta = ref_R.T @ R
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, delta.ravel())
    if q[0] < 0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arccos(w)
    return np.degrees(q[1:] / s * angle)
