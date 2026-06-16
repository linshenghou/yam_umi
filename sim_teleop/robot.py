"""YAM robot setup for MuJoCo teleop, IK, and replay-oriented model assets."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import mujoco
import numpy as np
import pyroki as pk
import yourdfpy

from .model_assets import (
    EE_SITE,
    I2RT_ROOT,
    PACKAGE_DIR,
    REPO_ROOT,
    TARGET_LINK_NAME,
    TRACKER_ALIGNED_EE_SITE,
    TRACKER_SITE,
    YAM_URDF_PATH,
    prepare_xml_with_tracker_site,
)
from .transform import T_EE_TRACK


def _first_existing_path(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists():
            return resolved
    tried = "\n  ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n  {tried}")


if str(I2RT_ROOT) not in sys.path:
    sys.path.insert(0, str(I2RT_ROOT))

from i2rt.robots.get_robot import get_yam_robot  # noqa: E402
from i2rt.robots.kinematics import Kinematics  # noqa: E402
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml  # noqa: E402


JPARSE_DIR = _first_existing_path(
    [
        REPO_ROOT / "yam_ik_controller",
        REPO_ROOT
        / "third_party"
        / "HuMI-main"
        / "humi_data_collection"
        / "packages"
        / "htc_interface"
        / "yam_ik_controller",
        PACKAGE_DIR / ".." / "yam_ik_controller",
    ],
    "yam_ik_controller",
)
if str(JPARSE_DIR) not in sys.path:
    sys.path.insert(0, str(JPARSE_DIR))

from _jparse import jparse_step  # noqa: E402  # type: ignore[import]


def _resolve_mesh(fname: str) -> str:
    if fname.startswith("package://"):
        fname = fname[len("package://") :]
    return str(YAM_URDF_PATH.parent / fname)


def build_pk_robot() -> tuple[pk.Robot, int]:
    """Load the YAM arm URDF into pyroki."""
    urdf = yourdfpy.URDF.load(str(YAM_URDF_PATH), filename_handler=_resolve_mesh)
    robot = pk.Robot.from_urdf(urdf)
    target_idx = robot.links.names.index(TARGET_LINK_NAME)
    return robot, target_idx


def pk_ee_pose(robot: pk.Robot, cfg: np.ndarray, target_idx: int) -> np.ndarray:
    """Return the 4x4 world pose of a pyroki link."""
    poses = robot.forward_kinematics(jnp.array(cfg, dtype=float))
    se3 = jaxlie.SE3(poses[target_idx])
    transform = np.eye(4)
    transform[:3, :3] = np.array(se3.rotation().as_matrix())
    transform[:3, 3] = np.array(se3.translation())
    return transform


class RobotBundle:
    """Robot objects needed by the teleoperation loop.

    Attributes:
        sim_robot: i2rt SimRobot with the YAM arm and LINEAR_4310 gripper.
        kin: MuJoCo/mink kinematics model at the selected control site.
        xml_path: Prepared MuJoCo XML with key sites.
        t_ee_track: Fixed transform from control site to tracker site.
        t_ee_track_inv: Inverse of ``t_ee_track``.
        pk_robot: pyroki arm-only robot for J-PARSE velocity IK.
        pk_target_idx: link_6 index in ``pk_robot``.
    """

    def __init__(
        self,
        control_site: str = EE_SITE,
        joint6_axis: np.ndarray | None = None,
    ) -> None:
        from .transform import T_EE_TRACK as configured_t_ee_track
        from .transform import tracker_pose_in_gripper_frame

        if control_site not in (EE_SITE, TRACKER_ALIGNED_EE_SITE):
            raise ValueError(
                f"Unsupported control_site {control_site!r}. "
                f"Use {EE_SITE!r} or {TRACKER_ALIGNED_EE_SITE!r}."
            )
        self.control_site = control_site

        self.sim_robot = get_yam_robot(
            arm_type=ArmType.YAM,
            gripper_type=GripperType.LINEAR_4310,
            sim=True,
        )

        tracker_pos, tracker_quat = tracker_pose_in_gripper_frame()
        self.xml_path = str(
            prepare_xml_with_tracker_site(
                self.sim_robot.xml_path,
                tracker_pos,
                tracker_quat,
                joint6_axis=joint6_axis,
            )
        )

        arm_xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
        arm_xml = str(
            prepare_xml_with_tracker_site(
                arm_xml,
                tracker_pos,
                tracker_quat,
                joint6_axis=joint6_axis,
            )
        )
        self.kin = Kinematics(arm_xml, control_site)

        q0 = self.sim_robot.get_joint_pos()[:6]
        t_w_ee = self.kin.fk(q0, control_site)
        t_w_tracker = self.kin.fk(q0, TRACKER_SITE)
        self.t_ee_track = np.linalg.inv(t_w_ee) @ t_w_tracker
        self.t_ee_track_inv = np.linalg.inv(self.t_ee_track)

        if control_site == EE_SITE:
            assert np.allclose(self.t_ee_track, configured_t_ee_track, atol=1e-5), (
                "FK-derived T_EE_TRACK does not match the configured tracker mount."
            )
        elif control_site == TRACKER_ALIGNED_EE_SITE:
            assert np.allclose(self.t_ee_track[:3, :3], np.eye(3), atol=1e-5), (
                "ee_site and tracker_site should have the same orientation."
            )

        self.pk_robot, self.pk_target_idx = build_pk_robot()

    def start(self) -> None:
        self.sim_robot.start_server()


def ik_jparse(
    bundle: RobotBundle,
    jparse_cfg: np.ndarray,
    T_target: np.ndarray,
    T_site_to_link6: np.ndarray,
    *,
    position_gain: float = 5.0,
    orientation_gain: float = 1.0,
    max_joint_velocity: float = 5.0,
    gamma: float = 0.1,
    dt: float = 0.02,
) -> tuple[np.ndarray, dict, bool]:
    """Run one J-PARSE velocity IK step.

    ``T_target`` is a control-site target. J-PARSE targets link_6, so the
    caller supplies the constant control-site-to-link_6 offset.
    """
    T_target_pk = T_target @ T_site_to_link6
    target_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(target_wxyz, T_target_pk[:3, :3].ravel())

    new_cfg, info = jparse_step(
        bundle.pk_robot,
        jparse_cfg,
        bundle.pk_target_idx,
        target_position=T_target_pk[:3, 3],
        target_wxyz=target_wxyz,
        position_gain=position_gain,
        orientation_gain=orientation_gain,
        max_joint_velocity=max_joint_velocity,
        gamma=gamma,
        dt=dt,
    )
    return new_cfg, info, bool(info["position_error"] < 0.01)


def ik_mink(
    bundle: RobotBundle,
    arm_qpos: np.ndarray,
    T_target: np.ndarray,
    *,
    position_cost: float = 1.0,
    orientation_cost: float = 0.1,
    posture_cost: float = 0.001,
    max_iters: int = 100,
) -> tuple[np.ndarray, bool]:
    """Run one MuJoCo/mink position IK step at the selected control site."""
    ok, q_sol = bundle.kin.ik(
        T_target,
        bundle.control_site,
        init_q=arm_qpos,
        pos_threshold=1e-3,
        ori_threshold=1e-3,
        max_iters=max_iters,
        position_cost=position_cost,
        orientation_cost=orientation_cost,
        posture_cost=posture_cost,
        posture_target=arm_qpos,
    )
    return q_sol, ok
