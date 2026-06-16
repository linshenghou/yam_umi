"""Stable YAM model assets for teleop recording and MuJoCo replay.

The vendored i2rt models are modular: the YAM arm XML, gripper XML, and
gripper mounting config are stored separately and combined at runtime.  This
module materializes the combination that this project actually uses:

    YAM arm + LINEAR_4310 gripper + grasp_site + tracker_site

The exported MuJoCo XML is the authoritative replay asset.  The exported URDF
is a lightweight kinematic reference that adds fixed grasp/tracker frames to
the YAM arm URDF; MuJoCo remains the source for gripper geometry and replay.
"""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .transform import (
    EE_TIP_OFFSET,
    GRASP_SITE_QUAT_WXYZ,
    T_EE_TRACK,
    tracker_pose_in_gripper_frame,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_MODEL_DIR = PACKAGE_DIR / "models"

EE_SITE = "grasp_site"
TRACKER_SITE = "tracker_site"
TRACKER_ALIGNED_EE_SITE = "ee_site"
TARGET_LINK_NAME = "link_6"
MUJOCO_LINK6_BODY = "link6"
GRIPPER_BODY = "gripper"

MODEL_BASENAME = "yam_linear_4310_tracker"
MODEL_XML_NAME = f"{MODEL_BASENAME}.xml"
MODEL_META_NAME = f"{MODEL_BASENAME}.meta.json"
MODEL_URDF_NAME = f"{MODEL_BASENAME}.frames.urdf"


def _first_existing_path(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists():
            return resolved
    tried = "\n  ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n  {tried}")


I2RT_ROOT = _first_existing_path(
    [
        REPO_ROOT / "third_party" / "i2rt-main",
        REPO_ROOT / "third_party" / "HuMI-main" / "third_party" / "i2rt-main",
        PACKAGE_DIR / ".." / ".." / ".." / ".." / "third_party" / "i2rt-main",
    ],
    "third_party/i2rt-main",
)
if str(I2RT_ROOT) not in sys.path:
    sys.path.insert(0, str(I2RT_ROOT))

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml  # noqa: E402


YAM_URDF_PATH = I2RT_ROOT / "i2rt" / "robot_models" / "arm" / "yam" / "yam.urdf"
LINEAR_4310_CONFIG_PATH = I2RT_ROOT / "i2rt" / "robots" / "config" / "linear_4310.yml"


@dataclass(frozen=True)
class ExportedModelAssets:
    """Paths for the materialized model assets."""

    xml_path: Path
    urdf_path: Path
    meta_path: Path
    metadata: dict[str, Any]


def matrix_to_nested_list(mat: np.ndarray) -> list[list[float]]:
    return np.asarray(mat, dtype=float).tolist()


def _format_vec(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).ravel())


def _transform_from_xpos_xmat(xpos: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(xmat, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(xpos, dtype=float)
    return transform


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def _matrix_to_rpy_xyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to URDF fixed-axis roll/pitch/yaw."""
    r = np.asarray(rotation, dtype=float)
    sy = -r[2, 0]
    cy = float(np.sqrt(max(0.0, 1.0 - sy * sy)))
    if cy > 1e-9:
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arctan2(sy, cy)
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = np.arctan2(-r[1, 2], r[1, 1])
        pitch = np.arctan2(sy, cy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def prepare_xml_with_tracker_site(
    xml_path: str | Path,
    tracker_pos: np.ndarray,
    tracker_quat: np.ndarray,
    *,
    out_path: str | Path | None = None,
    joint6_axis: np.ndarray | None = None,
) -> Path:
    """Write a MuJoCo XML with grasp_site and tracker_site configured.

    Args:
        xml_path: Combined YAM + gripper MuJoCo XML.
        tracker_pos: Tracker site position in the gripper body frame.
        tracker_quat: Tracker site orientation in the gripper body frame,
            MuJoCo quaternion order wxyz.
        out_path: Optional deterministic output path.  If omitted, a temporary
            XML path is created.
        joint6_axis: Optional override for the MuJoCo joint6 local axis.

    Returns:
        Path to the prepared XML.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for site in root.iter("site"):
        if site.get("name") == EE_SITE:
            site.set("pos", _format_vec(EE_TIP_OFFSET))
            site.set("quat", _format_vec(GRASP_SITE_QUAT_WXYZ))
            break
    else:
        raise ValueError(f"No site named {EE_SITE!r} found in {xml_path}")

    gripper_body = None
    for body in root.iter("body"):
        if body.get("name") == GRIPPER_BODY:
            gripper_body = body
            break
    if gripper_body is None:
        raise ValueError(f"No body named {GRIPPER_BODY!r} found in {xml_path}")

    if joint6_axis is not None:
        joint6 = None
        for joint in root.iter("joint"):
            if joint.get("name") == "joint6":
                joint6 = joint
                break
        if joint6 is None:
            raise ValueError(f"No joint named 'joint6' found in {xml_path}")
        joint6.set("axis", _format_vec(joint6_axis))

    def upsert_site(attrs: dict[str, str]) -> None:
        for site in root.iter("site"):
            if site.get("name") == attrs["name"]:
                for key, value in attrs.items():
                    site.set(key, value)
                return
        ET.SubElement(gripper_body, "site", attrs)

    upsert_site(
        {
            "name": TRACKER_ALIGNED_EE_SITE,
            "pos": _format_vec(EE_TIP_OFFSET),
            "quat": _format_vec(tracker_quat),
            "size": "0.01",
            "rgba": "0 1 1 1",
        }
    )
    upsert_site(
        {
            "name": TRACKER_SITE,
            "pos": _format_vec(tracker_pos),
            "quat": _format_vec(tracker_quat),
            "size": "0.01",
            "rgba": "1 0 1 1",
        }
    )

    ET.indent(tree, space="  ")
    if out_path is None:
        out_path = tempfile.NamedTemporaryFile(
            suffix=".xml", prefix="yam_ee_tracker_", delete=False
        ).name
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def validate_mujoco_model(xml_path: str | Path) -> dict[str, Any]:
    """Validate named frames and recover fixed transforms from MuJoCo FK."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    grasp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    tracker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TRACKER_SITE)
    aligned_ee_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, TRACKER_ALIGNED_EE_SITE
    )
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MUJOCO_LINK6_BODY)
    joint6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint6")
    if grasp_id < 0:
        raise ValueError(f"No MuJoCo site named {EE_SITE!r} in {xml_path}")
    if tracker_id < 0:
        raise ValueError(f"No MuJoCo site named {TRACKER_SITE!r} in {xml_path}")
    if aligned_ee_id < 0:
        raise ValueError(
            f"No MuJoCo site named {TRACKER_ALIGNED_EE_SITE!r} in {xml_path}"
        )
    if link6_id < 0:
        raise ValueError(f"No MuJoCo body named {MUJOCO_LINK6_BODY!r} in {xml_path}")
    if joint6_id < 0:
        raise ValueError(f"No MuJoCo joint named 'joint6' in {xml_path}")

    t_world_grasp = _transform_from_xpos_xmat(
        data.site_xpos[grasp_id], data.site_xmat[grasp_id]
    )
    t_world_tracker = _transform_from_xpos_xmat(
        data.site_xpos[tracker_id], data.site_xmat[tracker_id]
    )
    t_world_aligned_ee = _transform_from_xpos_xmat(
        data.site_xpos[aligned_ee_id], data.site_xmat[aligned_ee_id]
    )
    t_world_link6 = _transform_from_xpos_xmat(
        data.xpos[link6_id], data.xmat[link6_id]
    )

    t_grasp_tracker = np.linalg.inv(t_world_grasp) @ t_world_tracker
    t_aligned_ee_tracker = np.linalg.inv(t_world_aligned_ee) @ t_world_tracker
    t_grasp_aligned_ee = np.linalg.inv(t_world_grasp) @ t_world_aligned_ee
    t_link6_grasp = np.linalg.inv(t_world_link6) @ t_world_grasp
    allclose = bool(np.allclose(t_grasp_tracker, T_EE_TRACK, atol=1e-5))

    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "grasp_site_id": int(grasp_id),
        "tracker_site_id": int(tracker_id),
        "tracker_aligned_ee_site_id": int(aligned_ee_id),
        "link6_body_id": int(link6_id),
        "joint6_id": int(joint6_id),
        "joint6_axis": np.asarray(model.jnt_axis[joint6_id], dtype=float).tolist(),
        "t_grasp_tracker_fk": matrix_to_nested_list(t_grasp_tracker),
        "t_grasp_ee_tracker_aligned_fk": matrix_to_nested_list(t_grasp_aligned_ee),
        "t_ee_tracker_aligned_tracker_fk": matrix_to_nested_list(
            t_aligned_ee_tracker
        ),
        "t_link6_grasp_fk": matrix_to_nested_list(t_link6_grasp),
        "t_grasp_tracker_matches_config": allclose,
    }


def _remove_named_children(root: ET.Element, tag: str, names: set[str]) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == tag and child.get("name") in names:
                parent.remove(child)


def _append_fixed_frame(
    root: ET.Element,
    *,
    parent_link: str,
    child_link: str,
    joint_name: str,
    transform: np.ndarray,
) -> None:
    ET.SubElement(root, "link", {"name": child_link})
    joint = ET.SubElement(root, "joint", {"name": joint_name, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent_link})
    ET.SubElement(joint, "child", {"link": child_link})
    rpy = _matrix_to_rpy_xyz(transform[:3, :3])
    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": _format_vec(transform[:3, 3]),
            "rpy": _format_vec(rpy),
        },
    )


def export_frame_urdf(
    out_path: str | Path,
    *,
    t_link6_grasp: np.ndarray,
    t_link6_aligned_ee: np.ndarray,
    t_aligned_ee_tracker: np.ndarray,
) -> Path:
    """Export a YAM arm URDF with fixed grasp/tracker reference frames."""
    tree = ET.parse(YAM_URDF_PATH)
    root = tree.getroot()
    root.set("name", MODEL_BASENAME)

    _remove_named_children(root, "link", {EE_SITE, TRACKER_ALIGNED_EE_SITE, TRACKER_SITE})
    _remove_named_children(
        root,
        "joint",
        {
            f"{EE_SITE}_fixed",
            f"{TRACKER_ALIGNED_EE_SITE}_fixed",
            f"{TRACKER_SITE}_fixed",
        },
    )

    _append_fixed_frame(
        root,
        parent_link=TARGET_LINK_NAME,
        child_link=EE_SITE,
        joint_name=f"{EE_SITE}_fixed",
        transform=t_link6_grasp,
    )
    _append_fixed_frame(
        root,
        parent_link=TARGET_LINK_NAME,
        child_link=TRACKER_ALIGNED_EE_SITE,
        joint_name=f"{TRACKER_ALIGNED_EE_SITE}_fixed",
        transform=t_link6_aligned_ee,
    )
    _append_fixed_frame(
        root,
        parent_link=TRACKER_ALIGNED_EE_SITE,
        child_link=TRACKER_SITE,
        joint_name=f"{TRACKER_SITE}_fixed",
        transform=t_aligned_ee_tracker,
    )

    ET.indent(tree, space="  ")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def export_model_assets(
    out_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    overwrite: bool = True,
) -> ExportedModelAssets:
    """Export the stable MuJoCo XML, frame URDF, and metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / MODEL_XML_NAME
    urdf_path = out_dir / MODEL_URDF_NAME
    meta_path = out_dir / MODEL_META_NAME

    if not overwrite:
        existing = [p for p in (xml_path, urdf_path, meta_path) if p.exists()]
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise FileExistsError(f"Refusing to overwrite existing assets: {names}")

    combined_xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    tracker_pos, tracker_quat = tracker_pose_in_gripper_frame()
    prepare_xml_with_tracker_site(
        combined_xml,
        tracker_pos,
        tracker_quat,
        out_path=xml_path,
    )

    validation = validate_mujoco_model(xml_path)
    t_link6_grasp = np.asarray(validation["t_link6_grasp_fk"], dtype=float)
    t_grasp_tracker = np.asarray(validation["t_grasp_tracker_fk"], dtype=float)
    t_grasp_aligned_ee = np.asarray(
        validation["t_grasp_ee_tracker_aligned_fk"], dtype=float
    )
    t_aligned_ee_tracker = np.asarray(
        validation["t_ee_tracker_aligned_tracker_fk"], dtype=float
    )
    t_link6_aligned_ee = t_link6_grasp @ t_grasp_aligned_ee
    export_frame_urdf(
        urdf_path,
        t_link6_grasp=t_link6_grasp,
        t_link6_aligned_ee=t_link6_aligned_ee,
        t_aligned_ee_tracker=t_aligned_ee_tracker,
    )

    metadata: dict[str, Any] = {
        "schema": "yam_model_assets_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_BASENAME,
        "arm": "yam",
        "gripper": "linear_4310",
        "authoritative_replay_asset": MODEL_XML_NAME,
        "kinematic_reference_urdf": MODEL_URDF_NAME,
        "sites": {
            "grasp": EE_SITE,
            "tracker_aligned_ee": TRACKER_ALIGNED_EE_SITE,
            "tracker": TRACKER_SITE,
            "mujoco_link6_body": MUJOCO_LINK6_BODY,
            "urdf_link6": TARGET_LINK_NAME,
        },
        "source": {
            "i2rt_root": str(I2RT_ROOT),
            "arm_xml": ArmType.YAM.get_xml_path(),
            "gripper_xml": GripperType.LINEAR_4310.get_xml_path(),
            "gripper_mount_config": str(LINEAR_4310_CONFIG_PATH),
            "arm_urdf": str(YAM_URDF_PATH),
        },
        "transforms": {
            "t_grasp_tracker_config": matrix_to_nested_list(T_EE_TRACK),
            "t_grasp_tracker_fk": validation["t_grasp_tracker_fk"],
            "t_grasp_ee_tracker_aligned_fk": validation[
                "t_grasp_ee_tracker_aligned_fk"
            ],
            "t_ee_tracker_aligned_tracker_fk": validation[
                "t_ee_tracker_aligned_tracker_fk"
            ],
            "t_link6_grasp_fk": validation["t_link6_grasp_fk"],
            "tracker_site_in_gripper_body": {
                "pos": np.asarray(tracker_pos, dtype=float).tolist(),
                "quat_wxyz": np.asarray(tracker_quat, dtype=float).tolist(),
            },
            "grasp_site_in_gripper_body": {
                "pos": np.asarray(EE_TIP_OFFSET, dtype=float).tolist(),
                "quat_wxyz": np.asarray(GRASP_SITE_QUAT_WXYZ, dtype=float).tolist(),
            },
        },
        "validation": validation,
        "replay_pose_rule": {
            "mode": "relative_tracker_delta",
            "formula": (
                "T_world_grasp_target = T_world_grasp_0 @ "
                "(T_grasp_tracker @ inv(T_world_tracker_0) @ "
                "T_world_tracker_t @ inv(T_grasp_tracker))"
            ),
        },
        "notes": [
            "MuJoCo XML is the complete replay model with gripper geometry.",
            "URDF is a lightweight kinematic reference with fixed grasp/tracker frames.",
            "Do not edit generated assets by hand; regenerate them from sim_teleop.model_assets.",
        ],
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ExportedModelAssets(
        xml_path=xml_path,
        urdf_path=urdf_path,
        meta_path=meta_path,
        metadata=metadata,
    )
