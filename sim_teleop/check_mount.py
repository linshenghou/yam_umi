"""Check how the LINEAR_4310 gripper is mounted on the YAM arm in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from .model_assets import (
    DEFAULT_MODEL_DIR,
    LINEAR_4310_CONFIG_PATH,
    MODEL_XML_NAME,
    export_model_assets,
)


def _resolve_model_xml(path: Path | None) -> Path:
    if path is not None:
        return path
    default_xml = DEFAULT_MODEL_DIR / MODEL_XML_NAME
    if not default_xml.exists():
        export_model_assets(DEFAULT_MODEL_DIR)
    return default_xml


def _vec(text: str | None) -> np.ndarray:
    if text is None:
        return np.array([])
    return np.fromstring(text, sep=" ")


def _body_pose(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise ValueError(f"No body named {body!r}")
    transform = np.eye(4)
    transform[:3, :3] = data.xmat[bid].reshape(3, 3)
    transform[:3, 3] = data.xpos[bid]
    return transform


def _site_pose(model: mujoco.MjModel, data: mujoco.MjData, site: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        raise ValueError(f"No site named {site!r}")
    transform = np.eye(4)
    transform[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    transform[:3, 3] = data.site_xpos[sid]
    return transform


def _print_matrix(name: str, mat: np.ndarray) -> None:
    print(f"{name}:")
    print(np.array2string(mat, precision=6, suppress_small=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=None,
        help="MuJoCo XML to inspect. Defaults to the exported sim_teleop model.",
    )
    args = parser.parse_args()

    model_xml = _resolve_model_xml(args.model_xml)
    tree = ET.parse(model_xml)
    root = tree.getroot()

    cfg = yaml.safe_load(LINEAR_4310_CONFIG_PATH.read_text())
    mount = cfg["last_joint_mount"]["yam"]
    yaml_pos = _vec(mount["pos"])
    yaml_quat = _vec(mount["quat"])
    yaml_axis = _vec(mount["axis"])

    link6 = root.find(".//body[@name='link6']")
    gripper = root.find(".//body[@name='gripper']")
    if link6 is None:
        raise SystemExit("No body named link6 in XML")
    if gripper is None:
        raise SystemExit("No body named gripper in XML")
    joint6 = link6.find("joint")

    xml_link6_pos = _vec(link6.get("pos"))
    xml_link6_quat = _vec(link6.get("quat"))
    xml_joint6_axis = _vec(joint6.get("axis") if joint6 is not None else None)
    xml_gripper_pos = _vec(gripper.get("pos"))
    xml_gripper_quat = _vec(gripper.get("quat"))

    print(f"Model XML: {model_xml}")
    print(f"Mount config: {LINEAR_4310_CONFIG_PATH}")
    print()
    print("YAML last_joint_mount[yam]")
    print(f"  pos : {yaml_pos}")
    print(f"  quat: {yaml_quat}")
    print(f"  axis: {yaml_axis}")
    print()
    print("Generated XML")
    print(f"  link6.pos      : {xml_link6_pos}")
    print(f"  link6.quat     : {xml_link6_quat}")
    print(f"  joint6.axis    : {xml_joint6_axis}")
    print(f"  gripper.pos    : {xml_gripper_pos}")
    print(f"  gripper.quat   : {xml_gripper_quat}")
    print()
    print("Matches YAML")
    print(f"  link6.pos   : {bool(np.allclose(xml_link6_pos, yaml_pos))}")
    print(f"  link6.quat  : {bool(np.allclose(xml_link6_quat, yaml_quat))}")
    print(f"  joint6.axis : {bool(np.allclose(xml_joint6_axis, yaml_axis))}")
    print()

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    t_w_link6 = _body_pose(model, data, "link6")
    t_w_gripper = _body_pose(model, data, "gripper")
    t_w_grasp = _site_pose(model, data, "grasp_site")
    t_w_tracker = _site_pose(model, data, "tracker_site")

    _print_matrix("T_link6_gripper", np.linalg.inv(t_w_link6) @ t_w_gripper)
    _print_matrix("T_link6_grasp_site", np.linalg.inv(t_w_link6) @ t_w_grasp)
    _print_matrix("T_grasp_tracker_site", np.linalg.inv(t_w_grasp) @ t_w_tracker)


if __name__ == "__main__":
    main()
