"""Print YAM grasp/tracker frame diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from .model_assets import (
    DEFAULT_MODEL_DIR,
    MODEL_XML_NAME,
    EE_SITE,
    TRACKER_ALIGNED_EE_SITE,
    TRACKER_SITE,
    export_model_assets,
    validate_mujoco_model,
)
from .transform import GRASP_SITE_QUAT_WXYZ, T_EE_TRACK


AXIS_NAMES = ("+X", "+Y", "+Z")


def _resolve_model_xml(path: Path | None) -> Path:
    if path is not None:
        return path
    default_xml = DEFAULT_MODEL_DIR / MODEL_XML_NAME
    if not default_xml.exists():
        export_model_assets(DEFAULT_MODEL_DIR)
    return default_xml


def _axis_name(vec: np.ndarray) -> str:
    vec = np.asarray(vec, dtype=float)
    idx = int(np.argmax(np.abs(vec)))
    sign = "+" if vec[idx] >= 0 else "-"
    return f"{sign}{'XYZ'[idx]}"


def _print_axis_map(label: str, rotation: np.ndarray, target_frame: str) -> None:
    print(label)
    for i, axis in enumerate(AXIS_NAMES):
        vec = rotation[:, i]
        print(
            f"  source {axis} -> {target_frame} {_axis_name(vec)} "
            f"{np.round(vec, 6).tolist()}"
        )


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check grasp_site/tracker_site frame conventions."
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=None,
        help="MuJoCo XML to inspect. Defaults to the exported sim_teleop model.",
    )
    args = parser.parse_args()

    model_xml = _resolve_model_xml(args.model_xml)
    validation = validate_mujoco_model(model_xml)
    t_grasp_tracker = np.asarray(validation["t_grasp_tracker_fk"], dtype=float)
    t_link6_grasp = np.asarray(validation["t_link6_grasp_fk"], dtype=float)
    r_gripper_grasp = _quat_wxyz_to_matrix(GRASP_SITE_QUAT_WXYZ)
    grasp_tracker_offset = t_grasp_tracker[:3, 3]

    print(f"Model XML: {model_xml}")
    print(
        f"Sites: {EE_SITE} id={validation['grasp_site_id']}  "
        f"{TRACKER_ALIGNED_EE_SITE} id={validation['tracker_aligned_ee_site_id']}  "
        f"{TRACKER_SITE} id={validation['tracker_site_id']}"
    )
    print(
        "T_grasp_tracker validation: "
        f"{validation['t_grasp_tracker_matches_config']}"
    )
    print(
        "grasp_site -> tracker_site offset in grasp frame: "
        f"{np.round(grasp_tracker_offset, 6).tolist()} m, "
        f"distance={np.linalg.norm(grasp_tracker_offset):.6f} m"
    )
    print()

    _print_axis_map(
        "Tracker axes expressed in grasp_site frame:",
        t_grasp_tracker[:3, :3],
        "grasp",
    )
    print()
    t_grasp_aligned_ee = np.asarray(
        validation["t_grasp_ee_tracker_aligned_fk"], dtype=float
    )
    t_aligned_ee_tracker = np.asarray(
        validation["t_ee_tracker_aligned_tracker_fk"], dtype=float
    )
    _print_axis_map(
        "ee_site axes expressed in grasp_site frame:",
        t_grasp_aligned_ee[:3, :3],
        "grasp",
    )
    print()
    _print_axis_map(
        "Tracker axes expressed in ee_site frame:",
        t_aligned_ee_tracker[:3, :3],
        "ee_site",
    )
    print(
        "ee_site -> tracker_site offset in ee_site frame: "
        f"{np.round(t_aligned_ee_tracker[:3, 3], 6).tolist()} m"
    )
    print()
    _print_axis_map(
        "Grasp axes expressed in link6 frame:",
        t_link6_grasp[:3, :3],
        "link6",
    )
    print()
    _print_axis_map(
        "Grasp axes expressed in gripper body frame:",
        r_gripper_grasp,
        "gripper",
    )
    print()
    print("T_grasp_tracker matrix:")
    print(np.array2string(T_EE_TRACK, precision=6, suppress_small=True))


if __name__ == "__main__":
    main()
