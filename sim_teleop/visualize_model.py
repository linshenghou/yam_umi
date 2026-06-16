"""Visualize the YAM MuJoCo model with only key links and sites highlighted."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .model_assets import (
    DEFAULT_MODEL_DIR,
    MODEL_XML_NAME,
    EE_SITE,
    TRACKER_ALIGNED_EE_SITE,
    TRACKER_SITE,
    export_model_assets,
)

KEY_BODIES: tuple[str, ...] = ()
KEY_SITES = (TRACKER_ALIGNED_EE_SITE, TRACKER_SITE)

BODY_COLORS = {
    "link5": (0.7, 0.7, 0.7, 1.0),
    "link6": (1.0, 0.65, 0.1, 1.0),
    "gripper": (0.2, 0.7, 1.0, 1.0),
    "tip_left": (0.4, 0.8, 0.9, 1.0),
    "tip_right": (0.4, 0.8, 0.9, 1.0),
}

SITE_COLORS = {
    EE_SITE: (0.1, 1.0, 0.1, 1.0),
    TRACKER_ALIGNED_EE_SITE: (0.0, 1.0, 1.0, 1.0),
    TRACKER_SITE: (1.0, 0.0, 1.0, 1.0),
}


def _resolve_model_xml(path: Path | None) -> Path:
    if path is not None:
        return path
    default_xml = DEFAULT_MODEL_DIR / MODEL_XML_NAME
    if not default_xml.exists():
        export_model_assets(DEFAULT_MODEL_DIR)
    return default_xml


def _parse_pose(text: str, n: int) -> list[float]:
    if not text:
        return []
    try:
        values = [float(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --pose {text!r}: {exc}") from exc
    return values[:n]


def _name(model: mujoco.MjModel, objtype: mujoco.mjtObj, idx: int) -> str:
    name = mujoco.mj_id2name(model, objtype, idx)
    return name if name is not None else f"<{idx}>"


def _add_frame(
    scn: mujoco.MjvScene,
    pos: np.ndarray,
    mat: np.ndarray,
    *,
    length: float,
    width: float,
) -> None:
    colors = (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 0.2, 1.0, 1.0),
    )
    rot = mat.reshape(3, 3)
    for axis in range(3):
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array(colors[axis], dtype=np.float32),
        )
        tip = pos + length * rot[:, axis]
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, pos, tip)
        scn.ngeom += 1


def _add_sphere(
    scn: mujoco.MjvScene,
    pos: np.ndarray,
    *,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _draw_key_frames(viewer, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0

    for body_name in KEY_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            continue
        pos = data.xpos[bid]
        mat = data.xmat[bid]
        _add_sphere(
            scn,
            pos,
            radius=0.012,
            rgba=BODY_COLORS.get(body_name, (0.7, 0.7, 0.7, 1.0)),
        )
        _add_frame(scn, pos, mat, length=0.06, width=0.0035)

    for site_name in KEY_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            continue
        pos = data.site_xpos[sid]
        mat = data.site_xmat[sid]
        _add_sphere(
            scn,
            pos,
            radius=0.016 if site_name == TRACKER_SITE else 0.013,
            rgba=SITE_COLORS.get(site_name, (1.0, 1.0, 1.0, 1.0)),
        )
        _add_frame(
            scn,
            pos,
            mat,
            length=0.12
            if site_name in (EE_SITE, TRACKER_ALIGNED_EE_SITE, TRACKER_SITE)
            else 0.08,
            width=0.005,
        )


def _print_key_table(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    print("\nKey bodies")
    print(f"{'body':<12} {'id':>3} {'world_pos':<26} {'parent'}")
    for body_name in KEY_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            continue
        parent_id = model.body_parentid[bid]
        print(
            f"{body_name:<12} {bid:>3} "
            f"{str(np.round(data.xpos[bid], 5)):<26} "
            f"{_name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)}"
        )

    print("\nKey sites")
    print(f"{'site':<12} {'id':>3} {'body':<12} {'local_pos':<26} {'world_pos'}")
    for site_name in KEY_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            continue
        bid = model.site_bodyid[sid]
        print(
            f"{site_name:<12} {sid:>3} "
            f"{_name(model, mujoco.mjtObj.mjOBJ_BODY, bid):<12} "
            f"{str(np.round(model.site_pos[sid], 5)):<26} "
            f"{np.round(data.site_xpos[sid], 5)}"
        )
    tracker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TRACKER_SITE)
    aligned_ee_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, TRACKER_ALIGNED_EE_SITE
    )
    if aligned_ee_id >= 0 and tracker_id >= 0:
        delta_world = data.site_xpos[tracker_id] - data.site_xpos[aligned_ee_id]
        rel_rot = data.site_xmat[aligned_ee_id].reshape(3, 3).T @ data.site_xmat[
            tracker_id
        ].reshape(3, 3)
        print(
            "\ntracker_site - ee_site in world: "
            f"{np.round(delta_world, 6)} m, "
            f"distance={np.linalg.norm(delta_world):.6f} m"
        )
        print(
            "tracker_site rotation relative to ee_site:\n"
            f"{np.round(rel_rot, 6)}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=None,
        help="MuJoCo XML to inspect. Defaults to the exported sim_teleop model.",
    )
    parser.add_argument(
        "--pose",
        default="0,1.0,1.0,0,0.5,0",
        help="Comma-separated qpos for joints 1..6.",
    )
    parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="Show original mesh geoms behind the simplified markers.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print key body/site poses and exit without opening the viewer.",
    )
    args = parser.parse_args()

    model_xml = _resolve_model_xml(args.model_xml)
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)

    for idx, q in enumerate(_parse_pose(args.pose, min(6, model.nq))):
        data.qpos[idx] = q
    mujoco.mj_forward(model, data)

    print(f"Model XML: {model_xml}")
    _print_key_table(model, data)
    print("Legend: frame axes are X=red, Y=green, Z=blue.")
    print("Sites: ee_site=cyan, tracker=magenta.")

    if args.check_only:
        return

    def on_key(key: int) -> None:
        if key in (ord("M"), ord("m")):
            viewer.opt.geomgroup[:] = 1 - viewer.opt.geomgroup[:]
            print("[mesh] toggled geom groups", flush=True)
        elif key in (ord("F"), ord("f")):
            viewer.opt.frame = (
                mujoco.mjtFrame.mjFRAME_SITE
                if viewer.opt.frame == mujoco.mjtFrame.mjFRAME_NONE
                else mujoco.mjtFrame.mjFRAME_NONE
            )
            print("[frame] toggled site frames", flush=True)
        elif key in (ord("L"), ord("l")):
            viewer.opt.label = (
                mujoco.mjtLabel.mjLABEL_SITE
                if viewer.opt.label == mujoco.mjtLabel.mjLABEL_NONE
                else mujoco.mjtLabel.mjLABEL_NONE
            )
            print("[label] toggled site labels", flush=True)

    print("Keys: M=toggle mesh geoms  F=toggle built-in site frames  L=toggle labels")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        if not args.show_mesh:
            viewer.opt.geomgroup[:] = 0
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            _draw_key_frames(viewer, model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
