"""Render the YAM arm MuJoCo XML (yam.xml) to PNGs.

Unlike the URDF, yam.xml ships with all referenced meshes, so the full
arm geometry renders. Arm-only (no gripper) — that matches the URDF scope.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
XML = REPO / "third_party/HuMI-main/third_party/i2rt-main/i2rt/robot_models/arm/yam/yam.xml"
OUT_DIR = REPO / "sim_teleop/_renders"


def main() -> int:
    if not XML.exists():
        print(f"XML not found: {XML}", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(XML))
    model.vis.global_.offwidth = 1080
    model.vis.global_.offheight = 1080
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pos = data.geom_xpos
    center = pos[np.isfinite(pos).all(axis=1)].mean(axis=0)

    renderer = mujoco.Renderer(model, height=1080, width=1080)

    def shot(name: str, az: float, el: float, dist: float) -> None:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = center
        cam.distance = dist
        cam.azimuth = az
        cam.elevation = el
        renderer.update_scene(data, camera=cam)
        Image.fromarray(renderer.render()).save(OUT_DIR / f"{name}.png")
        print(f"  wrote {name}.png")

    print("Rendering YAM MJCF (yam.xml) views:")
    shot("yamxml_perspective", az=-130, el=-25, dist=1.5)
    shot("yamxml_front",       az=-90,  el=-5,  dist=1.4)
    shot("yamxml_side",        az=0,    el=-5,  dist=1.4)
    shot("yamxml_top",         az=0,    el=-89, dist=1.6)

    print(f"\nDone. Images in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
