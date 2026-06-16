"""Render the YAM arm URDF to PNGs (perspective + orthographic views).

Two issues are worked around so MuJoCo can load the URDF:
  * ``package://`` mesh URIs are rewritten to absolute asset paths.
  * <visual>/<collision> blocks whose mesh file is missing on disk
    (e.g. link_6 has no shipped STL) are stripped. The kinematic chain
    is untouched, so link_6 still renders as a frame.

Offscreen rendering only — no GUI window.
"""
from __future__ import annotations

import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
URDF = REPO / "third_party/HuMI-main/third_party/i2rt-main/i2rt/robot_models/arm/yam/yam.urdf"
ASSETS = URDF.parent / "assets"
OUT_DIR = REPO / "sim_teleop/_renders"


def _loadable_urdf() -> str:
    """Return a temp URDF path: package:// rewritten, missing meshes dropped."""
    text = URDF.read_text(encoding="utf-8")
    text = text.replace("package://assets/", ASSETS.as_posix() + "/")
    root = ET.fromstring(text)

    dropped: list[str] = []
    for link in root.findall("link"):
        for elem in list(link):  # copy, we mutate while iterating
            if elem.tag not in ("visual", "collision"):
                continue
            mesh = elem.find(".//mesh")
            fname = (mesh.get("filename") if mesh is not None else "")
            if not fname or not Path(fname).exists():
                link.remove(elem)
                dropped.append(f"{link.get('name')}/{elem.tag} -> {fname}")
    if dropped:
        print("Dropped geometry referencing missing meshes (kinematics kept):")
        for d in dropped:
            print(f"  - {d}")

    tmp = Path(tempfile.gettempdir()) / "yam_render.urdf"
    ET.ElementTree(root).write(tmp, encoding="utf-8", xml_declaration=True)
    return str(tmp)


def main() -> int:
    if not URDF.exists():
        print(f"URDF not found: {URDF}", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(_loadable_urdf())
    # Enlarge offscreen framebuffer (defaults to 640x480) before rendering.
    model.vis.global_.offwidth = 1080
    model.vis.global_.offheight = 1080
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Frame the camera on the geom centroid (robust across mujoco versions).
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

    print("Rendering YAM URDF views:")
    shot("yam_perspective", az=-130, el=-25, dist=1.5)
    shot("yam_front",       az=-90,  el=-5,  dist=1.4)
    shot("yam_side",        az=0,    el=-5,  dist=1.4)
    shot("yam_top",         az=0,    el=-89, dist=1.6)

    print(f"\nDone. Images in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
