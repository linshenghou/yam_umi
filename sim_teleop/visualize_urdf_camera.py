"""Show link_6 / camera / tracker relative geometry on the dual-YAM robot.

Loads the URDF, runs forward kinematics at a forward-reaching (or given) joint
pose, renders the arm meshes, and for each arm draws:

  - the camera AT the URDF CAD `*_camera_link` (its mesh + frame + true-FOV
    frustum): the CAD camera position is trusted,
  - the Vive tracker placed from the hand-eye result: X = T_tracker_camera, so
    tracker = camera @ inv(X), with X from configs/camera_mounts.json,
  - a schematic gripper proxy at the grasp frame,
  - relative-position lines link6 -> camera -> tracker.

This shows where the tracker sits relative to the wrist and camera (the thing the
hand-eye calibration measures). The camera<->tracker offset (~9 cm) is fixed, so
the joint pose does not affect that relationship.

Note: the tracker is placed treating the CAD camera_link frame as the camera
optical frame; if the CAD body frame differs from the optical (RDF) frame, the
tracker's orientation carries that offset (its distance from the camera is exact).

The URDF (dual_yam.urdf) is external; pass its path. Run from the repo root::

    python3 -m sim_teleop.visualize_urdf_camera --urdf /path/to/dual_yam/dual_yam.urdf
    python3 -m sim_teleop.visualize_urdf_camera --urdf .../dual_yam.urdf --side right --save urdf_cam.rrd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parent
MOUNTS_PATH = REPO / "configs" / "camera_mounts.json"
CAMERA_CONFIG_PATH = REPO / "configs" / "realsense_cameras.json"
MODEL_META_PATH = REPO / "models" / "yam_linear_4310_tracker.meta.json"
SIDE_TO_CAMERA_ROLE = {"left": "left_cam", "right": "right_cam"}
SIDE_TINT = {"left": (60, 160, 255), "right": (255, 140, 60)}

# A natural forward-reaching joint pose (joint1..joint6, radians): both wrist
# cameras point ~straight ahead (+X world) so the frustums are easy to read.
REACH_JOINTS = [-1.8, 1.0, 1.2, 0.0, 0.9, 0.0]


def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _R_to_quat_xyzw(m: np.ndarray) -> list[float]:
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def _inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _quat_wxyz_to_R(w, x, y, z) -> np.ndarray:
    n = float(np.linalg.norm([w, x, y, z])) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _origin(elem) -> np.ndarray:
    T = np.eye(4)
    o = elem.find("origin") if elem is not None else None
    if o is not None:
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        T[:3, :3] = _rpy_to_R(*rpy)
        T[:3, 3] = xyz
    return T


def _parse_urdf(path: Path):
    """Return (joints, links, root). joints[name] = dict(parent, child, T_origin, type, axis)."""
    root = ET.parse(path).getroot()
    joints, children = {}, set()
    for j in root.findall("joint"):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        axis_el = j.find("axis")
        axis = np.array([float(v) for v in axis_el.get("xyz").split()]) if axis_el is not None else np.array([0, 0, 1.0])
        joints[j.get("name")] = {
            "parent": parent, "child": child, "T": _origin(j),
            "type": j.get("type"), "axis": axis,
        }
        children.add(child)
    links = [l.get("name") for l in root.findall("link")]
    base = next(l for l in links if l not in children)
    return joints, links, base


def _forward_kinematics(joints: dict, base: str, cfg: dict) -> dict:
    """Absolute world transform per link at the given joint config (radians)."""
    by_parent: dict[str, list[str]] = {}
    for name, j in joints.items():
        by_parent.setdefault(j["parent"], []).append(name)
    world = {base: np.eye(4)}
    stack = [base]
    while stack:
        parent = stack.pop()
        for jname in by_parent.get(parent, []):
            j = joints[jname]
            T = j["T"].copy()
            if j["type"] == "revolute":
                q = float(cfg.get(jname, 0.0))
                a = j["axis"] / (np.linalg.norm(j["axis"]) or 1.0)
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
                M = np.eye(4); M[:3, :3] = R
                T = T @ M
            world[j["child"]] = world[parent] @ T
            stack.append(j["child"])
    return world


def _link_visual_meshes(urdf_path: Path) -> dict[str, list[tuple[Path, np.ndarray, np.ndarray]]]:
    """link -> [(mesh_path, T_visual_origin, scale)] for each <visual><mesh>."""
    root = ET.parse(urdf_path).getroot()
    base_dir = urdf_path.parent
    out: dict[str, list] = {}
    for link in root.findall("link"):
        vlist = []
        for vis in link.findall("visual"):
            geom = vis.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue
            fn = mesh.get("filename", "").replace("package://", "")
            scale = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
            vlist.append((base_dir / fn, _origin(vis), scale))
        if vlist:
            out[link.get("name")] = vlist
    return out


def _log_meshes(rr, urdf: Path, world: dict) -> None:
    try:
        import trimesh
    except ImportError:
        print("[URDF] trimesh not installed; skipping meshes (skeleton only).", flush=True)
        return
    missing = 0
    for link, vlist in _link_visual_meshes(urdf).items():
        if link not in world:
            continue
        for i, (path, T_org, scale) in enumerate(vlist):
            if not path.exists():
                missing += 1
                print(f"[URDF] missing mesh, skipping: {path.name}", flush=True)
                continue
            m = trimesh.load(path, force="mesh")
            try:  # vertex normals need scipy; fall back to flat shading without it
                normals = np.asarray(m.vertex_normals)
            except Exception:
                normals = None
            rr.log(
                f"robot/{link}/visual_{i}",
                rr.Transform3D(translation=T_org[:3, 3].tolist(),
                               quaternion=rr.Quaternion(xyzw=_R_to_quat_xyzw(T_org[:3, :3]))),
                rr.Mesh3D(vertex_positions=(np.asarray(m.vertices) * scale),
                          triangle_indices=np.asarray(m.faces),
                          vertex_normals=normals,
                          albedo_factor=(210, 214, 224)),
                static=True,
            )
    if missing:
        print(f"[URDF] {missing} mesh file(s) absent from assets/ (e.g. link_6) — frames still shown.", flush=True)


def _load_X(side: str) -> np.ndarray:
    role = SIDE_TO_CAMERA_ROLE[side]
    cam = json.loads(MOUNTS_PATH.read_text(encoding="utf-8"))["cameras"][role]
    w, x, y, z = cam["rotation_wxyz"]
    X = np.eye(4); X[:3, :3] = _quat_wxyz_to_R(w, x, y, z); X[:3, 3] = cam["translation_m"]
    return X


def _T_link6_grasp() -> np.ndarray:
    """Grasp (gripper TCP) frame in the link_6 frame, from the model metadata."""
    v = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))["validation"]
    return np.array(v["t_link6_grasp_fk"], float)


def _T_link6_camera(side: str) -> np.ndarray:
    """Measured camera-optical pose in the link_6 frame: T_link6_grasp @ T_grasp_tracker @ X."""
    v = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))["validation"]
    T_g_t = np.array(v["t_grasp_tracker_fk"], float)
    return _T_link6_grasp() @ T_g_t @ _load_X(side)


def _log_gripper_proxy(rr, side: str, link6: str, axis_length: float) -> None:
    """Schematic LINEAR_4310 gripper (body + two fingers) at the grasp frame.

    The real gripper.stl lives at an absolute path outside this repo, so this is
    a rough stand-in -- enough to see the camera relative to the jaws. Fingers
    extend along the grasp +Z (approach) axis.
    """
    ent = f"robot/{link6}/grasp_{side}"
    _log_transform(rr, ent, _T_link6_grasp(), axis_length * 0.8)
    rr.log(
        f"{ent}/gripper",
        rr.Boxes3D(
            centers=[[0, 0, -0.035], [0, 0.026, 0.02], [0, -0.026, 0.02]],
            half_sizes=[[0.022, 0.042, 0.022], [0.007, 0.007, 0.028], [0.007, 0.007, 0.028]],
            colors=[(80, 80, 90)],
        ),
        static=True,
    )
    rr.log(f"{ent}/glabel", rr.Points3D([[0, 0, 0]], labels=[f"{side} gripper (proxy)"],
           colors=[(120, 120, 130)]), static=True)


def _intrinsics(side: str) -> dict | None:
    role = SIDE_TO_CAMERA_ROLE[side]
    data = json.loads(CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
    return data.get("roles", {}).get(role, {}).get("intrinsics")


def _log_transform(rr, entity: str, T: np.ndarray, axis_length: float) -> None:
    rr.log(
        entity,
        rr.Transform3D(translation=T[:3, 3].tolist(),
                       quaternion=rr.Quaternion(xyzw=_R_to_quat_xyzw(T[:3, :3]))),
        rr.TransformAxes3D(axis_length=axis_length),
        static=True,
    )


def run(*, urdf: Path, sides: list[str], joint_vals: list[float], show_mesh: bool,
        show_gripper: bool, axis_length: float, plane_dist: float, save: Path | None) -> None:
    try:
        import rerun as rr
        from rerun import blueprint as rrb
    except ImportError as exc:  # optional dev dependency
        raise SystemExit("rerun-sdk is required for this viz: pip install rerun-sdk") from exc

    joints, links, base = _parse_urdf(urdf)
    cfg = {}
    for pre in ("left", "right"):
        for i, v in enumerate(joint_vals):
            cfg[f"{pre}_joint{i + 1}"] = v
    world = _forward_kinematics(joints, base, cfg)

    rr.init("yam_umi/urdf_camera", spawn=(save is None),
            default_blueprint=rrb.Blueprint(
                rrb.Spatial3DView(origin="/", contents="robot/**", name="URDF + hand-eye camera")))
    rr.log("robot", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # URDF is Z-up

    # Skeleton: a small triad per link + a bone from each parent to child.
    for link, T in world.items():
        _log_transform(rr, f"robot/{link}", T, axis_length * 0.5)
    bones = [[world[j["parent"]][:3, 3].tolist(), world[j["child"]][:3, 3].tolist()]
             for j in joints.values() if j["parent"] in world and j["child"] in world]
    rr.log("robot/bones", rr.LineStrips3D(bones, colors=[(140, 140, 140)]), static=True)

    if show_mesh:
        _log_meshes(rr, urdf, world)

    for side in sides:
        link6 = f"{side}_link_6"
        camlink = f"{side}_camera_link"
        if link6 not in world or camlink not in world:
            print(f"[URDF] missing {link6}/{camlink}; skipping {side}", flush=True)
            continue
        tint = SIDE_TINT[side]
        # Camera is trusted to sit at the URDF CAD camera_link (already placed by
        # FK, with its mesh). Draw its frame + true-FOV frustum coincident with it
        # (entity nested under camera_link with identity transform).
        cament = f"robot/{camlink}/optical_{side}"
        _log_transform(rr, cament, np.eye(4), axis_length * 1.8)
        intr = _intrinsics(side)
        fov_txt = ""
        if intr:
            rr.log(
                cament,
                rr.Pinhole(
                    image_from_camera=[[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]],
                    width=intr["width"], height=intr["height"],
                    camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=plane_dist),
                static=True)
            fov_h = 2 * np.degrees(np.arctan(intr["width"] / (2 * intr["fx"])))
            fov_v = 2 * np.degrees(np.arctan(intr["height"] / (2 * intr["fy"])))
            fov_txt = f" FOV {fov_h:.0f}x{fov_v:.0f}deg"
        rr.log(f"{cament}/label", rr.Points3D([[0.0, 0.0, 0.0]],
               labels=[f"{side} camera @ CAD camera_link{fov_txt}"], colors=[tint]), static=True)
        # Tracker located from the measured hand-eye result: X = T_tracker_camera,
        # so tracker = camera @ inv(X). Nested under camera_link with relative inv(X).
        Xi = _inv(_load_X(side))
        trkent = f"robot/{camlink}/tracker_{side}"
        _log_transform(rr, trkent, Xi, axis_length * 1.8)
        rr.log(f"{trkent}/label", rr.Points3D([[0.0, 0.0, 0.0]],
               labels=[f"{side} tracker (from hand-eye X)"], colors=[(90, 200, 90)]), static=True)
        # Relative-position lines: link6 -> camera -> tracker (world coords).
        p_l6, p_cam = world[link6][:3, 3], world[camlink][:3, 3]
        p_trk = (world[camlink] @ Xi)[:3, 3]
        rr.log(f"robot/relpos_{side}",
               rr.LineStrips3D([[p_l6.tolist(), p_cam.tolist()], [p_cam.tolist(), p_trk.tolist()]],
                               colors=[tint]), static=True)
        if show_gripper:
            _log_gripper_proxy(rr, side, link6, axis_length)
        print(f"[URDF] {side}: link6->camera={np.linalg.norm(p_cam - p_l6) * 1000:.0f}mm  "
              f"camera->tracker={np.linalg.norm(_load_X(side)[:3, 3]) * 1000:.0f}mm{fov_txt}", flush=True)

    print("[URDF] world = Z-up. Camera at CAD camera_link (with mesh); tracker placed via hand-eye X.", flush=True)
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save))
        print(f"[URDF] saved -> {save}", flush=True)
    else:
        print("[URDF] viewer launched; orbit to inspect the wrist cameras.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", type=Path, required=True, help="Path to dual_yam.urdf.")
    ap.add_argument("--side", choices=("left", "right", "both"), default="both")
    ap.add_argument("--pose", choices=("reach", "zero"), default="reach",
                    help="'reach' = forward-reaching pose (default), 'zero' = all joints 0.")
    ap.add_argument("--joints", default="",
                    help="Override with 6 comma-separated joint angles (rad), applied to both arms.")
    ap.add_argument("--no-mesh", action="store_true", help="Skeleton only (skip loading STL meshes).")
    ap.add_argument("--no-gripper", action="store_true", help="Skip the schematic gripper proxy.")
    ap.add_argument("--axis-length", type=float, default=0.05)
    ap.add_argument("--image-plane-distance", type=float, default=0.12)
    ap.add_argument("--save", type=Path, default=None, help="Save a .rrd instead of spawning the viewer.")
    args = ap.parse_args()
    if args.joints:
        joint_vals = [float(v) for v in args.joints.split(",")]
    else:
        joint_vals = [0.0] * 6 if args.pose == "zero" else list(REACH_JOINTS)
    sides = ["left", "right"] if args.side == "both" else [args.side]
    run(urdf=args.urdf, sides=sides, joint_vals=joint_vals, show_mesh=not args.no_mesh,
        show_gripper=not args.no_gripper, axis_length=args.axis_length,
        plane_dist=args.image_plane_distance, save=args.save)


if __name__ == "__main__":
    main()
