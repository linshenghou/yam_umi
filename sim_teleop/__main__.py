"""Vive Tracker → YAM arm + gripper teleoperation in MuJoCo sim.

Run from the htc_interface directory:
    & ".venv\\Scripts\\python.exe" -m sim_teleop [--port COM5] [--resolution 1024]

Keys (MuJoCo viewer):
    R — reset tracker reference frame → enter CONTROL mode
    O — record gripper OPEN  position (encoder calibration)
    C — record gripper CLOSED position (encoder calibration)
    Q — quit
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import openvr

from . import gripper as _gripper
from . import tracker as _tracker
from .model_assets import (
    DEFAULT_MODEL_DIR,
    MODEL_XML_NAME,
    TRACKER_ALIGNED_EE_SITE,
    TRACKER_SITE,
    export_model_assets,
    prepare_xml_with_tracker_site,
    validate_mujoco_model,
)
from .recording import EpisodeRecorder, array_to_list, matrix_to_list
from .robot import EE_SITE, RobotBundle, ik_jparse, ik_mink, pk_ee_pose
from .transform import delta_rotvec_deg, ee_delta, tracker_pose_in_gripper_frame

# ── Safety limits ─────────────────────────────────────────────────────────────
EE_BOUNDS_MIN = np.array([-0.4, -0.4, 0.05])
EE_BOUNDS_MAX = np.array([ 0.4,  0.4, 0.5])
MAX_EE_STEP_M = 0.03
CONTROL_HZ = 50
N_ARM = 6
N_CMD = 7  # arm joints + gripper motor


def _clip(pos: np.ndarray) -> np.ndarray:
    return np.clip(pos, EE_BOUNDS_MIN, EE_BOUNDS_MAX)


def _limit_step(pos: np.ndarray, prev: np.ndarray) -> np.ndarray:
    d = pos - prev
    n = np.linalg.norm(d)
    if n > MAX_EE_STEP_M and n > 0:
        pos = prev + d * (MAX_EE_STEP_M / n)
    return pos


def _resolve_model_xml(model_xml: Path | None) -> Path:
    if model_xml is not None:
        return model_xml
    default_xml = DEFAULT_MODEL_DIR / MODEL_XML_NAME
    if not default_xml.exists():
        export_model_assets(DEFAULT_MODEL_DIR)
    return default_xml


def _joint6_axis_override(choice: str) -> np.ndarray | None:
    if choice == "config":
        return None
    if choice == "positive":
        return np.array([0.0, 0.0, 1.0])
    if choice == "negative":
        return np.array([0.0, 0.0, -1.0])
    raise ValueError(f"Unsupported joint6 axis choice: {choice}")


def _print_model_check(model_xml: Path, control_site: str) -> dict:
    validation = validate_mujoco_model(model_xml)
    print(f"Model XML: {model_xml}", flush=True)
    print(
        f"Sites: {EE_SITE} id={validation['grasp_site_id']}  "
        f"{TRACKER_ALIGNED_EE_SITE} id={validation['tracker_aligned_ee_site_id']}  "
        f"{TRACKER_SITE} id={validation['tracker_site_id']}",
        flush=True,
    )
    print(f"Control site: {control_site}", flush=True)
    print(f"joint6 axis: {validation['joint6_axis']}", flush=True)
    print(
        "T_grasp_tracker validation: "
        f"{validation['t_grasp_tracker_matches_config']}",
        flush=True,
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vive Tracker → YAM arm + gripper teleoperation (MuJoCo sim)"
    )
    parser.add_argument(
        "-p", "--port", default=None,
        help="Serial port for BRT encoder (e.g. COM5). Auto-detect if omitted.",
    )
    parser.add_argument(
        "-R", "--resolution", type=int, default=1024,
        help="Encoder resolution (default 1024).",
    )
    parser.add_argument(
        "--ik-method", choices=["jparse", "mink"], default="jparse",
        help="IK solver: jparse (J-PARSE velocity IK, default) or mink.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Directory for YAM teleop episode recordings. Disabled if omitted.",
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=None,
        help=(
            "MuJoCo XML for viewer/replay. Defaults to the exported "
            "sim_teleop model with tracker_site."
        ),
    )
    parser.add_argument(
        "--check-model-only",
        action="store_true",
        help="Validate model XML sites/transforms and exit before OpenVR.",
    )
    parser.add_argument(
        "--control-site",
        choices=[EE_SITE, TRACKER_ALIGNED_EE_SITE],
        default=EE_SITE,
        help=(
            "Site controlled by tracker deltas. Use ee_site to test the "
            "tracker-aligned end-effector frame."
        ),
    )
    parser.add_argument(
        "--joint6-axis",
        choices=["config", "positive", "negative"],
        default="config",
        help=(
            "Experimental override for MuJoCo joint6 axis: config keeps the "
            "i2rt value, positive sets 0 0 1, negative sets 0 0 -1."
        ),
    )
    # J-PARSE tuning
    parser.add_argument("--position-gain",    type=float, default=5.0)
    parser.add_argument("--orientation-gain", type=float, default=1.0)
    parser.add_argument("--max-joint-vel",    type=float, default=5.0)
    parser.add_argument("--gamma",            type=float, default=0.1)
    # mink tuning
    parser.add_argument("--position-cost",    type=float, default=1.0)
    parser.add_argument("--orientation-cost", type=float, default=0.1)
    parser.add_argument("--posture-cost",     type=float, default=0.001)
    args = parser.parse_args()

    joint6_axis = _joint6_axis_override(args.joint6_axis)
    model_xml_path = _resolve_model_xml(args.model_xml)
    if joint6_axis is not None:
        tracker_pos, tracker_quat = tracker_pose_in_gripper_frame()
        model_xml_path = prepare_xml_with_tracker_site(
            model_xml_path,
            tracker_pos,
            tracker_quat,
            joint6_axis=joint6_axis,
        )
    model_validation = _print_model_check(model_xml_path, args.control_site)
    if args.check_model_only:
        return

    # ── OpenVR ────────────────────────────────────────────────────────────────
    print("Initializing OpenVR...", flush=True)
    openvr.init(openvr.VRApplication_Other)
    vr_system = openvr.VRSystem()
    time.sleep(2)

    T = _tracker.read_pose(vr_system)
    if T is None:
        print("ERROR: No tracker found!")
        openvr.shutdown()
        return
    print(f"Tracker at: {T[:3,3].round(4)}", flush=True)

    # ── Encoder / gripper ─────────────────────────────────────────────────────
    inst = None
    port = args.port or _gripper.find_serial_port()
    if port:
        print(f"Connecting to encoder on {port}...", flush=True)
        try:
            inst = _gripper.create_instrument(port)
            val = _gripper.read_raw(inst)
            if val is not None:
                print(f"Encoder OK. Raw={val}", flush=True)
            else:
                print("WARNING: Encoder no response. Gripper disabled.", flush=True)
                inst = None
        except Exception as e:
            print(f"WARNING: Cannot open {port}: {e}. Gripper disabled.", flush=True)
    else:
        print("WARNING: No serial port found. Gripper disabled.", flush=True)

    cal = _gripper.EncoderCalibration.load()
    print(f"Gripper calibration: {cal}", flush=True)

    # ── Robot ─────────────────────────────────────────────────────────────────
    print(f"Building YAM arm + gripper for {args.control_site} control...", flush=True)
    bundle = RobotBundle(control_site=args.control_site, joint6_axis=joint6_axis)
    bundle.start()

    model = mujoco.MjModel.from_xml_path(str(model_xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"Robot ready. nq={model.nq}", flush=True)
    print(
        f"Mount from FK: T_{args.control_site}_tracker "
        f"pos={bundle.t_ee_track[:3,3].round(4)}",
        flush=True,
    )

    recorder: EpisodeRecorder | None = None
    if args.record_dir is not None:
        arg_dict = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        }
        recorder = EpisodeRecorder(
            args.record_dir,
            metadata={
                "schema": "yam_teleop_session_v1",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "argv": sys.argv,
                "args": arg_dict,
                "control_hz": CONTROL_HZ,
                "n_arm": N_ARM,
                "n_cmd": N_CMD,
                "xml_path": str(model_xml_path),
                "control_site": args.control_site,
                "joint6_axis_mode": args.joint6_axis,
                "joint6_axis": array_to_list(joint6_axis)
                if joint6_axis is not None
                else None,
                "grasp_site": EE_SITE,
                "tracker_aligned_ee_site": TRACKER_ALIGNED_EE_SITE,
                "tracker_site": TRACKER_SITE,
                "ee_bounds_min": EE_BOUNDS_MIN.tolist(),
                "ee_bounds_max": EE_BOUNDS_MAX.tolist(),
                "max_ee_step_m": MAX_EE_STEP_M,
                "t_control_tracker": matrix_to_list(bundle.t_ee_track),
                "model_validation": model_validation,
            },
        )
        print(f"[REC] Session: {recorder.session_dir}", flush=True)

    # ── State ─────────────────────────────────────────────────────────────────
    tracker_init_inv: "np.ndarray | None" = None
    ee_init: "np.ndarray | None" = None
    T_site_to_link6: "np.ndarray | None" = None   # control site -> link_6 offset
    jparse_cfg = bundle.sim_robot.get_joint_pos()[:N_ARM].copy()

    mode = "VIS"
    ik_fail_streak = 0
    last_sent_pos: "np.ndarray | None" = None
    last_status_t = 0.0
    last_raw: "int | None" = None
    gripper_norm = 0.5
    should_quit = False

    def reset() -> None:
        nonlocal tracker_init_inv, ee_init, T_site_to_link6
        nonlocal jparse_cfg, ik_fail_streak, last_sent_pos

        T_now = _tracker.read_pose(vr_system)
        if T_now is None:
            print("[WARN] No tracker at reset.", flush=True)
            return

        arm_qpos = bundle.sim_robot.get_joint_pos()[:N_ARM]
        tracker_init_inv = np.linalg.inv(T_now)
        ee_init = bundle.kin.fk(arm_qpos, args.control_site)
        last_sent_pos = ee_init[:3, 3].copy()
        jparse_cfg = arm_qpos.copy()

        # Constant offset from the selected control site to link_6, derived
        # from both FK models at the same config.
        pk_mat = pk_ee_pose(bundle.pk_robot, jparse_cfg, bundle.pk_target_idx)
        T_site_to_link6 = np.linalg.inv(ee_init) @ pk_mat

        ik_fail_streak = 0
        print(
            f"[RESET] {args.control_site}={ee_init[:3,3].round(4)}  "
            f"tracker={T_now[:3,3].round(4)}",
            flush=True,
        )

    def on_key(key: int) -> None:
        nonlocal mode, should_quit
        if key == ord("R"):
            reset()
            mode = "CONTROL"
            print("[MODE] CONTROL", flush=True)
        elif key in (ord("S"), ord("s")):
            if recorder is None:
                print("[REC] Disabled. Run with --record-dir to enable.", flush=True)
            elif recorder.start():
                print(f"[REC] START -> {recorder.session_dir}", flush=True)
            else:
                print("[REC] Already recording.", flush=True)
        elif key in (ord("T"), ord("t")):
            if recorder is None:
                print("[REC] Disabled. Run with --record-dir to enable.", flush=True)
            else:
                out_path = recorder.stop()
                if out_path is None:
                    print("[REC] STOP: no frames saved.", flush=True)
                else:
                    print(f"[REC] SAVED {out_path}", flush=True)
        elif key in (ord("O"), ord("o")):
            if last_raw is not None:
                cal.raw_open = last_raw
                cal.save()
                print(f"[GRIPPER] OPEN raw={last_raw}  {cal}", flush=True)
        elif key in (ord("C"), ord("c")):
            if last_raw is not None:
                cal.raw_closed = last_raw
                cal.save()
                print(f"[GRIPPER] CLOSED raw={last_raw}  {cal}", flush=True)
        elif key in (ord("Q"), ord("q")):
            if recorder is not None and recorder.is_recording:
                out_path = recorder.close()
                if out_path is not None:
                    print(f"[REC] SAVED {out_path}", flush=True)
            print("[QUIT]", flush=True)
            should_quit = True

    print(
        "\n=== R=reset/control  S/T=record start/stop  "
        "O/C=gripper calib  Q=save+quit ===\n",
        flush=True,
    )

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running() and not should_quit:
            frame_tracker_pose: np.ndarray | None = None
            frame_target_pose: np.ndarray | None = None
            frame_realized_pose: np.ndarray | None = None
            frame_arm_q: np.ndarray | None = None
            frame_ik_ok: bool | None = None
            frame_ik_error: float | None = None
            frame_ik_orientation_error_deg: float | None = None

            # ── Encoder ──────────────────────────────────────────────────────
            if inst is not None:
                raw = _gripper.read_raw(inst)
                if raw is not None:
                    last_raw = raw
                    if cal.is_ready:
                        gripper_norm = cal.normalise(raw)

            # ── MuJoCo visualisation ─────────────────────────────────────────
            qpos_cmd = bundle.sim_robot.get_joint_pos()
            data.qpos[:N_ARM] = qpos_cmd[:N_ARM]
            for j in range(N_ARM, model.nq):
                lo, hi = model.jnt_range[j]
                data.qpos[j] = lo + gripper_norm * (hi - lo)
            mujoco.mj_forward(model, data)

            # ── Arm control ───────────────────────────────────────────────────
            if (
                mode == "CONTROL"
                and tracker_init_inv is not None
                and ee_init is not None
            ):
                T_curr = _tracker.read_pose(vr_system)
                if T_curr is not None:
                    frame_tracker_pose = T_curr.copy()

                    # Tracker delta → EE delta (similarity transform)
                    T_ee_delta = ee_delta(
                        tracker_init_inv, T_curr,
                        bundle.t_ee_track, bundle.t_ee_track_inv,
                    )
                    T_target = ee_init @ T_ee_delta

                    # Safety: clamp position + max step
                    target_pos = _clip(T_target[:3, 3])
                    if last_sent_pos is not None:
                        target_pos = _limit_step(target_pos, last_sent_pos)
                    T_target[:3, 3] = target_pos
                    frame_target_pose = T_target.copy()

                    # IK
                    if args.ik_method == "jparse" and T_site_to_link6 is not None:
                        q_sol, info, ok = ik_jparse(
                            bundle, jparse_cfg, T_target, T_site_to_link6,
                            position_gain=args.position_gain,
                            orientation_gain=args.orientation_gain,
                            max_joint_velocity=args.max_joint_vel,
                            gamma=args.gamma,
                            dt=1.0 / CONTROL_HZ,
                        )
                        jparse_cfg = q_sol
                    else:
                        arm_qpos = bundle.sim_robot.get_joint_pos()[:N_ARM]
                        q_sol, ok = ik_mink(
                            bundle, arm_qpos, T_target,
                            position_cost=args.position_cost,
                            orientation_cost=args.orientation_cost,
                            posture_cost=args.posture_cost,
                        )

                    ee_sol = bundle.kin.fk(q_sol, args.control_site)
                    err = float(np.linalg.norm(target_pos - ee_sol[:3, 3]))
                    ori_err_deg = float(
                        np.linalg.norm(
                            delta_rotvec_deg(T_target[:3, :3], ee_sol[:3, :3])
                        )
                    )
                    frame_realized_pose = ee_sol.copy()
                    frame_arm_q = q_sol[:N_ARM].copy()
                    frame_ik_ok = bool(ok)
                    frame_ik_error = err
                    frame_ik_orientation_error_deg = ori_err_deg

                    # Command arm + gripper
                    q_cmd = np.zeros(N_CMD)
                    q_cmd[:N_ARM] = q_sol[:N_ARM]
                    q_cmd[N_ARM] = gripper_norm

                    if ok:
                        bundle.sim_robot.command_joint_pos(q_cmd)
                        last_sent_pos = target_pos.copy()
                        ik_fail_streak = 0
                    elif err < 0.05:
                        bundle.sim_robot.command_joint_pos(q_cmd)
                        last_sent_pos = ee_sol[:3, 3].copy()
                        ik_fail_streak += 1
                    else:
                        ik_fail_streak += 1

                    # Status print at 1 Hz
                    now = time.time()
                    if now - last_status_t >= 1.0:
                        arm_now = bundle.sim_robot.get_joint_pos()[:N_ARM]
                        ee_now = bundle.kin.fk(arm_now, args.control_site)
                        e = float(np.linalg.norm(target_pos - ee_now[:3, 3]))
                        tag = "OK" if ok else f"FAIL(x{ik_fail_streak})"
                        ref_R = ee_init[:3, :3]
                        rv_tgt = delta_rotvec_deg(ref_R, T_target[:3, :3])
                        rv_ee = delta_rotvec_deg(ref_R, ee_now[:3, :3])
                        print(
                            f"[{tag}] err={e:.4f}m  ori={ori_err_deg:.1f}deg  "
                            f"gripper={gripper_norm:.2f}  "
                            f"target={target_pos.round(3)}  "
                            f"{args.control_site}={ee_now[:3,3].round(3)}",
                            flush=True,
                        )
                        print(
                            f"      rpy_target={rv_tgt.round(1)}  "
                            f"rpy_ee={rv_ee.round(1)}  (deg, z=roll)",
                            flush=True,
                        )
                        last_status_t = now

            if (
                recorder is not None
                and recorder.is_recording
                and frame_tracker_pose is not None
            ):
                recorder.append(
                    {
                        "timestamp": time.time(),
                        "mode": mode,
                        "control_site": args.control_site,
                        "tracker_pose": matrix_to_list(frame_tracker_pose),
                        "target_control_pose": matrix_to_list(frame_target_pose),
                        "realized_control_pose": matrix_to_list(frame_realized_pose),
                        "target_ee_pose": matrix_to_list(frame_target_pose),
                        "realized_ee_pose": matrix_to_list(frame_realized_pose),
                        "arm_q": array_to_list(frame_arm_q),
                        "ik_ok": frame_ik_ok,
                        "ik_error_m": frame_ik_error,
                        "ik_orientation_error_deg": frame_ik_orientation_error_deg,
                    }
                )

            viewer.sync()
            time.sleep(1 / CONTROL_HZ)

    if recorder is not None and recorder.is_recording:
        out_path = recorder.close()
        if out_path is not None:
            print(f"[REC] SAVED {out_path}", flush=True)

    openvr.shutdown()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
