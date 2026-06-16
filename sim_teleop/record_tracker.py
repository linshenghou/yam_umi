"""Record a single Vive Tracker pose stream to HuMI-style episode JSON."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import openvr

from .tracker import TRACKER_SERIAL_PREFIX, read_tracker_poses


def _get_key() -> str | None:
    try:
        import msvcrt  # type: ignore
    except ImportError:
        return None
    if not msvcrt.kbhit():
        return None
    key = msvcrt.getwch()
    if key == "\x1b":
        return "esc"
    return key.lower()


def _new_session(output_dir: Path, metadata: dict) -> Path:
    session_dir = output_dir / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return session_dir


def _save_episode(session_dir: Path, start_ts: float, frames: list[dict]) -> Path | None:
    if not frames:
        return None
    ts = datetime.fromtimestamp(start_ts).strftime("%Y.%m.%d_%H.%M.%S.%f")
    out_path = session_dir / f"tracker_recording_{ts}.json"
    payload = {
        "schema": "vive_tracker_pose_episode_v1",
        "metadata": "metadata.json",
        "episode": frames,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one Vive Tracker pose stream.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/tracker_poses"),
        help="Root directory for tracker pose sessions.",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        type=float,
        default=120.0,
        help="Recording frequency in Hz.",
    )
    parser.add_argument(
        "--serial-prefix",
        default=TRACKER_SERIAL_PREFIX,
        help="Only record trackers whose serial starts with this prefix.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start recording immediately instead of waiting for S.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop and save after this many seconds once recording starts.",
    )
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0.0:
        raise ValueError("--duration must be positive.")

    openvr.init(openvr.VRApplication_Other)
    vr_system = openvr.VRSystem()
    time.sleep(2.0)

    metadata = {
        "schema": "vive_tracker_pose_session_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frequency": args.frequency,
        "duration": args.duration,
        "serial_prefix": args.serial_prefix,
        "tracking_universe": "TrackingUniverseStanding",
        "controls": {
            "s": "start recording",
            "t": "stop and save",
            "q_or_esc": "save active recording and quit",
        },
    }
    session_dir = _new_session(args.output_dir, metadata)

    frames: list[dict] = []
    recording = False
    start_ts: float | None = None
    last_status = 0.0
    dt = 1.0 / args.frequency

    if args.auto_start:
        recording = True
        start_ts = time.time()

    print(f"[TRACKER] Session: {session_dir}", flush=True)
    print("[TRACKER] S=start  T=stop/save  Q/Esc=save+quit", flush=True)

    try:
        while True:
            loop_start = time.time()
            tracker_poses = read_tracker_poses(vr_system, args.serial_prefix)
            if tracker_poses:
                serial, pose = tracker_poses[0]
                if recording:
                    frames.append(
                        {
                            "timestamp": loop_start,
                            "serial": serial,
                            "tracker_pose": pose.tolist(),
                        }
                    )

            key = _get_key()
            if key == "s":
                if not recording:
                    frames.clear()
                    start_ts = time.time()
                    recording = True
                    print("[TRACKER] START", flush=True)
                else:
                    print("[TRACKER] Already recording.", flush=True)
            elif key == "t":
                if recording and start_ts is not None:
                    out_path = _save_episode(session_dir, start_ts, frames)
                    print(f"[TRACKER] SAVED {out_path}", flush=True)
                    frames.clear()
                    start_ts = None
                    recording = False
                else:
                    print("[TRACKER] Not recording.", flush=True)
            elif key in ("q", "esc"):
                if recording and start_ts is not None:
                    out_path = _save_episode(session_dir, start_ts, frames)
                    print(f"[TRACKER] SAVED {out_path}", flush=True)
                break

            now = time.time()
            if now - last_status >= 1.0:
                status = "REC" if recording else "IDLE"
                found = len(tracker_poses)
                count = len(frames)
                print(
                    f"[{status}] trackers={found} frames={count}",
                    flush=True,
                )
                last_status = now

            if (
                args.duration is not None
                and recording
                and start_ts is not None
                and loop_start - start_ts >= args.duration
            ):
                out_path = _save_episode(session_dir, start_ts, frames)
                print(f"[TRACKER] SAVED {out_path}", flush=True)
                break

            sleep_s = max(dt - (time.time() - loop_start), 0.0)
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        if recording and start_ts is not None:
            out_path = _save_episode(session_dir, start_ts, frames)
            print(f"[TRACKER] SAVED {out_path}", flush=True)
    finally:
        openvr.shutdown()
        print("[TRACKER] Done.", flush=True)


if __name__ == "__main__":
    main()
