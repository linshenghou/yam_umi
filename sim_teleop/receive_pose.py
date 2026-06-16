"""Receive streamed Vive Tracker poses over ZMQ from a LAN sender (Windows).

This receiver is standalone: it does NOT import openvr or any other
``sim_teleop`` module, so it runs on Ubuntu with only ``pyzmq`` installed.
The pose is a 4x4 homogeneous matrix in the SteamVR standing frame, identical
to what ``sim_teleop.record_tracker`` writes to disk.

Run on Ubuntu (from the repo root):

    python -m sim_teleop.receive_pose --host <windows_lan_ip> --port 1234

Or directly without the package machinery:

    python sim_teleop/receive_pose.py --host <windows_lan_ip> --port 1234

Wire message consumed (JSON)::

    {
      "timestamp": 1780000000.123,
      "trackers": [
        {"serial": "3B-A33M...", "tracker_pose": [[...4x4...]]},
        ...
      ]
    }

``tracker_pose`` row 3 = [x, y, z] translation; the top-left 3x3 block is the
rotation. Parse it with numpy if needed:

    import numpy as np
    pose = np.array(tracker["tracker_pose"])  # 4x4
    pos = pose[:3, 3]
    rot = pose[:3, :3]
"""
from __future__ import annotations

import argparse
import time

import zmq


def _format_frame(msg: dict) -> str:
    ts = msg["timestamp"]
    lines = [f"t={ts:.3f}"]
    for tk in msg["trackers"]:
        p = tk["tracker_pose"]  # 4x4 nested list
        x, y, z = p[0][3], p[1][3], p[2][3]
        lines.append(
            f"  {tk['serial']}: pos=({x:+.3f},{y:+.3f},{z:+.3f})"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Receive streamed Vive Tracker poses over ZMQ (LAN).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="IP of the Windows sender (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1234,
        help="TCP port the sender binds (default: 1234).",
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=0.5,
        help=(
            "Print one frame every N seconds. 0 = print every frame "
            "(default: 0.5)."
        ),
    )
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    # Keep only the most recent frame: critical for real-time use. Requires
    # RCVHWM == 1.
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(f"tcp://{args.host}:{args.port}")
    print(
        f"[RECV] SUB connected to tcp://{args.host}:{args.port} "
        "(waiting for first frame...)",
        flush=True,
    )

    last_print = 0.0
    count = 0
    try:
        while True:
            msg = socket.recv_json()
            count += 1
            now = time.time()
            if args.print_rate <= 0 or now - last_print >= args.print_rate:
                print(_format_frame(msg), flush=True)
                last_print = now
    except KeyboardInterrupt:
        print(f"[RECV] Stopping (received {count} frames)...", flush=True)
    finally:
        socket.close(linger=0)
        context.term()
        print("[RECV] Done.", flush=True)


if __name__ == "__main__":
    main()
