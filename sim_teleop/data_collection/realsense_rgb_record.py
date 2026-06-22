"""Record RGB-only RealSense streams for a raw data-collection smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def _import_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in this Python environment. "
            "Use third_party\\HuMI-main\\.realsense-env\\Scripts\\python.exe."
        ) from exc
    return rs


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required for RGB video recording.") from exc
    return cv2


def _host_time(wall0: float | None, perf0: float | None) -> float:
    if wall0 is None or perf0 is None:
        return time.time()
    return wall0 + (time.perf_counter() - perf0)


def _devices(rs) -> list:
    devices = []
    for dev in rs.context().query_devices():
        name = dev.get_info(rs.camera_info.name)
        if name.lower() == "platform camera":
            continue
        devices.append(dev)
    return devices


def _device_info(rs, dev) -> dict[str, str]:
    out = {}
    for field in ("name", "product_line", "serial_number", "firmware_version"):
        try:
            out[field] = dev.get_info(getattr(rs.camera_info, field))
        except Exception:
            pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument("--wall0", type=float, default=None)
    parser.add_argument("--perf0", type=float, default=None)
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    rs = _import_realsense()
    cv2 = _import_cv2()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    devices = _devices(rs)
    if args.max_cameras is not None:
        devices = devices[: args.max_cameras]
    if not devices:
        raise SystemExit("ERROR: no RealSense devices found.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    pipelines = []
    writers = []
    camera_meta = []
    host_timestamps: list[list[float]] = []
    device_timestamps: list[list[float]] = []
    receive_durations_ms: list[list[float]] = []
    last_images = {}

    try:
        for cam_idx, dev in enumerate(devices):
            info = _device_info(rs, dev)
            serial = info.get("serial_number", f"camera_{cam_idx}")
            cam_dir = args.output_dir / f"cam{cam_idx}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            print(f"[RS-REC] cam{cam_idx} {info}", flush=True)

            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(
                rs.stream.color,
                args.width,
                args.height,
                rs.format.bgr8,
                args.fps,
            )
            pipeline = rs.pipeline()
            pipeline.start(cfg)
            pipelines.append(pipeline)

            video_path = cam_dir / "color.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                args.fps,
                (args.width, args.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {video_path}")
            writers.append(writer)
            camera_meta.append(
                {
                    "camera_index": cam_idx,
                    "serial_number": serial,
                    "info": info,
                    "video_path": str(video_path),
                }
            )
            host_timestamps.append([])
            device_timestamps.append([])
            receive_durations_ms.append([])

        time.sleep(1.0)
        end_time = time.time() + args.duration
        while time.time() < end_time:
            for cam_idx, pipeline in enumerate(pipelines):
                recv_start = _host_time(args.wall0, args.perf0)
                frameset = pipeline.wait_for_frames(timeout_ms=5000)
                recv_end = _host_time(args.wall0, args.perf0)
                color_frame = frameset.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data())
                writers[cam_idx].write(image)
                last_images[cam_idx] = image
                host_timestamps[cam_idx].append(recv_start + 0.5 * (recv_end - recv_start))
                device_timestamps[cam_idx].append(color_frame.get_timestamp() / 1000.0)
                receive_durations_ms[cam_idx].append((recv_end - recv_start) * 1000.0)

        for cam_idx, cam_dir in enumerate(args.output_dir.glob("cam*")):
            np.save(cam_dir / "color_timestamps.npy", np.asarray(host_timestamps[cam_idx]))
            np.save(cam_dir / "device_timestamps.npy", np.asarray(device_timestamps[cam_idx]))
            np.save(
                cam_dir / "receive_durations_ms.npy",
                np.asarray(receive_durations_ms[cam_idx]),
            )
            if cam_idx in last_images:
                cv2.imwrite(str(cam_dir / "sample.png"), last_images[cam_idx])
            print(
                f"[RS-REC] cam{cam_idx} frames={len(host_timestamps[cam_idx])}",
                flush=True,
            )

        metadata = {
            "schema": "realsense_rgb_raw_v1",
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "duration": args.duration,
            "wall0": args.wall0,
            "perf0": args.perf0,
            "cameras": camera_meta,
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
    finally:
        for writer in writers:
            writer.release()
        for pipeline in pipelines:
            pipeline.stop()
        print("[RS-REC] stopped", flush=True)


if __name__ == "__main__":
    main()

