"""YAM teleoperation episode recording helpers."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def matrix_to_list(mat: np.ndarray | None) -> list[list[float]] | None:
    if mat is None:
        return None
    return np.asarray(mat, dtype=float).tolist()


def array_to_list(arr: np.ndarray | None) -> list[float] | None:
    if arr is None:
        return None
    return np.asarray(arr, dtype=float).tolist()


class EpisodeRecorder:
    """Write YAM teleop demonstrations as HuMI-style episode JSON files."""

    def __init__(self, root_dir: Path, metadata: dict[str, Any]) -> None:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = root_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.session_dir / "metadata.json"
        self.metadata_path.write_text(json.dumps(metadata, indent=2))

        self._frames: list[dict[str, Any]] = []
        self._recording = False
        self._start_ts: float | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def start(self) -> bool:
        if self._recording:
            return False
        self._frames.clear()
        self._start_ts = time.time()
        self._recording = True
        return True

    def append(self, frame: dict[str, Any]) -> None:
        if self._recording:
            self._frames.append(frame)

    def stop(self) -> Path | None:
        if not self._recording:
            return None
        self._recording = False
        if not self._frames:
            self._start_ts = None
            return None

        start_ts = self._start_ts or self._frames[0]["timestamp"]
        ts = datetime.fromtimestamp(start_ts).strftime("%Y.%m.%d_%H.%M.%S.%f")
        out_path = self.session_dir / f"recording_{ts}.json"
        payload = {
            "schema": "yam_teleop_episode_v1",
            "metadata": "metadata.json",
            "episode": self._frames,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        self._frames.clear()
        self._start_ts = None
        return out_path

    def close(self) -> Path | None:
        return self.stop() if self._recording else None
