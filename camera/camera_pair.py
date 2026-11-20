from __future__ import annotations

import time
import threading
from dataclasses import dataclass

import numpy as np

from .camera_base import CameraInfo
from .opencv_camera import OpenCVCamera


@dataclass(frozen=True)
class CameraPairInfo:
    global_camera: CameraInfo
    wrist_camera: CameraInfo


@dataclass(frozen=True)
class FramePair:
    global_frame: np.ndarray
    wrist_frame: np.ndarray
    timestamp: float
    capture_skew_ms: float


class OpenCVCameraPair:
    """Two USB cameras sampled while the robot is stopped at its inspection pose.

    Consumer webcams do not provide hardware triggering. The measured read skew
    is exposed so callers never mistake this adapter for synchronized stereo.
    """

    def __init__(self, global_camera: OpenCVCamera, wrist_camera: OpenCVCamera) -> None:
        self.global_camera = global_camera
        self.wrist_camera = wrist_camera
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray | None] = {"global": None, "wrist": None}
        self._timestamps: dict[str, float] = {"global": 0.0, "wrist": 0.0}

    def _capture_loop(self, name: str, camera: OpenCVCamera) -> None:
        while not self._stop_event.is_set():
            try:
                ok, frame = camera.read()
            except Exception:
                break
            if not ok or frame is None:
                continue
            timestamp = time.perf_counter()
            with self._lock:
                self._frames[name] = frame
                self._timestamps[name] = timestamp

    def open(self) -> CameraPairInfo:
        global_info = self.global_camera.open()
        try:
            wrist_info = self.wrist_camera.open()
        except Exception:
            self.global_camera.release()
            raise
        if global_info.index == wrist_info.index and global_info.backend == wrist_info.backend:
            self.release()
            raise RuntimeError("Global and wrist camera resolved to the same device")
        self._stop_event.clear()
        self._threads = [
            threading.Thread(target=self._capture_loop, args=("global", self.global_camera), daemon=True),
            threading.Thread(target=self._capture_loop, args=("wrist", self.wrist_camera), daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        return CameraPairInfo(global_info, wrist_info)

    def read(self) -> tuple[bool, FramePair | None]:
        with self._lock:
            global_frame = self._frames["global"]
            wrist_frame = self._frames["wrist"]
            global_timestamp = self._timestamps["global"]
            wrist_timestamp = self._timestamps["wrist"]
        if global_frame is None or wrist_frame is None:
            return False, None
        return True, FramePair(
            global_frame=global_frame,
            wrist_frame=wrist_frame,
            timestamp=time.time(),
            capture_skew_ms=abs(wrist_timestamp - global_timestamp) * 1000.0,
        )

    def release(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()
        self.global_camera.release()
        self.wrist_camera.release()
        with self._lock:
            self._frames = {"global": None, "wrist": None}
            self._timestamps = {"global": 0.0, "wrist": 0.0}

    def __enter__(self) -> "OpenCVCameraPair":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
