from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np

from .camera_base import CameraBase, CameraInfo

LOGGER = logging.getLogger(__name__)


class CameraOpenError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraCandidate:
    index: int
    backend: str
    width: int
    height: int
    fps: float

    def summary(self) -> str:
        return (
            f"index={self.index} backend={self.backend} "
            f"{self.width}x{self.height} @ {self.fps:.1f} FPS"
        )


def _backend_options(name: str) -> list[tuple[str, int]]:
    normalized = name.lower()
    directshow = ("DSHOW", cv2.CAP_DSHOW)
    msmf = ("MSMF", cv2.CAP_MSMF)
    any_backend = ("ANY", cv2.CAP_ANY)
    if os.name != "nt":
        return [any_backend]
    if normalized == "dshow":
        return [directshow]
    if normalized == "msmf":
        return [msmf]
    if normalized == "any":
        return [any_backend]
    return [directshow, msmf]


@contextmanager
def _quiet_videoio():
    """Hide low-level probe noise; our caller reports a useful final error."""
    get_level = getattr(cv2, "getLogLevel", None)
    set_level = getattr(cv2, "setLogLevel", None)
    previous = get_level() if get_level is not None else None
    if set_level is not None:
        set_level(0)
    try:
        yield
    finally:
        if set_level is not None and previous is not None:
            set_level(previous)


def _try_capture(index: int, backend_id: int, warmup_frames: int = 3) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
    capture = cv2.VideoCapture(index, backend_id)
    if not capture.isOpened():
        capture.release()
        return None, None
    frame: np.ndarray | None = None
    ok = False
    for _ in range(max(1, warmup_frames)):
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            break
    if not ok or frame is None or not frame.size:
        capture.release()
        return None, None
    return capture, frame


def discover_cameras(max_index: int = 8, backend: str = "auto") -> list[CameraCandidate]:
    """Probe usable camera indices and return only devices that yield a frame.

    Each physical device is returned once. On Windows automatic probing compares
    DirectShow and Media Foundation; CAP_ANY remains available when requested.
    """
    found: list[CameraCandidate] = []
    with _quiet_videoio():
        for index in range(max(0, max_index)):
            for backend_name, backend_id in _backend_options(backend):
                capture, frame = _try_capture(index, backend_id)
                if capture is None or frame is None:
                    continue
                height, width = frame.shape[:2]
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                found.append(CameraCandidate(index, backend_name, width, height, fps))
                capture.release()
                break
    return found


class OpenCVCamera(CameraBase):
    def __init__(
        self,
        index: int | str = "auto",
        backend: str = "auto",
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        scan_max_index: int = 8,
        warmup_frames: int = 8,
        source_name: str = "APC850",
        pixel_format: str | None = None,
    ) -> None:
        self.requested_index = index
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.scan_max_index = scan_max_index
        self.warmup_frames = warmup_frames
        self.source_name = source_name
        self.pixel_format = pixel_format.strip().upper() if pixel_format else None
        self.capture: cv2.VideoCapture | None = None
        self.info: CameraInfo | None = None

    def _indices(self) -> list[int]:
        if isinstance(self.requested_index, int):
            return [self.requested_index]
        text = str(self.requested_index).strip().lower()
        if text != "auto":
            try:
                return [int(text)]
            except ValueError as exc:
                raise CameraOpenError(f"Invalid camera index: {self.requested_index!r}") from exc
        return list(range(max(0, self.scan_max_index)))

    def open(self) -> CameraInfo:
        if self.capture is not None and self.capture.isOpened() and self.info is not None:
            return self.info

        attempts: list[str] = []
        with _quiet_videoio():
            for index in self._indices():
                for backend_name, backend_id in _backend_options(self.backend):
                    attempts.append(f"{index}/{backend_name}")
                    capture = cv2.VideoCapture(index, backend_id)
                    if not capture.isOpened():
                        capture.release()
                        continue

                    if self.pixel_format and len(self.pixel_format) == 4:
                        capture.set(
                            cv2.CAP_PROP_FOURCC,
                            cv2.VideoWriter_fourcc(*self.pixel_format),
                        )
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    capture.set(cv2.CAP_PROP_FPS, self.fps)
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    frame = None
                    ok = False
                    for _ in range(max(1, self.warmup_frames)):
                        ok, frame = capture.read()
                        if ok and frame is not None and frame.size:
                            break
                    if not ok or frame is None or not frame.size:
                        capture.release()
                        continue

                    actual_height, actual_width = frame.shape[:2]
                    actual_fps = float(capture.get(cv2.CAP_PROP_FPS)) or self.fps
                    self.capture = capture
                    self.info = CameraInfo(
                        source=self.source_name,
                        index=index,
                        backend=backend_name,
                        width=actual_width,
                        height=actual_height,
                        fps=actual_fps,
                    )
                    LOGGER.info("Opened camera: %s", self.info.as_dict())
                    return self.info

        tried = ", ".join(attempts) if attempts else "no indices"
        raise CameraOpenError(
            "No camera produced a valid frame. "
            f"Tried {tried}. Close apps using the webcam, check Windows camera "
            "privacy permission, or run `python camera_probe.py`."
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.capture is None or not self.capture.isOpened():
            raise CameraOpenError("Camera is not open")
        ok, frame = self.capture.read()
        if not ok or frame is None or not frame.size:
            return False, None
        return True, frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.info = None
