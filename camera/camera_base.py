from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraInfo:
    source: str
    index: int | str
    backend: str
    width: int
    height: int
    fps: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "index": self.index,
            "backend": self.backend,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
        }


class CameraBase(ABC):
    """Image-source-neutral interface used by the vision processor."""

    @abstractmethod
    def open(self) -> CameraInfo:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "CameraBase":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()

