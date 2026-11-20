from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class LensUndistorter:
    """Apply camera-specific radial/tangential lens correction before vision."""

    def __init__(self, camera_matrix: np.ndarray, coefficients: np.ndarray, image_size: tuple[int, int], alpha: float = 0.0) -> None:
        self.camera_matrix = camera_matrix.astype(np.float64)
        self.coefficients = coefficients.astype(np.float64)
        self.calibration_size = image_size
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._output_matrices: dict[tuple[int, int], np.ndarray] = {}

    @classmethod
    def from_config(cls, config: dict, project_root: Path) -> "LensUndistorter | None":
        if not bool(config.get("enabled", False)):
            return None
        path = Path(str(config.get("file", "config/camera_calibration.json")))
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            LOGGER.warning("Lens calibration is not created yet: %s", path)
            LOGGER.warning("Continuing with RAW image. Run `python calibrate_camera.py` after preparing a checkerboard.")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            matrix = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
            coefficients = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1)
            image_size_values = [int(value) for value in data["image_size"]]
            if len(image_size_values) != 2:
                raise ValueError("image_size must be [width, height]")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid camera calibration file {path}: {exc}") from exc
        LOGGER.info("Loaded lens calibration: %s", path)
        return cls(matrix, coefficients, tuple(image_size_values), float(config.get("alpha", 0.0)))

    def _build_maps(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        scale_x = width / max(self.calibration_size[0], 1)
        scale_y = height / max(self.calibration_size[1], 1)
        matrix = self.camera_matrix.copy()
        matrix[0, 0] *= scale_x
        matrix[0, 2] *= scale_x
        matrix[1, 1] *= scale_y
        matrix[1, 2] *= scale_y
        optimal, _ = cv2.getOptimalNewCameraMatrix(
            matrix,
            self.coefficients,
            (width, height),
            self.alpha,
            (width, height),
        )
        self._output_matrices[(width, height)] = np.asarray(optimal, np.float64)
        return cv2.initUndistortRectifyMap(
            matrix,
            self.coefficients,
            None,
            optimal,
            (width, height),
            cv2.CV_16SC2,
        )

    def output_camera_matrix(self, width: int, height: int) -> np.ndarray:
        """Intrinsics of the *corrected* image produced by :meth:`apply` (zero distortion)."""
        key = (int(width), int(height))
        if key not in self._maps:
            self._maps[key] = self._build_maps(*key)
        return self._output_matrices[key].copy()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        key = (width, height)
        if key not in self._maps:
            self._maps[key] = self._build_maps(width, height)
        map_x, map_y = self._maps[key]
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

