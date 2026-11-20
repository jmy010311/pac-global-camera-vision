from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from models.scene_state import OccupancyState

from .grid import GridCell


@dataclass(frozen=True)
class CellOccupancy:
    state: OccupancyState
    score: float
    confidence: float


class OccupancyDetector:
    """Appearance-independent first-pass occupancy detector.

    Each cell estimates its own floor colour from side/bottom bands, then scores
    colour change, foreground coverage, highlights and texture in the central
    manipulation region. No fixed blue/white colour is assumed.
    """

    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.empty_threshold = float(config.get("empty_threshold", 0.24))
        self.occupied_threshold = float(config.get("occupied_threshold", 0.42))
        self.inset_ratio = float(config.get("cell_inset_ratio", 0.08))
        self.color_delta_threshold = float(config.get("color_delta_threshold", 26.0))
        self.temporal_alpha = float(config.get("temporal_alpha", 0.4))
        self.filtered_scores: dict[int, float] = {}
        self.states: dict[int, OccupancyState] = {}
        self.last_shape: tuple[int, int] | None = None

    def _crop(self, image: np.ndarray, cell: GridCell) -> np.ndarray | None:
        polygon = cell.polygon
        x0, y0 = np.floor(polygon.min(axis=0)).astype(int)
        x1, y1 = np.ceil(polygon.max(axis=0)).astype(int)
        width, height = x1 - x0, y1 - y0
        inset_x = max(3, int(round(width * self.inset_ratio)))
        inset_y = max(3, int(round(height * self.inset_ratio)))
        x0, x1 = x0 + inset_x, x1 - inset_x
        y0, y1 = y0 + inset_y, y1 - inset_y
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        return image[y0:y1, x0:x1]

    def _background_centres(self, image: np.ndarray, cells: list[GridCell]) -> np.ndarray:
        samples: list[np.ndarray] = []
        for cell in cells:
            crop = self._crop(image, cell)
            if crop is None:
                continue
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            h, w = lab.shape[:2]
            region = lab[int(h * 0.28) : int(h * 0.88), int(w * 0.12) : int(w * 0.88)]
            samples.append(region.reshape(-1, 3)[::4])
        if not samples:
            return np.array([[128.0, 128.0, 128.0]], dtype=np.float32)
        pixels = np.vstack(samples).astype(np.float32)
        cluster_count = min(4, max(1, len(pixels) // 200))
        cv2.setRNGSeed(2026)
        _, labels, centres = cv2.kmeans(
            pixels,
            cluster_count,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2),
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        counts = np.bincount(labels.ravel(), minlength=cluster_count)
        dominant_index = int(np.argmax(counts))
        dominant = centres[dominant_index]
        selected = [dominant]
        for index, centre in enumerate(centres):
            chroma_distance = float(np.linalg.norm(centre[1:] - dominant[1:]))
            if index != dominant_index and chroma_distance < 12.0 and counts[index] >= counts[dominant_index] * 0.35:
                selected.append(centre)
        return np.asarray(selected, dtype=np.float32)

    def _raw_score(
        self,
        image: np.ndarray,
        cell: GridCell,
        background_centres: np.ndarray,
    ) -> float:
        crop = self._crop(image, cell)
        if crop is None:
            return 0.5

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w]
        nx = xx / max(w - 1, 1)
        ny = yy / max(h - 1, 1)

        evaluation = (nx >= 0.12) & (nx <= 0.88) & (ny >= 0.28) & (ny <= 0.88)
        if int(evaluation.sum()) < 20:
            return 0.5

        delta = np.min(
            np.linalg.norm(lab[:, :, None, :] - background_centres[None, None, :, :], axis=3),
            axis=2,
        )
        evaluated_delta = delta[evaluation]
        foreground_fraction = float(np.mean(evaluated_delta > self.color_delta_threshold))
        median_delta = float(np.median(evaluated_delta))

        background_l = float(np.max(background_centres[:, 0]))
        highlights = float(np.mean(lab[:, :, 0][evaluation] > background_l + 32.0))
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        texture = float(np.mean(laplacian[evaluation]))

        foreground_score = float(np.clip((foreground_fraction - 0.08) / 0.42, 0.0, 1.0))
        colour_score = float(np.clip((median_delta - 10.0) / 38.0, 0.0, 1.0))
        highlight_score = float(np.clip(highlights / 0.20, 0.0, 1.0))
        texture_score = float(np.clip((texture - 18.0) / 48.0, 0.0, 1.0))
        return float(
            np.clip(
                0.50 * foreground_score
                + 0.24 * colour_score
                + 0.16 * highlight_score
                + 0.10 * texture_score,
                0.0,
                1.0,
            )
        )

    def classify(
        self,
        image: np.ndarray,
        cells: list[GridCell],
        rows: int,
        cols: int,
    ) -> list[CellOccupancy]:
        shape = (rows, cols)
        if shape != self.last_shape:
            self.filtered_scores.clear()
            self.states.clear()
            self.last_shape = shape

        results: list[CellOccupancy] = []
        background_centres = self._background_centres(image, cells)
        for cell in cells:
            raw = self._raw_score(image, cell, background_centres) if self.enabled else 0.5
            previous = self.filtered_scores.get(cell.id)
            if previous is None:
                score = raw
            else:
                alpha = max(0.45, self.temporal_alpha) if raw > previous + 0.18 else self.temporal_alpha
                score = (1.0 - alpha) * previous + alpha * raw
            self.filtered_scores[cell.id] = score

            if not self.enabled:
                state = OccupancyState.UNKNOWN
                confidence = 0.0
            elif self.states.get(cell.id) == OccupancyState.EMPTY and score <= self.empty_threshold + 0.05:
                state = OccupancyState.EMPTY
                confidence = float(np.clip(1.0 - score / max(self.empty_threshold + 0.05, 1e-6), 0.0, 1.0))
            elif self.states.get(cell.id) == OccupancyState.OCCUPIED and score >= self.occupied_threshold - 0.08:
                state = OccupancyState.OCCUPIED
                confidence = float(np.clip(score, 0.0, 1.0))
            elif score <= self.empty_threshold:
                state = OccupancyState.EMPTY
                confidence = float(np.clip(1.0 - score / max(self.empty_threshold, 1e-6), 0.0, 1.0))
            elif score >= self.occupied_threshold:
                state = OccupancyState.OCCUPIED
                confidence = float(
                    np.clip((score - self.occupied_threshold) / max(1.0 - self.occupied_threshold, 1e-6), 0.0, 1.0)
                )
            else:
                state = OccupancyState.UNKNOWN
                midpoint = (self.empty_threshold + self.occupied_threshold) / 2.0
                half_range = max((self.occupied_threshold - self.empty_threshold) / 2.0, 1e-6)
                confidence = float(np.clip(1.0 - abs(score - midpoint) / half_range, 0.0, 1.0))
            self.states[cell.id] = state
            results.append(CellOccupancy(state, round(score, 4), round(confidence, 4)))
        return results
