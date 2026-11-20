from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MeshRectification:
    image: np.ndarray
    nodes: np.ndarray
    confidence: float
    applied: bool


class GridMeshRectifier:
    """Non-rigid grid rectification using observed partition intersections.

    A global homography provides coarse alignment. Grid intersections are then
    used as control points and every cell receives its own local homography.
    This piecewise-planar model tolerates bowed cardboard that cannot be
    represented by one four-corner projective transform.
    """

    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.search_ratio = float(config.get("search_ratio", 0.22))
        self.max_displacement_ratio = float(config.get("max_displacement_ratio", 0.10))
        self.regularization_strength = float(config.get("regularization_strength", 0.30))
        self.regularization_iterations = max(1, int(config.get("regularization_iterations", 1)))
        self.min_confidence = float(config.get("min_confidence", 0.28))
        self.temporal_alpha = float(config.get("temporal_alpha", 0.14))
        self.deadband_px = float(config.get("deadband_px", 1.5))
        self.lock_after_frames = max(2, int(config.get("lock_after_frames", 10)))
        self.unlock_px = float(config.get("unlock_px", 8.0))
        self.unlock_frames = max(2, int(config.get("unlock_frames", 5)))
        self.previous_nodes: np.ndarray | None = None
        self.previous_shape: tuple[int, int] | None = None
        self.tracked_frames = 0
        self.motion_frames = 0
        self.locked = False

    @staticmethod
    def _regular_nodes(width: int, height: int, rows: int, cols: int) -> np.ndarray:
        xs = np.linspace(0, width - 1, cols + 1, dtype=np.float32)
        ys = np.linspace(0, height - 1, rows + 1, dtype=np.float32)
        return np.stack(np.meshgrid(xs, ys), axis=-1)

    @staticmethod
    def _corner_response(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        harris = cv2.cornerHarris(np.float32(blurred) / 255.0, 5, 3, 0.04)
        harris = cv2.GaussianBlur(np.maximum(harris, 0), (5, 5), 0)
        gradient_x = cv2.GaussianBlur(np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)), (7, 7), 0)
        gradient_y = cv2.GaussianBlur(np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)), (7, 7), 0)
        crossing = gradient_x * gradient_y
        if float(harris.max()) > 0:
            harris /= float(harris.max())
        if float(crossing.max()) > 0:
            crossing /= float(crossing.max())
        return gray, 0.65 * harris + 0.35 * crossing

    def _detect_nodes(
        self,
        image: np.ndarray,
        rows: int,
        cols: int,
        row_lines: list[float],
        col_lines: list[float],
    ) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        regular = self._regular_nodes(width, height, rows, cols)
        base_x = np.asarray([0.0, *sorted(col_lines), float(width - 1)], dtype=np.float32)
        base_y = np.asarray([0.0, *sorted(row_lines), float(height - 1)], dtype=np.float32)
        if len(base_x) != cols + 1:
            base_x = regular[0, :, 0]
        if len(base_y) != rows + 1:
            base_y = regular[:, 0, 1]
        base_nodes = np.stack(np.meshgrid(base_x, base_y), axis=-1).astype(np.float32)

        gray, response = self._corner_response(image)
        nodes = np.zeros_like(regular)
        accepted = 0
        quality_sum = 0.0
        cell_width = width / cols
        cell_height = height / rows
        radius_x = max(8, int(round(cell_width * self.search_ratio)))
        radius_y = max(8, int(round(cell_height * self.search_ratio)))

        for row in range(rows + 1):
            for col in range(cols + 1):
                expected = np.array([base_x[col], base_y[row]], dtype=np.float32)
                if row in (0, rows) or col in (0, cols):
                    if row == 0:
                        expected[1] = 0.0
                    elif row == rows:
                        expected[1] = float(height - 1)
                    if col == 0:
                        expected[0] = 0.0
                    elif col == cols:
                        expected[0] = float(width - 1)
                    nodes[row, col] = expected
                    continue
                x0 = max(0, int(round(expected[0])) - radius_x)
                x1 = min(width, int(round(expected[0])) + radius_x + 1)
                y0 = max(0, int(round(expected[1])) - radius_y)
                y1 = min(height, int(round(expected[1])) + radius_y + 1)
                window = response[y0:y1, x0:x1]
                if window.size == 0:
                    nodes[row, col] = expected
                    continue
                grid_y, grid_x = np.mgrid[0 : window.shape[0], 0 : window.shape[1]]
                expected_x = expected[0] - x0
                expected_y = expected[1] - y0
                sigma_x = max(3.0, radius_x * 0.48)
                sigma_y = max(3.0, radius_y * 0.48)
                prior = np.exp(
                    -0.5
                    * (
                        ((grid_x - expected_x) / sigma_x) ** 2
                        + ((grid_y - expected_y) / sigma_y) ** 2
                    )
                ).astype(np.float32)
                weighted = window * prior
                _, _, _, location = cv2.minMaxLoc(weighted)
                peak = float(window[location[1], location[0]])
                candidate = np.array([x0 + location[0], y0 + location[1]], dtype=np.float32)
                local_reference = float(np.percentile(window, 92)) + 1e-8
                quality = float(np.clip(peak / (local_reference * 3.0), 0.0, 1.0))
                displacement = np.abs(candidate - expected)
                max_dx = max(6.0, cell_width * self.max_displacement_ratio)
                max_dy = max(6.0, cell_height * self.max_displacement_ratio)
                within = displacement[0] <= max_dx and displacement[1] <= max_dy
                if peak > 0.035 and within:
                    point = candidate.reshape(1, 1, 2)
                    try:
                        cv2.cornerSubPix(
                            gray,
                            point,
                            (5, 5),
                            (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03),
                        )
                        candidate = point.reshape(2)
                    except cv2.error:
                        pass
                    refined_displacement = np.abs(candidate - expected)
                    if refined_displacement[0] > max_dx or refined_displacement[1] > max_dy:
                        candidate = expected
                        within = False
                    if not within:
                        nodes[row, col] = expected
                        continue
                    nodes[row, col] = candidate
                    accepted += 1
                    quality_sum += quality
                else:
                    nodes[row, col] = expected

        nodes = self._regularize_nodes(nodes, base_nodes, rows, cols, min(cell_width, cell_height))
        total = max(1, (rows - 1) * (cols - 1))
        confidence = 0.65 * accepted / total + 0.35 * quality_sum / total
        return nodes, float(np.clip(confidence, 0.0, 1.0))

    def _regularize_nodes(
        self,
        nodes: np.ndarray,
        base_nodes: np.ndarray,
        rows: int,
        cols: int,
        cell_size: float,
    ) -> np.ndarray:
        """Smooth the control-point displacement field and suppress one-point kinks."""
        displacement = nodes - base_nodes
        strength = float(np.clip(self.regularization_strength, 0.0, 0.9))
        max_shift = max(5.0, cell_size * self.max_displacement_ratio)
        for _ in range(self.regularization_iterations):
            updated = displacement.copy()
            for row in range(1, rows):
                for col in range(1, cols):
                    neighbours = np.array(
                        [
                            displacement[row - 1, col],
                            displacement[row + 1, col],
                            displacement[row, col - 1],
                            displacement[row, col + 1],
                        ],
                        dtype=np.float32,
                    )
                    neighbour_mean = neighbours.mean(axis=0)
                    residual = float(np.linalg.norm(displacement[row, col] - neighbour_mean))
                    local_strength = min(0.85, strength + 0.20) if residual > max_shift * 0.55 else strength
                    updated[row, col] = (
                        (1.0 - local_strength) * displacement[row, col]
                        + local_strength * neighbour_mean
                    )
                    norm = float(np.linalg.norm(updated[row, col]))
                    if norm > max_shift:
                        updated[row, col] *= max_shift / norm
            displacement = updated
        displacement[0, :, :] = 0
        displacement[-1, :, :] = 0
        displacement[:, 0, :] = 0
        displacement[:, -1, :] = 0
        return (base_nodes + displacement).astype(np.float32)

    def _stabilize(self, nodes: np.ndarray, rows: int, cols: int) -> np.ndarray:
        shape = (rows, cols)
        if self.previous_nodes is None or self.previous_shape != shape:
            self.previous_nodes = nodes.copy()
            self.previous_shape = shape
            self.tracked_frames = 1
            self.motion_frames = 0
            self.locked = False
            return nodes
        displacement = np.linalg.norm(nodes - self.previous_nodes, axis=2)
        if self.locked:
            if float(np.mean(displacement)) < self.unlock_px:
                self.motion_frames = 0
                return self.previous_nodes
            self.motion_frames += 1
            if self.motion_frames < self.unlock_frames:
                return self.previous_nodes
            self.locked = False
            self.tracked_frames = 0
            self.motion_frames = 0
        target = nodes.copy()
        target[displacement <= self.deadband_px] = self.previous_nodes[displacement <= self.deadband_px]
        alpha = self.temporal_alpha
        stabilized = (1.0 - alpha) * self.previous_nodes + alpha * target
        self.previous_nodes = stabilized.astype(np.float32)
        self.tracked_frames += 1
        if self.tracked_frames >= self.lock_after_frames:
            self.locked = True
            self.motion_frames = 0
        return self.previous_nodes

    @staticmethod
    def _piecewise_warp(
        image: np.ndarray,
        nodes: np.ndarray,
        rows: int,
        cols: int,
        destination_nodes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Warp every cell with its own homography onto (non-uniform) destination cells."""
        height, width = image.shape[:2]
        if destination_nodes is None:
            destination_nodes = GridMeshRectifier._regular_nodes(width, height, rows, cols)
        result = np.zeros_like(image)
        for row in range(rows):
            for col in range(cols):
                source = np.array(
                    [nodes[row, col], nodes[row, col + 1], nodes[row + 1, col + 1], nodes[row + 1, col]],
                    dtype=np.float32,
                )
                destination = np.array(
                    [
                        destination_nodes[row, col],
                        destination_nodes[row, col + 1],
                        destination_nodes[row + 1, col + 1],
                        destination_nodes[row + 1, col],
                    ],
                    dtype=np.float32,
                )
                x0 = max(0, int(np.floor(destination[:, 0].min())))
                y0 = max(0, int(np.floor(destination[:, 1].min())))
                x1 = min(width, int(np.ceil(destination[:, 0].max())) + 1)
                y1 = min(height, int(np.ceil(destination[:, 1].max())) + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                shifted = destination - np.array([x0, y0], np.float32)
                transform = cv2.getPerspectiveTransform(source, shifted)
                warped = cv2.warpPerspective(image, transform, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
                result[y0:y1, x0:x1] = warped
        return result

    def rectify_with_nodes(
        self,
        image: np.ndarray,
        nodes: np.ndarray,
        destination_nodes: np.ndarray,
        confidence: float,
    ) -> MeshRectification:
        """Piecewise rectification from *observed* graph junctions.

        ``nodes`` are the measured junctions (rectified pixels) and
        ``destination_nodes`` place every row/column at its measured mean
        position, so unequal cells stay unequal.
        """
        rows, cols = nodes.shape[0] - 1, nodes.shape[1] - 1
        if not self.enabled or rows < 1 or cols < 1:
            return MeshRectification(image, destination_nodes, confidence, False)
        stabilized = self._stabilize(nodes.astype(np.float32), rows, cols)
        height, width = image.shape[:2]
        if confidence < self.min_confidence or not self._valid_nodes(stabilized, destination_nodes):
            return MeshRectification(image, destination_nodes, confidence, False)
        return MeshRectification(
            self._piecewise_warp(image, stabilized, rows, cols, destination_nodes),
            stabilized,
            confidence,
            True,
        )

    @staticmethod
    def _valid_nodes(nodes: np.ndarray, destination: np.ndarray) -> bool:
        rows, cols = nodes.shape[0] - 1, nodes.shape[1] - 1
        for row in range(rows):
            for col in range(cols):
                quad = np.array(
                    [nodes[row, col], nodes[row, col + 1], nodes[row + 1, col + 1], nodes[row + 1, col]], np.float32
                )
                target = np.array(
                    [destination[row, col], destination[row, col + 1], destination[row + 1, col + 1], destination[row + 1, col]],
                    np.float32,
                )
                if not cv2.isContourConvex(quad):
                    return False
                area = abs(float(cv2.contourArea(quad)))
                expected = abs(float(cv2.contourArea(target)))
                if expected <= 1.0 or not expected * 0.35 <= area <= expected * 2.5:
                    return False
        return True

    def rectify(
        self,
        image: np.ndarray,
        rows: int,
        cols: int,
        row_lines: list[float],
        col_lines: list[float],
    ) -> MeshRectification:
        if not self.enabled:
            return MeshRectification(image, self._base_nodes(image.shape[1], image.shape[0], rows, cols, row_lines, col_lines), 0.0, False)
        nodes, confidence = self._detect_nodes(image, rows, cols, row_lines, col_lines)
        nodes = self._stabilize(nodes, rows, cols)
        destination = self._base_nodes(image.shape[1], image.shape[0], rows, cols, row_lines, col_lines)
        if confidence < self.min_confidence or not self._valid_nodes(nodes, destination):
            return MeshRectification(image, destination, confidence, False)
        return MeshRectification(
            self._piecewise_warp(image, nodes, rows, cols, destination),
            nodes,
            confidence,
            True,
        )

    @classmethod
    def _base_nodes(cls, width: int, height: int, rows: int, cols: int, row_lines: list[float], col_lines: list[float]) -> np.ndarray:
        regular = cls._regular_nodes(width, height, rows, cols)
        base_x = np.asarray([0.0, *sorted(col_lines), float(width - 1)], dtype=np.float32)
        base_y = np.asarray([0.0, *sorted(row_lines), float(height - 1)], dtype=np.float32)
        if len(base_x) != cols + 1:
            base_x = regular[0, :, 0]
        if len(base_y) != rows + 1:
            base_y = regular[:, 0, 1]
        return np.stack(np.meshgrid(base_x, base_y), axis=-1).astype(np.float32)
