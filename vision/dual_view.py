from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from .stereo_height import (
    CellHeight,
    HeightCell,
    HeightState,
    StereoCalibration,
    StereoHeightEstimator,
    StereoResult,
    TemporalHeightFilter,
)


class SurfaceHeightState(str, Enum):
    FLOOR = "FLOOR"
    ABOVE_FLOOR = "ABOVE_FLOOR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SurfaceCell:
    id: int
    row: int
    col: int
    polygon: np.ndarray
    center: tuple[float, float]


@dataclass(frozen=True)
class SurfaceMeasurement:
    id: int
    row: int
    col: int
    state: SurfaceHeightState
    parallax_px: float
    raised_fraction: float
    match_count: int
    confidence: float
    direction_coherence: float = 0.0


@dataclass
class DualViewResult:
    cells: list[SurfaceMeasurement]
    global_points: np.ndarray
    predicted_floor_points: np.ndarray
    residuals_px: np.ndarray
    warped_wrist: np.ndarray
    valid_mask: np.ndarray
    total_matches: int


@dataclass
class DualViewCalibration:
    """Fixed global grid plus the floor-plane relation between two cameras.

    ``grid_nodes`` contains every vertex of the detected partition,
    not just the four outside corners. ``homography_wrist_to_global`` aligns
    points on the cell-floor plane. Points above that plane retain parallax.
    """

    rows: int
    cols: int
    global_image_size: tuple[int, int]
    wrist_image_size: tuple[int, int]
    grid_nodes: np.ndarray
    edge_samples: dict[str, np.ndarray]
    global_floor_points: np.ndarray
    wrist_floor_points: np.ndarray
    homography_wrist_to_global: np.ndarray
    reprojection_error_px: float

    def validate(self) -> None:
        expected_shape = (self.rows + 1, self.cols + 1, 2)
        if self.grid_nodes.shape != expected_shape:
            raise ValueError(f"grid_nodes must have shape {expected_shape}, got {self.grid_nodes.shape}")
        if self.rows < 1 or self.cols < 1:
            raise ValueError("Surface grid must contain at least one row and one column")
        if len(self.global_floor_points) < 4 or len(self.wrist_floor_points) < 4:
            raise ValueError("At least four corresponding floor-plane points are required")
        if self.global_floor_points.shape != self.wrist_floor_points.shape:
            raise ValueError("Global/wrist floor point counts must match")
        if self.homography_wrist_to_global.shape != (3, 3):
            raise ValueError("homography_wrist_to_global must be 3x3")
        for key, samples in self.edge_samples.items():
            if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 2:
                raise ValueError(f"Edge {key} must contain at least two 2D points")
        for cell in self.cells():
            if abs(float(cv2.contourArea(cell.polygon.astype(np.float32)))) < 20.0:
                raise ValueError(f"Cell {cell.id} is degenerate; check the detected boundary points")

    @classmethod
    def from_correspondences(
        cls,
        grid_nodes: np.ndarray,
        global_floor_points: np.ndarray,
        wrist_floor_points: np.ndarray,
        global_image_size: tuple[int, int],
        wrist_image_size: tuple[int, int],
        edge_samples: dict[str, np.ndarray] | None = None,
        rows: int = 3,
        cols: int = 3,
        ransac_threshold_px: float = 3.0,
    ) -> "DualViewCalibration":
        global_points = np.asarray(global_floor_points, dtype=np.float32).reshape(-1, 2)
        wrist_points = np.asarray(wrist_floor_points, dtype=np.float32).reshape(-1, 2)
        if len(global_points) < 4 or global_points.shape != wrist_points.shape:
            raise ValueError("Use at least four matching floor points in both camera images")
        homography, inliers = cv2.findHomography(
            wrist_points,
            global_points,
            cv2.RANSAC,
            float(ransac_threshold_px),
        )
        if homography is None or not np.isfinite(homography).all():
            raise ValueError("Could not estimate a floor-plane homography")
        predicted = cv2.perspectiveTransform(wrist_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(predicted - global_points, axis=1)
        if inliers is not None:
            keep = inliers.reshape(-1).astype(bool)
            if int(keep.sum()) >= 4:
                errors = errors[keep]
        calibration = cls(
            rows=int(rows),
            cols=int(cols),
            global_image_size=tuple(int(value) for value in global_image_size),
            wrist_image_size=tuple(int(value) for value in wrist_image_size),
            grid_nodes=np.asarray(grid_nodes, dtype=np.float32).reshape(rows + 1, cols + 1, 2),
            edge_samples={
                str(key): np.asarray(value, dtype=np.float32).reshape(-1, 2)
                for key, value in (edge_samples or {}).items()
            },
            global_floor_points=global_points,
            wrist_floor_points=wrist_points,
            homography_wrist_to_global=homography.astype(np.float64),
            reprojection_error_px=float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else float("inf"),
        )
        calibration.validate()
        return calibration

    @staticmethod
    def horizontal_edge_key(row: int, col: int) -> str:
        return f"h_{row}_{col}"

    @staticmethod
    def vertical_edge_key(row: int, col: int) -> str:
        return f"v_{row}_{col}"

    def _edge(self, key: str, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        samples = self.edge_samples.get(key)
        if samples is None or len(samples) < 2:
            return np.asarray([start, end], dtype=np.float32)
        return np.asarray(samples, dtype=np.float32)

    def cells(self) -> list[SurfaceCell]:
        result: list[SurfaceCell] = []
        for row in range(self.rows):
            for col in range(self.cols):
                top = self._edge(
                    self.horizontal_edge_key(row, col),
                    self.grid_nodes[row, col],
                    self.grid_nodes[row, col + 1],
                )
                right = self._edge(
                    self.vertical_edge_key(row, col + 1),
                    self.grid_nodes[row, col + 1],
                    self.grid_nodes[row + 1, col + 1],
                )
                bottom = self._edge(
                    self.horizontal_edge_key(row + 1, col),
                    self.grid_nodes[row + 1, col],
                    self.grid_nodes[row + 1, col + 1],
                )[::-1]
                left = self._edge(
                    self.vertical_edge_key(row, col),
                    self.grid_nodes[row, col],
                    self.grid_nodes[row + 1, col],
                )[::-1]
                polygon = np.concatenate([top, right[1:], bottom[1:], left[1:-1]], axis=0).astype(np.float32)
                center = np.asarray(
                    [
                        self.grid_nodes[row, col],
                        self.grid_nodes[row, col + 1],
                        self.grid_nodes[row + 1, col + 1],
                        self.grid_nodes[row + 1, col],
                    ],
                    dtype=np.float32,
                ).mean(axis=0)
                result.append(
                    SurfaceCell(
                        id=row * self.cols + col,
                        row=row,
                        col=col,
                        polygon=polygon,
                        center=(float(center[0]), float(center[1])),
                    )
                )
        return result

    def scaled(
        self,
        global_image_size: tuple[int, int],
        wrist_image_size: tuple[int, int],
    ) -> "DualViewCalibration":
        global_scale = np.diag(
            [
                global_image_size[0] / max(self.global_image_size[0], 1),
                global_image_size[1] / max(self.global_image_size[1], 1),
                1.0,
            ]
        )
        wrist_scale = np.diag(
            [
                wrist_image_size[0] / max(self.wrist_image_size[0], 1),
                wrist_image_size[1] / max(self.wrist_image_size[1], 1),
                1.0,
            ]
        )
        nodes = self.grid_nodes.copy()
        nodes[..., 0] *= float(global_scale[0, 0])
        nodes[..., 1] *= float(global_scale[1, 1])
        global_points = self.global_floor_points.copy()
        global_points[:, 0] *= float(global_scale[0, 0])
        global_points[:, 1] *= float(global_scale[1, 1])
        wrist_points = self.wrist_floor_points.copy()
        wrist_points[:, 0] *= float(wrist_scale[0, 0])
        wrist_points[:, 1] *= float(wrist_scale[1, 1])
        homography = global_scale @ self.homography_wrist_to_global @ np.linalg.inv(wrist_scale)
        edge_samples: dict[str, np.ndarray] = {}
        for key, samples in self.edge_samples.items():
            scaled_samples = samples.copy()
            scaled_samples[:, 0] *= float(global_scale[0, 0])
            scaled_samples[:, 1] *= float(global_scale[1, 1])
            edge_samples[key] = scaled_samples.astype(np.float32)
        return DualViewCalibration(
            rows=self.rows,
            cols=self.cols,
            global_image_size=global_image_size,
            wrist_image_size=wrist_image_size,
            grid_nodes=nodes.astype(np.float32),
            edge_samples=edge_samples,
            global_floor_points=global_points.astype(np.float32),
            wrist_floor_points=wrist_points.astype(np.float32),
            homography_wrist_to_global=homography.astype(np.float64),
            reprojection_error_px=self.reprojection_error_px * float(
                (global_scale[0, 0] + global_scale[1, 1]) / 2.0
            ),
        )

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "rows": self.rows,
            "cols": self.cols,
            "global_image_size": list(self.global_image_size),
            "wrist_image_size": list(self.wrist_image_size),
            "grid_nodes": self.grid_nodes.tolist(),
            "edge_samples": {key: value.tolist() for key, value in self.edge_samples.items()},
            "global_floor_points": self.global_floor_points.tolist(),
            "wrist_floor_points": self.wrist_floor_points.tolist(),
            "homography_wrist_to_global": self.homography_wrist_to_global.tolist(),
            "reprojection_error_px": self.reprojection_error_px,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "DualViewCalibration":
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        calibration = cls(
            rows=int(data["rows"]),
            cols=int(data["cols"]),
            global_image_size=tuple(int(value) for value in data["global_image_size"]),
            wrist_image_size=tuple(int(value) for value in data["wrist_image_size"]),
            grid_nodes=np.asarray(data["grid_nodes"], dtype=np.float32),
            edge_samples={
                str(key): np.asarray(value, dtype=np.float32)
                for key, value in data.get("edge_samples", {}).items()
            },
            global_floor_points=np.asarray(data["global_floor_points"], dtype=np.float32),
            wrist_floor_points=np.asarray(data["wrist_floor_points"], dtype=np.float32),
            homography_wrist_to_global=np.asarray(data["homography_wrist_to_global"], dtype=np.float64),
            reprojection_error_px=float(data["reprojection_error_px"]),
        )
        calibration.validate()
        return calibration


def height_cells(calibration: DualViewCalibration, global_size: tuple[int, int] | None = None) -> list[HeightCell]:
    """Global-camera cell polygons (the only boundary reference) for the stereo estimator."""
    active = calibration if global_size is None else calibration.scaled(global_size, calibration.wrist_image_size)
    return [HeightCell(cell.id, cell.row, cell.col, cell.polygon) for cell in active.cells()]


class PlaneParallaxDetector:
    """AUXILIARY ONLY: single floor-homography residual ("differs from floor").

    The residual of a point after warping by the floor homography is non-zero
    for anything off the floor plane, but its magnitude depends on image
    position and camera geometry, so it is *not* a height.  Metric heights come
    from :class:`vision.stereo_height.StereoHeightEstimator`.  This detector is
    kept for the floor-alignment debug view and as a cross-check.
    """

    def __init__(self, config: dict) -> None:
        self.ratio_test = float(config.get("ratio_test", 0.78))
        self.max_descriptor_distance = float(config.get("max_descriptor_distance", 72.0))
        self.floor_threshold_px = float(config.get("floor_threshold_px", 2.5))
        self.raised_threshold_px = float(config.get("raised_threshold_px", 5.0))
        self.raised_fraction_threshold = float(config.get("raised_fraction_threshold", 0.35))
        self.direction_coherence_threshold = float(config.get("direction_coherence_threshold", 0.55))
        self.max_parallax_px = float(config.get("max_parallax_px", 80.0))
        self.min_matches_per_cell = max(2, int(config.get("min_matches_per_cell", 4)))
        self.cell_inset_ratio = float(config.get("cell_inset_ratio", 0.12))
        self.orb = cv2.ORB_create(
            nfeatures=max(500, int(config.get("orb_features", 2200))),
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=9,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    @staticmethod
    def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(points.astype(np.float32).reshape(-1, 1, 2), homography).reshape(-1, 2)

    def _match(self, global_frame: np.ndarray, wrist_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        global_keypoints, global_descriptors = self.orb.detectAndCompute(self._gray(global_frame), None)
        wrist_keypoints, wrist_descriptors = self.orb.detectAndCompute(self._gray(wrist_frame), None)
        if global_descriptors is None or wrist_descriptors is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        pairs = self.matcher.knnMatch(wrist_descriptors, global_descriptors, k=2)
        accepted = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            best, second = pair
            if best.distance <= self.max_descriptor_distance and best.distance < self.ratio_test * second.distance:
                accepted.append(best)
        if not accepted:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        global_points = np.asarray([global_keypoints[match.trainIdx].pt for match in accepted], dtype=np.float32)
        wrist_points = np.asarray([wrist_keypoints[match.queryIdx].pt for match in accepted], dtype=np.float32)
        return global_points, wrist_points

    @staticmethod
    def _inset(polygon: np.ndarray, ratio: float) -> np.ndarray:
        center = polygon.mean(axis=0, keepdims=True)
        return center + (polygon - center) * float(np.clip(1.0 - ratio, 0.45, 1.0))

    def classify_correspondences(
        self,
        calibration: DualViewCalibration,
        global_points: np.ndarray,
        wrist_points: np.ndarray,
    ) -> list[SurfaceMeasurement]:
        global_points = np.asarray(global_points, dtype=np.float32).reshape(-1, 2)
        wrist_points = np.asarray(wrist_points, dtype=np.float32).reshape(-1, 2)
        if global_points.shape != wrist_points.shape:
            raise ValueError("Global and wrist correspondence arrays must have the same shape")
        predicted = self._project(wrist_points, calibration.homography_wrist_to_global)
        vectors = global_points - predicted
        residuals = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(residuals) & (residuals <= self.max_parallax_px)
        global_points = global_points[valid]
        vectors = vectors[valid]
        residuals = residuals[valid]
        measurements: list[SurfaceMeasurement] = []
        for cell in calibration.cells():
            polygon = self._inset(cell.polygon, self.cell_inset_ratio).astype(np.float32)
            inside = np.asarray(
                [cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0 for point in global_points],
                dtype=bool,
            )
            values = residuals[inside]
            cell_vectors = vectors[inside]
            count = int(len(values))
            raised_mask = values >= self.raised_threshold_px
            raised_vectors = cell_vectors[raised_mask]
            if len(raised_vectors):
                lengths = np.linalg.norm(raised_vectors, axis=1, keepdims=True)
                unit_vectors = raised_vectors / np.maximum(lengths, 1e-6)
                direction_coherence = float(np.linalg.norm(unit_vectors.mean(axis=0)))
            else:
                direction_coherence = 0.0
            if count < self.min_matches_per_cell:
                state = SurfaceHeightState.UNKNOWN
                parallax = float(np.median(values)) if count else 0.0
                raised_fraction = float(np.mean(values >= self.raised_threshold_px)) if count else 0.0
                confidence = min(0.45, count / self.min_matches_per_cell * 0.45)
            else:
                parallax = float(np.median(values))
                raised_fraction = float(np.mean(values >= self.raised_threshold_px))
                if (
                    raised_fraction >= self.raised_fraction_threshold
                    and float(np.percentile(values, 65)) >= self.raised_threshold_px
                    and direction_coherence >= self.direction_coherence_threshold
                ):
                    state = SurfaceHeightState.ABOVE_FLOOR
                    evidence = min(1.0, parallax / max(self.raised_threshold_px, 1e-6))
                    confidence = float(
                        np.clip(0.35 * evidence + 0.40 * raised_fraction + 0.25 * direction_coherence, 0.0, 1.0)
                    )
                elif float(np.percentile(values, 75)) <= self.floor_threshold_px:
                    state = SurfaceHeightState.FLOOR
                    confidence = float(np.clip(1.0 - parallax / max(self.floor_threshold_px * 1.5, 1e-6), 0.0, 1.0))
                else:
                    state = SurfaceHeightState.UNKNOWN
                    confidence = 0.35
            measurements.append(
                SurfaceMeasurement(
                    id=cell.id,
                    row=cell.row,
                    col=cell.col,
                    state=state,
                    parallax_px=round(parallax, 3),
                    raised_fraction=round(raised_fraction, 3),
                    match_count=count,
                    confidence=round(confidence, 3),
                    direction_coherence=round(direction_coherence, 3),
                )
            )
        return measurements

    def process(
        self,
        global_frame: np.ndarray,
        wrist_frame: np.ndarray,
        calibration: DualViewCalibration,
    ) -> DualViewResult:
        global_size = (global_frame.shape[1], global_frame.shape[0])
        wrist_size = (wrist_frame.shape[1], wrist_frame.shape[0])
        active = calibration.scaled(global_size, wrist_size)
        global_points, wrist_points = self._match(global_frame, wrist_frame)
        predicted = self._project(wrist_points, active.homography_wrist_to_global)
        residuals = np.linalg.norm(global_points - predicted, axis=1) if len(global_points) else np.empty(0, np.float32)
        measurements = self.classify_correspondences(active, global_points, wrist_points)
        warped = cv2.warpPerspective(
            wrist_frame,
            active.homography_wrist_to_global,
            global_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        source_mask = np.full(wrist_frame.shape[:2], 255, dtype=np.uint8)
        valid_mask = cv2.warpPerspective(
            source_mask,
            active.homography_wrist_to_global,
            global_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return DualViewResult(
            cells=measurements,
            global_points=global_points,
            predicted_floor_points=predicted,
            residuals_px=residuals,
            warped_wrist=warped,
            valid_mask=valid_mask,
            total_matches=len(global_points),
        )


class TemporalSurfaceFilter:
    """Require repeated evidence before a cell changes height state."""

    def __init__(self, confirm_frames: int = 3, hold_unknown_frames: int = 4) -> None:
        self.confirm_frames = max(1, int(confirm_frames))
        self.hold_unknown_frames = max(0, int(hold_unknown_frames))
        self.stable: dict[int, SurfaceHeightState] = {}
        self.pending: dict[int, tuple[SurfaceHeightState, int]] = {}
        self.unknown_counts: dict[int, int] = {}

    def update(self, measurements: list[SurfaceMeasurement]) -> list[SurfaceMeasurement]:
        output: list[SurfaceMeasurement] = []
        for measurement in measurements:
            cell_id = measurement.id
            stable = self.stable.get(cell_id, SurfaceHeightState.UNKNOWN)
            raw = measurement.state
            if raw == SurfaceHeightState.UNKNOWN:
                unknown_count = self.unknown_counts.get(cell_id, 0) + 1
                self.unknown_counts[cell_id] = unknown_count
                self.pending.pop(cell_id, None)
                if stable != SurfaceHeightState.UNKNOWN and unknown_count <= self.hold_unknown_frames:
                    output.append(replace(measurement, state=stable, confidence=round(measurement.confidence * 0.75, 3)))
                else:
                    self.stable[cell_id] = SurfaceHeightState.UNKNOWN
                    output.append(measurement)
                continue
            self.unknown_counts[cell_id] = 0
            if raw == stable:
                self.pending.pop(cell_id, None)
                output.append(measurement)
                continue
            pending_state, count = self.pending.get(cell_id, (raw, 0))
            count = count + 1 if pending_state == raw else 1
            self.pending[cell_id] = (raw, count)
            if count >= self.confirm_frames:
                self.stable[cell_id] = raw
                self.pending.pop(cell_id, None)
                output.append(measurement)
            else:
                output.append(replace(measurement, state=stable, confidence=round(measurement.confidence * 0.6, 3)))
        return output


def draw_dual_view(
    global_frame: np.ndarray,
    result: DualViewResult,
    calibration: DualViewCalibration,
) -> np.ndarray:
    canvas = global_frame.copy()
    active = calibration.scaled(
        (global_frame.shape[1], global_frame.shape[0]),
        calibration.wrist_image_size,
    )
    colours = {
        SurfaceHeightState.FLOOR: (70, 230, 120),
        SurfaceHeightState.ABOVE_FLOOR: (55, 65, 245),
        SurfaceHeightState.UNKNOWN: (0, 190, 255),
    }
    for cell, measurement in zip(active.cells(), result.cells):
        polygon = np.round(cell.polygon).astype(np.int32)
        colour = colours[measurement.state]
        cv2.polylines(canvas, [polygon], True, colour, 3, cv2.LINE_AA)
        center = (int(round(cell.center[0])), int(round(cell.center[1])))
        cv2.circle(canvas, center, 5, colour, -1, cv2.LINE_AA)
        label = f"{measurement.id} {measurement.state.value} {measurement.parallax_px:.1f}px"
        cv2.putText(canvas, label, (center[0] - 55, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.43, colour, 1, cv2.LINE_AA)
    for point, predicted, residual in zip(result.global_points, result.predicted_floor_points, result.residuals_px):
        if residual < 1.0:
            continue
        p1 = tuple(np.round(point).astype(int))
        p0 = tuple(np.round(predicted).astype(int))
        colour = (55, 65, 245) if residual >= 5.0 else (0, 190, 255)
        cv2.arrowedLine(canvas, p0, p1, colour, 1, cv2.LINE_AA, tipLength=0.3)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (12, 12), (505, 92), (24, 28, 34), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0.0, canvas)
    above = sum(cell.state == SurfaceHeightState.ABOVE_FLOOR for cell in result.cells)
    cv2.putText(canvas, f"PAC FLOOR RESIDUAL (AUX) {calibration.rows}x{calibration.cols}", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"MATCHES {result.total_matches}   OFF-FLOOR CUE {above}/{len(result.cells)}",
        (28, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 220, 40),
        1,
        cv2.LINE_AA,
    )
    return canvas


HEIGHT_COLOURS = {
    HeightState.FLOOR: (70, 230, 120),
    HeightState.ABOVE_FLOOR: (55, 65, 245),
    HeightState.BELOW_FLOOR: (230, 120, 60),
    HeightState.UNKNOWN: (0, 190, 255),
    HeightState.UNCALIBRATED: (160, 160, 160),
}


def draw_height_view(
    global_frame: np.ndarray,
    cells: list[HeightCell],
    measurements: list[CellHeight],
    *,
    rows: int,
    cols: int,
    boundary_state: str,
    height_status: str,
    edge_samples: dict[str, np.ndarray] | None = None,
    match_count: int = 0,
) -> np.ndarray:
    """Global main view: dense boundaries + per-cell geometric height labels.

    Colour is applied only after the geometric decision, for display.
    """
    canvas = global_frame.copy()
    uncertain = boundary_state == "UNCERTAIN"
    for samples in (edge_samples or {}).values():
        for xy in np.round(samples).astype(np.int32):
            cv2.circle(canvas, (int(xy[0]), int(xy[1])), 1, (150, 150, 150) if uncertain else (90, 255, 145), -1, cv2.LINE_AA)
    for cell, measurement in zip(cells, measurements):
        colour = HEIGHT_COLOURS[measurement.height_state]
        polygon = np.round(cell.polygon).astype(np.int32)
        if not edge_samples:
            cv2.polylines(canvas, [polygon], True, colour, 1, cv2.LINE_AA)
        center = polygon.mean(axis=0).astype(int)
        if measurement.height_state == HeightState.UNCALIBRATED:
            height_text = "UNCAL"
        elif measurement.height_mm is None or measurement.height_state == HeightState.UNKNOWN:
            height_text = "UNKNOWN"
        else:
            height_text = f"{measurement.height_mm:+.1f}mm"
        lines = [f"#{measurement.id} {height_text}", f"c{measurement.confidence:.2f} n{measurement.valid_3d_points}"]
        if measurement.uncertainty_mm is not None and measurement.height_state not in (HeightState.UNCALIBRATED,):
            lines[1] += f" +-{measurement.uncertainty_mm:.1f}"
        for index, text in enumerate(lines):
            origin = (int(center[0]) - 48, int(center[1]) - 6 + 16 * index)
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (12, 12), (600, 118), (24, 28, 34), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0.0, canvas)
    grid_text = f"DETECTED GRID: {rows} x {cols}" if not uncertain else "TOPOLOGY UNCERTAIN"
    cv2.putText(canvas, grid_text, (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 245) if not uncertain else (0, 190, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"BOUNDARY: {boundary_state}", (28, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 40), 1, cv2.LINE_AA)
    height_colour = (160, 160, 160) if height_status.startswith("UNCALIBRATED") else (255, 220, 40)
    cv2.putText(canvas, f"HEIGHT: {height_status}   3D MATCHES {match_count}", (28, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, height_colour, 1, cv2.LINE_AA)
    if height_status.startswith("UNCALIBRATED"):
        cv2.putText(canvas, "UNCALIBRATED - no metric height", (28, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 70, 245), 2, cv2.LINE_AA)
    return canvas
