from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from models.scene_state import CameraState, CellState, OccupancyState, PartitionState, SceneState

from .dense_edges import DenseGridTrace, graph_to_camera
from .grid import GridCell, make_grid
from .grid_topology import GridTopology, GridTopologyDetector
from .mesh_rectifier import GridMeshRectifier
from .occupancy import OccupancyDetector
from .outer_boundary import OuterBoundaryRefiner
from .partition_detector import PartitionDetection, PartitionDetector
from .perspective import project_points, rectify


@dataclass
class VisionResult:
    detection: PartitionDetection
    rectified: np.ndarray | None
    inverse_homography: np.ndarray | None
    grid_cells: list[GridCell]
    topology: GridTopology
    mesh_nodes: np.ndarray | None
    mesh_applied: bool
    scene: SceneState
    dense_trace: DenseGridTrace | None = None
    analysis_homography: np.ndarray | None = None
    outer_refined: bool = False


class VisionProcessor:
    """Camera-independent, frame-in/scene-state-out vision pipeline.

    Order: outer quad -> outer boundary refinement -> padded rectification ->
    boundary evidence graph (multi-frame) -> topology derived from the graph ->
    mesh rectification on observed junctions -> dense camera-space boundaries.
    """

    def __init__(self, config: dict) -> None:
        grid = config["grid"]
        self.rows = int(grid.get("rows", 3))
        self.cols = int(grid.get("cols", 3))
        self.rectified_width = int(grid.get("rectified_width", 720))
        self.rectified_height = int(grid.get("rectified_height", 720))
        boundary = dict(config.get("boundary", {}))
        self.boundary_interval = max(1, int(boundary.pop("update_interval_frames", 1)))
        self.detector = PartitionDetector(config.get("partition", {}))
        self.outer_refiner = OuterBoundaryRefiner(config.get("outer_boundary", {}))
        self.topology_detector = GridTopologyDetector(grid, boundary)
        self.mesh_rectifier = GridMeshRectifier(config.get("mesh_rectification", {}))
        self.occupancy_detector = OccupancyDetector(config.get("occupancy", {}))
        self._refined_corners: np.ndarray | None = None
        self._calls = 0
        self._last_topology: GridTopology | None = None

    def _fallback_topology(self) -> GridTopology:
        return GridTopology(rows=self.rows, cols=self.cols, confidence=0.0, automatic=False, state="UNCERTAIN")

    def _smooth_outer(self, corners: np.ndarray) -> np.ndarray:
        if self._refined_corners is None:
            self._refined_corners = corners.copy()
            return corners
        scale = float(max(np.ptp(corners[:, 0]), np.ptp(corners[:, 1]), 1.0))
        jump = float(np.max(np.linalg.norm(corners - self._refined_corners, axis=1)))
        if jump > 0.08 * scale:
            self._refined_corners = corners.copy()
        elif jump > 0.6:
            self._refined_corners = (0.7 * self._refined_corners + 0.3 * corners).astype(np.float32)
        return self._refined_corners.copy()

    def process(
        self,
        frame: np.ndarray,
        frame_id: int,
        measured_fps: float,
        camera_metadata: dict | None = None,
    ) -> VisionResult:
        detection = self.detector.detect(frame)
        rectified = None
        inverse = None
        cells: list[CellState] = []
        topology = self.topology_detector.current or self._fallback_topology()
        grid_cells: list[GridCell] = []
        mesh_nodes: np.ndarray | None = None
        mesh_applied = False
        mesh_confidence = 0.0
        dense_trace: DenseGridTrace | None = None
        analysis_h: np.ndarray | None = None
        outer_refined = False

        if detection.detected and detection.corners is not None:
            refinement = self.outer_refiner.refine(frame, detection.corners)
            outer_refined = refinement.applied
            detection.corners = self._smooth_outer(refinement.corners)
            rectified, _, inverse = rectify(frame, detection.corners, self.rectified_width, self.rectified_height)
            boundary = self.topology_detector.boundary
            analysis_image, analysis_h = boundary.warp(frame, detection.corners)
            run_boundary = self._calls % self.boundary_interval == 0 or self._last_topology is None
            self._calls += 1
            if run_boundary or not self.topology_detector.auto_detect:
                topology = self.topology_detector.detect(analysis_image, (self.rectified_width, self.rectified_height))
                self._last_topology = topology
            else:
                topology = self._last_topology

            if topology.graph is not None:
                graph = topology.graph
                try:
                    dense_trace = graph_to_camera(
                        frame, graph, boundary.geometry, analysis_h, state=topology.state
                    )
                except (ValueError, cv2.error, np.linalg.LinAlgError):
                    dense_trace = None
                if topology.regular:
                    scale = np.array([self.rectified_width - 1, self.rectified_height - 1], np.float32)
                    observed = graph.nodes * scale
                    xs = np.asarray(graph.col_positions(), np.float32) * scale[0]
                    ys = np.asarray(graph.row_positions(), np.float32) * scale[1]
                    xs[0], xs[-1], ys[0], ys[-1] = 0.0, scale[0], 0.0, scale[1]
                    destination = np.stack(np.meshgrid(xs, ys), axis=-1).astype(np.float32)
                    mesh = self.mesh_rectifier.rectify_with_nodes(
                        rectified, observed, destination, float(topology.confidence)
                    )
                    rectified = mesh.image
                    mesh_nodes = observed
                    mesh_applied = mesh.applied
                    mesh_confidence = mesh.confidence
                    grid_cells = make_grid(
                        self.rectified_width,
                        self.rectified_height,
                        topology.rows,
                        topology.cols,
                        [float(y) for y in ys[1:-1]],
                        [float(x) for x in xs[1:-1]],
                    )
            elif not self.topology_detector.auto_detect:
                grid_cells = make_grid(self.rectified_width, self.rectified_height, topology.rows, topology.cols)

            if grid_cells:
                occupancy = self.occupancy_detector.classify(rectified, grid_cells, topology.rows, topology.cols)
                for cell, cell_occupancy in zip(grid_cells, occupancy):
                    cells.append(
                        CellState(
                            id=cell.id,
                            row=cell.row,
                            col=cell.col,
                            center_px=[round(cell.center[0], 2), round(cell.center[1], 2)],
                            state=cell_occupancy.state,
                            occupancy_score=cell_occupancy.score,
                            confidence=cell_occupancy.confidence,
                        )
                    )

        metadata = camera_metadata or {}
        height, width = frame.shape[:2]
        scene = SceneState(
            timestamp=time.time(),
            frame_id=frame_id,
            partition=PartitionState(
                detected=detection.detected,
                rows=topology.rows,
                cols=topology.cols,
                confidence=round(float(detection.confidence), 4),
                topology_confidence=round(float(topology.confidence), 4),
                grid_automatic=topology.automatic,
                mesh_rectified=mesh_applied,
                mesh_confidence=round(float(mesh_confidence), 4),
                corners_px=(np.round(detection.corners, 2).tolist() if detection.corners is not None else []),
                stale=detection.stale,
                topology_state=topology.state if detection.detected else "UNCERTAIN",
                regular=bool(topology.regular),
                face_count=int(topology.face_count),
                boundary_points=dense_trace.point_count if dense_trace is not None else 0,
            ),
            cells=cells,
            blocks=[],
            camera=CameraState(
                source=str(metadata.get("source", "APC850")),
                index=metadata.get("index", "unknown"),
                backend=str(metadata.get("backend", "unknown")),
                fps=round(float(measured_fps), 2),
                resolution=[width, height],
                lens_corrected=bool(metadata.get("lens_corrected", False)),
            ),
        )
        return VisionResult(
            detection,
            rectified,
            inverse,
            grid_cells,
            topology,
            mesh_nodes,
            mesh_applied,
            scene,
            dense_trace,
            analysis_h,
            outer_refined,
        )

    @staticmethod
    def grid_in_camera(result: VisionResult) -> list[np.ndarray]:
        """Curved camera polygons of every cell (dense real-edge boundaries)."""
        if result.dense_trace is not None and result.dense_trace.regular and result.grid_cells:
            if len(result.grid_cells) == result.dense_trace.rows * result.dense_trace.cols:
                return result.dense_trace.cell_polygons()
        if result.inverse_homography is None:
            return []
        return [project_points(cell.polygon, result.inverse_homography) for cell in result.grid_cells]
