"""Grid topology derived from the evidence-driven planar boundary graph.

The previous implementation averaged whole-image row/column profiles and fitted
*equally spaced* divisions (``_fit_divisions``).  With only the two centre
dividers visible that fit preferred 2x2 and every later stage inherited it.
That code is removed.  Rows and columns are now counted from boundary curves
that were actually observed (coverage, continuity, T/X junctions) and
accumulated over frames; they are derived only when the boundary graph is a
regular grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boundary_graph import BoundaryConfig, BoundaryDetection, BoundaryGraphDetector, PlanarBoundaryGraph


@dataclass
class GridTopology:
    rows: int
    cols: int
    confidence: float
    row_confidence: float = 0.0
    col_confidence: float = 0.0
    row_lines_px: list[float] = field(default_factory=list)
    col_lines_px: list[float] = field(default_factory=list)
    automatic: bool = False
    state: str = "UNCERTAIN"  # CONFIRMED / TENTATIVE / UNCERTAIN / FIXED
    regular: bool = False
    graph: PlanarBoundaryGraph | None = None
    stable_frames: int = 0
    face_count: int = 0

    @property
    def uncertain(self) -> bool:
        return self.state == "UNCERTAIN"


class GridTopologyDetector:
    """Rows/cols from accumulated boundary evidence (no spacing assumption)."""

    def __init__(self, config: dict, boundary_config: dict | None = None) -> None:
        self.fallback_rows = int(config.get("rows", 3))
        self.fallback_cols = int(config.get("cols", 3))
        self.auto_detect = bool(config.get("auto_detect", True))
        self.min_rows = int(config.get("min_rows", 1))
        self.max_rows = int(config.get("max_rows", 12))
        self.min_cols = int(config.get("min_cols", 1))
        self.max_cols = int(config.get("max_cols", 12))
        self.min_confidence = float(config.get("topology_min_confidence", 0.45))
        boundary = dict(boundary_config or {})
        boundary.setdefault("min_topology_confidence", self.min_confidence)
        boundary.setdefault("confirm_frames", int(config.get("switch_confirm_frames", 3)))
        self.boundary = BoundaryGraphDetector(BoundaryConfig.from_dict(boundary))
        self.current: GridTopology | None = None
        self.last_detection: BoundaryDetection | None = None

    @property
    def geometry(self):
        return self.boundary.geometry

    def reset(self) -> None:
        self.boundary.reset()
        self.current = None
        self.last_detection = None

    def fixed(self) -> GridTopology:
        return GridTopology(
            rows=self.fallback_rows,
            cols=self.fallback_cols,
            confidence=0.0,
            automatic=False,
            state="FIXED",
            regular=True,
        )

    def _from_detection(self, detection: BoundaryDetection, width: int, height: int) -> GridTopology:
        graph = detection.graph
        assert graph is not None
        rows, cols = graph.rows, graph.cols
        row_positions = graph.row_positions()[1:-1]
        col_positions = graph.col_positions()[1:-1]
        row_conf = [c for c in graph.row_confidence[1:-1]]
        col_conf = [c for c in graph.col_confidence[1:-1]]
        in_range = self.min_rows <= rows <= self.max_rows and self.min_cols <= cols <= self.max_cols
        state = detection.state if in_range else "UNCERTAIN"
        automatic = graph.regular and in_range and state != "UNCERTAIN"
        return GridTopology(
            rows=rows,
            cols=cols,
            confidence=float(detection.confidence),
            row_confidence=float(np.mean(row_conf)) if row_conf else 1.0,
            col_confidence=float(np.mean(col_conf)) if col_conf else 1.0,
            row_lines_px=[float(v * (height - 1)) for v in row_positions],
            col_lines_px=[float(u * (width - 1)) for u in col_positions],
            automatic=bool(automatic),
            state=state,
            regular=graph.regular,
            graph=graph,
            stable_frames=detection.stable_frames,
            face_count=len(graph.faces),
        )

    def detect(self, analysis_image: np.ndarray, output_size: tuple[int, int] | None = None) -> GridTopology:
        """``analysis_image`` is the padded rectification from ``BoundaryGraphDetector.warp``."""
        if not self.auto_detect:
            return self.fixed()
        width, height = output_size or (self.geometry.width, self.geometry.height)
        detection = self.boundary.detect_rectified(analysis_image)
        self.last_detection = detection
        topology = self._from_detection(detection, width, height)
        self.current = topology
        return topology
