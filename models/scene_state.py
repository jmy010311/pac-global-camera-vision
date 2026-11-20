from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class OccupancyState(str, Enum):
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"
    UNKNOWN = "UNKNOWN"


@dataclass
class PartitionState:
    detected: bool = False
    rows: int = 3
    cols: int = 3
    confidence: float = 0.0
    topology_confidence: float = 0.0
    grid_automatic: bool = False
    mesh_rectified: bool = False
    mesh_confidence: float = 0.0
    corners_px: list[list[float]] = field(default_factory=list)
    stale: bool = False
    topology_state: str = "UNCERTAIN"
    regular: bool = False
    face_count: int = 0
    boundary_points: int = 0


@dataclass
class CellState:
    id: int
    row: int
    col: int
    center_px: list[float]
    state: OccupancyState = OccupancyState.UNKNOWN
    occupancy_score: float = 0.0
    confidence: float = 0.0
    occupancy_source: str = "appearance"  # single-camera appearance cue, never a height


@dataclass
class CameraState:
    source: str = "APC850"
    index: int | str = "unknown"
    backend: str = "unknown"
    fps: float = 0.0
    resolution: list[int] = field(default_factory=list)
    lens_corrected: bool = False


@dataclass
class SceneState:
    timestamp: float
    frame_id: int
    partition: PartitionState
    cells: list[CellState]
    blocks: list[dict[str, Any]]
    camera: CameraState

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for cell in data["cells"]:
            if isinstance(cell["state"], OccupancyState):
                cell["state"] = cell["state"].value
        return data

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(self.to_json(), encoding="utf-8")
        temporary.replace(target)
