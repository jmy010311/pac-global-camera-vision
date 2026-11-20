from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridCell:
    id: int
    row: int
    col: int
    polygon: np.ndarray
    center: tuple[float, float]


def make_grid(
    width: int,
    height: int,
    rows: int,
    cols: int,
    row_lines_px: list[float] | None = None,
    col_lines_px: list[float] | None = None,
) -> list[GridCell]:
    if rows <= 0 or cols <= 0:
        raise ValueError("Grid rows and columns must be positive")
    if col_lines_px is not None and len(col_lines_px) == cols - 1:
        xs = np.asarray([0.0, *sorted(col_lines_px), float(width - 1)], dtype=np.float32)
    else:
        xs = np.linspace(0, width - 1, cols + 1, dtype=np.float32)
    if row_lines_px is not None and len(row_lines_px) == rows - 1:
        ys = np.asarray([0.0, *sorted(row_lines_px), float(height - 1)], dtype=np.float32)
    else:
        ys = np.linspace(0, height - 1, rows + 1, dtype=np.float32)
    cells: list[GridCell] = []
    for row in range(rows):
        for col in range(cols):
            polygon = np.array(
                [
                    [xs[col], ys[row]],
                    [xs[col + 1], ys[row]],
                    [xs[col + 1], ys[row + 1]],
                    [xs[col], ys[row + 1]],
                ],
                dtype=np.float32,
            )
            cells.append(
                GridCell(
                    id=row * cols + col,
                    row=row,
                    col=col,
                    polygon=polygon,
                    center=(float((xs[col] + xs[col + 1]) / 2), float((ys[row] + ys[row + 1]) / 2)),
                )
            )
    return cells
