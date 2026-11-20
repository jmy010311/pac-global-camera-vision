from __future__ import annotations

import cv2
import numpy as np

from models.scene_state import OccupancyState

from .processor import VisionProcessor, VisionResult

GREEN = (70, 230, 120)
AMBER = (0, 190, 255)
CYAN = (255, 220, 40)
RED = (60, 70, 245)
WHITE = (245, 245, 245)
DARK = (24, 28, 34)


def _state_color(state: OccupancyState) -> tuple[int, int, int]:
    if state == OccupancyState.EMPTY:
        return GREEN
    if state == OccupancyState.OCCUPIED:
        return RED
    return AMBER


def _label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = WHITE,
    scale: float = 0.6,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


GRAY = (150, 150, 150)


def draw_dense_boundaries(canvas: np.ndarray, trace, confirmed: bool, min_edge_confidence: float = 0.35) -> None:
    """Small dots on every observed boundary sample (node-to-node only)."""
    for key, samples in trace.edge_samples.items():
        if not trace.edge_present.get(key, True):
            continue
        confidence = trace.edge_confidence.get(key, 1.0)
        colour = (GREEN if confidence >= min_edge_confidence else AMBER) if confirmed else GRAY
        for xy in np.round(samples).astype(np.int32):
            cv2.circle(canvas, (int(xy[0]), int(xy[1])), 1, colour, -1, cv2.LINE_AA)


def boundary_status(result: VisionResult) -> tuple[str, tuple[int, int, int]]:
    partition = result.scene.partition
    if not result.detection.detected:
        return "BOUNDARY: SEARCHING", RED
    state = partition.topology_state
    if state == "UNCERTAIN":
        return "TOPOLOGY UNCERTAIN", AMBER
    colour = GREEN if state in ("CONFIRMED", "FIXED") else CYAN
    return f"BOUNDARY: {state}  CONF {partition.topology_confidence:.2f}", colour


def draw_global_view(frame: np.ndarray, result: VisionResult, fps: float) -> np.ndarray:
    canvas = frame.copy()
    detection = result.detection
    partition = result.scene.partition
    uncertain = partition.topology_state == "UNCERTAIN"

    if detection.detected and detection.corners is not None:
        trace = result.dense_trace
        if trace is not None:
            draw_dense_boundaries(canvas, trace, confirmed=not uncertain)
            corners = np.asarray(
                [trace.nodes[0, 0], trace.nodes[0, trace.cols], trace.nodes[trace.rows, trace.cols], trace.nodes[trace.rows, 0]]
            )
        else:
            corners = detection.corners
            cv2.polylines(canvas, [np.round(corners).astype(np.int32)], True, AMBER if detection.stale else GREEN, 2, cv2.LINE_AA)
        for name, point in zip(("TL", "TR", "BR", "BL"), np.round(corners).astype(np.int32)):
            xy = (int(point[0]), int(point[1]))
            cv2.circle(canvas, xy, 7, CYAN, -1, cv2.LINE_AA)
            _label(canvas, name, (xy[0] + 9, xy[1] - 9), CYAN, 0.55, 2)

        if not uncertain:
            camera_cells = VisionProcessor.grid_in_camera(result)
            for cell, scene_cell, polygon in zip(result.grid_cells, result.scene.cells, camera_cells):
                state_color = _state_color(scene_cell.state)
                center = tuple(np.round(np.asarray(polygon).mean(axis=0)).astype(int))
                cv2.circle(canvas, center, 3, state_color, -1, cv2.LINE_AA)
                short_state = "OCC" if scene_cell.state == OccupancyState.OCCUPIED else (
                    "EMPTY" if scene_cell.state == OccupancyState.EMPTY else "?"
                )
                _label(canvas, f"{cell.id} {short_state}", (center[0] + 5, center[1] - 5), state_color, 0.43, 1)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (12, 12), (520, 160), DARK, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    status = "DETECTED" if detection.detected else "SEARCHING"
    status_color = AMBER if detection.stale else (GREEN if detection.detected else RED)
    _label(canvas, "PAC GLOBAL CAMERA", (28, 40), WHITE, 0.72, 2)
    _label(canvas, f"FPS: {fps:5.1f}", (380, 40), CYAN, 0.57, 1)
    _label(canvas, f"PARTITION: {status}", (28, 68), status_color, 0.58, 2)
    if detection.detected and not uncertain:
        grid_text = f"DETECTED GRID: {partition.rows} x {partition.cols}"
    elif detection.detected:
        grid_text = "DETECTED GRID: -- (not confirmed)"
    else:
        grid_text = "DETECTED GRID: --"
    _label(canvas, grid_text, (28, 94), WHITE, 0.6, 2)
    text, colour = boundary_status(result)
    _label(canvas, f"{text}   POINTS {partition.boundary_points}", (28, 118), colour, 0.5, 1)
    _label(
        canvas,
        f"OCCUPANCY: appearance cue only (no height)   LENS: {'CAL' if result.scene.camera.lens_corrected else 'RAW'}",
        (28, 142),
        WHITE,
        0.45,
        1,
    )
    return canvas


def draw_rectified_view(result: VisionResult, width: int, height: int) -> np.ndarray:
    if result.rectified is None:
        canvas = np.full((height, width, 3), (22, 25, 30), dtype=np.uint8)
        _label(canvas, "RECTIFIED PARTITION VIEW", (30, 50), WHITE, 0.78, 2)
        _label(canvas, "Waiting for a stable partition candidate...", (30, 92), AMBER, 0.58, 1)
        _label(canvas, "Keep all four outer corners visible.", (30, 122), WHITE, 0.52, 1)
        return canvas

    canvas = result.rectified.copy()
    for cell, scene_cell in zip(result.grid_cells, result.scene.cells):
        polygon = np.round(cell.polygon).astype(np.int32)
        state_color = _state_color(scene_cell.state)
        if scene_cell.state == OccupancyState.OCCUPIED:
            tint = canvas.copy()
            cv2.fillPoly(tint, [polygon], state_color)
            cv2.addWeighted(tint, 0.12, canvas, 0.88, 0, canvas)
        cv2.polylines(canvas, [polygon], True, state_color, 3, cv2.LINE_AA)
        center = tuple(int(round(value)) for value in cell.center)
        cv2.circle(canvas, center, 6, state_color, -1, cv2.LINE_AA)
        text = f"CELL {cell.id}"
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        _label(canvas, text, (center[0] - text_width // 2, center[1] - 14), WHITE, 0.62, 2)
        state_text = f"{scene_cell.state.value}  {scene_cell.occupancy_score:.2f}"
        (state_width, _), _ = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        _label(canvas, state_text, (center[0] - state_width // 2, center[1] + 22), state_color, 0.48, 1)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, 42), DARK, -1)
    cv2.addWeighted(overlay, 0.76, canvas, 0.24, 0, canvas)
    _label(
        canvas,
        f"RECTIFIED PARTITION   {result.scene.partition.rows} x {result.scene.partition.cols}   "
        f"MESH {result.scene.partition.mesh_confidence:.2f}",
        (16, 29),
        WHITE,
        0.68,
        2,
    )
    return canvas
