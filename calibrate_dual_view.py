from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from camera import CameraOpenError, LensUndistorter, OpenCVCamera, OpenCVCameraPair
from vision import VisionProcessor, VisionResult
from vision.dense_edges import DenseGridTrace
from vision.dual_view import DualViewCalibration
from vision.stereo_height import StereoCalibration, plane_from_board_pose, plane_from_points

ROOT = Path(__file__).resolve().parent
WINDOW = "PAC Dual-Camera Surface Setup"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
HEADER_HEIGHT = 96

def grid_click_order(rows: int, cols: int) -> list[tuple[str, int, int]]:
    """Manual fallback order for any R x C: outer corners, then sides, then inner nodes."""
    order = [("TL", 0, 0), ("TR", 0, cols), ("BR", rows, cols), ("BL", rows, 0)]
    order += [(f"TOP-{c}", 0, c) for c in range(1, cols)]
    order += [(f"RIGHT-{r}", r, cols) for r in range(1, rows)]
    order += [(f"BOTTOM-{c}", rows, c) for c in range(cols - 1, 0, -1)]
    order += [(f"LEFT-{r}", r, 0) for r in range(rows - 1, 0, -1)]
    order += [(f"INNER-{r}{c}", r, c) for r in range(1, rows) for c in range(1, cols)]
    return order


def edge_click_order(rows: int, cols: int) -> list[tuple[str, str, tuple[int, int], tuple[int, int]]]:
    return [
        *((f"H{r}-{c}", f"h_{r}_{c}", (r, c), (r, c + 1)) for r in range(rows + 1) for c in range(cols)),
        *((f"V{r}-{c}", f"v_{r}_{c}", (r, c), (r + 1, c)) for c in range(cols + 1) for r in range(rows)),
    ]


GRID_CLICK_ORDER = grid_click_order(3, 3)
EDGE_CLICK_ORDER = edge_click_order(3, 3)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def camera_from_config(config: dict, index_override: str | None, label: str) -> OpenCVCamera:
    return OpenCVCamera(
        index=index_override if index_override is not None else config.get("index", "auto"),
        backend=str(config.get("backend", "auto")),
        width=int(config.get("width", 1280)),
        height=int(config.get("height", 720)),
        fps=float(config.get("fps", 30)),
        scan_max_index=int(config.get("scan_max_index", 8)),
        warmup_frames=int(config.get("warmup_frames", 8)),
        source_name=label,
        pixel_format=config.get("pixel_format"),
    )


def encode_image(path: Path, image: np.ndarray) -> None:
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode reference image: {path}")
    buffer.tofile(str(path))


class DenseGuideWorker:
    """Run expensive edge following away from the live camera display loop."""

    def __init__(self, config: dict) -> None:
        self.processor = VisionProcessor(config)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending: tuple[np.ndarray, int] | None = None
        self._result: VisionResult | None = None
        self._trace: DenseGridTrace | None = None
        self._thread = threading.Thread(target=self._run, name="dense-edge-guide", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, frame_id: int) -> None:
        with self._lock:
            self._pending = (frame.copy(), frame_id)
        self._wake.set()

    def snapshot(self) -> tuple[VisionResult | None, DenseGridTrace | None]:
        with self._lock:
            return self._result, self._trace

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.2)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                continue
            frame, frame_id = pending
            result = self.processor.process(
                frame,
                frame_id=frame_id,
                measured_fps=0.0,
                camera_metadata={"lens_corrected": True},
            )
            trace = result.dense_trace
            if trace is not None and (not trace.regular or result.scene.partition.topology_state == "UNCERTAIN"):
                trace = None  # never lock an unconfirmed / irregular topology
            with self._lock:
                self._result = result
                self._trace = trace

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2.0)


@dataclass
class SetupState:
    global_frame: np.ndarray | None = None
    global_result: VisionResult | None = None
    wrist_frame: np.ndarray | None = None
    active_view: str = "global"
    auto_trace: DenseGridTrace | None = None
    frozen: bool = False
    grid_points: list[tuple[float, float]] = field(default_factory=list)
    locked_nodes: np.ndarray | None = None
    locked_edge_samples: dict[str, np.ndarray] = field(default_factory=dict)
    edge_index: int = 0
    edge_points: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    min_edge_points: int = 3
    floor_global: list[tuple[float, float]] = field(default_factory=list)
    floor_wrist: list[tuple[float, float]] = field(default_factory=list)
    pending_global: tuple[float, float] | None = None
    message: str = "SPACE: freeze both cameras at the inspection pose"
    rows: int = 3
    cols: int = 3

    @property
    def grid_order(self) -> list[tuple[str, int, int]]:
        return grid_click_order(self.rows, self.cols)

    @property
    def edge_order(self) -> list[tuple[str, str, tuple[int, int], tuple[int, int]]]:
        return edge_click_order(self.rows, self.cols)

    @property
    def grid_complete(self) -> bool:
        return self.locked_nodes is not None or len(self.grid_points) == len(self.grid_order)

    @property
    def edges_complete(self) -> bool:
        return self.locked_nodes is not None or self.edge_index >= len(self.edge_order)

    @property
    def current_edge(self) -> tuple[str, str, tuple[int, int], tuple[int, int]] | None:
        return None if self.edges_complete else self.edge_order[self.edge_index]

    def _raw_point(self, x: int, y: int) -> tuple[str, tuple[float, float]] | None:
        if y < HEADER_HEIGHT or y >= HEADER_HEIGHT + DISPLAY_HEIGHT:
            return None
        if self.global_frame is None or self.wrist_frame is None:
            return None
        if x < 0 or x >= DISPLAY_WIDTH:
            return None
        image = self.global_frame if self.active_view == "global" else self.wrist_frame
        height, width = image.shape[:2]
        return self.active_view, (x * width / DISPLAY_WIDTH, (y - HEADER_HEIGHT) * height / DISPLAY_HEIGHT)

    def click(self, x: int, y: int) -> None:
        if not self.frozen:
            self.message = "Press SPACE first to freeze a stable pair"
            return
        mapped = self._raw_point(x, y)
        if mapped is None:
            return
        side, point = mapped
        if not self.grid_complete:
            if side != "global":
                self.message = "Grid vertices are registered in the GLOBAL main view (press V)"
                return
            self.grid_points.append(point)
            if self.grid_complete:
                self.message = f"Grid complete. Add {self.min_edge_points}+ points along edge {self.edge_order[0][0]}, then N/ENTER"
            else:
                self.message = f"Next global vertex: {self.grid_order[len(self.grid_points)][0]}"
            return
        if not self.edges_complete:
            if side != "global":
                self.message = "Curved edge points are registered in the GLOBAL main view (press V)"
                return
            assert self.current_edge is not None
            name, key, _, _ = self.current_edge
            self.edge_points.setdefault(key, []).append(point)
            count = len(self.edge_points[key])
            self.message = f"Edge {name}: {count} points. Add more or N/ENTER when >= {self.min_edge_points}"
            return
        if self.pending_global is None:
            if side != "global":
                self.message = "First click the floor feature in the GLOBAL main view (press V)"
                return
            self.pending_global = point
            self.active_view = "wrist"
            self.message = "AUX view: click the SAME floor feature"
        else:
            if side != "wrist":
                self.pending_global = point
                self.active_view = "wrist"
                self.message = "Global floor point replaced; click its match in AUX"
                return
            self.floor_global.append(self.pending_global)
            self.floor_wrist.append(point)
            self.pending_global = None
            self.active_view = "global"
            self.message = f"Floor pairs: {len(self.floor_global)}. Add more in GLOBAL or press ENTER to save"

    def undo(self) -> None:
        if self.pending_global is not None:
            self.pending_global = None
            self.active_view = "global"
        elif self.floor_global:
            self.floor_global.pop()
            self.floor_wrist.pop()
        elif self.locked_nodes is not None:
            self.locked_nodes = None
            self.locked_edge_samples.clear()
        elif self.grid_complete and not self.edges_complete:
            assert self.current_edge is not None
            key = self.current_edge[1]
            if self.edge_points.get(key):
                self.edge_points[key].pop()
            elif self.edge_index > 0:
                self.edge_index -= 1
                previous_key = self.edge_order[self.edge_index][1]
                if self.edge_points.get(previous_key):
                    self.edge_points[previous_key].pop()
        elif self.edges_complete:
            self.edge_index = len(self.edge_order) - 1
            key = self.edge_order[self.edge_index][1]
            if self.edge_points.get(key):
                self.edge_points[key].pop()
        elif self.grid_points:
            self.grid_points.pop()
        self.message = "Last point removed"

    def advance_edge(self) -> bool:
        if self.edges_complete:
            return True
        assert self.current_edge is not None
        name, key, _, _ = self.current_edge
        count = len(self.edge_points.get(key, []))
        if count < self.min_edge_points:
            self.message = f"Edge {name} needs at least {self.min_edge_points} points ({count}/{self.min_edge_points})"
            return False
        self.edge_index += 1
        if self.edges_complete:
            self.message = f"All {len(self.edge_order)} curved edges complete. Click FLOOR point in GLOBAL, then its match in AUX"
        else:
            self.message = f"Next edge {self.edge_order[self.edge_index][0]}: add points, then N/ENTER"
        return True

    def reset(self) -> None:
        self.grid_points.clear()
        self.locked_nodes = None
        self.locked_edge_samples.clear()
        self.edge_index = 0
        self.edge_points.clear()
        self.floor_global.clear()
        self.floor_wrist.clear()
        self.pending_global = None
        self.active_view = "global"
        self.message = "Reset. Click TL in the GLOBAL main view"

    def lock_dense_trace(self) -> bool:
        if self.auto_trace is None:
            return False
        self.rows, self.cols = self.auto_trace.rows, self.auto_trace.cols
        self.locked_nodes = self.auto_trace.nodes.copy()
        self.locked_edge_samples = {
            key: path.copy() for key, path in self.auto_trace.edge_samples.items()
        }
        point_count = sum(len(path) for path in self.locked_edge_samples.values())
        self.message = (
            f"Locked {self.auto_trace.rows}x{self.auto_trace.cols}, {point_count} edge points. "
            "Click FLOOR feature in GLOBAL"
        )
        return True


def mouse_callback(event: int, x: int, y: int, flags: int, state: SetupState) -> None:
    del flags
    if event == cv2.EVENT_LBUTTONDOWN:
        state.click(x, y)


def display_point(point: tuple[float, float], image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    return (
        int(round(point[0] * DISPLAY_WIDTH / width)),
        int(round(point[1] * DISPLAY_HEIGHT / height)) + HEADER_HEIGHT,
    )


def draw_global_guide(
    frame: np.ndarray,
    result: VisionResult,
    trace: DenseGridTrace | None,
) -> np.ndarray:
    """Draw geometry hints only; occupancy remains a separate runtime decision."""
    canvas = frame.copy()
    if trace is None:
        cv2.putText(canvas, "DENSE EDGES: SEARCHING", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 190, 255), 2, cv2.LINE_AA)
        return canvas
    # Full row/column paths are deliberately extended to find reliable
    # intersections. Only node-to-node physical segments are visible.
    for line in trace.edge_samples.values():
        integer_line = np.round(line).astype(np.int32)
        cv2.polylines(canvas, [integer_line], False, (70, 230, 120), 1, cv2.LINE_AA)
        for xy in integer_line:
            cv2.circle(canvas, tuple(int(value) for value in xy), 1, (90, 255, 145), -1, cv2.LINE_AA)
    corners = (
        trace.nodes[0, 0],
        trace.nodes[0, trace.cols],
        trace.nodes[trace.rows, trace.cols],
        trace.nodes[trace.rows, 0],
    )
    for name, point in zip(("TL", "TR", "BR", "BL"), corners):
        point = np.round(point).astype(np.int32)
        xy = tuple(int(value) for value in point)
        cv2.circle(canvas, xy, 7, (255, 220, 40), -1, cv2.LINE_AA)
        cv2.putText(canvas, name, (xy[0] + 9, xy[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 40), 2, cv2.LINE_AA)
    point_count = sum(len(path) for path in trace.edge_samples.values())
    cv2.putText(
        canvas,
        f"DENSE {trace.rows}x{trace.cols} EDGES: {point_count} POINTS",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (70, 230, 120),
        2,
        cv2.LINE_AA,
    )
    return canvas


def render(state: SetupState) -> np.ndarray:
    if state.global_frame is None or state.wrist_frame is None:
        return np.zeros((HEADER_HEIGHT + DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    canvas = np.zeros((HEADER_HEIGHT + DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    if state.active_view == "global":
        source = (
            draw_global_guide(state.global_frame, state.global_result, state.auto_trace)
            if state.global_result is not None
            else state.global_frame
        )
        view_name = "GLOBAL / MAIN / FIXED"
    else:
        source = state.wrist_frame
        view_name = "WRIST / AUXILIARY / GRIPPER"
    canvas[HEADER_HEIGHT:] = cv2.resize(source, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, view_name, (DISPLAY_WIDTH - 375, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, state.message, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 245), 2, cv2.LINE_AA)
    controls = "V global/aux | SPACE freeze/live | U undo | R reset | N/ENTER next edge | ENTER save | Q cancel"
    cv2.putText(canvas, controls, (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 210, 255), 1, cv2.LINE_AA)

    node_map: dict[tuple[int, int], tuple[int, int]] = {}
    if state.active_view == "global":
        for index, point in enumerate(state.grid_points):
            name, row, col = state.grid_order[index]
            xy = display_point(point, state.global_frame)
            node_map[(row, col)] = xy
            cv2.circle(canvas, xy, 6, (255, 220, 40), -1, cv2.LINE_AA)
            cv2.putText(canvas, name, (xy[0] + 7, xy[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 220, 40), 1, cv2.LINE_AA)
        for row in range(state.rows + 1):
            for col in range(state.cols):
                if (row, col) in node_map and (row, col + 1) in node_map:
                    cv2.line(canvas, node_map[(row, col)], node_map[(row, col + 1)], (70, 230, 120), 2, cv2.LINE_AA)
        for col in range(state.cols + 1):
            for row in range(state.rows):
                if (row, col) in node_map and (row + 1, col) in node_map:
                    cv2.line(canvas, node_map[(row, col)], node_map[(row + 1, col)], (70, 230, 120), 2, cv2.LINE_AA)

    if state.active_view == "global" and state.grid_complete and state.locked_nodes is None:
        nodes = build_grid_nodes(state.grid_points, state.rows, state.cols)
        for edge_index, (name, key, start_rc, end_rc) in enumerate(state.edge_order):
            controls = state.edge_points.get(key, [])
            if not controls and edge_index > state.edge_index:
                continue
            raw_line = edge_polyline(nodes, key, start_rc, end_rc, controls)
            display_line = np.asarray(
                [display_point((float(point[0]), float(point[1])), state.global_frame) for point in raw_line],
                dtype=np.int32,
            )
            colour = (0, 190, 255) if edge_index == state.edge_index and not state.edges_complete else (80, 170, 255)
            dense = len(controls) > 10
            cv2.polylines(canvas, [display_line], False, colour, 1 if dense else 2, cv2.LINE_AA)
            for control in controls:
                xy = display_point(control, state.global_frame)
                cv2.circle(canvas, xy, 1 if dense else 4, colour, -1, cv2.LINE_AA)
            if edge_index == state.edge_index and not state.edges_complete:
                midpoint = display_line[len(display_line) // 2]
                cv2.putText(canvas, name, tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2, cv2.LINE_AA)

    points = state.floor_global if state.active_view == "global" else state.floor_wrist
    image = state.global_frame if state.active_view == "global" else state.wrist_frame
    for index, point in enumerate(points, start=1):
        xy = display_point(point, image)
        cv2.circle(canvas, xy, 5, (220, 80, 230), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (xy[0] + 5, xy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 80, 230), 1)
    if state.pending_global is not None and state.active_view == "global":
        xy = display_point(state.pending_global, state.global_frame)
        cv2.circle(canvas, xy, 7, (0, 190, 255), 2, cv2.LINE_AA)
    return canvas


def build_grid_nodes(points: list[tuple[float, float]], rows: int = 3, cols: int = 3) -> np.ndarray:
    order = grid_click_order(rows, cols)
    if len(points) != len(order):
        raise ValueError(f"All {len(order)} grid vertices must be registered")
    nodes = np.zeros((rows + 1, cols + 1, 2), dtype=np.float32)
    for point, (_, row, col) in zip(points, order):
        nodes[row, col] = point
    return nodes


def edge_polyline(
    nodes: np.ndarray,
    key: str,
    start_rc: tuple[int, int],
    end_rc: tuple[int, int],
    controls: list[tuple[float, float]],
) -> np.ndarray:
    del key
    start = nodes[start_rc]
    end = nodes[end_rc]
    points = np.asarray([start, *controls, end], dtype=np.float32).reshape(-1, 2)
    direction = end - start
    denominator = max(float(np.dot(direction, direction)), 1e-6)
    order = np.argsort(((points - start) @ direction) / denominator)
    ordered = points[order]
    ordered[0] = start
    ordered[-1] = end
    return ordered


def build_edge_samples(
    nodes: np.ndarray,
    controls: dict[str, list[tuple[float, float]]],
) -> dict[str, np.ndarray]:
    rows, cols = nodes.shape[0] - 1, nodes.shape[1] - 1
    return {
        key: edge_polyline(nodes, key, start_rc, end_rc, controls.get(key, []))
        for _, key, start_rc, end_rc in edge_click_order(rows, cols)
    }


def checkerboard_view_signature(corners: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Compact pose signature used to reject near-duplicate calibration views."""
    points = np.asarray(corners, np.float64).reshape(-1, 2)
    width, height = max(int(image_size[0]), 1), max(int(image_size[1]), 1)
    centre = points.mean(axis=0) / np.asarray([width, height], np.float64)
    span = np.ptp(points, axis=0) / np.asarray([width, height], np.float64)
    axis = points[-1] - points[0]
    angle = float(np.arctan2(axis[1], axis[0]))
    return np.asarray([centre[0], centre[1], span[0], span[1], np.sin(angle), np.cos(angle)], np.float64)


def run_stereo_calibration(args: argparse.Namespace, config: dict) -> int:
    """Stereo extrinsics (global -> wrist at the inspection pose) + floor plane.

    SPACE: store a checkerboard view seen by BOTH cameras (vary pose, >= 10).
    F: store the floor-plane pose (board lying flat on the cell floor).
    C: compute and save.  Q: cancel.  Lens calibration of both cameras is required.
    """
    dual = config.get("dual_camera", {})
    stereo_cfg = dual.get("stereo", {})
    min_pairs = max(6, int(stereo_cfg.get("min_calibration_pairs", 10)))
    max_rms = float(stereo_cfg.get("max_stereo_rms_px", 1.5))
    min_baseline = float(stereo_cfg.get("min_baseline_mm", 20.0))
    global_lens = LensUndistorter.from_config(dual.get("global_calibration", {}), ROOT)
    wrist_lens = LensUndistorter.from_config(dual.get("wrist_calibration", {}), ROOT)
    if global_lens is None or wrist_lens is None:
        print("Lens calibration missing. Run calibrate_camera.py for BOTH cameras first (heights stay UNCALIBRATED).")
        return 2
    pattern = (int(args.inner_cols), int(args.inner_rows))
    template = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    template[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * float(args.square_mm)
    pair = OpenCVCameraPair(
        camera_from_config(dual.get("global_camera", config.get("camera", {})), args.global_index, "GLOBAL"),
        camera_from_config(dual.get("wrist_camera", {}), args.wrist_index, "WRIST"),
    )
    try:
        pair.open()
    except (CameraOpenError, RuntimeError) as exc:
        print(f"Camera pair could not open: {exc}")
        return 2
    object_points: list[np.ndarray] = []
    global_points: list[np.ndarray] = []
    wrist_points: list[np.ndarray] = []
    floor_pose: np.ndarray | None = None
    view_signatures: list[np.ndarray] = []
    notice = ""
    sizes: tuple[tuple[int, int], tuple[int, int]] | None = None
    window = "PAC Stereo Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frames = pair.read()
            if not ok or frames is None:
                if cv2.waitKey(20) & 0xFF in (27, ord("q")):
                    return 1
                continue
            g = global_lens.apply(frames.global_frame)
            w = wrist_lens.apply(frames.wrist_frame)
            sizes = ((g.shape[1], g.shape[0]), (w.shape[1], w.shape[0]))
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
            found_g, corners_g = cv2.findChessboardCornersSB(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), pattern, flags=flags)
            found_w, corners_w = cv2.findChessboardCornersSB(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY), pattern, flags=flags)
            view = np.hstack((cv2.resize(g, (640, 360)), cv2.resize(w, (640, 360))))
            status = f"PAIRS {len(object_points)}/{min_pairs}  FLOOR {'OK' if floor_pose is not None else '--'}  BOARD G:{int(found_g)} W:{int(found_w)}"
            cv2.putText(view, status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 80) if found_g and found_w else (40, 180, 255), 2)
            cv2.putText(view, "SPACE pair | F floor pose | C compute+save | Q cancel", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)
            if notice:
                cv2.putText(view, notice, (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 180, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 1
            if key == 32 and found_g and found_w:
                signature = np.concatenate((checkerboard_view_signature(corners_g, sizes[0]), checkerboard_view_signature(corners_w, sizes[1])))
                if view_signatures and min(float(np.linalg.norm(signature - previous)) for previous in view_signatures) < 0.045:
                    notice = "DUPLICATE POSE: move/tilt the checkerboard before SPACE"
                else:
                    object_points.append(template.copy())
                    global_points.append(corners_g.astype(np.float32))
                    wrist_points.append(corners_w.astype(np.float32))
                    view_signatures.append(signature)
                    notice = "PAIR SAVED"
            elif key == ord("f") and found_g:
                floor_pose = corners_g.astype(np.float32)
                notice = "FLOOR PLANE POSE SAVED"
            elif key == ord("c"):
                if len(object_points) >= min_pairs and sizes is not None:
                    break
                notice = f"NEED {min_pairs - len(object_points)} MORE DIVERSE PAIRS"
    finally:
        pair.release()
        cv2.destroyAllWindows()
    assert sizes is not None
    k_global = global_lens.output_camera_matrix(*sizes[0])
    k_wrist = wrist_lens.output_camera_matrix(*sizes[1])
    zeros = np.zeros(5)
    plane = plane_from_board_pose(template, floor_pose, k_global, zeros) if floor_pose is not None else None
    calibration = StereoCalibration.from_stereo_views(
        object_points, global_points, wrist_points, k_global, zeros, k_wrist, zeros, sizes[0], sizes[1], floor_plane=plane
    )
    if not np.isfinite(calibration.stereo_rms_px) or calibration.stereo_rms_px > max_rms:
        print(f"Calibration rejected: RMS {calibration.stereo_rms_px:.3f}px exceeds {max_rms:.3f}px. Capture more varied, sharper views.")
        return 2
    if not np.isfinite(calibration.baseline_mm) or calibration.baseline_mm < min_baseline:
        print(f"Calibration rejected: baseline {calibration.baseline_mm:.1f}mm is below {min_baseline:.1f}mm.")
        return 2
    if plane is not None:
        calibration = calibration.with_floor_plane(*plane, source="checkerboard_on_floor")
    output = resolve_path(stereo_cfg.get("calibration_file", "config/stereo_calibration.json"))
    calibration.save(output)
    print(f"Saved stereo calibration: {output}  RMS {calibration.stereo_rms_px:.3f}px  baseline {calibration.baseline_mm:.1f}mm")
    if plane is None:
        print("Floor plane not measured yet: run the grid setup and click >= 8 floor pairs (triangulated plane).")
    return 0


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def update_floor_plane_from_pairs(config: dict, floor_global: list, floor_wrist: list) -> str:
    """Triangulate clicked floor pairs with the stereo rig and fit the reference plane."""
    stereo_path = resolve_path(config.get("dual_camera", {}).get("stereo", {}).get("calibration_file", "config/stereo_calibration.json"))
    calibration, reason = StereoCalibration.try_load(stereo_path)
    if calibration is None:
        return f"stereo calibration not available ({reason}); heights stay UNCALIBRATED"
    if calibration.notes.get("floor_plane_source") == "checkerboard_on_floor":
        return "floor plane already measured with the checkerboard"
    points = calibration.triangulate_pixels(np.asarray(floor_global, np.float64), np.asarray(floor_wrist, np.float64))
    normal, offset, rms, inliers = plane_from_points(points)
    calibration.with_floor_plane(normal, offset, rms, source=f"triangulated_floor_pairs({int(inliers.sum())})").save(stereo_path)
    return f"floor plane from {int(inliers.sum())} triangulated pairs, RMS {rms:.2f}mm"


def main() -> int:
    parser = argparse.ArgumentParser(description="Register an automatically sized surface grid and two-camera floor plane")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "vision.json")
    parser.add_argument("--global-index", default=None)
    parser.add_argument("--wrist-index", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "config" / "dual_surface_calibration.json")
    parser.add_argument("--stereo", action="store_true", help="Calibrate stereo extrinsics + floor plane (checkerboard)")
    parser.add_argument("--rows", type=int, default=3, help="Manual fallback rows (auto trace overrides)")
    parser.add_argument("--cols", type=int, default=3, help="Manual fallback cols (auto trace overrides)")
    parser.add_argument("--inner-cols", type=int, default=9)
    parser.add_argument("--inner-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=25.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    config = load_json(args.config)
    if args.stereo:
        return run_stereo_calibration(args, config)
    dual = config.get("dual_camera", {})
    global_config = dual.get("global_camera", config.get("camera", {}))
    wrist_config = dual.get("wrist_camera", {})
    global_camera = camera_from_config(global_config, args.global_index, "GLOBAL")
    wrist_camera = camera_from_config(wrist_config, args.wrist_index, "WRIST")
    pair = OpenCVCameraPair(global_camera, wrist_camera)
    try:
        pair.open()
    except (CameraOpenError, RuntimeError) as exc:
        print(f"Camera pair could not open: {exc}")
        return 2

    global_lens = LensUndistorter.from_config(dual.get("global_calibration", {}), ROOT)
    wrist_lens = LensUndistorter.from_config(dual.get("wrist_calibration", {}), ROOT)
    guide_worker = DenseGuideWorker(config)
    guide_frame_id = 0
    last_guide_time = 0.0
    state = SetupState(min_edge_points=max(1, int(dual.get("edge_points_per_segment", 3))), rows=args.rows, cols=args.cols)
    min_points = max(4, int(dual.get("min_plane_points", 8)))
    ransac_threshold = float(dual.get("homography_ransac_threshold_px", 3.0))
    max_error = float(dual.get("max_reprojection_error_px", 3.0))
    calibration: DualViewCalibration | None = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse_callback, state)
    try:
        while True:
            if not state.frozen:
                ok, frames = pair.read()
                if ok and frames is not None:
                    global_frame = frames.global_frame
                    wrist_frame = frames.wrist_frame
                    state.global_frame = global_lens.apply(global_frame) if global_lens is not None else global_frame
                    state.wrist_frame = wrist_lens.apply(wrist_frame) if wrist_lens is not None else wrist_frame
                    now = time.perf_counter()
                    if now - last_guide_time >= 0.70:
                        guide_worker.submit(state.global_frame, guide_frame_id)
                        guide_frame_id += 1
                        last_guide_time = now
                    guide_result, guide_trace = guide_worker.snapshot()
                    if guide_result is not None:
                        state.global_result = guide_result
                        state.auto_trace = guide_trace
            cv2.imshow(WINDOW, render(state))
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q")):
                return 1
            if key == 32 and state.global_frame is not None:
                state.frozen = not state.frozen
                state.active_view = "global"
                if state.frozen and state.locked_nodes is None and not state.grid_points:
                    if not state.lock_dense_trace():
                        state.message = (
                            f"TOPOLOGY UNCERTAIN - not locked. Manual {state.rows}x{state.cols} fallback: click TL"
                        )
                elif not state.frozen:
                    state.message = "Live dense-edge preview"
            elif key == ord("v"):
                state.active_view = "wrist" if state.active_view == "global" else "global"
                state.message = "AUX camera (height verification only)" if state.active_view == "wrist" else "GLOBAL main camera"
            elif key == ord("u"):
                state.undo()
            elif key == ord("r"):
                state.reset()
            elif key == ord("n") or (key in (10, 13) and state.grid_complete and not state.edges_complete):
                if state.grid_complete:
                    state.advance_edge()
            elif key in (10, 13):
                if not state.grid_complete:
                    total = len(state.grid_order)
                    state.message = f"Need all {total} grid vertices ({len(state.grid_points)}/{total})"
                    continue
                if state.pending_global is not None:
                    state.message = "Complete the pending wrist match first"
                    continue
                if len(state.floor_global) < min_points:
                    state.message = f"Need at least {min_points} floor point pairs ({len(state.floor_global)}/{min_points})"
                    continue
                assert state.global_frame is not None and state.wrist_frame is not None
                try:
                    if state.locked_nodes is not None:
                        grid_nodes = state.locked_nodes
                        edge_samples = state.locked_edge_samples
                    else:
                        grid_nodes = build_grid_nodes(state.grid_points, state.rows, state.cols)
                        edge_samples = build_edge_samples(grid_nodes, state.edge_points)
                    rows = int(grid_nodes.shape[0] - 1)
                    cols = int(grid_nodes.shape[1] - 1)
                    calibration = DualViewCalibration.from_correspondences(
                        grid_nodes=grid_nodes,
                        edge_samples=edge_samples,
                        global_floor_points=np.asarray(state.floor_global, dtype=np.float32),
                        wrist_floor_points=np.asarray(state.floor_wrist, dtype=np.float32),
                        global_image_size=(state.global_frame.shape[1], state.global_frame.shape[0]),
                        wrist_image_size=(state.wrist_frame.shape[1], state.wrist_frame.shape[0]),
                        rows=rows,
                        cols=cols,
                        ransac_threshold_px=ransac_threshold,
                    )
                except ValueError as exc:
                    state.message = str(exc)
                    continue
                if calibration.reprojection_error_px > max_error:
                    state.message = (
                        f"RMS {calibration.reprojection_error_px:.2f}px > {max_error:.2f}px; "
                        "undo inaccurate floor pairs"
                    )
                    calibration = None
                    continue
                break
    finally:
        guide_worker.close()
        pair.release()
        cv2.destroyAllWindows()

    if calibration is None or state.global_frame is None or state.wrist_frame is None:
        return 1
    output = args.output if args.output.is_absolute() else ROOT / args.output
    calibration.save(output)
    encode_image(output.with_name("dual_reference_global.png"), state.global_frame)
    encode_image(output.with_name("dual_reference_wrist.png"), state.wrist_frame)
    print(f"Saved dual-view calibration: {output}")
    print("Stereo floor plane:", update_floor_plane_from_pairs(config, state.floor_global, state.floor_wrist))
    print(
        f"Grid: {calibration.rows}x{calibration.cols}, vertices: {(calibration.rows + 1) * (calibration.cols + 1)}, "
        f"floor pairs: {len(state.floor_global)}, RMS: {calibration.reprojection_error_px:.3f}px"
    )
    print("Keep both cameras rigid. Move the gripper to this same inspection pose before measuring height.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
