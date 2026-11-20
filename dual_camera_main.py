from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from camera import CameraOpenError, LensUndistorter, OpenCVCamera, OpenCVCameraPair, discover_cameras
from vision.dual_view import DualViewCalibration, PlaneParallaxDetector, draw_height_view, height_cells
from vision.processor import VisionProcessor
from vision.stereo_height import (
    CellHeight,
    HeightCell,
    HeightState,
    StereoCalibration,
    StereoHeightEstimator,
    StereoResult,
    TemporalHeightFilter,
)

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("pac_dual_camera")
MAIN_WINDOW = "PAC Dual-Camera Height"
DEBUG_WINDOW = "PAC Floor Alignment Debug"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_camera(config: dict, override: str | None, name: str) -> OpenCVCamera:
    return OpenCVCamera(
        index=override if override is not None else config.get("index", "auto"),
        backend=str(config.get("backend", "auto")),
        width=int(config.get("width", 1280)),
        height=int(config.get("height", 720)),
        fps=float(config.get("fps", 30)),
        scan_max_index=int(config.get("scan_max_index", 8)),
        warmup_frames=int(config.get("warmup_frames", 8)),
        source_name=name,
        pixel_format=config.get("pixel_format"),
    )


def resolve(path_text: str | None, default: str) -> Path:
    path = Path(str(path_text or default))
    return path if path.is_absolute() else ROOT / path


def load_stereo(dual: dict, lens_ok: bool) -> tuple[StereoCalibration | None, str]:
    """Metric height is only allowed with lens + stereo + floor-plane calibration."""
    if not lens_ok:
        return None, "UNCALIBRATED (lens calibration missing)"
    path = resolve(dual.get("stereo", {}).get("calibration_file"), "config/stereo_calibration.json")
    calibration, reason = StereoCalibration.try_load(path)
    if calibration is None:
        return None, f"UNCALIBRATED ({reason})"
    if not calibration.has_floor_plane:
        return None, "UNCALIBRATED (floor plane not measured)"
    stereo_cfg = dual.get("stereo", {})
    max_rms = float(stereo_cfg.get("max_stereo_rms_px", 1.5))
    min_baseline = float(stereo_cfg.get("min_baseline_mm", 20.0))
    if not np.isfinite(calibration.stereo_rms_px) or calibration.stereo_rms_px <= 0.0 or calibration.stereo_rms_px > max_rms:
        return None, f"UNCALIBRATED (stereo RMS {calibration.stereo_rms_px:.2f}px; limit {max_rms:.2f}px)"
    if not np.isfinite(calibration.baseline_mm) or calibration.baseline_mm < min_baseline:
        return None, f"UNCALIBRATED (baseline {calibration.baseline_mm:.1f}mm; minimum {min_baseline:.1f}mm)"
    return calibration, f"STEREO  baseline {calibration.baseline_mm:.0f}mm  rms {calibration.stereo_rms_px:.2f}px"


def scene_json(cells: list[CellHeight], frame_id: int, skew_ms: float, rows: int, cols: int, boundary: str, height_status: str, matches: int) -> dict:
    return {
        "timestamp": time.time(),
        "frame_id": frame_id,
        "capture_skew_ms": round(skew_ms, 3),
        "rows": rows,
        "cols": cols,
        "boundary_state": boundary,
        "height_status": height_status,
        "match_count": matches,
        "height_source": "stereo_triangulation" if not height_status.startswith("UNCALIBRATED") else None,
        "cells": [cell.to_dict() for cell in cells],
    }


def write_json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        temporary.replace(path)
    except PermissionError:
        # OneDrive/indexers can briefly lock the previous JSON on Windows.
        # Telemetry must never terminate the live camera loop.
        temporary.unlink(missing_ok=True)
        LOGGER.warning("Scene JSON is temporarily locked; skipped this update: %s", path)


def cell_geometry_signature(cells: list[HeightCell], rows: int, cols: int) -> tuple:
    """Stable identity for the currently detected physical grid.

    Cell ids alone are unsafe: a replacement box can have the same number of
    cells while its dividers are in different places.  Quantising the box and
    normalised cell centres ignores pixel jitter but invalidates stale stereo
    results after a box/topology change.
    """
    if not cells:
        return (0, 0)
    points = np.concatenate([np.asarray(cell.polygon, np.float32).reshape(-1, 2) for cell in cells], axis=0)
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = np.maximum(high - low, 1.0)
    box = (
        int(round(float((low[0] + high[0]) * 0.5) / 16.0)),
        int(round(float((low[1] + high[1]) * 0.5) / 16.0)),
        int(round(float(span[0]) / 16.0)),
        int(round(float(span[1]) / 16.0)),
    )
    centres = []
    for cell in sorted(cells, key=lambda item: (item.row, item.col, item.id)):
        centre = np.asarray(cell.polygon, np.float32).reshape(-1, 2).mean(axis=0)
        normalised = (centre - low) / span
        centres.append((cell.row, cell.col, int(round(float(normalised[0]) * 25.0)), int(round(float(normalised[1]) * 25.0))))
    return (int(rows), int(cols), *box, tuple(centres))


class LiveBoundaryWorker:
    """Global-camera boundary graph in the background (grid reference)."""

    def __init__(self, config: dict, lens_corrected: bool) -> None:
        self.processor = VisionProcessor(config)
        self.lens_corrected = bool(lens_corrected)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._result = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="boundary-graph", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()

    def snapshot(self):
        with self._lock:
            return self._result

    def _run(self) -> None:
        frame_id = 0
        while not self._stop.is_set():
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                time.sleep(0.02)
                continue
            result = self.processor.process(frame, frame_id, 0.0, {"lens_corrected": self.lens_corrected})
            frame_id += 1
            with self._lock:
                self._result = result

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class HeightWorker:
    """Stereo height on the most recent frame pair; every cell is independent."""

    def __init__(self, estimator: StereoHeightEstimator, calibration: StereoCalibration | None) -> None:
        self.estimator = estimator
        self.calibration = calibration
        self._lock = threading.Lock()
        self._job: tuple[np.ndarray, np.ndarray, list[HeightCell], tuple] | None = None
        self._result: tuple[tuple, StereoResult] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stereo-height", daemon=True)
        self._thread.start()

    def submit(self, global_frame: np.ndarray, wrist_frame: np.ndarray, cells: list[HeightCell], signature: tuple) -> None:
        with self._lock:
            self._job = (global_frame.copy(), wrist_frame.copy(), cells, signature)

    def snapshot(self) -> tuple[tuple, StereoResult] | None:
        with self._lock:
            return self._result

    def invalidate(self) -> None:
        with self._lock:
            self._job = None
            self._result = None

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                time.sleep(0.02)
                continue
            result = self.estimator.process(job[0], job[1], job[2], self.calibration)
            with self._lock:
                self._result = (job[3], result)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dual = config.get("dual_camera", {})
    grid_path = resolve(dual.get("surface_calibration_file"), "config/dual_surface_calibration.json")
    fixed_grid: DualViewCalibration | None = None
    if args.fixed_grid:
        if not grid_path.exists():
            LOGGER.error("--fixed-grid requested but setup file is missing: %s", grid_path)
            return 2
        fixed_grid = DualViewCalibration.load(grid_path)

    global_config = dual.get("global_camera", config.get("camera", {}))
    wrist_config = dual.get("wrist_camera", {})
    pair = OpenCVCameraPair(
        build_camera(global_config, args.global_index, "APC850-GLOBAL"),
        build_camera(wrist_config, args.wrist_index, "AU100-WRIST"),
    )
    try:
        info = pair.open()
    except (CameraOpenError, RuntimeError) as exc:
        LOGGER.error("Could not open camera pair: %s", exc)
        return 3

    global_lens = LensUndistorter.from_config(dual.get("global_calibration", {}), ROOT)
    wrist_lens = LensUndistorter.from_config(dual.get("wrist_calibration", {}), ROOT)
    stereo, height_status = load_stereo(dual, global_lens is not None and wrist_lens is not None)
    if stereo is None:
        LOGGER.warning("%s -> heights are NOT metric; run calibrate_dual_view.py --stereo", height_status)
    estimator = StereoHeightEstimator(dual.get("stereo", {}))
    height_worker = HeightWorker(estimator, stereo)
    boundary_worker = LiveBoundaryWorker(config, lens_corrected=global_lens is not None) if fixed_grid is None else None
    temporal = TemporalHeightFilter(
        window=int(dual.get("stereo", {}).get("temporal_window", 5)),
        hold_unknown_frames=int(dual.get("hold_unknown_frames", 4)),
    )
    aux = PlaneParallaxDetector(dual) if fixed_grid is not None else None
    output = ROOT / "latest_dual_scene.json"
    write_scene_json = bool(config.get("output", {}).get("write_json", False))
    LOGGER.info("Global camera: %s", info.global_camera.as_dict())
    LOGGER.info("Wrist camera: %s", info.wrist_camera.as_dict())
    LOGGER.info("Grid source: %s", "fixed setup file" if fixed_grid is not None else "live global boundary graph")
    LOGGER.info("Robot must be stopped at the calibrated inspection pose. V=floor debug, Q/ESC=exit")

    frame_id = 0
    failures = 0
    debug_visible = bool(args.show_aligned)
    last_height_submit = 0.0
    last_boundary_submit = 0.0
    shown_cells: list[CellHeight] = []
    active_signature: tuple | None = None
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frames = pair.read()
            if not ok or frames is None:
                failures += 1
                if failures >= 20:
                    LOGGER.error("Camera pair stopped returning frames")
                    return 4
                if cv2.waitKey(20) & 0xFF in (27, ord("q")):
                    break
                continue
            failures = 0
            global_frame = global_lens.apply(frames.global_frame) if global_lens is not None else frames.global_frame
            wrist_frame = wrist_lens.apply(frames.wrist_frame) if wrist_lens is not None else frames.wrist_frame
            now = time.perf_counter()

            # ---- boundary reference: global camera only
            edge_samples: dict[str, np.ndarray] = {}
            if fixed_grid is not None:
                active_grid = fixed_grid.scaled((global_frame.shape[1], global_frame.shape[0]), fixed_grid.wrist_image_size)
                cells = height_cells(active_grid)
                rows, cols, boundary_state = active_grid.rows, active_grid.cols, "LOCKED (setup)"
                edge_samples = active_grid.edge_samples
            else:
                assert boundary_worker is not None
                if now - last_boundary_submit > 0.5:
                    boundary_worker.submit(global_frame)
                    last_boundary_submit = now
                result = boundary_worker.snapshot()
                trace = result.dense_trace if result is not None else None
                state = result.scene.partition.topology_state if result is not None else "SEARCHING"
                if trace is not None and trace.regular and state != "UNCERTAIN":
                    cells = [
                        HeightCell(index, index // trace.cols, index % trace.cols, polygon)
                        for index, polygon in enumerate(trace.cell_polygons())
                    ]
                    rows, cols = trace.rows, trace.cols
                    edge_samples = {k: v for k, v in trace.edge_samples.items() if trace.edge_present.get(k, True)}
                else:
                    cells, rows, cols = [], 0, 0
                    state = "UNCERTAIN" if result is not None else "SEARCHING"
                boundary_state = state

            # ---- height: stereo geometry, per cell
            signature = cell_geometry_signature(cells, rows, cols)
            if signature != active_signature:
                active_signature = signature
                temporal.reset()
                height_worker.invalidate()
                shown_cells = []
            if cells and now - last_height_submit > float(dual.get("stereo", {}).get("min_interval_s", 0.3)):
                height_worker.submit(global_frame, wrist_frame, cells, signature)
                last_height_submit = now
            height_snapshot = height_worker.snapshot()
            stereo_result = height_snapshot[1] if height_snapshot is not None and height_snapshot[0] == signature else None
            if stereo_result is not None and len(stereo_result.cells) == len(cells):
                shown_cells = temporal.update(stereo_result.cells) if stereo_result.calibrated else stereo_result.cells
                matches = stereo_result.total_matches
            else:
                shown_cells = [
                    CellHeight(c.id, c.row, c.col, HeightState.UNCALIBRATED if stereo is None else HeightState.UNKNOWN, None, 0.0, 0, None, reason="PENDING")
                    for c in cells
                ]
                matches = 0
            view = draw_height_view(
                global_frame,
                cells,
                shown_cells,
                rows=rows,
                cols=cols,
                boundary_state=boundary_state,
                height_status=height_status,
                edge_samples=edge_samples,
                match_count=matches,
            )
            cv2.putText(view, f"SKEW {frames.capture_skew_ms:.1f}ms", (440, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.imshow(MAIN_WINDOW, view)
            if debug_visible and fixed_grid is not None and aux is not None:
                aux_result = aux.process(global_frame, wrist_frame, fixed_grid)
                cv2.imshow(DEBUG_WINDOW, cv2.addWeighted(global_frame, 0.52, aux_result.warped_wrist, 0.48, 0.0))
            if write_scene_json and frame_id % 5 == 0 and shown_cells:
                write_json_atomic(output, scene_json(shown_cells, frame_id, frames.capture_skew_ms, rows, cols, boundary_state, height_status, matches))
            frame_id += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("v"):
                debug_visible = not debug_visible
                if debug_visible:
                    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
                else:
                    try:
                        cv2.destroyWindow(DEBUG_WINDOW)
                    except cv2.error:
                        pass
    finally:
        height_worker.close()
        if boundary_worker is not None:
            boundary_worker.close()
        pair.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PAC dual-camera stereo height per detected cell")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "vision.json")
    parser.add_argument("--global-index", default=None)
    parser.add_argument("--wrist-index", default=None)
    parser.add_argument("--show-aligned", action="store_true", help="Floor-homography residual debug view (auxiliary)")
    parser.add_argument("--fixed-grid", action="store_true", help="Use the saved setup grid (live adaptive detection is the default)")
    parser.add_argument("--live-grid", action="store_true", help=argparse.SUPPRESS)  # backward-compatible no-op
    parser.add_argument("--list-cameras", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    if args.list_cameras:
        candidates = discover_cameras(max_index=8, backend="auto")
        for candidate in candidates:
            print(candidate.summary())
        return 0 if candidates else 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
