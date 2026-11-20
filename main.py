from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2

from camera import CameraOpenError, LensUndistorter, OpenCVCamera, discover_cameras
from vision.processor import VisionProcessor
from vision.synthetic import make_partition_frame
from vision.visualization import draw_global_view, draw_rectified_view

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "vision.json"
LOGGER = logging.getLogger("pac_vision")


def load_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON config {path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PAC 2026 global-camera partition vision")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera-index", default=None, help="Integer index or 'auto' (default: config)")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf", "any"), default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run once on a generated 3x3 scene; no camera needed")
    parser.add_argument("--show-rectified", action="store_true", help="Also open the debug Top View (toggle with v)")
    parser.add_argument("--debug", action="store_true")
    return parser


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.camera_index is not None:
        config["camera"]["index"] = args.camera_index
    if args.backend is not None:
        config["camera"]["backend"] = args.backend
    if args.rows is not None:
        config["grid"]["rows"] = args.rows
    if args.cols is not None:
        config["grid"]["cols"] = args.cols
    if args.rows is not None or args.cols is not None:
        config["grid"]["auto_detect"] = False


def run_self_test(config: dict) -> int:
    rows = int(config["grid"]["rows"])
    cols = int(config["grid"]["cols"])
    frame, expected = make_partition_frame(rows, cols)
    processor = VisionProcessor(config)
    result = processor.process(frame, frame_id=0, measured_fps=30.0, camera_metadata={"source": "SYNTHETIC"})
    if not result.detection.detected or result.rectified is None:
        LOGGER.error("Self-test failed: partition was not detected (confidence %.3f)", result.detection.confidence)
        return 1
    if len(result.scene.cells) != rows * cols:
        LOGGER.error("Self-test failed: expected %d cells, got %d", rows * cols, len(result.scene.cells))
        return 1
    LOGGER.info(
        "Self-test passed: confidence=%.3f, cells=%d, mean corner error=%.1f px",
        result.detection.confidence,
        len(result.scene.cells),
        float(abs(result.detection.corners - expected).mean()) if result.detection.corners is not None else -1.0,
    )
    print(result.scene.to_json())
    return 0


def run_live(config: dict, show_rectified: bool = False) -> int:
    camera_config = config["camera"]
    camera = OpenCVCamera(
        index=camera_config.get("index", "auto"),
        backend=camera_config.get("backend", "auto"),
        width=int(camera_config.get("width", 1280)),
        height=int(camera_config.get("height", 720)),
        fps=float(camera_config.get("fps", 30)),
        scan_max_index=int(camera_config.get("scan_max_index", 8)),
        warmup_frames=int(camera_config.get("warmup_frames", 8)),
        source_name=str(camera_config.get("name", "APC850")),
    )
    processor = VisionProcessor(config)
    undistorter = LensUndistorter.from_config(config.get("calibration", {}), PROJECT_ROOT)
    output_config = config.get("output", {})
    json_path = PROJECT_ROOT / str(output_config.get("json_path", "latest_scene.json"))
    write_json = bool(output_config.get("write_json", False))
    json_interval = max(1, int(output_config.get("json_interval_frames", 10)))
    read_fail_limit = max(1, int(camera_config.get("read_fail_limit", 20)))

    try:
        camera_info = camera.open()
    except CameraOpenError as exc:
        LOGGER.error("%s", exc)
        return 2

    metadata = camera_info.as_dict()
    metadata["lens_corrected"] = undistorter is not None
    LOGGER.info("Original camera view is fixed. Press v to toggle debug Top View; q or ESC exits")
    frame_id = 0
    failures = 0
    fps_ema = 0.0
    previous_time = time.perf_counter()

    try:
        cv2.namedWindow("PAC Global Camera", cv2.WINDOW_NORMAL)
        rectified_visible = show_rectified
        if rectified_visible:
            cv2.namedWindow("PAC Rectified Partition", cv2.WINDOW_NORMAL)
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                failures += 1
                LOGGER.warning("Camera frame read failed (%d/%d)", failures, read_fail_limit)
                if failures >= read_fail_limit:
                    LOGGER.error("Camera stopped returning frames; exiting cleanly")
                    return 3
                if cv2.waitKey(30) & 0xFF in (27, ord("q")):
                    break
                continue
            failures = 0
            if undistorter is not None:
                frame = undistorter.apply(frame)

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            fps_ema = instantaneous_fps if fps_ema <= 0 else 0.9 * fps_ema + 0.1 * instantaneous_fps

            result = processor.process(frame, frame_id, fps_ema, metadata)
            global_view = draw_global_view(frame, result, fps_ema)
            cv2.imshow("PAC Global Camera", global_view)
            if rectified_visible:
                rectified_view = draw_rectified_view(
                    result,
                    processor.rectified_width,
                    processor.rectified_height,
                )
                cv2.imshow("PAC Rectified Partition", rectified_view)

            if write_json and frame_id % json_interval == 0:
                result.scene.write_json(json_path)
            frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("v"):
                rectified_visible = not rectified_visible
                if rectified_visible:
                    cv2.namedWindow("PAC Rectified Partition", cv2.WINDOW_NORMAL)
                else:
                    try:
                        cv2.destroyWindow("PAC Rectified Partition")
                    except cv2.error:
                        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        config = load_config(args.config.resolve())
        apply_overrides(config, args)
        if args.list_cameras:
            cameras = discover_cameras(
                int(config["camera"].get("scan_max_index", 8)),
                str(config["camera"].get("backend", "auto")),
            )
            if not cameras:
                LOGGER.error("No camera returned a valid frame")
                return 1
            for camera in cameras:
                print(camera.summary())
            return 0
        if args.self_test:
            return run_self_test(config)
        return run_live(config, show_rectified=args.show_rectified)
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
