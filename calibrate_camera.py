from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from camera import CameraOpenError, OpenCVCamera

ROOT = Path(__file__).resolve().parent


def make_board(path: Path, inner_cols: int, inner_rows: int, square_px: int = 220) -> None:
    cols, rows = inner_cols + 1, inner_rows + 1
    board = np.full((rows * square_px, cols * square_px), 255, dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                cv2.rectangle(
                    board,
                    (col * square_px, row * square_px),
                    ((col + 1) * square_px - 1, (row + 1) * square_px - 1),
                    0,
                    -1,
                )
    encoded, buffer = cv2.imencode(".png", board)
    if not encoded:
        raise RuntimeError(f"Could not encode checkerboard: {path}")
    buffer.tofile(str(path))
    print(f"Checkerboard created: {path}")
    print("Print at 100% scale on a flat surface and measure the real square size.")


def calibrate(args: argparse.Namespace) -> int:
    camera = OpenCVCamera(
        index=args.camera_index,
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=30,
        source_name="APC850-CALIBRATION",
    )
    try:
        camera.open()
    except CameraOpenError as exc:
        print(exc)
        return 2

    pattern = (args.inner_cols, args.inner_rows)
    object_template = np.zeros((args.inner_cols * args.inner_rows, 3), dtype=np.float32)
    object_template[:, :2] = np.mgrid[0 : args.inner_cols, 0 : args.inner_rows].T.reshape(-1, 2)
    object_template *= args.square_mm
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    print("Show the flat checkerboard at varied positions/angles. SPACE=capture, q/ESC=finish.")
    try:
        cv2.namedWindow("PAC Camera Calibration", cv2.WINDOW_NORMAL)
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCornersSB(
                gray,
                pattern,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
            )
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, pattern, corners, found)
            colour = (60, 220, 80) if found else (40, 180, 255)
            cv2.putText(
                display,
                f"SAMPLES {len(image_points)}/{args.samples}  {'READY - SPACE' if found else 'FINDING BOARD'}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                colour,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("PAC Camera Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and found:
                object_points.append(object_template.copy())
                image_points.append(corners.astype(np.float32))
                print(f"Captured {len(image_points)}/{args.samples}")
            if key in (27, ord("q")) or len(image_points) >= args.samples:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if image_size is None or len(image_points) < max(8, args.samples // 2):
        print(f"Not enough valid views: {len(image_points)}. Capture at least {max(8, args.samples // 2)}.")
        return 1

    rms, matrix, coefficients, rotations, translations = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    errors: list[float] = []
    for object_point, image_point, rotation, translation in zip(object_points, image_points, rotations, translations):
        projected, _ = cv2.projectPoints(object_point, rotation, translation, matrix, coefficients)
        errors.append(float(cv2.norm(image_point, projected, cv2.NORM_L2) / len(projected)))

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    data = {
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": coefficients.reshape(-1).tolist(),
        "image_size": list(image_size),
        "rms": float(rms),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "pattern_inner_corners": [args.inner_cols, args.inner_rows],
        "square_size_mm": args.square_mm,
        "samples": len(image_points),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    print(f"RMS={rms:.4f}, mean reprojection error={np.mean(errors):.4f}px")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate APC850 lens distortion")
    parser.add_argument("--camera-index", default="0")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf", "any"), default="dshow")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--inner-cols", type=int, default=9)
    parser.add_argument("--inner-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--output", default="config/camera_calibration.json")
    parser.add_argument("--make-board", action="store_true")
    parser.add_argument("--board-output", default="calibration_checkerboard.png")
    args = parser.parse_args()
    if args.make_board:
        path = Path(args.board_output)
        if not path.is_absolute():
            path = ROOT / path
        make_board(path, args.inner_cols, args.inner_rows)
        return 0
    return calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
