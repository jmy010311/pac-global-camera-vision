from __future__ import annotations

import argparse
import logging

from camera.opencv_camera import discover_cameras


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Windows camera indices that return real frames")
    parser.add_argument("--max-index", type=int, default=8)
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf", "any"), default="auto")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print(f"Scanning camera indices 0..{args.max_index - 1} ({args.backend})...")
    cameras = discover_cameras(args.max_index, args.backend)
    if not cameras:
        print("No camera returned a valid frame.")
        print("Check the USB cable, close Camera/Teams/Zoom, and allow desktop camera access in Windows Settings.")
        return 1
    for camera in cameras:
        print(camera.summary())
    print("Use one with: python main.py --camera-index <index> --backend <backend>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

