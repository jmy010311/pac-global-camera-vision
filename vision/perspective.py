from __future__ import annotations

import cv2
import numpy as np


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return four points in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if ordered[1, 0] < ordered[-1, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def destination_corners(width: int, height: int, margin_ratio: float = 0.0) -> np.ndarray:
    margin_x = float(width) * margin_ratio
    margin_y = float(height) * margin_ratio
    return np.array(
        [
            [margin_x, margin_y],
            [width - 1 - margin_x, margin_y],
            [width - 1 - margin_x, height - 1 - margin_y],
            [margin_x, height - 1 - margin_y],
        ],
        dtype=np.float32,
    )


def rectify(
    frame: np.ndarray,
    corners: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = order_corners(corners)
    destination = destination_corners(width, height)
    homography = cv2.getPerspectiveTransform(source, destination)
    inverse = cv2.getPerspectiveTransform(destination, source)
    warped = cv2.warpPerspective(
        frame,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return warped, homography, inverse


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)

