"""Refine the outer partition boundary before any interior analysis.

The contour-based quad from :mod:`partition_detector` can latch onto an inner
wall face, a cast shadow or a single cell corner.  Each side is re-estimated in
a generously padded rectification of the coarse quad:

* candidate straight lines near the side are scored by oriented-evidence
  coverage (gap tolerant, polarity free), and
* by a *termination* score: perpendicular dividers exist just inside the true
  outer wall and stop at it, whereas an inner wall face is crossed by the
  dividers on both sides and a cast shadow has dividers on neither side.

The refined sides are mapped back to the camera and intersected.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .boundary_evidence import compute_evidence
from .perspective import order_corners


@dataclass
class OuterRefinement:
    corners: np.ndarray
    applied: bool
    side_scores: list[float]
    movement_px: float


def _line_candidates(
    evidence: np.ndarray,
    perpendicular: np.ndarray,
    threshold: float,
    margin: int,
    size: int,
    near: float,
    search: tuple[float, float],
    tilt: float,
    inward: int,
) -> list[tuple[float, float, float, float]]:
    """Vertical straight-line candidates near normalized position ``near``.

    Returns (u0, tilt, coverage, termination) with u(s) = u0 + tilt*(s-0.5).
    ``inward`` is +1 when the box interior is at larger u.
    """

    band = max(1, int(round(0.006 * size)))
    dilated = cv2.dilate(evidence, np.ones((1, 2 * band + 1), np.uint8))
    s = np.linspace(0.08, 0.92, 90, dtype=np.float32)
    ys = np.rint(margin + s * (size - 1)).astype(np.int64)
    outward, inward_limit = search
    low, high = (near - outward, near + inward_limit) if inward > 0 else (near - inward_limit, near + outward)
    u0s = np.linspace(low, high, max(8, int((high - low) * size)), dtype=np.float32)
    tilts = np.linspace(-tilt, tilt, 13, dtype=np.float32)
    height, width = evidence.shape
    results: list[tuple[float, float, float]] = []
    for t in tilts:
        us = u0s[:, None] + t * (s[None, :] - 0.5)
        xs = np.rint(margin + us * (size - 1)).astype(np.int64)
        valid = (xs >= 0) & (xs < width)
        values = np.zeros(xs.shape, np.float32)
        rows = np.broadcast_to(ys[None, :], xs.shape)
        values[valid] = dilated[rows[valid], xs[valid]]
        coverage = (values >= threshold).mean(axis=1)
        for u0, value in zip(u0s, coverage):
            results.append((float(u0), float(t), float(value)))
    if not results:
        return []
    results.sort(key=lambda item: -item[2])
    best = results[0][2]
    picked: list[tuple[float, float, float]] = []
    for u0, t, value in results:
        if value < max(0.25, 0.45 * best):
            break
        if any(abs(u0 - other[0]) < 0.012 for other in picked):
            continue
        picked.append((u0, t, value))
        if len(picked) >= 8:
            break

    per_dilated = perpendicular >= threshold
    gap, strip = 0.012, 0.07
    output: list[tuple[float, float, float, float]] = []
    for u0, t, value in picked:
        densities = []
        for side in (inward, -inward):
            lo, hi = (gap, gap + strip) if side > 0 else (-gap - strip, -gap)
            count = 0
            total = 0
            for offset in np.linspace(lo, hi, 6):
                us = u0 + t * (s - 0.5) + offset
                xs = np.rint(margin + us * (size - 1)).astype(np.int64)
                valid = (xs >= 0) & (xs < width)
                total += int(valid.sum())
                count += int(per_dilated[ys[valid], xs[valid]].sum())
            densities.append(count / max(total, 1))
        inside, outside = densities
        termination = float(np.clip((inside - outside) / (inside + outside + 0.03), 0.0, 1.0))
        output.append((u0, t, value, termination))
    return output


def _choose(
    candidates: list[tuple[float, float, float, float]],
    near: float,
    min_termination: float,
) -> tuple[float, float, float] | None:
    """Pick the side line; keep the current side unless a structural wall is found."""
    if not candidates:
        return None
    scored = [
        (np.sqrt(coverage) * (0.25 + 0.75 * termination), u0, t)
        for u0, t, coverage, termination in candidates
        if termination >= min_termination
    ]
    if not scored:
        return None
    score, u0, t = max(scored)
    return float(u0), float(t), float(score)


def _intersection(line_a: tuple[np.ndarray, np.ndarray], line_b: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
    (p, d), (q, e) = line_a, line_b
    matrix = np.array([[d[0], -e[0]], [d[1], -e[1]]], dtype=np.float64)
    if abs(np.linalg.det(matrix)) < 1e-9:
        return None
    t, _ = np.linalg.solve(matrix, (q - p).astype(np.float64))
    return (p + t * d).astype(np.float32)


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    return np.array([x0, y0], np.float32), np.array([vx, vy], np.float32)


class OuterBoundaryRefiner:
    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.size = int(config.get("analysis_size", 384))
        self.margin_ratio = float(config.get("margin_ratio", 0.18))
        self.search_ratio = float(config.get("search_ratio", 0.15))
        self.tilt_ratio = float(config.get("tilt_ratio", 0.15))
        self.iterations = max(1, int(config.get("iterations", 2)))
        self.max_move_ratio = float(config.get("max_move_ratio", 0.22))
        self.max_area_shrink_ratio = float(config.get("max_area_shrink_ratio", 0.04))
        self.min_termination = float(config.get("min_termination", 0.35))
        self.inward_ratio = float(config.get("inward_ratio", 0.09))

    def refine(self, frame: np.ndarray, corners: np.ndarray) -> OuterRefinement:
        original = order_corners(np.asarray(corners, np.float32))
        if not self.enabled:
            return OuterRefinement(original, False, [], 0.0)
        current = original.copy()
        scores: list[float] = []
        for iteration in range(self.iterations):
            search = (
                (self.search_ratio, self.inward_ratio)
                if iteration == 0
                else (0.3 * self.search_ratio, 0.3 * self.inward_ratio)
            )
            refined, scores = self._refine_once(frame, current, search)
            if refined is None:
                break
            current = refined
        box_scale = float(max(np.ptp(original[:, 0]), np.ptp(original[:, 1]), 1.0))
        movement = float(np.max(np.linalg.norm(current - original, axis=1)))
        area_old = abs(float(cv2.contourArea(original)))
        area_new = abs(float(cv2.contourArea(current)))
        valid = (
            cv2.isContourConvex(current.reshape(-1, 1, 2).astype(np.float32))
            and movement <= self.max_move_ratio * box_scale
            and (area_old - area_new) <= self.max_area_shrink_ratio * max(area_old, 1.0)
            and area_new <= 1.8 * area_old
        )
        if not valid:
            return OuterRefinement(original, False, scores, 0.0)
        return OuterRefinement(current, bool(movement > 0.5), scores, movement)

    def _refine_once(self, frame: np.ndarray, corners: np.ndarray, search: tuple[float, float]) -> tuple[np.ndarray | None, list[float]]:
        size = self.size
        margin = int(round(size * self.margin_ratio))
        destination = np.array(
            [[margin, margin], [margin + size - 1, margin], [margin + size - 1, margin + size - 1], [margin, margin + size - 1]],
            np.float32,
        )
        homography = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
        inverse = np.linalg.inv(homography)
        full = size + 2 * margin
        image = cv2.warpPerspective(frame, homography, (full, full), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        evidence = compute_evidence(image, roi=(margin, margin, margin + size, margin + size), use_lsd=False)
        vertical, horizontal = evidence.vertical, evidence.horizontal
        threshold = evidence.threshold
        lines_cam: list[tuple[np.ndarray, np.ndarray]] = []
        scores: list[float] = []
        # order: top, right, bottom, left
        specs = [("h", 0.0, +1), ("v", 1.0, -1), ("h", 1.0, -1), ("v", 0.0, +1)]
        s = np.linspace(0.0, 1.0, 40, dtype=np.float32)
        for axis, near, inward in specs:
            if axis == "v":
                candidates = _line_candidates(vertical, horizontal, threshold, margin, size, near, search, self.tilt_ratio, inward)
            else:
                candidates = _line_candidates(
                    np.ascontiguousarray(horizontal.T), np.ascontiguousarray(vertical.T), threshold, margin, size, near, search, self.tilt_ratio, inward
                )
            chosen = _choose(candidates, near, self.min_termination)
            if chosen is None:
                u0, tilt, score = near, 0.0, 0.0  # no better structural wall: keep side
            else:
                u0, tilt, score = chosen
            scores.append(score)
            us = u0 + tilt * (s - 0.5)
            if axis == "v":
                points = np.column_stack((margin + us * (size - 1), margin + s * (size - 1)))
            else:
                points = np.column_stack((margin + s * (size - 1), margin + us * (size - 1)))
            camera = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), inverse).reshape(-1, 2)
            lines_cam.append(_fit_line(camera))
        refined = []
        for index in range(4):
            point = _intersection(lines_cam[index - 1], lines_cam[index])  # (left,top)->TL, (top,right)->TR ...
            if point is None:
                return None, scores
            refined.append(point)
        # intersections are ordered TL, TR, BR, BL because specs start at top
        return order_corners(np.asarray(refined, np.float32)), scores
