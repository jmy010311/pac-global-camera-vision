from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .perspective import order_corners


@dataclass
class PartitionDetection:
    detected: bool
    corners: np.ndarray | None
    confidence: float
    stale: bool = False
    candidate_count: int = 0
    debug_edges: np.ndarray | None = None


def _corner_quality_score(quad: np.ndarray) -> float:
    """Reject near-collinear corners without requiring fronto-parallel right angles."""
    points = order_corners(quad)
    scores: list[float] = []
    for index in range(4):
        previous = points[(index - 1) % 4] - points[index]
        following = points[(index + 1) % 4] - points[index]
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denominator < 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(previous, following) / denominator, -1.0, 1.0))
        sine = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
        scores.append(float(np.clip((sine - 0.28) / 0.72, 0.0, 1.0)))
    return float(np.mean(scores))


def _quad_aspect_ratio(quad: np.ndarray) -> float:
    tl, tr, br, bl = order_corners(quad)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(width / max(height, 1e-6))


def _grid_evidence(gray: np.ndarray, quad: np.ndarray) -> float:
    """Evidence of long interior dividers, independent of their count and spacing.

    Earlier versions scored equally spaced 2..6 divisions, which biased the
    outer-quad choice towards regular grids.  Here each interior column/row of
    the rectified candidate is scored by the fraction of its length carrying an
    oriented gradient (gap tolerant); the best two peaks per axis are averaged.
    """
    size = 200
    source = order_corners(quad).astype(np.float32)
    destination = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(gray, matrix, (size, size))
    warped = cv2.GaussianBlur(np.log1p(warped.astype(np.float32)) * 40.0, (3, 3), 0)
    grad_x = np.abs(cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3))
    oriented_x = np.clip(grad_x - grad_y, 0, None)
    oriented_y = np.clip(grad_y - grad_x, 0, None)
    reference = float(np.percentile(np.maximum(oriented_x, oriented_y), 97)) + 1e-6
    lo, hi = int(size * 0.08), int(size * 0.92)

    def axis_score(oriented: np.ndarray) -> float:
        band = cv2.dilate(oriented, np.ones((1, 5), np.uint8))
        coverage = (band[lo:hi] >= 0.35 * reference).mean(axis=0)[lo:hi]
        if coverage.size == 0:
            return 0.0
        top = np.sort(coverage)[-2:]
        return float(np.clip((np.mean(top) - 0.2) / 0.6, 0.0, 1.0))

    return 0.5 * axis_score(oriented_x) + 0.5 * axis_score(np.ascontiguousarray(oriented_y.T))


def _dominant_side_quad(contour: np.ndarray) -> np.ndarray | None:
    """Quad from the four longest nearly-straight runs of the convex hull.

    ``approxPolyDP`` picks four *vertices*; a clipped or rounded corner (marker,
    shadow, damaged cardboard) then moves a vertex far along one side.  Fitting
    four *sides* and intersecting them is insensitive to such local corner
    damage and does not assume a right angle or a square.
    """
    hull = cv2.convexHull(contour.reshape(-1, 1, 2).astype(np.float32))
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        return None
    polygon = cv2.approxPolyDP(hull, 0.004 * perimeter, True).reshape(-1, 2).astype(np.float32)
    count = len(polygon)
    if count < 4:
        return None
    vectors = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    # start the chaining at the sharpest turn so no chain wraps around index 0
    turns = np.abs(np.angle(np.exp(1j * (angles - np.roll(angles, 1)))))
    start = int(np.argmax(turns))
    order = [(start + k) % count for k in range(count)]
    chains: list[list[int]] = [[order[0]]]
    for index in order[1:]:
        previous = chains[-1][-1]
        difference = abs(float(np.angle(np.exp(1j * (angles[index] - angles[previous])))))
        chain_angle = float(np.angle(np.sum(lengths[chains[-1]] * np.exp(1j * angles[chains[-1]]))))
        drift = abs(float(np.angle(np.exp(1j * (angles[index] - chain_angle)))))
        if difference < np.deg2rad(14.0) and drift < np.deg2rad(18.0):
            chains[-1].append(index)
        else:
            chains.append([index])
    if len(chains) < 4:
        return None
    chain_lengths = [float(lengths[chain].sum()) for chain in chains]
    selected = sorted(np.argsort(chain_lengths)[-4:])
    if min(chain_lengths[k] for k in selected) < 0.06 * perimeter:
        return None
    lines = []
    for k in selected:
        chain = chains[k]
        points = np.vstack([polygon[chain], polygon[(chain[-1] + 1) % count][None, :]])
        dense = np.vstack([np.linspace(points[i], points[i + 1], 8) for i in range(len(points) - 1)])
        vx, vy, x0, y0 = cv2.fitLine(dense.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    corners = []
    for index in range(4):
        (p, d), (q, e) = lines[index - 1], lines[index]
        matrix = np.array([[d[0], -e[0]], [d[1], -e[1]]])
        if abs(np.linalg.det(matrix)) < 1e-6:
            return None
        t, _ = np.linalg.solve(matrix, q - p)
        corners.append(p + t * d)
    quad = np.asarray(corners, np.float32)
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None
    return order_corners(quad)


def _region_iou(quad: np.ndarray, contour: np.ndarray) -> float:
    points = np.vstack([quad.reshape(-1, 2), contour.reshape(-1, 2)]).astype(np.float32)
    x0, y0 = np.floor(points.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(points.max(axis=0)).astype(int) + 2
    scale = min(1.0, 300.0 / max(x1 - x0, y1 - y0, 1))
    size = (int((x1 - x0) * scale) + 1, int((y1 - y0) * scale) + 1)
    a = np.zeros((size[1], size[0]), np.uint8)
    b = np.zeros_like(a)
    offset = np.array([x0, y0], np.float32)
    cv2.fillPoly(a, [np.round((quad.reshape(-1, 2) - offset) * scale).astype(np.int32)], 1)
    hull = cv2.convexHull(contour.reshape(-1, 1, 2).astype(np.float32)).reshape(-1, 2)
    cv2.fillPoly(b, [np.round((hull - offset) * scale).astype(np.int32)], 1)
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b)) / max(union, 1)


class TemporalCornerFilter:
    def __init__(
        self,
        alpha: float = 0.15,
        hold_frames: int = 6,
        median_window: int = 5,
        deadband_ratio: float = 0.005,
        lock_after_frames: int = 10,
        unlock_ratio: float = 0.045,
        unlock_frames: int = 5,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.hold_frames = max(0, int(hold_frames))
        self.deadband_ratio = max(0.0, float(deadband_ratio))
        self.history: deque[np.ndarray] = deque(maxlen=max(1, int(median_window)))
        self.lock_after_frames = max(2, int(lock_after_frames))
        self.unlock_ratio = max(0.01, float(unlock_ratio))
        self.unlock_frames = max(2, int(unlock_frames))
        self.locked = False
        self.tracked_frames = 0
        self.motion_frames = 0
        self.corners: np.ndarray | None = None
        self.confidence = 0.0
        self.misses = 0

    def update(self, detection: PartitionDetection) -> PartitionDetection:
        if detection.detected and detection.corners is not None:
            new_corners = order_corners(detection.corners)
            self.history.append(new_corners)
            stable_corners = np.median(np.stack(tuple(self.history)), axis=0).astype(np.float32)
            if self.locked and self.corners is not None:
                box_scale = max(np.ptp(self.corners[:, 0]), np.ptp(self.corners[:, 1]), 1.0)
                jump = float(np.mean(np.linalg.norm(stable_corners - self.corners, axis=1)))
                if jump < box_scale * self.unlock_ratio:
                    self.motion_frames = 0
                    detection.corners = self.corners.copy()
                    self.confidence = detection.confidence
                    self.misses = 0
                    return detection
                self.motion_frames += 1
                if self.motion_frames < self.unlock_frames:
                    detection.corners = self.corners.copy()
                    self.confidence = detection.confidence
                    self.misses = 0
                    return detection
                self.locked = False
                self.tracked_frames = 0
                self.motion_frames = 0
                self.history.clear()
                self.history.append(new_corners)
                stable_corners = new_corners
            if self.corners is None:
                self.corners = stable_corners
            else:
                box_scale = max(np.ptp(stable_corners[:, 0]), np.ptp(stable_corners[:, 1]), 1.0)
                jump = float(np.mean(np.linalg.norm(stable_corners - self.corners, axis=1)))
                if jump <= box_scale * self.deadband_ratio:
                    stable_corners = self.corners
                if jump > box_scale * 0.22:
                    self.history.clear()
                    self.history.append(new_corners)
                    stable_corners = new_corners
                    alpha = 0.5
                elif jump > box_scale * 0.05:
                    alpha = min(0.3, self.alpha * 1.8)
                else:
                    alpha = self.alpha
                self.corners = (1.0 - alpha) * self.corners + alpha * stable_corners
            self.tracked_frames += 1
            if self.tracked_frames >= self.lock_after_frames:
                self.corners = np.median(np.stack(tuple(self.history)), axis=0).astype(np.float32)
                self.locked = True
                self.motion_frames = 0
            self.confidence = detection.confidence
            self.misses = 0
            detection.corners = self.corners.copy()
            return detection

        self.misses += 1
        allowed_misses = max(self.hold_frames, 30) if self.locked else self.hold_frames
        if self.corners is not None and self.misses <= allowed_misses:
            decay = 1.0 - self.misses / float(self.hold_frames + 1)
            return PartitionDetection(
                detected=True,
                corners=self.corners.copy(),
                confidence=self.confidence * decay,
                stale=True,
                candidate_count=detection.candidate_count,
                debug_edges=detection.debug_edges,
            )
        self.corners = None
        self.history.clear()
        self.locked = False
        self.tracked_frames = 0
        self.motion_frames = 0
        self.confidence = 0.0
        return detection


class PartitionDetector:
    """Scores all plausible quadrilaterals; it does not assume largest == box."""

    def __init__(self, config: dict) -> None:
        self.min_area_ratio = float(config.get("min_area_ratio", 0.08))
        self.max_area_ratio = float(config.get("max_area_ratio", 0.92))
        self.min_confidence = float(config.get("min_confidence", 0.42))
        self.canny_sigma = float(config.get("canny_sigma", 0.33))
        self.epsilon_ratio = float(config.get("approx_epsilon_ratio", 0.025))
        self.max_aspect_ratio = max(1.5, float(config.get("max_aspect_ratio", 4.0)))
        self.grid_weight = float(np.clip(config.get("grid_evidence_weight", 0.25), 0.0, 0.5))
        self.filter = TemporalCornerFilter(
            alpha=float(config.get("temporal_alpha", 0.15)),
            hold_frames=int(config.get("hold_frames", 6)),
            median_window=int(config.get("median_window", 5)),
            deadband_ratio=float(config.get("corner_deadband_ratio", 0.005)),
            lock_after_frames=int(config.get("lock_after_frames", 10)),
            unlock_ratio=float(config.get("unlock_ratio", 0.045)),
            unlock_frames=int(config.get("unlock_frames", 5)),
        )

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blurred = cv2.GaussianBlur(clahe, (5, 5), 0)
        median = float(np.median(blurred))
        lower = int(max(0, (1.0 - self.canny_sigma) * median))
        upper = int(min(255, (1.0 + self.canny_sigma) * median))
        if upper - lower < 30:
            lower, upper = max(0, int(median - 30)), min(255, int(median + 30))
        edges = cv2.Canny(blurred, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            7,
        )
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
        return gray, edges, adaptive

    def _candidates(self, masks: tuple[np.ndarray, ...], frame_area: float) -> list[tuple[np.ndarray, float]]:
        candidates: list[tuple[np.ndarray, float]] = []
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = abs(float(cv2.contourArea(contour)))
                area_ratio = area / frame_area
                if not self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue
                polygon = None
                epsilon_candidates = sorted(
                    {0.012, 0.018, self.epsilon_ratio, 0.035, 0.05}
                )
                for epsilon in epsilon_candidates:
                    approximated = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                    if len(approximated) == 4 and cv2.isContourConvex(approximated):
                        polygon = approximated
                        break
                sided = _dominant_side_quad(contour)
                if polygon is None and sided is None:
                    continue
                quad = order_corners(polygon.reshape(4, 2)) if polygon is not None else sided
                if polygon is not None and sided is not None and _region_iou(sided, contour) > _region_iou(quad, contour) + 0.005:
                    quad = sided
                if any(float(np.mean(np.linalg.norm(quad - old, axis=1))) < 8.0 for old, _ in candidates):
                    continue
                candidates.append((quad, area))
        for mask in masks:
            for quad, area in self._cell_corner_candidates(mask, frame_area):
                if any(float(np.mean(np.linalg.norm(quad - old, axis=1))) < 8.0 for old, _ in candidates):
                    continue
                candidates.append((quad, area))
        return candidates

    def _cell_corner_candidates(
        self,
        mask: np.ndarray,
        frame_area: float,
    ) -> list[tuple[np.ndarray, float]]:
        """Recover an outer quad by connecting corners of multiple visible cells.

        This is the fallback for cardboard edges that are curved, broken or not a
        single closed contour. It requires a spatial cluster of at least four
        similarly sized quadrilateral cells, then validates the resulting hull
        with the same geometry/grid score as normal candidates.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cell_quads: list[tuple[np.ndarray, float, np.ndarray]] = []
        lower_area = max(450.0, frame_area * 0.0015)
        upper_area = frame_area * min(0.08, self.min_area_ratio * 1.8)
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            if not lower_area <= area <= upper_area:
                continue
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            if perimeter <= 0:
                continue
            polygon = None
            for epsilon in (0.018, 0.025, 0.035, 0.05, 0.07):
                approximated = cv2.approxPolyDP(hull, epsilon * perimeter, True)
                if len(approximated) == 4 and cv2.isContourConvex(approximated):
                    polygon = approximated
                    break
            if polygon is None:
                continue
            quad = order_corners(polygon.reshape(4, 2))
            if _corner_quality_score(quad) < 0.28:
                continue
            center = quad.mean(axis=0)
            cell_quads.append((quad, area, center))

        if len(cell_quads) < 4:
            return []

        parents = list(range(len(cell_quads)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parents[root_right] = root_left

        for left in range(len(cell_quads)):
            _, area_left, center_left = cell_quads[left]
            for right in range(left + 1, len(cell_quads)):
                _, area_right, center_right = cell_quads[right]
                size_ratio = max(area_left, area_right) / max(min(area_left, area_right), 1.0)
                typical_size = max(np.sqrt(area_left), np.sqrt(area_right))
                distance = float(np.linalg.norm(center_left - center_right))
                if size_ratio <= 2.8 and 0.28 * typical_size <= distance <= 2.25 * typical_size:
                    union(left, right)

        groups: dict[int, list[int]] = {}
        for index in range(len(cell_quads)):
            groups.setdefault(find(index), []).append(index)

        recovered: list[tuple[np.ndarray, float]] = []
        for indices in groups.values():
            if len(indices) < 4:
                continue
            points = np.vstack([cell_quads[index][0] for index in indices]).astype(np.float32)
            hull = cv2.convexHull(points.reshape(-1, 1, 2))
            hull_perimeter = cv2.arcLength(hull, True)
            outer = None
            for epsilon in (0.015, 0.025, 0.04, 0.06, 0.09):
                approximated = cv2.approxPolyDP(hull, epsilon * hull_perimeter, True)
                if len(approximated) == 4 and cv2.isContourConvex(approximated):
                    outer = approximated.reshape(4, 2)
                    break
            if outer is None:
                outer = cv2.boxPoints(cv2.minAreaRect(points))
            outer = order_corners(outer)
            # A missing corner cell clips the hull; four fitted sides restore it.
            sided = _dominant_side_quad(hull)
            if sided is not None and _region_iou(sided, hull) >= 0.88:
                if abs(float(cv2.contourArea(sided))) >= abs(float(cv2.contourArea(outer))):
                    outer = sided
            outer_area = abs(float(cv2.contourArea(outer)))
            area_ratio = outer_area / frame_area
            if self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                recovered.append((outer, outer_area))
        return recovered

    def _score(self, gray: np.ndarray, quad: np.ndarray, contour_area: float) -> float:
        height, width = gray.shape[:2]
        frame_area = float(height * width)
        area_ratio = contour_area / frame_area
        area_score = float(np.clip((area_ratio - self.min_area_ratio) / 0.18, 0.0, 1.0))

        rectangle = cv2.minAreaRect(quad.astype(np.float32))
        rect_area = max(float(rectangle[1][0] * rectangle[1][1]), 1.0)
        rectangularity = float(np.clip(contour_area / rect_area, 0.0, 1.0))
        corner_score = _corner_quality_score(quad)

        aspect = _quad_aspect_ratio(quad)
        symmetric_aspect = max(aspect, 1.0 / max(aspect, 1e-6))
        aspect_score = float(
            np.clip((self.max_aspect_ratio - symmetric_aspect) / (self.max_aspect_ratio - 1.0), 0.0, 1.0)
        )

        ordered = order_corners(quad)
        edge_lengths = np.array(
            [np.linalg.norm(ordered[(index + 1) % 4] - ordered[index]) for index in range(4)]
        )
        edge_balance = float(np.clip(edge_lengths.min() / max(edge_lengths.max(), 1e-6) / 0.22, 0.0, 1.0))

        distances = np.concatenate(
            [quad[:, 0], width - 1 - quad[:, 0], quad[:, 1], height - 1 - quad[:, 1]]
        )
        clearance_score = float(np.clip(np.min(distances) / (0.035 * min(width, height)), 0.0, 1.0))
        grid_score = _grid_evidence(gray, quad)

        base_weight = 1.0 - self.grid_weight
        shape_score = (
            0.22 * area_score
            + 0.14 * rectangularity
            + 0.25 * corner_score
            + 0.10 * aspect_score
            + 0.20 * edge_balance
            + 0.09 * clearance_score
        )
        return float(np.clip(base_weight * shape_score + self.grid_weight * grid_score, 0.0, 1.0))

    def detect(self, frame: np.ndarray) -> PartitionDetection:
        if frame is None or frame.size == 0:
            return self.filter.update(PartitionDetection(False, None, 0.0))
        gray, edges, adaptive = self._preprocess(frame)
        candidates = self._candidates((edges, adaptive), float(frame.shape[0] * frame.shape[1]))
        if not candidates:
            return self.filter.update(
                PartitionDetection(False, None, 0.0, candidate_count=0, debug_edges=edges)
            )

        scored = [(self._score(gray, quad, area), quad) for quad, area in candidates]
        confidence, best_quad = max(scored, key=lambda item: item[0])
        if confidence >= self.min_confidence:
            window = max(3, int(round(min(frame.shape[:2]) * 0.008)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.02)
            try:
                refined = cv2.cornerSubPix(
                    gray,
                    best_quad.reshape(-1, 1, 2).astype(np.float32),
                    (window, window),
                    (-1, -1),
                    criteria,
                )
                best_quad = order_corners(refined.reshape(4, 2))
            except cv2.error:
                pass
        raw = PartitionDetection(
            detected=confidence >= self.min_confidence,
            corners=best_quad if confidence >= self.min_confidence else None,
            confidence=confidence,
            candidate_count=len(candidates),
            debug_edges=edges,
        )
        return self.filter.update(raw)
