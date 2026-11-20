"""Dense, real-image boundary points in the global camera image.

Boundaries are *not* seeded from a rows/cols topology.  They come from the
planar boundary graph that is searched directly from the refined outer
boundary (:mod:`boundary_graph`).  Each graph curve is projected into the
camera image, resampled every ~3 px and re-snapped to camera-image gradients
with a smooth dynamic-programming path.  Full-length (extended) curves are
kept only for intersection computation; published edge samples run strictly
from junction node to junction node.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .boundary_graph import (
    AnalysisGeometry,
    BoundaryConfig,
    BoundaryGraphDetector,
    PlanarBoundaryGraph,
)


@dataclass(frozen=True)
class DenseGridTrace:
    """Image-evidence paths for every physical boundary of an R x C partition."""

    rows: int
    cols: int
    nodes: np.ndarray
    edge_samples: dict[str, np.ndarray]
    row_lines: list[np.ndarray]  # extended curves, internal use only (not drawn)
    col_lines: list[np.ndarray]
    edge_confidence: dict[str, float] = field(default_factory=dict)
    edge_present: dict[str, bool] = field(default_factory=dict)
    node_confidence: np.ndarray | None = None
    node_types: list[list[str]] | None = None
    regular: bool = True
    state: str = "TENTATIVE"
    confidence: float = 0.0
    faces: list[tuple[int, list[tuple[int, int]], np.ndarray]] = field(default_factory=list)

    @property
    def point_count(self) -> int:
        return int(sum(len(path) for key, path in self.edge_samples.items() if self.edge_present.get(key, True)))

    def cell_polygons(self) -> list[np.ndarray]:
        """Curved camera polygons of every cell (regular grids only)."""
        polygons: list[np.ndarray] = []
        for row in range(self.rows):
            for col in range(self.cols):
                top = self.edge_samples[f"h_{row}_{col}"]
                right = self.edge_samples[f"v_{row}_{col + 1}"]
                bottom = self.edge_samples[f"h_{row + 1}_{col}"][::-1]
                left = self.edge_samples[f"v_{row}_{col}"][::-1]
                polygons.append(np.concatenate([top, right[1:], bottom[1:], left[1:-1]], axis=0).astype(np.float32))
        return polygons


def _sample_map(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    map_x = points[..., 0].astype(np.float32)
    map_y = points[..., 1].astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if len(points) < 2:
        return points
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total < 1e-6:
        return points[:1]
    count = max(2, int(np.ceil(total / max(spacing, 0.5))) + 1)
    targets = np.linspace(0.0, total, count)
    return np.column_stack((np.interp(targets, cumulative, points[:, 0]), np.interp(targets, cumulative, points[:, 1]))).astype(np.float32)


class GradientField:
    """Signed gradients of log-luminance and chroma; projections are polarity free."""

    def __init__(self, image: np.ndarray, sigma: float = 0.9) -> None:
        if image.ndim == 2:
            channels = [(40.0 * np.log1p(image.astype(np.float32)), 1.0)]
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            channels = [
                (40.0 * np.log1p(lab[:, :, 0]), 1.0),
                (lab[:, :, 1] - 128.0, 0.7),
                (lab[:, :, 2] - 128.0, 0.7),
            ]
        self.maps: list[tuple[np.ndarray, np.ndarray, float]] = []
        for channel, weight in channels:
            blurred = cv2.GaussianBlur(channel, (0, 0), sigma)
            self.maps.append(
                (cv2.Scharr(blurred, cv2.CV_32F, 1, 0) / 16.0, cv2.Scharr(blurred, cv2.CV_32F, 0, 1) / 16.0, weight)
            )

    def oriented(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """|grad . n| summed over channels at ``points`` (..., 2) with ``normals`` broadcastable."""
        total = np.zeros(points.shape[:-1], np.float32)
        for gx, gy, weight in self.maps:
            sx = _sample_map(gx, points)
            sy = _sample_map(gy, points)
            total += weight * np.abs(sx * normals[..., 0] + sy * normals[..., 1])
        return total


def trace_guided_polyline(
    field_or_gray: GradientField | np.ndarray,
    guide: np.ndarray,
    *,
    spacing_px: float = 3.0,
    search_radius_px: int = 8,
    smoothness: float = 0.055,
    max_step: float = 5.0,
) -> np.ndarray:
    """Snap a (curved) guide polyline to real image boundary evidence.

    The guide only defines a search corridor; every returned point is chosen
    from pixel evidence by a continuity/curvature penalised DP.
    """

    gradient = field_or_gray if isinstance(field_or_gray, GradientField) else GradientField(field_or_gray)
    base = resample_polyline(guide, spacing_px)
    count = len(base)
    if count < 3:
        return base
    tangent = np.gradient(base, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])).astype(np.float32)
    radius = max(1, int(search_radius_px))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    candidates = base[:, None, :] + offsets[None, :, None] * normal[:, None, :]
    evidence = gradient.oriented(candidates, normal[:, None, :])
    scale = float(np.percentile(evidence, 97.0))
    if scale > 1e-6:
        evidence = np.clip(evidence / scale, 0.0, 1.0)
    proximity = 0.10 * np.square(offsets / float(radius))[None, :]
    local_cost = -evidence + proximity
    states = len(offsets)
    backtrack = np.zeros((count, states), dtype=np.int16)
    previous = local_cost[0]
    delta = offsets[:, None] - offsets[None, :]
    transition = smoothness * np.square(delta)
    transition[np.abs(delta) > max_step] += 5.0
    for index in range(1, count):
        options = previous[:, None] + transition
        best = np.argmin(options, axis=0)
        previous = local_cost[index] + options[best, np.arange(states)]
        backtrack[index] = best.astype(np.int16)
    state = int(np.argmin(previous))
    selected = np.empty(count, dtype=np.float32)
    for index in range(count - 1, -1, -1):
        selected[index] = offsets[state]
        if index:
            state = int(backtrack[index, state])
    if count >= 5:
        padded = np.pad(selected, 2, mode="edge")
        selected = np.convolve(padded, np.ones(5, np.float32) / 5.0, mode="valid")
    return (base + selected[:, None] * normal).astype(np.float32)


def trace_divider_band_centerline(
    gradient: GradientField,
    guide: np.ndarray,
    *,
    spacing_px: float = 3.0,
    search_radius_px: int = 12,
) -> np.ndarray:
    """Trace the centre of a raised divider's two visible parallel rims.

    A polarity-free single-edge tracer can change from one side of a thick
    divider to the other while crossing a perpendicular divider.  Here every
    centre candidate is scored by *paired* edge evidence on opposite sides of
    the divider band.  The DP therefore follows one physical raised strip and
    cannot take a floor shadow or swap rims at a junction.
    """
    base = resample_polyline(guide, spacing_px)
    count = len(base)
    if count < 3:
        return base
    tangent = np.gradient(base, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])).astype(np.float32)
    radius = max(4, int(search_radius_px))
    centres = np.arange(-radius, radius + 1, dtype=np.float32)
    # Cardboard/plastic divider faces are narrow compared with a cell.  Wide
    # dark illumination bands are deliberately excluded from pair formation.
    half_widths = np.arange(1.5, max(3.0, 0.72 * radius) + 0.1, 1.0, dtype=np.float32)
    pair_score = np.zeros((count, len(centres)), np.float32)
    for half_width in half_widths:
        left_offsets = centres - half_width
        right_offsets = centres + half_width
        left_points = base[:, None, :] + left_offsets[None, :, None] * normal[:, None, :]
        right_points = base[:, None, :] + right_offsets[None, :, None] * normal[:, None, :]
        left = gradient.oriented(left_points, normal[:, None, :])
        right = gradient.oriented(right_points, normal[:, None, :])
        # Geometric mean requires both rims while tolerating unequal contrast.
        pair_score = np.maximum(pair_score, np.sqrt(np.maximum(left * right, 0.0)))
    scale = float(np.percentile(pair_score, 97.0))
    if scale <= 1e-6:
        return trace_guided_polyline(gradient, guide, spacing_px=spacing_px, search_radius_px=radius)
    pair_score = np.clip(pair_score / scale, 0.0, 1.0)
    # If almost no sample has a paired rim, this is a thin single boundary.
    if float(np.mean(np.max(pair_score, axis=1) >= 0.35)) < 0.28:
        return trace_guided_polyline(gradient, guide, spacing_px=spacing_px, search_radius_px=radius)

    local = -pair_score + 0.08 * np.square(centres / float(radius))[None, :]
    states = len(centres)
    backtrack = np.zeros((count, states), dtype=np.int16)
    previous = local[0]
    delta = centres[:, None] - centres[None, :]
    transition = 0.12 * np.square(delta)
    transition[np.abs(delta) > 2.5] += 8.0
    for index in range(1, count):
        options = previous[:, None] + transition
        best = np.argmin(options, axis=0)
        previous = local[index] + options[best, np.arange(states)]
        backtrack[index] = best.astype(np.int16)
    state = int(np.argmin(previous))
    selected = np.empty(count, np.float32)
    for index in range(count - 1, -1, -1):
        selected[index] = centres[state]
        if index:
            state = int(backtrack[index, state])
    if count >= 7:
        padded = np.pad(selected, 3, mode="edge")
        selected = np.convolve(padded, np.ones(7, np.float32) / 7.0, mode="valid")
    return (base + selected[:, None] * normal).astype(np.float32)


def trace_edge_polyline(
    gray: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    spacing_px: float = 3.0,
    search_radius_px: int = 34,
    extend_px: float = 42.0,
) -> np.ndarray:
    """Follow real image gradients between two points (straight corridor)."""

    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 2.0:
        return np.asarray([start, end], dtype=np.float32)
    tangent = direction / length
    guide = np.asarray([start - tangent * float(extend_px), end + tangent * float(extend_px)], np.float32)
    return trace_guided_polyline(gray, guide, spacing_px=spacing_px, search_radius_px=search_radius_px)


def _segment_intersections(horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    p = horizontal[:-1][:, None, :]
    r = (horizontal[1:] - horizontal[:-1])[:, None, :]
    q = vertical[:-1][None, :, :]
    s = (vertical[1:] - vertical[:-1])[None, :, :]
    denominator = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    safe = np.where(np.abs(denominator) < 1e-9, np.nan, denominator)
    qp = q - p
    t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / safe
    u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / safe
    mask = (t >= -1e-6) & (t <= 1 + 1e-6) & (u >= -1e-6) & (u <= 1 + 1e-6)
    i, j = np.nonzero(mask)
    return (p[i, 0] + t[i, j][:, None] * r[i, 0]).astype(np.float32)


def _closest_intersection(horizontal: np.ndarray, vertical: np.ndarray, seed: np.ndarray, search_radius: float) -> np.ndarray:
    """Intersection of two dense polylines near ``seed`` (segment level)."""
    h_near = np.flatnonzero(np.linalg.norm(horizontal - seed, axis=1) <= search_radius)
    v_near = np.flatnonzero(np.linalg.norm(vertical - seed, axis=1) <= search_radius)
    if len(h_near) and len(v_near):
        h_part = horizontal[max(0, h_near.min() - 1) : min(len(horizontal), h_near.max() + 2)]
        v_part = vertical[max(0, v_near.min() - 1) : min(len(vertical), v_near.max() + 2)]
        if len(h_part) >= 2 and len(v_part) >= 2:
            hits = _segment_intersections(h_part, v_part)
            if len(hits):
                return hits[int(np.argmin(np.linalg.norm(hits - seed, axis=1)))]
    distances = np.sum(np.square(horizontal[:, None, :] - vertical[None, :, :]), axis=2)
    h_index, v_index = np.unravel_index(int(np.argmin(distances)), distances.shape)
    if float(np.sqrt(distances[h_index, v_index])) > search_radius:
        return seed.astype(np.float32)
    return ((horizontal[h_index] + vertical[v_index]) * 0.5).astype(np.float32)


def _locate(line: np.ndarray, point: np.ndarray) -> float:
    a = line[:-1]
    segment = line[1:] - a
    denominator = np.maximum(np.sum(segment * segment, axis=1), 1e-9)
    t = np.clip(np.sum((point - a) * segment, axis=1) / denominator, 0.0, 1.0)
    distance = np.linalg.norm(a + t[:, None] * segment - point, axis=1)
    index = int(np.argmin(distance))
    return index + float(t[index])


def _cut(line: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Sub-polyline of ``line`` between the projections of two node points."""
    a, b = _locate(line, start), _locate(line, end)
    if a <= b:
        inner = line[int(np.floor(a)) + 1 : int(np.ceil(b))]
    else:
        inner = line[int(np.floor(b)) + 1 : int(np.ceil(a))][::-1]
    return np.vstack((start[None, :], inner.reshape(-1, 2), end[None, :])).astype(np.float32)


def graph_to_camera(
    frame: np.ndarray,
    graph: PlanarBoundaryGraph,
    geometry: AnalysisGeometry,
    analysis_homography: np.ndarray,
    *,
    state: str = "TENTATIVE",
    spacing_px: float | None = None,
    search_radius_px: int | None = None,
) -> DenseGridTrace:
    inverse = np.linalg.inv(analysis_homography)

    def to_camera(normalized: np.ndarray) -> np.ndarray:
        pixels = geometry.to_pixels(normalized)
        return cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), inverse).reshape(-1, 2).astype(np.float32)

    rows, cols = graph.rows, graph.cols
    corners = to_camera(np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
    box_scale = float(np.mean([np.linalg.norm(corners[k] - corners[(k + 1) % 4]) for k in range(4)]))
    spacing = float(np.clip(box_scale / 250.0, 2.0, 4.0)) if spacing_px is None else float(spacing_px)
    radius = int(round(np.clip(box_scale * 0.018, 5.0, 18.0))) if search_radius_px is None else int(search_radius_px)

    x0, y0 = np.floor(corners.min(axis=0) - 0.08 * box_scale).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0) + 0.08 * box_scale).astype(int)
    height, width = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    roi = frame[y0:y1, x0:x1]
    offset = np.array([x0, y0], np.float32)
    field_roi = GradientField(roi)

    def refine(normalized: np.ndarray, divider_band: bool = False) -> np.ndarray:
        guide = to_camera(normalized) - offset
        if divider_band:
            return trace_divider_band_centerline(field_roi, guide, spacing_px=spacing, search_radius_px=radius) + offset
        return trace_guided_polyline(field_roi, guide, spacing_px=spacing, search_radius_px=radius) + offset

    row_lines = [refine(graph.full_line_points("h", i), 0 < i < rows) for i in range(rows + 1)]
    col_lines = [refine(graph.full_line_points("v", j), 0 < j < cols) for j in range(cols + 1)]
    seeds = to_camera(graph.nodes.reshape(-1, 2)).reshape(rows + 1, cols + 1, 2)
    nodes = np.zeros_like(seeds)
    for i in range(rows + 1):
        for j in range(cols + 1):
            nodes[i, j] = _closest_intersection(row_lines[i], col_lines[j], seeds[i, j], max(3.0 * radius, 10.0))

    edge_samples: dict[str, np.ndarray] = {}
    edge_conf: dict[str, float] = {}
    edge_present: dict[str, bool] = {}
    for key, edge in graph.edges.items():
        axis, i_text, j_text = key.split("_")
        i, j = int(i_text), int(j_text)
        if axis == "h":
            path = _cut(row_lines[i], nodes[i, j], nodes[i, j + 1])
        else:
            path = _cut(col_lines[j], nodes[i, j], nodes[i + 1, j])
        edge_samples[key] = resample_polyline(path, spacing)
        edge_samples[key][0], edge_samples[key][-1] = path[0], path[-1]
        edge_conf[key] = float(edge.confidence)
        edge_present[key] = bool(edge.present)
    faces = [(face.id, face.cells, to_camera(face.polygon) if len(face.polygon) else face.polygon) for face in graph.faces]
    return DenseGridTrace(
        rows,
        cols,
        nodes.astype(np.float32),
        edge_samples,
        row_lines,
        col_lines,
        edge_conf,
        edge_present,
        graph.node_confidence.copy(),
        graph.node_types,
        graph.regular,
        state,
        float(graph.confidence),
        faces,
    )


def trace_partition(
    frame: np.ndarray,
    outer_corners: np.ndarray,
    *,
    detector: BoundaryGraphDetector | None = None,
    config: BoundaryConfig | dict | None = None,
    spacing_px: float | None = None,
    search_radius_px: int | None = None,
) -> DenseGridTrace:
    """Search interior boundaries directly from the outer boundary."""
    active = detector if detector is not None else BoundaryGraphDetector(config)
    detection, homography = active.detect(frame, np.asarray(outer_corners, np.float32).reshape(4, 2))
    if detection.graph is None:
        raise ValueError("No boundary graph could be built")
    return graph_to_camera(
        frame,
        detection.graph,
        active.geometry,
        homography,
        state=detection.state,
        spacing_px=spacing_px,
        search_radius_px=search_radius_px,
    )


def trace_dense_grid(
    frame: np.ndarray,
    result,
    *,
    spacing_px: float | None = None,
    search_radius_px: int | None = None,
    detector: BoundaryGraphDetector | None = None,
) -> DenseGridTrace:
    """Dense trace from a processor result's *outer boundary only*.

    ``result.topology`` (rows/cols) is intentionally ignored; interior
    boundaries are searched again from image evidence.
    """
    corners = getattr(getattr(result, "detection", None), "corners", None)
    if corners is None:
        corners = getattr(result, "outer_corners", None)
    if corners is None:
        raise ValueError("Dense boundary tracing requires a detected outer boundary")
    return trace_partition(frame, corners, detector=detector, spacing_px=spacing_px, search_radius_px=search_radius_px)
