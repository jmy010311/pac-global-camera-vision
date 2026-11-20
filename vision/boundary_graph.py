"""Evidence-driven partition structure as a planar boundary graph.

Pipeline (all in the *rectified analysis image* of the refined outer boundary):

1. ``compute_evidence``: polarity-free oriented boundary evidence.
2. Line hypotheses: every tilted / bowed curve crossing the box is scored by
   the *fraction of its length* supported by oriented evidence (gap tolerant).
   Nothing is sampled at predicted equal-spacing positions.
3. Dynamic-programming refinement inside a narrow corridor gives the real
   curved path; continuity and curvature are penalised so the path cannot jump
   into cell-interior shadows or object edges.
4. Near-parallel duplicates (divider top edge + visible wall face) are merged by
   a scale-free gap-ratio rule; the outer walls absorb their inner faces.
5. Junctions (X/T/L) between vertical and horizontal paths are measured from
   the arms of both curves; each line gets a junction-support score.
6. Lines are accumulated over frames (existence probability per line track).
7. ``PlanarBoundaryGraph``: junction nodes + curved polyline edges + enclosed
   faces.  Rows/cols are derived only when every face is a simple cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .boundary_evidence import EvidenceMaps, compute_evidence


# --------------------------------------------------------------------------- config


@dataclass
class BoundaryConfig:
    analysis_size: int = 512
    margin_ratio: float = 0.06
    gradient_sigma: float = 1.0
    tensor_sigma: float = 1.5
    ridge_sigma: float = 1.6
    evidence_ratio: float = 0.30
    noise_factor: float = 4.0
    use_lsd: bool = True
    band_ratio: float = 0.008
    tilt_ratio: float = 0.05
    tilt_steps: int = 9
    bow_ratio: float = 0.035
    bow_steps: int = 5
    search_row_step_px: int = 5
    min_coverage: float = 0.45
    nms_ratio: float = 0.025
    corridor_ratio: float = 0.022
    border_corridor_ratio: float = 0.028
    path_step_px: float = 3.0
    extension_ratio: float = 0.03
    smoothness: float = 0.08
    min_cell_ratio: float = 0.30
    border_merge_ratio: float = 0.45
    min_cell_size_ratio: float = 0.035
    accept_score: float = 0.50
    min_line_coverage: float = 0.35
    min_junction_support: float = 0.82
    min_structural_coverage: float = 0.90
    shadow_excess_ratio: float = 1.70
    arm_ratio: float = 0.07
    junction_gap_ratio: float = 0.015
    segment_min_coverage: float = 0.22
    segment_hysteresis: float = 0.5  # weak threshold (x tau) along already verified lines
    max_lines_per_axis: int = 11
    temporal_alpha: float = 0.25
    association_ratio: float = 0.035
    path_alpha: float = 0.35
    confirm_frames: int = 3
    min_topology_confidence: float = 0.45

    @classmethod
    def from_dict(cls, data: dict | None) -> "BoundaryConfig":
        config = cls()
        for key, value in (data or {}).items():
            if hasattr(config, key):
                current = getattr(config, key)
                setattr(config, key, type(current)(value) if not isinstance(current, bool) else bool(value))
        return config


@dataclass(frozen=True)
class AnalysisGeometry:
    """Box occupies pixels [mx, mx+width-1] x [my, my+height-1] of the analysis image."""

    margin_x: int
    margin_y: int
    width: int
    height: int

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.width + 2 * self.margin_x, self.height + 2 * self.margin_y)

    def to_pixels(self, normalized: np.ndarray) -> np.ndarray:
        points = np.asarray(normalized, dtype=np.float32).reshape(-1, 2)
        return np.column_stack(
            (
                self.margin_x + points[:, 0] * (self.width - 1),
                self.margin_y + points[:, 1] * (self.height - 1),
            )
        ).astype(np.float32)

    def to_normalized(self, pixels: np.ndarray) -> np.ndarray:
        points = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
        return np.column_stack(
            (
                (points[:, 0] - self.margin_x) / max(self.width - 1, 1),
                (points[:, 1] - self.margin_y) / max(self.height - 1, 1),
            )
        ).astype(np.float32)

    def transposed(self) -> "AnalysisGeometry":
        return AnalysisGeometry(self.margin_y, self.margin_x, self.height, self.width)


def analysis_geometry(config: BoundaryConfig) -> AnalysisGeometry:
    size = int(config.analysis_size)
    margin = int(round(size * config.margin_ratio))
    return AnalysisGeometry(margin, margin, size, size)


def analysis_homography(corners: np.ndarray, geometry: AnalysisGeometry) -> np.ndarray:
    """Camera -> padded analysis image homography for TL, TR, BR, BL corners."""
    destination = geometry.to_pixels(np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
    return cv2.getPerspectiveTransform(np.asarray(corners, np.float32).reshape(4, 2), destination)


# --------------------------------------------------------------------------- lines


@dataclass
class LineObservation:
    """One boundary curve in normalized box coordinates.

    For a vertical line ``path[k]`` is u at s=v=s_grid[k]; for a horizontal
    line it is v at s=u=s_grid[k].
    """

    axis: str  # "v" or "h"
    position: float  # mean normalized position across the box
    path: np.ndarray
    hits: np.ndarray
    coverage: float
    strength: float
    border: bool = False
    junction_support: float = 0.0
    score: float = 0.0
    longest_gap: float = 0.0


def s_grid_for(config: BoundaryConfig, length_px: int) -> np.ndarray:
    extension = config.extension_ratio
    count = max(24, int(round((1.0 + 2.0 * extension) * length_px / max(config.path_step_px, 1.0))) + 1)
    return np.linspace(-extension, 1.0 + extension, count, dtype=np.float32)


def _gather(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup; samples outside the image read as zero evidence."""
    height, width = image.shape
    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)
    outside = (xi < 0) | (xi >= width) | (yi < 0) | (yi >= height)
    np.clip(xi, 0, width - 1, out=xi)
    np.clip(yi, 0, height - 1, out=yi)
    values = image.ravel()[yi * width + xi]
    if outside.any():
        values = np.where(outside, np.float32(0.0), values)
    return values


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _search_hypotheses(
    evidence: np.ndarray,
    threshold: float,
    geometry: AnalysisGeometry,
    config: BoundaryConfig,
) -> list[tuple[float, float, float, float]]:
    """Return (mean_u, x0, tilt, bow, coverage) peaks for curves crossing the box.

    Works on the "vertical" orientation; callers transpose for horizontal lines.
    """

    width, height = geometry.width, geometry.height
    band = max(1, int(round(config.band_ratio * width)))
    dilated = cv2.dilate(evidence, np.ones((1, 2 * band + 1), np.uint8))
    step = max(1, int(config.search_row_step_px))
    ys = np.arange(geometry.margin_y + int(0.02 * height), geometry.margin_y + int(0.98 * height), step)
    s = ((ys - geometry.margin_y) / max(height - 1, 1)).astype(np.float32)
    x0_px = np.arange(int(0.015 * width), int(0.985 * width), dtype=np.float32)
    tilts = np.linspace(-config.tilt_ratio, config.tilt_ratio, max(1, config.tilt_steps)) * (width - 1)
    bows = np.linspace(-config.bow_ratio, config.bow_ratio, max(1, config.bow_steps)) * (width - 1)
    bins = len(x0_px) + 2 * int(np.ceil(abs(bows).max() if len(bows) else 0)) + 4
    offset = int(np.ceil(abs(bows).max() if len(bows) else 0)) + 2
    best_cov = np.zeros(bins, np.float32)
    best_param = np.zeros((bins, 3), np.float32)
    shape_bow = (1.0 - np.square(2.0 * s - 1.0))[None, :]
    shape_tilt = (s - 0.5)[None, :]
    rows = ys[None, :].repeat(len(x0_px), axis=0)
    for tilt in tilts:
        for bow in bows:
            xs = geometry.margin_x + x0_px[:, None] + tilt * shape_tilt + bow * shape_bow
            hits = _gather(dilated, xs, rows) >= threshold
            coverage = hits.mean(axis=1)
            mean_pos = x0_px + bow * 2.0 / 3.0
            index = np.clip(np.rint(mean_pos).astype(np.int64) - int(x0_px[0]) + offset, 0, bins - 1)
            order = np.argsort(coverage)
            index_sorted = index[order]
            coverage_sorted = coverage[order]
            better = coverage_sorted > best_cov[index_sorted]
            # later (larger) writes win because of the ascending sort
            best_cov[index_sorted[better]] = coverage_sorted[better]
            best_param[index_sorted[better]] = np.column_stack(
                (x0_px[order][better], np.full(int(better.sum()), tilt), np.full(int(better.sum()), bow))
            )
    smoothed = np.convolve(best_cov, np.ones(3, np.float32) / 3.0, mode="same")
    radius = max(2, int(round(config.nms_ratio * width)))
    peaks: list[tuple[float, float, float, float, float]] = []
    order = np.argsort(-smoothed)
    taken = np.zeros(bins, bool)
    for index in order:
        value = float(smoothed[index])
        if value < config.min_coverage:
            break
        if taken[index]:
            continue
        lo, hi = max(0, index - radius), min(bins, index + radius + 1)
        taken[lo:hi] = True
        # centre of the plateau around the maximum
        plateau = np.flatnonzero(best_cov[lo:hi] >= best_cov[index] - 0.02) + lo
        center = int(plateau[len(plateau) // 2]) if len(plateau) else int(index)
        x0, tilt, bow = best_param[center]
        if best_cov[center] <= 0:
            x0, tilt, bow = best_param[index]
        mean_u = float((x0 + bow * 2.0 / 3.0) / max(width - 1, 1))
        peaks.append((mean_u, float(x0 / (width - 1)), float(tilt / (width - 1)), float(bow / (width - 1)), value))
    return peaks


def _refine_path(
    evidence_dilated: np.ndarray,
    threshold: float,
    geometry: AnalysisGeometry,
    config: BoundaryConfig,
    s_grid: np.ndarray,
    guide_u: np.ndarray,
    corridor_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic-programming path (u as function of s) around ``guide_u``."""

    width, height = geometry.width, geometry.height
    radius = max(2, int(round(corridor_ratio * width)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    ys = geometry.margin_y + s_grid * (height - 1)
    guide_px = geometry.margin_x + guide_u * (width - 1)
    xs = guide_px[:, None] + offsets[None, :]
    values = _gather(evidence_dilated, xs, ys[:, None].repeat(len(offsets), axis=1))
    capped = np.minimum(values, 3.0 * threshold) / max(threshold, 1e-6)
    local = np.where(values >= 0.5 * threshold, -capped, 0.0).astype(np.float32)
    local += 0.03 * np.square(offsets / radius)[None, :]
    count, states = local.shape
    total = local[0].copy()
    back = np.zeros((count, states), np.int8)
    penalty = float(config.smoothness)
    big = np.float32(1e9)
    for k in range(1, count):
        stay = total
        left = np.concatenate(([big], total[:-1])) + penalty  # came from offset-1
        right = np.concatenate((total[1:], [big])) + penalty  # came from offset+1
        stacked = np.stack((left, stay, right))
        choice = np.argmin(stacked, axis=0)
        total = stacked[choice, np.arange(states)] + local[k]
        back[k] = (choice - 1).astype(np.int8)
    state = int(np.argmin(total))
    chosen = np.empty(count, np.float32)
    for k in range(count - 1, -1, -1):
        chosen[k] = offsets[state]
        if k:
            state = int(np.clip(state + back[k, state], 0, states - 1))
    if count >= 7:
        kernel = np.ones(5, np.float32) / 5.0
        padded = np.pad(chosen, 2, mode="edge")
        chosen = np.convolve(padded, kernel, mode="valid")
    path_px = guide_px + chosen
    hits = _gather(evidence_dilated, path_px, ys) >= threshold
    path_u = ((path_px - geometry.margin_x) / max(width - 1, 1)).astype(np.float32)
    return path_u, hits


def _path_statistics(path: np.ndarray, hits: np.ndarray, s_grid: np.ndarray, values: np.ndarray | None = None) -> tuple[float, float, float]:
    inside = (s_grid >= 0.02) & (s_grid <= 0.98)
    if not np.any(inside):
        return 0.0, 0.0, 1.0
    coverage = float(hits[inside].mean())
    gap = _longest_run(~hits[inside]) / max(1, int(inside.sum()))
    strength = 0.0
    if values is not None:
        selected = values[inside][hits[inside]]
        strength = float(np.mean(selected)) if selected.size else 0.0
    return coverage, strength, float(gap)


def path_hits(evidence_dilated: np.ndarray, threshold: float, geometry: AnalysisGeometry, s_grid: np.ndarray, path_u: np.ndarray) -> np.ndarray:
    xs = geometry.margin_x + path_u * (geometry.width - 1)
    ys = geometry.margin_y + s_grid * (geometry.height - 1)
    return _gather(evidence_dilated, xs, ys) >= threshold


def _merge_duplicates(lines: list[LineObservation], config: BoundaryConfig) -> list[LineObservation]:
    """Merge the two visible sides of a divider into its raised-face centre.

    A thick divider normally produces two strong, nearly parallel curves.  If
    one of them is simply discarded, the surviving curve is often the
    floor/wall contact edge.  Averaging the paired paths keeps the result on
    the physical divider band and retains all of its dense samples.  Outer
    borders are different: their coarse contour is already the outside rim,
    so an adjacent inner wall edge is absorbed without moving the border.
    """
    ordered = sorted(lines, key=lambda line: line.position)
    changed = True
    while changed and len(ordered) > 2:
        changed = False
        positions = np.asarray([line.position for line in ordered], np.float32)
        gaps = np.diff(positions)
        reference = float(np.percentile(gaps, 75)) if len(gaps) else 1.0
        index = int(np.argmin(gaps))
        gap = float(gaps[index])
        left, right = ordered[index], ordered[index + 1]
        touches_border = left.border or right.border
        ratio = config.border_merge_ratio if touches_border else config.min_cell_ratio
        if gap < max(ratio * reference, config.min_cell_size_ratio):
            if left.border and right.border:
                break
            if left.border:
                del ordered[index + 1]
            elif right.border:
                del ordered[index]
            else:
                weight_left = max(left.score, left.coverage, 0.1)
                weight_right = max(right.score, right.coverage, 0.1)
                total = weight_left + weight_right
                path = ((weight_left * left.path + weight_right * right.path) / total).astype(np.float32)
                stronger = left if (left.score, left.coverage) >= (right.score, right.coverage) else right
                # Geometry is centred on the raised band, but edge existence is
                # inherited from one actually observed side.  OR-ing both hit
                # masks could bridge a genuinely missing divider segment.
                hits = stronger.hits.copy()
                merged = LineObservation(
                    axis=left.axis,
                    position=float(np.mean(path)),
                    path=path,
                    hits=hits,
                    coverage=float(stronger.coverage),
                    strength=float(max(left.strength, right.strength)),
                    border=False,
                    junction_support=float(max(left.junction_support, right.junction_support)),
                    score=float(max(left.score, right.score)),
                    longest_gap=float(stronger.longest_gap),
                )
                ordered[index : index + 2] = [merged]
            changed = True
    return ordered


def _interp(path: np.ndarray, s_grid: np.ndarray, value: float) -> float:
    return float(np.interp(value, s_grid, path))


def intersect(vertical: LineObservation, horizontal: LineObservation, s_grid_v: np.ndarray, s_grid_h: np.ndarray) -> tuple[float, float]:
    u = vertical.position
    v = horizontal.position
    for _ in range(8):
        v = _interp(horizontal.path, s_grid_h, u)
        u = _interp(vertical.path, s_grid_v, v)
    return u, v


def _arm(hits: np.ndarray, s_grid: np.ndarray, start: float, stop: float) -> bool | None:
    lo, hi = sorted((start, stop))
    lo = max(lo, -0.004)
    hi = min(hi, 1.004)
    if hi - lo < 0.01:
        return None
    mask = (s_grid >= lo) & (s_grid <= hi)
    if not np.any(mask):
        return None
    return bool(hits[mask].mean() >= 0.5)


def junction_arms(
    vertical: LineObservation,
    horizontal: LineObservation,
    s_grid_v: np.ndarray,
    s_grid_h: np.ndarray,
    config: BoundaryConfig,
) -> tuple[tuple[float, float], list[bool | None]]:
    u, v = intersect(vertical, horizontal, s_grid_v, s_grid_h)
    arm, gap = config.arm_ratio, config.junction_gap_ratio
    arms = [
        _arm(vertical.hits, s_grid_v, v - arm, v - gap),  # up
        _arm(horizontal.hits, s_grid_h, u + gap, u + arm),  # right
        _arm(vertical.hits, s_grid_v, v + gap, v + arm),  # down
        _arm(horizontal.hits, s_grid_h, u - arm, u - gap),  # left
    ]
    return (u, v), arms


def junction_type(arms: list[bool | None]) -> str:
    vertical = sum(bool(a) for a in (arms[0], arms[2]))
    horizontal = sum(bool(a) for a in (arms[1], arms[3]))
    if vertical == 0 or horizontal == 0:
        return "NONE"
    total = vertical + horizontal
    return {4: "X", 3: "T", 2: "L"}.get(total, "NONE")


# --------------------------------------------------------------------------- detection


@dataclass
class RawStructure:
    evidence: EvidenceMaps
    geometry: AnalysisGeometry
    s_grid_v: np.ndarray  # parameter grid of vertical lines (along v)
    s_grid_h: np.ndarray  # parameter grid of horizontal lines (along u)
    vertical: list[LineObservation]  # borders + interior, ordered by u
    horizontal: list[LineObservation]
    rejected: list[LineObservation] = field(default_factory=list)
    dilated_v: np.ndarray | None = None
    dilated_h_t: np.ndarray | None = None  # transposed horizontal evidence


def _dilate_small(evidence: np.ndarray) -> np.ndarray:
    # +-2 px: a divider of a few pixels counts as one boundary even when only
    # one of its two edges has contrast (e.g. deep shadow on one side).
    return cv2.dilate(evidence, np.ones((1, 5), np.uint8))


def _axis_lines(
    evidence: np.ndarray,
    threshold: float,
    geometry: AnalysisGeometry,
    config: BoundaryConfig,
    axis: str,
    s_grid: np.ndarray,
) -> tuple[list[LineObservation], np.ndarray]:
    dilated = _dilate_small(evidence)
    lines: list[LineObservation] = []
    for border_u in (0.0, 1.0):
        guide = np.full(len(s_grid), border_u, np.float32)
        path, hits = _refine_path(dilated, threshold, geometry, config, s_grid, guide, config.border_corridor_ratio)
        coverage, strength, gap = _path_statistics(path, hits, s_grid, _gather(dilated, geometry.margin_x + path * (geometry.width - 1), geometry.margin_y + s_grid * (geometry.height - 1)) / max(threshold, 1e-6))
        lines.append(
            LineObservation(axis, float(np.mean(path[(s_grid >= 0) & (s_grid <= 1)])), path, hits, coverage, strength, True, 1.0, 1.0, gap)
        )
    for mean_u, x0, tilt, bow, _ in _search_hypotheses(evidence, threshold, geometry, config)[: 2 * config.max_lines_per_axis]:
        guide = (x0 + tilt * (s_grid - 0.5) + bow * (1.0 - np.square(2.0 * s_grid - 1.0))).astype(np.float32)
        path, hits = _refine_path(dilated, threshold, geometry, config, s_grid, guide, config.corridor_ratio)
        values = _gather(dilated, geometry.margin_x + path * (geometry.width - 1), geometry.margin_y + s_grid * (geometry.height - 1)) / max(threshold, 1e-6)
        coverage, strength, gap = _path_statistics(path, hits, s_grid, values)
        inside = (s_grid >= 0) & (s_grid <= 1)
        position = float(np.mean(path[inside]))
        if coverage < config.min_line_coverage or not 0.0 < position < 1.0:
            continue
        lines.append(LineObservation(axis, position, path, hits, coverage, min(1.0, strength / 3.0), False, 0.0, 0.0, gap))
    # remove duplicates of the same physical curve found from different hypotheses
    unique: list[LineObservation] = []
    for line in sorted(lines, key=lambda item: (not item.border, -item.coverage)):
        if any(abs(line.position - other.position) < 0.5 * config.nms_ratio and not (line.border and other.border) for other in unique):
            continue
        unique.append(line)
    return sorted(unique, key=lambda item: item.position), dilated


def _line_support(line: LineObservation, other: LineObservation, structure: "RawStructure", config: BoundaryConfig) -> float:
    """How well ``line`` behaves as a structural divider at its crossing with ``other``.

    Interior crossings must be X-like (the divider passes through), crossings
    with the outer wall must be T-like (the divider reaches the wall).
    """
    if line.axis == "v":
        _, arms = junction_arms(line, other, structure.s_grid_v, structure.s_grid_h, config)
        own, perpendicular = (arms[0], arms[2]), (arms[1], arms[3])
    else:
        _, arms = junction_arms(other, line, structure.s_grid_v, structure.s_grid_h, config)
        own, perpendicular = (arms[1], arms[3]), (arms[0], arms[2])
    if not any(bool(arm) for arm in perpendicular):
        return 0.0
    applicable = [arm for arm in own if arm is not None]
    if not applicable:
        return 0.0
    return float(sum(bool(arm) for arm in applicable) / len(applicable))


def _score_lines(structure: RawStructure, config: BoundaryConfig) -> None:
    for line in structure.vertical + structure.horizontal:
        if line.border:
            line.score = 1.0
            line.junction_support = 1.0
            continue
        perpendicular = structure.horizontal if line.axis == "v" else structure.vertical
        support = [_line_support(line, other, structure, config) for other in perpendicular]
        line.junction_support = float(np.mean(support)) if support else 0.0
        line.score = float(line.coverage * (0.55 + 0.45 * line.junction_support) * (0.85 + 0.15 * line.strength))


def detect_structure(analysis_image: np.ndarray, geometry: AnalysisGeometry, config: BoundaryConfig) -> RawStructure:
    """Single-frame line detection in the padded rectified analysis image."""
    roi = (geometry.margin_x, geometry.margin_y, geometry.margin_x + geometry.width, geometry.margin_y + geometry.height)
    evidence = compute_evidence(
        analysis_image,
        gradient_sigma=config.gradient_sigma,
        tensor_sigma=config.tensor_sigma,
        ridge_sigma=config.ridge_sigma,
        evidence_ratio=config.evidence_ratio,
        noise_factor=config.noise_factor,
        use_lsd=config.use_lsd,
        roi=roi,
    )
    s_grid_v = s_grid_for(config, geometry.height)
    s_grid_h = s_grid_for(config, geometry.width)
    vertical, dilated_v = _axis_lines(evidence.vertical, evidence.threshold, geometry, config, "v", s_grid_v)
    transposed = geometry.transposed()
    horizontal_t = np.ascontiguousarray(evidence.horizontal.T)
    horizontal, dilated_h = _axis_lines(horizontal_t, evidence.threshold, transposed, config, "h", s_grid_h)
    structure = RawStructure(evidence, geometry, s_grid_v, s_grid_h, vertical, horizontal, [], dilated_v, dilated_h)
    for _ in range(2):
        _score_lines(structure, config)
        def structural(line: LineObservation) -> bool:
            # A floor shadow may run through every cell but it stops at each
            # raised horizontal wall, producing weak/non-X junctions.  A real
            # divider continues into the wall junctions.  Very high coverage
            # remains a fallback for a partly occluded divider.
            return line.border or (
                line.junction_support >= config.min_junction_support
                or line.coverage >= config.min_structural_coverage
            )

        def keep_axis(lines: list[LineObservation]) -> list[LineObservation]:
            scored = [line for line in lines if line.border or line.score >= config.accept_score]
            strong = [line for line in scored if structural(line)]
            # Do not discard a lone weak/occluded physical divider.  Suppress
            # weak tracks only when they form a large second population around
            # an already complete structural lattice (the repeated cell-floor
            # shadows in the live camera do exactly this).
            if len(strong) >= 3 and len(scored) >= config.shadow_excess_ratio * len(strong):
                return strong
            return scored

        kept_v = keep_axis(structure.vertical)
        kept_h = keep_axis(structure.horizontal)
        structure.rejected.extend(line for line in structure.vertical + structure.horizontal if line not in kept_v + kept_h)
        structure.vertical = _merge_duplicates(kept_v, config)
        structure.horizontal = _merge_duplicates(kept_h, config)
    _score_lines(structure, config)
    for axis in ("vertical", "horizontal"):
        lines = getattr(structure, axis)
        interior = sorted((line for line in lines if not line.border), key=lambda line: -line.score)
        keep = set(id(line) for line in interior[: config.max_lines_per_axis])
        setattr(structure, axis, [line for line in lines if line.border or id(line) in keep])
    return structure


# --------------------------------------------------------------------------- temporal accumulation


@dataclass
class LineTrack:
    position: float
    path: np.ndarray
    probability: float
    reference: float
    hits: int = 1
    misses: int = 0
    accepted: bool = False
    observed: bool = True
    coverage: float = 0.0
    junction_support: float = 0.0

    @property
    def confidence(self) -> float:
        return float(np.clip(self.probability / max(self.reference, 1e-6), 0.0, 1.0) * self.reference)


class LineTracker:
    """Per-axis existence probability of interior boundary lines.

    A line seen repeatedly rises towards its observation score; an unseen line
    decays.  Acceptance is relative to the line's own typical score, so a newly
    appearing set of lines replaces an old set after the same number of frames
    (``1 - (1-alpha)^n > 0.5``) instead of leaving a gap or freezing the first
    guess.
    """

    def __init__(self, config: BoundaryConfig) -> None:
        self.alpha = float(np.clip(config.temporal_alpha, 0.01, 1.0))
        self.path_alpha = float(np.clip(config.path_alpha, 0.01, 1.0))
        self.association = float(config.association_ratio)
        self.accept = float(config.accept_score)
        self.tracks: list[LineTrack] = []
        self.frames = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.frames = 0

    def update(self, observations: list[LineObservation]) -> list[LineTrack]:
        bootstrap = self.frames == 0
        self.frames += 1
        unmatched = list(range(len(observations)))
        pairs: list[tuple[float, int, int]] = []
        for t_index, track in enumerate(self.tracks):
            for o_index, observation in enumerate(observations):
                distance = abs(track.position - observation.position)
                if distance < self.association:
                    pairs.append((distance, t_index, o_index))
        used_tracks: set[int] = set()
        used_obs: set[int] = set()
        for _, t_index, o_index in sorted(pairs):
            if t_index in used_tracks or o_index in used_obs:
                continue
            used_tracks.add(t_index)
            used_obs.add(o_index)
            track = self.tracks[t_index]
            observation = observations[o_index]
            score = observation.score
            track.probability = (1.0 - self.alpha) * track.probability + self.alpha * score
            track.reference = 0.8 * track.reference + 0.2 * score
            track.path = ((1.0 - self.path_alpha) * track.path + self.path_alpha * observation.path).astype(np.float32)
            track.position = (1.0 - self.path_alpha) * track.position + self.path_alpha * observation.position
            track.hits += 1
            track.misses = 0
            track.observed = True
            track.coverage = observation.coverage
            track.junction_support = observation.junction_support
        for t_index, track in enumerate(self.tracks):
            if t_index not in used_tracks:
                track.probability *= 1.0 - self.alpha
                track.misses += 1
                track.observed = False
        for o_index in unmatched:
            if o_index in used_obs:
                continue
            observation = observations[o_index]
            probability = observation.score if bootstrap else self.alpha * observation.score
            self.tracks.append(
                LineTrack(
                    observation.position,
                    observation.path.copy(),
                    probability,
                    max(observation.score, self.accept),
                    coverage=observation.coverage,
                    junction_support=observation.junction_support,
                )
            )
        survivors: list[LineTrack] = []
        for track in self.tracks:
            ratio = track.probability / max(track.reference, 1e-6)
            track.accepted = ratio >= 0.5 - 1e-6
            if ratio >= 0.08:
                survivors.append(track)
        self.tracks = sorted(survivors, key=lambda track: track.position)
        return [track for track in self.tracks if track.accepted]


# --------------------------------------------------------------------------- graph


@dataclass
class BoundaryEdge:
    key: str
    start: int
    end: int
    points: np.ndarray  # normalized (n, 2)
    confidence: float
    present: bool


@dataclass
class BoundaryFace:
    id: int
    cells: list[tuple[int, int]]
    polygon: np.ndarray  # normalized (n, 2)
    confidence: float


@dataclass
class PlanarBoundaryGraph:
    """Junction nodes + curved polyline edges + enclosed faces (normalized coordinates)."""

    nodes: np.ndarray  # (nh, nv, 2) grid of line intersections
    node_confidence: np.ndarray  # (nh, nv)
    node_types: list[list[str]]
    edges: dict[str, BoundaryEdge]
    faces: list[BoundaryFace]
    row_paths: list[np.ndarray]  # horizontal lines: v(u) on s_grid_h
    col_paths: list[np.ndarray]  # vertical lines: u(v) on s_grid_v
    s_grid_v: np.ndarray
    s_grid_h: np.ndarray
    row_confidence: list[float]
    col_confidence: list[float]
    regular: bool
    confidence: float

    @property
    def rows(self) -> int:
        return len(self.row_paths) - 1

    @property
    def cols(self) -> int:
        return len(self.col_paths) - 1

    def row_positions(self) -> list[float]:
        return [float(np.mean(path[(self.s_grid_h >= 0) & (self.s_grid_h <= 1)])) for path in self.row_paths]

    def col_positions(self) -> list[float]:
        return [float(np.mean(path[(self.s_grid_v >= 0) & (self.s_grid_v <= 1)])) for path in self.col_paths]

    def full_line_points(self, axis: str, index: int) -> np.ndarray:
        """Normalized (u, v) points of a full (extended) line, internal use only."""
        if axis == "h":
            return np.column_stack((self.s_grid_h, self.row_paths[index])).astype(np.float32)
        return np.column_stack((self.col_paths[index], self.s_grid_v)).astype(np.float32)


def _segment_points(axis: str, path: np.ndarray, s_grid: np.ndarray, start: float, stop: float, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    mask = (s_grid > start) & (s_grid < stop)
    inner = s_grid[mask]
    values = path[mask]
    if axis == "h":
        middle = np.column_stack((inner, values))
    else:
        middle = np.column_stack((values, inner))
    return np.vstack((first[None, :], middle, last[None, :])).astype(np.float32)


def _coverage_between(hits: np.ndarray, s_grid: np.ndarray, start: float, stop: float, gap: float) -> float:
    lo, hi = start + gap, stop - gap
    if hi <= lo:
        lo, hi = start, stop
    mask = (s_grid >= lo) & (s_grid <= hi)
    return float(hits[mask].mean()) if np.any(mask) else 0.0


def build_graph(
    vertical: list[LineObservation],
    horizontal: list[LineObservation],
    s_grid_v: np.ndarray,
    s_grid_h: np.ndarray,
    config: BoundaryConfig,
    line_confidence_v: list[float] | None = None,
    line_confidence_h: list[float] | None = None,
) -> PlanarBoundaryGraph:
    vertical = sorted(vertical, key=lambda line: line.position)
    horizontal = sorted(horizontal, key=lambda line: line.position)
    nv, nh = len(vertical), len(horizontal)
    conf_v = line_confidence_v or [1.0 if line.border else line.score for line in vertical]
    conf_h = line_confidence_h or [1.0 if line.border else line.score for line in horizontal]
    nodes = np.zeros((nh, nv, 2), np.float32)
    node_conf = np.zeros((nh, nv), np.float32)
    types: list[list[str]] = [["NONE"] * nv for _ in range(nh)]
    for i, h_line in enumerate(horizontal):
        for j, v_line in enumerate(vertical):
            (u, v), arms = junction_arms(v_line, h_line, s_grid_v, s_grid_h, config)
            nodes[i, j] = (u, v)
            kind = junction_type(arms)
            expected = sum(arm is not None for arm in arms)
            present = sum(bool(arm) for arm in arms)
            types[i][j] = kind
            node_conf[i, j] = present / max(expected, 1) * min(conf_h[i], conf_v[j]) ** 0.5
    edges: dict[str, BoundaryEdge] = {}
    gap = config.junction_gap_ratio
    for i, h_line in enumerate(horizontal):
        for j in range(nv - 1):
            start, stop = float(nodes[i, j, 0]), float(nodes[i, j + 1, 0])
            coverage = _coverage_between(h_line.hits, s_grid_h, start, stop, gap)
            points = _segment_points("h", h_line.path, s_grid_h, start, stop, nodes[i, j], nodes[i, j + 1])
            present = coverage >= config.segment_min_coverage or h_line.border and coverage >= 0.5 * config.segment_min_coverage
            edges[f"h_{i}_{j}"] = BoundaryEdge(f"h_{i}_{j}", i * nv + j, i * nv + j + 1, points, float(coverage * conf_h[i]), bool(present))
    for j, v_line in enumerate(vertical):
        for i in range(nh - 1):
            start, stop = float(nodes[i, j, 1]), float(nodes[i + 1, j, 1])
            coverage = _coverage_between(v_line.hits, s_grid_v, start, stop, gap)
            points = _segment_points("v", v_line.path, s_grid_v, start, stop, nodes[i, j], nodes[i + 1, j])
            present = coverage >= config.segment_min_coverage or v_line.border and coverage >= 0.5 * config.segment_min_coverage
            edges[f"v_{i}_{j}"] = BoundaryEdge(f"v_{i}_{j}", i * nv + j, (i + 1) * nv + j, points, float(coverage * conf_v[j]), bool(present))
    # Outer boundary edges are part of the verified outer wall; keep them even
    # if a short piece is shadowed so that faces stay closed.
    for key, edge in edges.items():
        _, i, j = key.split("_")
        i, j = int(i), int(j)
        if (key.startswith("h") and i in (0, nh - 1)) or (key.startswith("v") and j in (0, nv - 1)):
            edge.present = True

    rows, cols = nh - 1, nv - 1
    parent = list(range(rows * cols))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols and not edges[f"v_{r}_{c + 1}"].present:
                parent[find(r * cols + c + 1)] = find(r * cols + c)
            if r + 1 < rows and not edges[f"h_{r + 1}_{c}"].present:
                parent[find((r + 1) * cols + c)] = find(r * cols + c)
    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            groups.setdefault(find(r * cols + c), []).append((r, c))
    faces: list[BoundaryFace] = []
    for face_id, cells in enumerate(sorted(groups.values(), key=lambda cells: min(cells))):
        polygon, confidence = _face_polygon(cells, edges, nodes, rows, cols)
        faces.append(BoundaryFace(face_id, sorted(cells), polygon, confidence))
    regular = len(faces) == rows * cols and rows >= 1 and cols >= 1
    interior_conf = [c for c, line in zip(conf_v, vertical) if not line.border] + [
        c for c, line in zip(conf_h, horizontal) if not line.border
    ]
    edge_conf = [edge.confidence for edge in edges.values() if edge.present]
    confidence = float(np.mean(interior_conf)) if interior_conf else float(np.mean(edge_conf) if edge_conf else 0.0)
    return PlanarBoundaryGraph(
        nodes,
        node_conf,
        types,
        edges,
        faces,
        [line.path.copy() for line in horizontal],
        [line.path.copy() for line in vertical],
        s_grid_v,
        s_grid_h,
        list(map(float, conf_h)),
        list(map(float, conf_v)),
        bool(regular),
        confidence,
    )


def _face_polygon(
    cells: list[tuple[int, int]],
    edges: dict[str, BoundaryEdge],
    nodes: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, float]:
    """Chain the directed boundary edges of a (possibly merged) face into a cycle."""
    cell_set = set(cells)
    nv = cols + 1
    directed: dict[int, tuple[int, np.ndarray, float]] = {}
    for r, c in cells:
        sides = [
            (f"h_{r}_{c}", False, (r - 1, c)),
            (f"v_{r}_{c + 1}", False, (r, c + 1)),
            (f"h_{r + 1}_{c}", True, (r + 1, c)),
            (f"v_{r}_{c}", True, (r, c - 1)),
        ]
        for key, reverse, neighbour in sides:
            if neighbour in cell_set:
                continue
            edge = edges[key]
            points = edge.points[::-1] if reverse else edge.points
            start, end = (edge.end, edge.start) if reverse else (edge.start, edge.end)
            directed[start] = (end, points, edge.confidence)
    if not directed:
        return np.zeros((0, 2), np.float32), 0.0
    start = min(directed)
    current = start
    chain: list[np.ndarray] = []
    confidences: list[float] = []
    for _ in range(len(directed) + 1):
        if current not in directed:
            break
        end, points, confidence = directed[current]
        chain.append(points[:-1])
        confidences.append(confidence)
        current = end
        if current == start:
            break
    polygon = np.vstack(chain).astype(np.float32) if chain else np.zeros((0, 2), np.float32)
    del nodes, nv
    return polygon, float(np.mean(confidences)) if confidences else 0.0


# --------------------------------------------------------------------------- stateful detector


@dataclass
class BoundaryDetection:
    graph: PlanarBoundaryGraph | None
    raw: RawStructure
    state: str  # CONFIRMED / TENTATIVE / UNCERTAIN
    stable_frames: int
    confidence: float


class BoundaryGraphDetector:
    """Frame-to-frame accumulation of evidence-driven boundaries."""

    def __init__(self, config: BoundaryConfig | dict | None = None) -> None:
        self.config = config if isinstance(config, BoundaryConfig) else BoundaryConfig.from_dict(config)
        self.geometry = analysis_geometry(self.config)
        self.v_tracker = LineTracker(self.config)
        self.h_tracker = LineTracker(self.config)
        self.border_paths: dict[str, np.ndarray] = {}
        self.last_shape: tuple[int, int] | None = None
        self.stable_frames = 0

    def reset(self) -> None:
        self.v_tracker.reset()
        self.h_tracker.reset()
        self.border_paths.clear()
        self.last_shape = None
        self.stable_frames = 0

    def warp(self, frame: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        homography = analysis_homography(corners, self.geometry)
        image = cv2.warpPerspective(frame, homography, self.geometry.image_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return image, homography

    def _tracked_lines(self, raw: RawStructure, axis: str) -> tuple[list[LineObservation], list[float]]:
        observed = raw.vertical if axis == "v" else raw.horizontal
        tracker = self.v_tracker if axis == "v" else self.h_tracker
        s_grid = raw.s_grid_v if axis == "v" else raw.s_grid_h
        dilated = raw.dilated_v if axis == "v" else raw.dilated_h_t
        geometry = raw.geometry if axis == "v" else raw.geometry.transposed()
        assert dilated is not None
        interior = [line for line in observed if not line.border]
        borders = [line for line in observed if line.border]
        accepted = tracker.update(interior)
        # Hysteresis: a line is *accepted* on the strong threshold; its
        # individual segments are then verified with a weaker one (Canny-like),
        # so a divider stays connected through deep shadow but an erased
        # segment (no edge at all) is still reported as missing.
        weak = raw.evidence.threshold * self.config.segment_hysteresis
        lines: list[LineObservation] = []
        confidences: list[float] = []
        for index, border in enumerate(sorted(borders, key=lambda line: line.position)):
            key = f"{axis}{index}"
            previous = self.border_paths.get(key)
            path = border.path if previous is None or previous.shape != border.path.shape else (
                (1.0 - self.config.path_alpha) * previous + self.config.path_alpha * border.path
            ).astype(np.float32)
            self.border_paths[key] = path
            hits = path_hits(dilated, weak, geometry, s_grid, path)
            lines.append(LineObservation(axis, float(np.mean(path)), path, hits, border.coverage, border.strength, True, 1.0, 1.0))
            confidences.append(1.0)
        for track in accepted:
            hits = path_hits(dilated, weak, geometry, s_grid, track.path)
            inside = (s_grid >= 0.02) & (s_grid <= 0.98)
            coverage = float(hits[inside].mean())
            lines.append(LineObservation(axis, track.position, track.path, hits, coverage, 0.0, False, track.junction_support, track.confidence))
            confidences.append(track.confidence)
        order = np.argsort([line.position for line in lines])
        return [lines[k] for k in order], [confidences[k] for k in order]

    def detect_rectified(self, analysis_image: np.ndarray) -> BoundaryDetection:
        raw = detect_structure(analysis_image, self.geometry, self.config)
        vertical, conf_v = self._tracked_lines(raw, "v")
        horizontal, conf_h = self._tracked_lines(raw, "h")
        graph = build_graph(vertical, horizontal, raw.s_grid_v, raw.s_grid_h, self.config, conf_v, conf_h)
        shape = (graph.rows, graph.cols) if graph.regular else None
        if shape is not None and shape == self.last_shape:
            self.stable_frames += 1
        else:
            self.stable_frames = 1 if shape is not None else 0
        self.last_shape = shape
        confidence = graph.confidence
        if not graph.regular or confidence < self.config.min_topology_confidence:
            state = "UNCERTAIN"
        elif self.stable_frames >= self.config.confirm_frames:
            state = "CONFIRMED"
        else:
            state = "TENTATIVE"
        return BoundaryDetection(graph, raw, state, self.stable_frames, confidence)

    def detect(self, frame: np.ndarray, corners: np.ndarray) -> tuple[BoundaryDetection, np.ndarray]:
        image, homography = self.warp(frame, corners)
        return self.detect_rectified(image), homography


def detect_single_frame(frame: np.ndarray, corners: np.ndarray, config: BoundaryConfig | dict | None = None) -> tuple[BoundaryDetection, np.ndarray, AnalysisGeometry]:
    detector = BoundaryGraphDetector(config)
    detection, homography = detector.detect(frame, corners)
    return detection, homography, detector.geometry
