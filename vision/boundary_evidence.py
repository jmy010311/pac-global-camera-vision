"""Polarity-free boundary evidence maps.

The maps answer "is there a sharp, oriented, structural boundary here?" and not
"is this pixel bright?".  They are built from

* log-luminance gradients (multiplicative shading/shadow is removed by log),
* chroma gradients (a*, b*), so a dark/bright or colour polarity flip between
  neighbouring cells gives the same response,
* a multi-channel structure tensor (orientation + coherence),
* a Hessian ridge term for thin dividers of either polarity,
* optional LSD line segments as confirmation of weak but straight edges.

No absolute brightness or colour value is used as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EvidenceMaps:
    vertical: np.ndarray  # boundaries running along image y (gradient along x)
    horizontal: np.ndarray  # boundaries running along image x (gradient along y)
    grad_x: np.ndarray  # combined signed-free gradient components for DP tracing
    grad_y: np.ndarray
    coherence: np.ndarray
    threshold: float  # adaptive hit threshold on vertical/horizontal maps
    noise: float


def _log_lab(image: np.ndarray) -> list[tuple[np.ndarray, float]]:
    if image.ndim == 2:
        gray = image.astype(np.float32)
        return [(40.0 * np.log1p(gray), 1.0)]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    return [
        (40.0 * np.log1p(lab[:, :, 0]), 1.0),
        (lab[:, :, 1] - 128.0, 0.7),
        (lab[:, :, 2] - 128.0, 0.7),
    ]


def compute_evidence(
    image: np.ndarray,
    *,
    gradient_sigma: float = 1.0,
    tensor_sigma: float = 1.5,
    ridge_sigma: float = 1.6,
    evidence_ratio: float = 0.30,
    noise_factor: float = 4.0,
    use_lsd: bool = True,
    roi: tuple[int, int, int, int] | None = None,
) -> EvidenceMaps:
    """Compute oriented structural boundary evidence for ``image``.

    ``roi`` (x0, y0, x1, y1) limits the statistics used for normalisation to the
    region that actually contains the partition (the padding outside the outer
    wall must not dominate the adaptive threshold).
    """

    channels = _log_lab(image)
    jxx = jxy = jyy = None
    gx_sum = gy_sum = None
    for channel, weight in channels:
        blurred = cv2.GaussianBlur(channel, (0, 0), gradient_sigma)
        gx = cv2.Scharr(blurred, cv2.CV_32F, 1, 0) / 16.0
        gy = cv2.Scharr(blurred, cv2.CV_32F, 0, 1) / 16.0
        if jxx is None:
            jxx, jxy, jyy = weight * gx * gx, weight * gx * gy, weight * gy * gy
            gx_sum, gy_sum = weight * np.abs(gx), weight * np.abs(gy)
        else:
            jxx += weight * gx * gx
            jxy += weight * gx * gy
            jyy += weight * gy * gy
            gx_sum += weight * np.abs(gx)
            gy_sum += weight * np.abs(gy)
    assert jxx is not None and jxy is not None and jyy is not None
    jxx = cv2.GaussianBlur(jxx, (0, 0), tensor_sigma)
    jxy = cv2.GaussianBlur(jxy, (0, 0), tensor_sigma)
    jyy = cv2.GaussianBlur(jyy, (0, 0), tensor_sigma)
    trace = jxx + jyy
    discriminant = np.sqrt(np.square(jxx - jyy) + 4.0 * np.square(jxy))
    magnitude = np.sqrt(np.maximum((trace + discriminant) * 0.5, 0.0))
    coherence = discriminant / (trace + 1e-6)
    # cos(2*theta) of the dominant gradient direction; +1 => gradient along x.
    cos2 = (jxx - jyy) / (discriminant + 1e-6)

    # Thin-divider ridge term (either polarity) on the log-luminance channel.
    luminance = cv2.GaussianBlur(channels[0][0], (0, 0), ridge_sigma)
    dxx = cv2.Sobel(luminance, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(luminance, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(luminance, cv2.CV_32F, 1, 1, ksize=3)
    h_disc = np.sqrt(np.square(dxx - dyy) + 4.0 * np.square(dxy))
    ridge = 0.5 * h_disc  # |lambda1 - lambda2| / 2: line-like curvature, polarity free
    ridge_cos2 = (dxx - dyy) / (h_disc + 1e-6)  # +1 => curvature along x => vertical ridge

    height, width = magnitude.shape
    if roi is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = (int(value) for value in roi)
    inner = magnitude[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
    scale = float(np.percentile(inner, 90)) if inner.size else float(np.percentile(magnitude, 90))
    scale = max(scale, 1e-3)
    noise = float(np.median(inner)) / scale if inner.size else 0.0
    edge = np.clip(magnitude / scale, 0.0, 4.0)
    ridge_inner = ridge[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
    ridge_scale = max(float(np.percentile(ridge_inner, 97)) if ridge_inner.size else 1.0, 1e-3)
    ridge_n = np.clip(ridge / ridge_scale, 0.0, 4.0) * 0.8

    w_vertical = np.clip(cos2, 0.0, 1.0)
    w_horizontal = np.clip(-cos2, 0.0, 1.0)
    vertical = edge * coherence * w_vertical
    horizontal = edge * coherence * w_horizontal
    vertical = np.maximum(vertical, ridge_n * np.clip(ridge_cos2, 0.0, 1.0))
    horizontal = np.maximum(horizontal, ridge_n * np.clip(-ridge_cos2, 0.0, 1.0))

    both = np.maximum(vertical, horizontal)[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
    high = float(np.percentile(both, 99.5)) if both.size else 1.0
    threshold = max(evidence_ratio * high, noise_factor * noise, 0.08)

    if use_lsd:
        vertical, horizontal = _add_lsd_support(channels[0][0], vertical, horizontal, threshold)

    return EvidenceMaps(
        vertical=vertical.astype(np.float32),
        horizontal=horizontal.astype(np.float32),
        grad_x=gx_sum.astype(np.float32),
        grad_y=gy_sum.astype(np.float32),
        coherence=coherence.astype(np.float32),
        threshold=float(threshold),
        noise=float(noise),
    )


_LSD = None


def _add_lsd_support(
    log_luminance: np.ndarray,
    vertical: np.ndarray,
    horizontal: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """LSD confirms weak-but-straight edges; it never creates evidence alone."""
    global _LSD
    try:
        if _LSD is None:
            _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        gray = cv2.normalize(log_luminance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lines = _LSD.detect(gray)[0]
    except cv2.error:
        return vertical, horizontal
    if lines is None or not len(lines):
        return vertical, horizontal
    height, width = vertical.shape
    minimum = 0.03 * min(height, width)
    mask_v = np.zeros((height, width), np.uint8)
    mask_h = np.zeros((height, width), np.uint8)
    for x_a, y_a, x_b, y_b in lines.reshape(-1, 4):
        dx, dy = x_b - x_a, y_b - y_a
        length = float(np.hypot(dx, dy))
        if length < minimum:
            continue
        target = mask_v if abs(dy) >= abs(dx) * 2.0 else (mask_h if abs(dx) >= abs(dy) * 2.0 else None)
        if target is None:
            continue
        cv2.line(target, (int(round(x_a)), int(round(y_a))), (int(round(x_b)), int(round(y_b))), 255, 2, cv2.LINE_8)
    boost = np.float32(threshold * 1.05)
    support_v = (mask_v > 0) & (vertical >= 0.5 * threshold)
    support_h = (mask_h > 0) & (horizontal >= 0.5 * threshold)
    vertical = vertical.copy()
    horizontal = horizontal.copy()
    vertical[support_v] = np.maximum(vertical[support_v], boost)
    horizontal[support_h] = np.maximum(horizontal[support_h], boost)
    return vertical, horizontal
