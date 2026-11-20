"""Metric per-cell height from calibrated global + gripper (wrist) cameras.

Order of operations (per inspection frame pair):

1. lens distortion correction (``LensUndistorter`` upstream, or ``D`` here),
2. epipolar rectification (``cv2.stereoRectify``; keypoint coordinates are
   rectified with R1/P1 and R2/P2),
3. feature correspondences restricted to the same rectified epipolar line and
   a physically valid disparity range, ratio test + mutual check,
4. outlier removal: calibrated epipolar gating, triangulation reprojection
   check and per-cell RANSAC height consensus,
5. 3D triangulation (``cv2.triangulatePoints``),
6. signed distance of every 3D point to the calibrated reference floor plane.

Every cell uses only the 3D points whose global-image location lies inside its
own polygon; no cell's result is used as a reference for another cell.  Colour
and brightness are never used: features are gradient descriptors on
contrast-normalised grayscale and only geometry decides the height.
Without a lens + stereo calibration no metric height is produced
(``UNCALIBRATED``); without enough valid 3D points a cell is ``UNKNOWN``.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import cv2
import numpy as np


class HeightState(str, Enum):
    FLOOR = "FLOOR"
    ABOVE_FLOOR = "ABOVE_FLOOR"
    BELOW_FLOOR = "BELOW_FLOOR"
    UNKNOWN = "UNKNOWN"
    UNCALIBRATED = "UNCALIBRATED"


# --------------------------------------------------------------------------- calibration


def _normalize_plane(normal: np.ndarray, offset: float) -> tuple[np.ndarray, float]:
    """Unit normal pointing to the camera side, so heights above the floor are positive."""
    normal = np.asarray(normal, np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("Degenerate floor-plane normal")
    normal, offset = normal / norm, float(offset) / norm
    if offset < 0:  # camera centre (origin) must be on the positive side
        normal, offset = -normal, -offset
    return normal, offset


def plane_from_points(points: np.ndarray, threshold_mm: float = 2.0, iterations: int = 400, seed: int = 2026) -> tuple[np.ndarray, float, float, np.ndarray]:
    """RANSAC + SVD plane through 3D points; returns (n, d, rms_mm, inlier mask)."""
    points = np.asarray(points, np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise ValueError("At least three 3D floor points are required")
    rng = np.random.default_rng(seed)
    best = np.ones(len(points), bool)
    best_count = 0
    if len(points) > 3:
        for _ in range(iterations):
            sample = points[rng.choice(len(points), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            if np.linalg.norm(normal) < 1e-9:
                continue
            normal /= np.linalg.norm(normal)
            distance = np.abs((points - sample[0]) @ normal)
            inliers = distance <= threshold_mm
            if int(inliers.sum()) > best_count:
                best, best_count = inliers, int(inliers.sum())
        if best_count < 3:
            best = np.ones(len(points), bool)
    selected = points[best]
    centroid = selected.mean(axis=0)
    _, _, vt = np.linalg.svd(selected - centroid)
    normal = vt[-1]
    normal, offset = _normalize_plane(normal, -float(normal @ centroid))
    rms = float(np.sqrt(np.mean(np.square(selected @ normal + offset))))
    return normal, offset, rms, best


def plane_from_board_pose(object_points: np.ndarray, image_points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray) -> tuple[np.ndarray, float]:
    """Floor plane from a checkerboard lying flat on the reference floor."""
    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(object_points, np.float64).reshape(-1, 3),
        np.asarray(image_points, np.float64).reshape(-1, 1, 2),
        np.asarray(camera_matrix, np.float64),
        np.asarray(distortion, np.float64),
    )
    if not ok:
        raise ValueError("solvePnP failed for the floor board")
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2]
    return _normalize_plane(normal, -float(normal @ tvec.reshape(3)))


@dataclass
class StereoCalibration:
    """Intrinsics/extrinsics of the fixed global camera and the wrist camera at
    its fixed inspection pose, plus the reference floor plane.

    Convention: ``X_wrist = R @ X_global + T`` (mm).  The plane is
    ``n . X + d = 0`` in global-camera coordinates with ``d > 0``; the signed
    height of a point is ``n . X + d``.
    ``K_*``/``D_*`` describe the images passed to the estimator.  When the
    images are already lens-corrected by ``LensUndistorter`` ``D_*`` is zero and
    ``K_*`` is the undistorter's output matrix.
    """

    global_image_size: tuple[int, int]
    wrist_image_size: tuple[int, int]
    K_global: np.ndarray
    D_global: np.ndarray
    K_wrist: np.ndarray
    D_wrist: np.ndarray
    R: np.ndarray
    T: np.ndarray
    plane_normal: np.ndarray
    plane_offset: float
    stereo_rms_px: float = 0.0
    plane_rms_mm: float = 0.0
    lens_corrected_inputs: bool = True
    created_at: str = ""
    notes: dict = field(default_factory=dict)

    def validate(self) -> None:
        for name in ("K_global", "K_wrist", "R"):
            if np.asarray(getattr(self, name)).shape != (3, 3):
                raise ValueError(f"{name} must be 3x3")
        if np.asarray(self.T).reshape(-1).shape != (3,):
            raise ValueError("T must have three elements")
        if float(np.linalg.norm(self.T)) < 1e-6:
            raise ValueError("Stereo baseline is zero; heights cannot be triangulated")
        if abs(float(np.linalg.det(self.R)) - 1.0) > 1e-3:
            raise ValueError("R is not a rotation matrix")
        if self.plane_offset <= 0:
            raise ValueError("Floor plane must be in front of the global camera")

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    def scaled(self, global_size: tuple[int, int], wrist_size: tuple[int, int]) -> "StereoCalibration":
        def scale(matrix: np.ndarray, source: tuple[int, int], target: tuple[int, int]) -> np.ndarray:
            output = np.asarray(matrix, np.float64).copy()
            sx, sy = target[0] / max(source[0], 1), target[1] / max(source[1], 1)
            output[0, :] *= sx
            output[1, :] *= sy
            return output

        if tuple(global_size) == tuple(self.global_image_size) and tuple(wrist_size) == tuple(self.wrist_image_size):
            return self
        return replace(
            self,
            global_image_size=tuple(global_size),
            wrist_image_size=tuple(wrist_size),
            K_global=scale(self.K_global, self.global_image_size, global_size),
            K_wrist=scale(self.K_wrist, self.wrist_image_size, wrist_size),
        )

    def floor_homography_global_to_wrist(self) -> np.ndarray:
        """Plane-induced homography of the floor (auxiliary residual only)."""
        n = self.plane_normal.reshape(3, 1)
        # points on the plane satisfy n.X = -d  ->  X_w = (R - T n^T / d) X
        matrix = self.R - (self.T.reshape(3, 1) @ n.T) / self.plane_offset
        return self.K_wrist @ matrix @ np.linalg.inv(self.K_global)

    def signed_height(self, points_global: np.ndarray) -> np.ndarray:
        return np.asarray(points_global, np.float64).reshape(-1, 3) @ self.plane_normal + self.plane_offset

    # ---- construction

    @classmethod
    def from_stereo_views(
        cls,
        object_points: list[np.ndarray],
        global_points: list[np.ndarray],
        wrist_points: list[np.ndarray],
        K_global: np.ndarray,
        D_global: np.ndarray,
        K_wrist: np.ndarray,
        D_wrist: np.ndarray,
        global_image_size: tuple[int, int],
        wrist_image_size: tuple[int, int],
        floor_plane: tuple[np.ndarray, float] | None = None,
        lens_corrected_inputs: bool = True,
    ) -> "StereoCalibration":
        if len(object_points) < 3:
            raise ValueError("Capture at least three simultaneous checkerboard views")
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
            [np.asarray(p, np.float32).reshape(-1, 3) for p in object_points],
            [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in global_points],
            [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in wrist_points],
            np.asarray(K_global, np.float64),
            np.asarray(D_global, np.float64),
            np.asarray(K_wrist, np.float64),
            np.asarray(D_wrist, np.float64),
            tuple(int(v) for v in global_image_size),
            criteria=criteria,
            flags=flags,
        )
        if floor_plane is None:
            normal, offset = np.array([0.0, 0.0, 1.0]), 1.0
            plane_known = False
        else:
            normal, offset = _normalize_plane(*floor_plane)
            plane_known = True
        calibration = cls(
            global_image_size=tuple(int(v) for v in global_image_size),
            wrist_image_size=tuple(int(v) for v in wrist_image_size),
            K_global=np.asarray(K_global, np.float64),
            D_global=np.asarray(D_global, np.float64).reshape(-1),
            K_wrist=np.asarray(K_wrist, np.float64),
            D_wrist=np.asarray(D_wrist, np.float64).reshape(-1),
            R=np.asarray(rotation, np.float64),
            T=np.asarray(translation, np.float64).reshape(3),
            plane_normal=normal,
            plane_offset=float(offset),
            stereo_rms_px=float(rms),
            lens_corrected_inputs=lens_corrected_inputs,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            notes={"floor_plane_measured": plane_known},
        )
        calibration.validate()
        return calibration

    def with_floor_plane(self, normal: np.ndarray, offset: float, rms_mm: float = 0.0, source: str = "") -> "StereoCalibration":
        normal, offset = _normalize_plane(normal, offset)
        notes = dict(self.notes)
        notes.update({"floor_plane_measured": True, "floor_plane_source": source})
        return replace(self, plane_normal=normal, plane_offset=float(offset), plane_rms_mm=float(rms_mm), notes=notes)

    @property
    def has_floor_plane(self) -> bool:
        return bool(self.notes.get("floor_plane_measured", True))

    def triangulate_pixels(self, global_points: np.ndarray, wrist_points: np.ndarray) -> np.ndarray:
        """3D points (global camera frame) from pixel correspondences (no rectification)."""
        g = cv2.undistortPoints(np.asarray(global_points, np.float64).reshape(-1, 1, 2), self.K_global, self.D_global).reshape(-1, 2)
        w = cv2.undistortPoints(np.asarray(wrist_points, np.float64).reshape(-1, 1, 2), self.K_wrist, self.D_wrist).reshape(-1, 2)
        p1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        p2 = np.hstack((self.R, self.T.reshape(3, 1)))
        homogeneous = cv2.triangulatePoints(p1, p2, g.T, w.T)
        return (homogeneous[:3] / homogeneous[3]).T

    # ---- persistence

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "global_image_size": list(self.global_image_size),
            "wrist_image_size": list(self.wrist_image_size),
            "K_global": np.asarray(self.K_global).tolist(),
            "D_global": np.asarray(self.D_global).reshape(-1).tolist(),
            "K_wrist": np.asarray(self.K_wrist).tolist(),
            "D_wrist": np.asarray(self.D_wrist).reshape(-1).tolist(),
            "R": np.asarray(self.R).tolist(),
            "T_mm": np.asarray(self.T).reshape(-1).tolist(),
            "floor_plane_normal": np.asarray(self.plane_normal).tolist(),
            "floor_plane_offset_mm": float(self.plane_offset),
            "stereo_rms_px": float(self.stereo_rms_px),
            "plane_rms_mm": float(self.plane_rms_mm),
            "lens_corrected_inputs": bool(self.lens_corrected_inputs),
            "created_at": self.created_at,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "StereoCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        calibration = cls(
            global_image_size=tuple(int(v) for v in data["global_image_size"]),
            wrist_image_size=tuple(int(v) for v in data["wrist_image_size"]),
            K_global=np.asarray(data["K_global"], np.float64),
            D_global=np.asarray(data.get("D_global", [0, 0, 0, 0, 0]), np.float64),
            K_wrist=np.asarray(data["K_wrist"], np.float64),
            D_wrist=np.asarray(data.get("D_wrist", [0, 0, 0, 0, 0]), np.float64),
            R=np.asarray(data["R"], np.float64),
            T=np.asarray(data["T_mm"], np.float64).reshape(3),
            plane_normal=np.asarray(data["floor_plane_normal"], np.float64),
            plane_offset=float(data["floor_plane_offset_mm"]),
            stereo_rms_px=float(data.get("stereo_rms_px", 0.0)),
            plane_rms_mm=float(data.get("plane_rms_mm", 0.0)),
            lens_corrected_inputs=bool(data.get("lens_corrected_inputs", True)),
            created_at=str(data.get("created_at", "")),
            notes=dict(data.get("notes", {})),
        )
        calibration.validate()
        return calibration

    @classmethod
    def try_load(cls, path: str | Path | None) -> tuple["StereoCalibration | None", str]:
        if path is None:
            return None, "no stereo calibration path configured"
        source = Path(path)
        if not source.exists():
            return None, f"missing {source.name}"
        try:
            return cls.load(source), "ok"
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return None, f"invalid {source.name}: {exc}"


# --------------------------------------------------------------------------- measurement


@dataclass(frozen=True)
class CellHeight:
    id: int
    row: int
    col: int
    height_state: HeightState
    height_mm: float | None
    confidence: float
    valid_3d_points: int
    uncertainty_mm: float | None
    p10_mm: float | None = None
    p90_mm: float | None = None
    mad_mm: float | None = None
    inlier_ratio: float = 0.0
    floor_residual_px: float | None = None  # auxiliary "differs from floor" cue only
    reason: str = ""

    def to_dict(self) -> dict:
        def rounded(value: float | None, digits: int = 2) -> float | None:
            return None if value is None else round(float(value), digits)

        return {
            "id": self.id,
            "row": self.row,
            "col": self.col,
            "height_state": self.height_state.value,
            "height_mm": rounded(self.height_mm),
            "confidence": round(float(self.confidence), 3),
            "valid_3d_points": int(self.valid_3d_points),
            "uncertainty_mm": rounded(self.uncertainty_mm),
            "p10_mm": rounded(self.p10_mm),
            "p90_mm": rounded(self.p90_mm),
            "mad_mm": rounded(self.mad_mm),
            "inlier_ratio": round(float(self.inlier_ratio), 3),
            "floor_residual_px": rounded(self.floor_residual_px),
            "reason": self.reason,
        }


@dataclass
class StereoResult:
    cells: list[CellHeight]
    global_points: np.ndarray  # matched global pixels (valid 3D)
    heights_mm: np.ndarray
    total_matches: int
    calibrated: bool
    status: str


@dataclass(frozen=True)
class HeightCell:
    id: int
    row: int
    col: int
    polygon: np.ndarray


def _rectification_axis(p2: np.ndarray) -> int:
    return 0 if abs(float(p2[0, 3])) >= abs(float(p2[1, 3])) else 1


class StereoHeightEstimator:
    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.feature = str(config.get("feature", "sift")).lower()
        self.max_features = int(config.get("max_features", 6000))
        self.ratio_test = float(config.get("ratio_test", 0.80))
        self.epipolar_tolerance_px = float(config.get("epipolar_tolerance_px", 1.5))
        self.fundamental_ransac_px = float(config.get("fundamental_ransac_px", 1.25))
        self.fundamental_confidence = float(config.get("fundamental_confidence", 0.995))
        self.max_reprojection_px = float(config.get("max_reprojection_px", 2.0))
        self.min_depth_mm = float(config.get("min_depth_mm", 50.0))
        self.max_depth_mm = float(config.get("max_depth_mm", 3000.0))
        self.min_points = int(config.get("min_points_per_cell", 8))
        self.floor_tolerance_mm = float(config.get("floor_tolerance_mm", 4.0))
        self.max_uncertainty_mm = float(config.get("max_uncertainty_mm", 6.0))
        self.min_inlier_ratio = float(config.get("min_inlier_ratio", 0.55))
        self.ransac_tolerance_mm = float(config.get("ransac_tolerance_mm", 3.0))
        self.ransac_iterations = int(config.get("ransac_iterations", 120))
        self.cell_inset_ratio = float(config.get("cell_inset_ratio", 0.12))
        self.disparity_noise_px = float(config.get("disparity_noise_px", 0.5))
        self.significance = float(config.get("height_significance_sigma", 2.5))
        self._detector = self._make_detector()
        self._cache_key: tuple | None = None
        self._rect: dict | None = None

    def _make_detector(self):
        if self.feature == "sift" and hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=self.max_features, contrastThreshold=0.01, edgeThreshold=12)
        self.feature = "orb"
        return cv2.ORB_create(nfeatures=self.max_features, fastThreshold=7, edgeThreshold=15)

    # ---- geometry

    def _prepare(self, calibration: StereoCalibration) -> dict:
        key = (
            tuple(calibration.global_image_size),
            tuple(calibration.wrist_image_size),
            calibration.K_global.tobytes(),
            calibration.K_wrist.tobytes(),
            calibration.R.tobytes(),
            calibration.T.tobytes(),
        )
        if self._cache_key == key and self._rect is not None:
            return self._rect
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            calibration.K_global,
            calibration.D_global,
            calibration.K_wrist,
            calibration.D_wrist,
            tuple(calibration.global_image_size),
            calibration.R,
            calibration.T.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1,
        )
        axis = _rectification_axis(p2)
        baseline_term = float(p2[axis, 3])
        self._rect = {"R1": r1, "R2": r2, "P1": p1, "P2": p2, "Q": q, "axis": axis, "sign": -np.sign(baseline_term) or 1.0}
        self._cache_key = key
        return self._rect

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        # Descriptors need texture only.  Value/max-channel intensity preserves
        # the same printed texture when a surface colour changes; the resulting
        # correspondences are still accepted or rejected exclusively by stereo
        # geometry, reprojection error and 3-D floor distance.
        gray = np.max(image, axis=2).astype(np.uint8) if image.ndim == 3 else np.asarray(image, np.uint8)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    def _features(self, image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        keypoints, descriptors = self._detector.detectAndCompute(self._gray(image), mask)
        if descriptors is None or not keypoints:
            return np.empty((0, 2), np.float32), None
        return np.asarray([kp.pt for kp in keypoints], np.float32), descriptors

    def match(
        self,
        global_frame: np.ndarray,
        wrist_frame: np.ndarray,
        calibration: StereoCalibration,
        global_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Epipolar-constrained correspondences (pixel coordinates in both images)."""
        rect = self._prepare(calibration)
        g_points, g_desc = self._features(global_frame, global_mask)
        w_points, w_desc = self._features(wrist_frame, None)
        if g_desc is None or w_desc is None or len(g_points) < 2 or len(w_points) < 2:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        g_rect = cv2.undistortPoints(g_points.reshape(-1, 1, 2).astype(np.float64), calibration.K_global, calibration.D_global, R=rect["R1"], P=rect["P1"]).reshape(-1, 2)
        w_rect = cv2.undistortPoints(w_points.reshape(-1, 1, 2).astype(np.float64), calibration.K_wrist, calibration.D_wrist, R=rect["R2"], P=rect["P2"]).reshape(-1, 2)
        axis = rect["axis"]
        other = 1 - axis
        # physically valid disparity range from the configured depth range
        focal = float(rect["P1"][axis, axis])
        baseline = abs(float(rect["P2"][axis, 3])) / max(focal, 1e-9)
        d_min = focal * baseline / self.max_depth_mm
        d_max = focal * baseline / self.min_depth_mm
        if self.feature == "sift":
            wd = w_desc.astype(np.float32)
            w_norm = (wd * wd).sum(1)
        else:
            wd = np.unpackbits(w_desc, axis=1).astype(np.float32)
            w_norm = wd.sum(1)
        count_g, count_w = len(g_points), len(w_points)
        best_index = np.full(count_g, -1, np.int64)
        best = np.full(count_g, np.inf)
        second = np.full(count_g, np.inf)
        column_best = np.full(count_w, np.inf)
        column_arg = np.full(count_w, -1, np.int64)
        chunk = 512
        for start in range(0, count_g, chunk):
            stop = min(count_g, start + chunk)
            gr = g_rect[start:stop]
            epipolar = np.abs(gr[:, None, other] - w_rect[None, :, other]) <= self.epipolar_tolerance_px
            disparity = (gr[:, None, axis] - w_rect[None, :, axis]) * rect["sign"]
            valid = epipolar & (disparity >= d_min) & (disparity <= d_max)
            if self.feature == "sift":
                gd = g_desc[start:stop].astype(np.float32)
                distances = np.sqrt(np.maximum((gd * gd).sum(1)[:, None] + w_norm[None, :] - 2.0 * gd @ wd.T, 0.0))
            else:
                gd = np.unpackbits(g_desc[start:stop], axis=1).astype(np.float32)
                distances = gd.sum(1)[:, None] + w_norm[None, :] - 2.0 * gd @ wd.T
            distances = np.where(valid, distances, np.inf)
            if count_w >= 2:
                two = np.argpartition(distances, 1, axis=1)[:, :2]
                d_two = np.take_along_axis(distances, two, axis=1)
                swap = d_two[:, 0] > d_two[:, 1]
                two[swap] = two[swap][:, ::-1]
                d_two[swap] = d_two[swap][:, ::-1]
                best_index[start:stop] = two[:, 0]
                best[start:stop] = d_two[:, 0]
                second[start:stop] = d_two[:, 1]
            else:
                best_index[start:stop] = 0
                best[start:stop] = distances[:, 0]
            rows = np.argmin(distances, axis=0)
            values = distances[rows, np.arange(count_w)]
            improved = values < column_best
            column_best[improved] = values[improved]
            column_arg[improved] = rows[improved] + start
        ratio_ok = np.isfinite(best) & ((~np.isfinite(second)) | (best < self.ratio_test * second))
        safe_index = np.clip(best_index, 0, count_w - 1)
        mutual = column_arg[safe_index] == np.arange(count_g)
        keep = ratio_ok & mutual
        matched_g = g_points[keep]
        matched_w = w_points[safe_index[keep]]

        # Ratio + mutual checks remove descriptor ambiguity; this independent
        # two-view RANSAC removes the remaining geometrically inconsistent
        # correspondences before triangulation.  It never uses colour or
        # brightness as a height signal.
        if len(matched_g) >= 8:
            _, inlier_mask = cv2.findFundamentalMat(
                matched_g,
                matched_w,
                cv2.FM_RANSAC,
                self.fundamental_ransac_px,
                self.fundamental_confidence,
            )
            if inlier_mask is not None:
                inliers = inlier_mask.reshape(-1).astype(bool)
                if int(inliers.sum()) >= 8:
                    matched_g = matched_g[inliers]
                    matched_w = matched_w[inliers]
        return matched_g, matched_w

    def triangulate(self, g_points: np.ndarray, w_points: np.ndarray, calibration: StereoCalibration) -> tuple[np.ndarray, np.ndarray]:
        """Returns (3D points in global camera frame, validity mask)."""
        if not len(g_points):
            return np.empty((0, 3)), np.zeros(0, bool)
        rect = self._prepare(calibration)
        g_rect = cv2.undistortPoints(g_points.reshape(-1, 1, 2).astype(np.float64), calibration.K_global, calibration.D_global, R=rect["R1"], P=rect["P1"]).reshape(-1, 2)
        w_rect = cv2.undistortPoints(w_points.reshape(-1, 1, 2).astype(np.float64), calibration.K_wrist, calibration.D_wrist, R=rect["R2"], P=rect["P2"]).reshape(-1, 2)
        homogeneous = cv2.triangulatePoints(rect["P1"], rect["P2"], g_rect.T, w_rect.T)
        w = homogeneous[3]
        safe = np.abs(w) > 1e-12
        rectified = np.zeros((len(g_points), 3))
        rectified[safe] = (homogeneous[:3, safe] / w[safe]).T
        points = rectified @ rect["R1"]  # R1^T * X_rect, row-vector form
        # reprojection check in both original images
        rvec1 = np.zeros(3)
        proj_g, _ = cv2.projectPoints(points, rvec1, np.zeros(3), calibration.K_global, calibration.D_global)
        rvec2, _ = cv2.Rodrigues(calibration.R)
        proj_w, _ = cv2.projectPoints(points, rvec2, calibration.T, calibration.K_wrist, calibration.D_wrist)
        error = np.maximum(
            np.linalg.norm(proj_g.reshape(-1, 2) - g_points, axis=1),
            np.linalg.norm(proj_w.reshape(-1, 2) - w_points, axis=1),
        )
        depth_w = (points @ calibration.R.T + calibration.T)[:, 2]
        valid = (
            safe
            & (points[:, 2] > self.min_depth_mm)
            & (points[:, 2] < self.max_depth_mm)
            & (depth_w > self.min_depth_mm)
            & (error <= self.max_reprojection_px)
        )
        return points, valid

    # ---- per-cell statistics

    def _ransac_height(self, heights: np.ndarray, seed: int) -> np.ndarray:
        """1-point RANSAC consensus for a single dominant surface height."""
        count = len(heights)
        if count == 0:
            return np.zeros(0, bool)
        rng = np.random.default_rng(seed)
        trials = heights if count <= self.ransac_iterations else heights[rng.choice(count, self.ransac_iterations, replace=False)]
        best_mask = np.zeros(count, bool)
        best_key = (-1, 0.0)
        for hypothesis in trials:
            mask = np.abs(heights - hypothesis) <= self.ransac_tolerance_mm
            spread = float(np.median(np.abs(heights[mask] - np.median(heights[mask])))) if mask.any() else np.inf
            key = (int(mask.sum()), -spread)
            if key > best_key:
                best_key, best_mask = key, mask
        return best_mask

    def _geometric_sigma(self, points: np.ndarray, calibration: StereoCalibration) -> float:
        if not len(points):
            return np.inf
        rect = self._prepare(calibration)
        axis = rect["axis"]
        focal = float(rect["P1"][axis, axis])
        depth = float(np.median(points[:, 2]))
        sigma_z = depth * depth / max(focal * calibration.baseline_mm, 1e-9) * self.disparity_noise_px
        return sigma_z * abs(float(calibration.plane_normal[2]))

    def measure_cell(
        self,
        cell: HeightCell,
        global_points: np.ndarray,
        points_3d: np.ndarray,
        heights: np.ndarray,
        residuals: np.ndarray,
        calibration: StereoCalibration,
    ) -> CellHeight:
        polygon = np.asarray(cell.polygon, np.float32)
        center = polygon.mean(axis=0, keepdims=True)
        inset = (center + (polygon - center) * float(np.clip(1.0 - self.cell_inset_ratio, 0.3, 1.0))).astype(np.float32)
        if len(global_points):
            inside = np.asarray([cv2.pointPolygonTest(inset, (float(x), float(y)), False) >= 0 for x, y in global_points], bool)
        else:
            inside = np.zeros(0, bool)
        values = heights[inside]
        count = int(len(values))
        residual = float(np.median(residuals[inside])) if count else None
        if count < self.min_points:
            return CellHeight(cell.id, cell.row, cell.col, HeightState.UNKNOWN, None, 0.0, count, None, floor_residual_px=residual, reason="INSUFFICIENT_3D_POINTS")
        consensus = self._ransac_height(values, seed=cell.id + 17)
        inliers = values[consensus]
        inlier_ratio = float(len(inliers) / count)
        height = float(np.median(inliers))
        mad = float(1.4826 * np.median(np.abs(inliers - height)))
        statistical = mad / np.sqrt(max(len(inliers), 1))
        geometric = self._geometric_sigma(points_3d[inside][consensus], calibration) / np.sqrt(max(len(inliers), 1)) * 2.0
        uncertainty = float(np.hypot(statistical, geometric) + 0.5 * calibration.plane_rms_mm)
        p10, p90 = (float(v) for v in np.percentile(values, [10, 90]))
        confidence = float(
            np.clip(min(1.0, len(inliers) / (3.0 * self.min_points)) * inlier_ratio * np.exp(-uncertainty / max(self.max_uncertainty_mm, 1e-6)), 0.0, 1.0)
        )
        base = dict(
            id=cell.id,
            row=cell.row,
            col=cell.col,
            valid_3d_points=count,
            uncertainty_mm=uncertainty,
            p10_mm=p10,
            p90_mm=p90,
            mad_mm=mad,
            inlier_ratio=inlier_ratio,
            floor_residual_px=residual,
        )
        if len(inliers) < self.min_points or inlier_ratio < self.min_inlier_ratio:
            return CellHeight(height_state=HeightState.UNKNOWN, height_mm=height, confidence=confidence * 0.5, reason="NO_DOMINANT_SURFACE", **base)
        if uncertainty > self.max_uncertainty_mm:
            return CellHeight(height_state=HeightState.UNKNOWN, height_mm=height, confidence=confidence * 0.5, reason="HIGH_UNCERTAINTY", **base)
        margin = self.significance * uncertainty
        if abs(height) <= self.floor_tolerance_mm:
            state = HeightState.FLOOR
        elif height > self.floor_tolerance_mm and height > margin:
            state = HeightState.ABOVE_FLOOR
        elif height < -self.floor_tolerance_mm and -height > margin:
            state = HeightState.BELOW_FLOOR
        else:
            return CellHeight(height_state=HeightState.UNKNOWN, height_mm=height, confidence=confidence * 0.5, reason="NOT_SIGNIFICANT", **base)
        return CellHeight(height_state=state, height_mm=height, confidence=confidence, reason="OK", **base)

    def process(
        self,
        global_frame: np.ndarray,
        wrist_frame: np.ndarray,
        cells: list[HeightCell],
        calibration: StereoCalibration | None,
    ) -> StereoResult:
        if calibration is None or not calibration.has_floor_plane:
            reason = "NO_STEREO_CALIBRATION" if calibration is None else "NO_FLOOR_PLANE"
            return StereoResult(
                [CellHeight(c.id, c.row, c.col, HeightState.UNCALIBRATED, None, 0.0, 0, None, reason=reason) for c in cells],
                np.empty((0, 2), np.float32),
                np.empty(0),
                0,
                False,
                "UNCALIBRATED",
            )
        active = calibration.scaled(
            (global_frame.shape[1], global_frame.shape[0]),
            (wrist_frame.shape[1], wrist_frame.shape[0]),
        )
        mask = None
        if cells:
            mask = np.zeros(global_frame.shape[:2], np.uint8)
            for cell in cells:
                cv2.fillPoly(mask, [np.round(cell.polygon).astype(np.int32)], 255)
            mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        g_points, w_points = self.match(global_frame, wrist_frame, active, mask)
        points, valid = self.triangulate(g_points, w_points, active)
        g_valid, w_valid, p_valid = g_points[valid], w_points[valid], points[valid]
        heights = active.signed_height(p_valid)
        homography = active.floor_homography_global_to_wrist()
        if len(g_valid):
            predicted = cv2.perspectiveTransform(g_valid.reshape(-1, 1, 2).astype(np.float64), homography).reshape(-1, 2)
            residuals = np.linalg.norm(predicted - w_valid, axis=1)
        else:
            residuals = np.empty(0)
        measurements = [self.measure_cell(cell, g_valid, p_valid, heights, residuals, active) for cell in cells]
        return StereoResult(measurements, g_valid, heights, int(len(g_points)), True, "STEREO")


class TemporalHeightFilter:
    """Independent per-cell temporal median of valid metric heights."""

    def __init__(self, window: int = 5, hold_unknown_frames: int = 3, floor_tolerance_mm: float = 4.0) -> None:
        self.window = max(1, int(window))
        self.hold = max(0, int(hold_unknown_frames))
        self.floor_tolerance_mm = float(floor_tolerance_mm)
        self.history: dict[int, deque] = {}
        self.unknown: dict[int, int] = {}
        self.last: dict[int, CellHeight] = {}

    def reset(self) -> None:
        self.history.clear()
        self.unknown.clear()
        self.last.clear()

    def update(self, cells: list[CellHeight]) -> list[CellHeight]:
        output: list[CellHeight] = []
        for cell in cells:
            if cell.height_state == HeightState.UNCALIBRATED:
                output.append(cell)
                continue
            if cell.height_state == HeightState.UNKNOWN:
                count = self.unknown.get(cell.id, 0) + 1
                self.unknown[cell.id] = count
                previous = self.last.get(cell.id)
                if previous is not None and count <= self.hold:
                    output.append(replace(previous, confidence=round(previous.confidence * 0.7, 3), reason="HELD"))
                else:
                    self.history.pop(cell.id, None)
                    self.last.pop(cell.id, None)
                    output.append(cell)
                continue
            self.unknown[cell.id] = 0
            history = self.history.setdefault(cell.id, deque(maxlen=self.window))
            history.append(cell)
            heights = np.asarray([item.height_mm for item in history], np.float64)
            median = float(np.median(heights))
            representative = min(history, key=lambda item: abs(item.height_mm - median))
            filtered = replace(cell, height_mm=median, height_state=representative.height_state)
            self.last[cell.id] = filtered
            output.append(filtered)
        return output
