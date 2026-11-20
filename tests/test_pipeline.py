from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera.camera_base import CameraInfo
from camera.camera_pair import OpenCVCameraPair
from vision.dense_edges import trace_dense_grid, trace_edge_polyline
from vision.grid import make_grid
from vision.dual_view import DualViewCalibration, PlaneParallaxDetector, SurfaceHeightState
from vision.mesh_rectifier import GridMeshRectifier
from vision.partition_detector import PartitionDetection, TemporalCornerFilter
from vision.perspective import order_corners
from vision.processor import VisionProcessor
from vision.synthetic import make_partition_frame

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def config(self, rows: int = 3, cols: int = 3) -> dict:
        config = json.loads((ROOT / "config" / "vision.json").read_text(encoding="utf-8"))
        config["grid"]["rows"] = rows
        config["grid"]["cols"] = cols
        return config

    def test_corner_order(self) -> None:
        shuffled = np.array([[90, 90], [10, 10], [10, 90], [90, 10]], dtype=np.float32)
        expected = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
        np.testing.assert_allclose(order_corners(shuffled), expected)

    def test_camera_pair_reads_independently(self) -> None:
        class FakeCamera:
            def __init__(self, index: int) -> None:
                self.index = index
                self.counter = 0
                self.released = False

            def open(self) -> CameraInfo:
                return CameraInfo("FAKE", self.index, "TEST", 8, 6, 30.0)

            def read(self) -> tuple[bool, np.ndarray]:
                time.sleep(0.002)
                self.counter += 1
                return True, np.full((6, 8, 3), self.index + self.counter, np.uint8)

            def release(self) -> None:
                self.released = True

        global_camera = FakeCamera(0)
        wrist_camera = FakeCamera(1)
        pair = OpenCVCameraPair(global_camera, wrist_camera)  # type: ignore[arg-type]
        pair.open()
        frames = None
        deadline = time.perf_counter() + 0.5
        while frames is None and time.perf_counter() < deadline:
            ok, frames = pair.read()
            if not ok:
                time.sleep(0.005)
        pair.release()
        self.assertIsNotNone(frames)
        self.assertGreater(global_camera.counter, 0)
        self.assertGreater(wrist_camera.counter, 0)
        self.assertTrue(global_camera.released)
        self.assertTrue(wrist_camera.released)

    def test_parameterized_grid(self) -> None:
        cells = make_grid(800, 800, 4, 4)
        self.assertEqual(len(cells), 16)
        self.assertEqual((cells[-1].id, cells[-1].row, cells[-1].col), (15, 3, 3))

    def test_synthetic_detection_and_scene_state(self) -> None:
        config = self.config()
        frame, expected_corners = make_partition_frame()
        result = VisionProcessor(config).process(frame, 7, 29.8, {"source": "TEST"})
        self.assertTrue(result.detection.detected)
        self.assertIsNotNone(result.rectified)
        self.assertEqual(len(result.scene.cells), 9)
        self.assertEqual(result.scene.cells[-1].id, 8)
        self.assertLess(float(np.mean(np.linalg.norm(result.detection.corners - expected_corners, axis=1))), 30.0)
        parsed = json.loads(result.scene.to_json())
        self.assertTrue(parsed["partition"]["detected"])
        self.assertTrue(parsed["partition"]["grid_automatic"])
        self.assertTrue(parsed["partition"]["mesh_rectified"])
        self.assertEqual(parsed["cells"][0]["state"], "EMPTY")

    def test_scale_and_oblique_view_do_not_require_fixed_pixels(self) -> None:
        poses = [
            np.array([[430, 220], [680, 225], [690, 430], [420, 425]], dtype=np.float32),
            np.array([[380, 170], [720, 155], [745, 455], [350, 470]], dtype=np.float32),
            np.array([[250, 90], [895, 230], [760, 650], [150, 505]], dtype=np.float32),
            np.array([[330, 70], [890, 250], [720, 670], [130, 440]], dtype=np.float32),
        ]
        for corners in poses:
            with self.subTest(corners=corners.tolist()):
                config = self.config(rows=2, cols=2)
                frame, expected = make_partition_frame(3, 3, target_corners=corners)
                result = VisionProcessor(config).process(frame, 0, 30.0, {"source": "TEST"})
                self.assertTrue(result.detection.detected)
                self.assertTrue(result.scene.partition.grid_automatic)
                self.assertEqual((result.scene.partition.rows, result.scene.partition.cols), (3, 3))
                error = float(np.mean(np.linalg.norm(result.detection.corners - expected, axis=1)))
                self.assertLess(error, 35.0)

    def test_auto_topology_detects_4x4_instead_of_fallback_3x3(self) -> None:
        config = self.config(rows=3, cols=3)
        frame, _ = make_partition_frame(4, 4)
        result = VisionProcessor(config).process(frame, 0, 30.0, {"source": "TEST"})
        self.assertTrue(result.scene.partition.grid_automatic)
        self.assertEqual((result.scene.partition.rows, result.scene.partition.cols), (4, 4))
        self.assertEqual(len(result.scene.cells), 16)

    def test_auto_topology_supports_rectangular_3x4_workspace(self) -> None:
        config = self.config(rows=2, cols=2)
        target = np.array([[165, 160], [935, 105], [900, 585], [135, 630]], dtype=np.float32)
        frame, _ = make_partition_frame(3, 4, target_corners=target)
        result = VisionProcessor(config).process(frame, 0, 30.0, {"source": "TEST"})
        self.assertTrue(result.detection.detected)
        self.assertTrue(result.scene.partition.grid_automatic)
        self.assertEqual((result.scene.partition.rows, result.scene.partition.cols), (3, 4))
        self.assertEqual(len(result.scene.cells), 12)

    def test_topology_switch_requires_consecutive_frames(self) -> None:
        processor = VisionProcessor(self.config(rows=3, cols=3))
        frame3, _ = make_partition_frame(3, 3)
        initial = processor.process(frame3, 0, 30.0, {"source": "TEST"})
        self.assertEqual((initial.scene.partition.rows, initial.scene.partition.cols), (3, 3))

        frame4, _ = make_partition_frame(4, 4)
        first = processor.process(frame4, 1, 30.0, {"source": "TEST"})
        second = processor.process(frame4, 2, 30.0, {"source": "TEST"})
        third = processor.process(frame4, 3, 30.0, {"source": "TEST"})
        self.assertEqual((first.scene.partition.rows, first.scene.partition.cols), (3, 3))
        self.assertEqual((second.scene.partition.rows, second.scene.partition.cols), (3, 3))
        self.assertEqual((third.scene.partition.rows, third.scene.partition.cols), (4, 4))

    def test_occupancy_marks_inserted_block(self) -> None:
        config = self.config()
        frame, _ = make_partition_frame(3, 3, occupied_cells=[4])
        result = VisionProcessor(config).process(frame, 0, 30.0, {"source": "TEST"})
        states = [cell.state.value for cell in result.scene.cells]
        self.assertEqual(states[4], "OCCUPIED")
        self.assertTrue(all(state == "EMPTY" for index, state in enumerate(states) if index != 4))

    def test_corner_filter_suppresses_subpixel_jitter(self) -> None:
        rng = np.random.default_rng(2026)
        base = np.array([[100, 100], [500, 100], [500, 500], [100, 500]], dtype=np.float32)
        corner_filter = TemporalCornerFilter(alpha=0.15, median_window=5, deadband_ratio=0.005)
        outputs = []
        raw = []
        for _ in range(30):
            noisy = base + rng.normal(0, 1.3, base.shape).astype(np.float32)
            raw.append(noisy)
            result = corner_filter.update(PartitionDetection(True, noisy, 0.9))
            outputs.append(result.corners)
        raw_std = float(np.std(np.stack(raw)[8:], axis=0).mean())
        output_std = float(np.std(np.stack(outputs)[8:], axis=0).mean())
        self.assertLess(output_std, raw_std * 0.45)

    def test_control_point_mesh_handles_bowed_grid(self) -> None:
        size = 720
        image = np.full((size, size, 3), (112, 68, 32), dtype=np.uint8)
        nodes = np.array(
            [
                [[0, 0], [240, 0], [480, 0], [719, 0]],
                [[0, 240], [226, 231], [492, 248], [719, 240]],
                [[0, 480], [232, 493], [496, 468], [719, 480]],
                [[0, 719], [240, 719], [480, 719], [719, 719]],
            ],
            dtype=np.int32,
        )
        for row in range(4):
            cv2.polylines(image, [nodes[row]], False, (235, 235, 235), 12, cv2.LINE_AA)
        for col in range(4):
            cv2.polylines(image, [np.ascontiguousarray(nodes[:, col])], False, (235, 235, 235), 12, cv2.LINE_AA)
        rectifier = GridMeshRectifier({"enabled": True, "min_confidence": 0.25})
        result = rectifier.rectify(image, 3, 3, [240, 480], [240, 480])
        self.assertTrue(result.applied)
        self.assertGreater(result.confidence, 0.5)
        self.assertLess(float(np.linalg.norm(result.nodes[1, 1] - nodes[1, 1])), 15.0)

    def test_dense_edge_tracer_follows_bowed_pixels(self) -> None:
        gray = np.full((300, 640), 25, dtype=np.uint8)
        xs = np.arange(45, 596, dtype=np.int32)
        ys = np.round(145 + 18 * np.sin((xs - 45) * np.pi / 550)).astype(np.int32)
        cv2.polylines(gray, [np.column_stack((xs, ys))], False, 235, 9, cv2.LINE_AA)
        traced = trace_edge_polyline(
            gray,
            np.asarray([45, 145], np.float32),
            np.asarray([595, 145], np.float32),
            spacing_px=3.0,
            search_radius_px=28,
            extend_px=0.0,
        )
        expected_y = 145 + 18 * np.sin((traced[:, 0] - 45) * np.pi / 550)
        self.assertGreater(len(traced), 150)
        self.assertLess(float(np.mean(np.abs(traced[:, 1] - expected_y))), 8.0)

    def test_dense_grid_contains_many_real_edge_samples(self) -> None:
        config = self.config()
        target = np.asarray([[140, 95], [920, 125], [865, 625], [105, 590]], np.float32)
        frame, _ = make_partition_frame(3, 3, target_corners=target)
        result = VisionProcessor(config).process(frame, 0, 30.0, {"source": "TEST"})
        trace = trace_dense_grid(frame, result)
        self.assertEqual(trace.nodes.shape, (4, 4, 2))
        self.assertEqual(len(trace.edge_samples), 24)
        self.assertGreater(sum(len(path) for path in trace.edge_samples.values()), 1200)
        self.assertTrue(all(len(path) > 20 for path in trace.edge_samples.values()))

    def test_dense_grid_adapts_to_four_by_four_structure(self) -> None:
        config = self.config()
        target = np.asarray([[260, 90], [865, 125], [845, 625], [235, 600]], np.float32)
        frame, _ = make_partition_frame(4, 4, target_corners=target)
        result = VisionProcessor(config).process(frame, 0, 30.0, {"source": "TEST"})
        trace = trace_dense_grid(frame, result)
        self.assertEqual((trace.rows, trace.cols), (4, 4))
        self.assertEqual(trace.nodes.shape, (5, 5, 2))
        self.assertEqual(len(trace.edge_samples), 40)
        self.assertTrue(all(len(path) > 15 for path in trace.edge_samples.values()))
        calibration = DualViewCalibration.from_correspondences(
            trace.nodes,
            target,
            target.copy(),
            (frame.shape[1], frame.shape[0]),
            (frame.shape[1], frame.shape[0]),
            edge_samples=trace.edge_samples,
            rows=trace.rows,
            cols=trace.cols,
        )
        self.assertEqual(len(calibration.cells()), 16)

    def test_dense_curved_edges_define_all_nine_surfaces(self) -> None:
        nodes = np.stack(np.meshgrid(np.linspace(80, 560, 4), np.linspace(60, 420, 4)), axis=-1).astype(np.float32)
        edges: dict[str, np.ndarray] = {}
        for row in range(4):
            for col in range(3):
                start, end = nodes[row, col], nodes[row, col + 1]
                midpoint = (start + end) * 0.5 + np.array([0, (-1) ** (row + col) * 8], np.float32)
                edges[f"h_{row}_{col}"] = np.asarray([start, midpoint, end], np.float32)
        for col in range(4):
            for row in range(3):
                start, end = nodes[row, col], nodes[row + 1, col]
                midpoint = (start + end) * 0.5 + np.array([(-1) ** (row + col) * 7, 0], np.float32)
                edges[f"v_{row}_{col}"] = np.asarray([start, midpoint, end], np.float32)
        global_floor = np.asarray(
            [[90, 70], [550, 70], [550, 410], [90, 410], [320, 70], [550, 240], [320, 410], [90, 240]],
            np.float32,
        )
        wrist_corners = np.asarray([[55, 95], [575, 65], [535, 440], [70, 425]], np.float32)
        homography = cv2.getPerspectiveTransform(wrist_corners, global_floor[:4])
        wrist_floor = cv2.perspectiveTransform(
            global_floor.reshape(-1, 1, 2), np.linalg.inv(homography)
        ).reshape(-1, 2)
        calibration = DualViewCalibration.from_correspondences(
            nodes,
            global_floor,
            wrist_floor,
            (640, 480),
            (640, 480),
            edge_samples=edges,
        )
        cells = calibration.cells()
        self.assertEqual(len(cells), 9)
        self.assertTrue(all(len(cell.polygon) > 4 for cell in cells))
        self.assertLess(calibration.reprojection_error_px, 0.01)

    def test_plane_parallax_marks_only_raised_cell(self) -> None:
        nodes = np.stack(np.meshgrid(np.linspace(80, 560, 4), np.linspace(60, 420, 4)), axis=-1).astype(np.float32)
        global_floor = np.asarray(
            [[90, 70], [550, 70], [550, 410], [90, 410], [320, 70], [550, 240], [320, 410], [90, 240]],
            np.float32,
        )
        wrist_corners = np.asarray([[55, 95], [575, 65], [535, 440], [70, 425]], np.float32)
        homography = cv2.getPerspectiveTransform(wrist_corners, global_floor[:4])
        wrist_floor = cv2.perspectiveTransform(
            global_floor.reshape(-1, 1, 2), np.linalg.inv(homography)
        ).reshape(-1, 2)
        calibration = DualViewCalibration.from_correspondences(
            nodes,
            global_floor,
            wrist_floor,
            (640, 480),
            (640, 480),
        )
        global_points: list[np.ndarray] = []
        wrist_points: list[np.ndarray] = []
        inverse = np.linalg.inv(calibration.homography_wrist_to_global)
        offsets = np.asarray([[-18, -18], [0, -18], [18, -18], [-18, 18], [0, 18], [18, 18]], np.float32)
        for cell in calibration.cells():
            points = np.asarray(cell.center, np.float32) + offsets
            plane_targets = points.copy()
            if cell.id == 4:
                plane_targets -= np.asarray([10.0, 7.0], np.float32)
            wrists = cv2.perspectiveTransform(plane_targets.reshape(-1, 1, 2), inverse).reshape(-1, 2)
            global_points.extend(points)
            wrist_points.extend(wrists)
        detector = PlaneParallaxDetector(
            {
                "min_matches_per_cell": 4,
                "floor_threshold_px": 2.5,
                "raised_threshold_px": 5.0,
                "raised_fraction_threshold": 0.35,
                "cell_inset_ratio": 0.08,
            }
        )
        measurements = detector.classify_correspondences(
            calibration,
            np.asarray(global_points, np.float32),
            np.asarray(wrist_points, np.float32),
        )
        self.assertEqual(measurements[4].state, SurfaceHeightState.ABOVE_FLOOR)
        self.assertTrue(
            all(cell.state == SurfaceHeightState.FLOOR for index, cell in enumerate(measurements) if index != 4)
        )

        varied_global: list[np.ndarray] = []
        varied_wrist: list[np.ndarray] = []
        offsets_by_cell = {
            1: np.asarray([7.0, 5.0], np.float32),
            4: np.asarray([15.0, 10.0], np.float32),
        }
        for cell in calibration.cells():
            points = np.asarray(cell.center, np.float32) + offsets
            plane_targets = points - offsets_by_cell.get(cell.id, np.zeros(2, np.float32))
            wrists = cv2.perspectiveTransform(plane_targets.reshape(-1, 1, 2), inverse).reshape(-1, 2)
            varied_global.extend(points)
            varied_wrist.extend(wrists)
        varied = detector.classify_correspondences(
            calibration,
            np.asarray(varied_global, np.float32),
            np.asarray(varied_wrist, np.float32),
        )
        self.assertEqual(varied[1].state, SurfaceHeightState.ABOVE_FLOOR)
        self.assertEqual(varied[4].state, SurfaceHeightState.ABOVE_FLOOR)
        self.assertGreater(varied[4].parallax_px, varied[1].parallax_px)
        self.assertTrue(
            all(cell.state == SurfaceHeightState.FLOOR for cell in varied if cell.id not in (1, 4))
        )




# ============================================================================ boundary graph

from vision.dense_edges import trace_partition  # noqa: E402
from vision.grid_topology import GridTopology  # noqa: E402
from vision.synthetic import make_structure_frame, render_stereo_cells  # noqa: E402
from vision.stereo_height import (  # noqa: E402
    HeightCell,
    HeightState,
    StereoCalibration,
    StereoHeightEstimator,
    plane_from_points,
)
from dual_camera_main import cell_geometry_signature  # noqa: E402

REAL_4X4 = ROOT / "tests" / "data" / "real_4x4_global.png"
CURRENT_RAW_4X4 = ROOT / "tests" / "data" / "current_4x4_raw.png"


def _config() -> dict:
    return json.loads((ROOT / "config" / "vision.json").read_text(encoding="utf-8"))


def _detect(frame: np.ndarray, frames: int = 1):
    processor = VisionProcessor(_config())
    result = None
    for index in range(frames):
        result = processor.process(frame, index, 30.0, {"source": "TEST"})
    return result


class BoundaryTopologyTests(unittest.TestCase):
    def assert_structure(self, result, truth: dict, tolerance: float = 0.035) -> None:
        topology = result.topology
        self.assertTrue(result.detection.detected)
        self.assertTrue(topology.regular, topology.state)
        self.assertNotEqual(topology.state, "UNCERTAIN")
        self.assertEqual((topology.rows, topology.cols), (truth["rows"], truth["cols"]))
        graph = topology.graph
        np.testing.assert_allclose(graph.col_positions(), truth["col_positions"], atol=tolerance)
        np.testing.assert_allclose(graph.row_positions(), truth["row_positions"], atol=tolerance)
        self.assertEqual(len(result.scene.cells), truth["rows"] * truth["cols"])

    def test_shapes_2x2_3x3_4x4_3x4_4x3(self) -> None:
        for rows, cols in ((2, 2), (3, 3), (4, 4), (3, 4), (4, 3)):
            with self.subTest(shape=(rows, cols)):
                frame, _, truth = make_structure_frame(rows, cols, seed=rows * 10 + cols)
                self.assert_structure(_detect(frame), truth)

    def test_unequal_cell_sizes_are_not_forced_equal(self) -> None:
        frame, _, truth = make_structure_frame([0.4, 0.25, 0.35], [0.15, 0.35, 0.2, 0.3])
        self.assert_structure(_detect(frame), truth)

    def test_curved_outer_and_inner_dividers(self) -> None:
        frame, _, truth = make_structure_frame(4, 4, bow=0.03)
        result = _detect(frame)
        self.assert_structure(result, truth, tolerance=0.045)
        # dense camera polylines really bend (not straight node-to-node segments)
        trace = result.dense_trace
        path = trace.edge_samples["v_1_2"]
        chord = path[-1] - path[0]
        normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
        self.assertGreater(float(np.max(np.abs((path - path[0]) @ normal))), 1.0)

    def test_strong_oblique_perspective(self) -> None:
        corners = np.float32([[380, 110], [700, 120], [1010, 650], [70, 630]])
        frame, _, truth = make_structure_frame(4, 4, target_corners=corners)
        self.assert_structure(_detect(frame), truth)

    def test_every_cell_different_colour_and_brightness(self) -> None:
        frame, _, truth = make_structure_frame(4, 4, cell_colors="random", seed=3)
        self.assert_structure(_detect(frame), truth)

    def test_strong_shadow_and_illumination_gradient(self) -> None:
        frame, _, truth = make_structure_frame(4, 4, cell_colors="random", shadow=True, seed=5)
        self.assert_structure(_detect(frame), truth)

    def test_wall_faces_do_not_create_extra_rows_or_columns(self) -> None:
        frame, _, truth = make_structure_frame(4, 4, wall_faces=True)
        self.assert_structure(_detect(frame), truth, tolerance=0.05)

    def test_broken_and_occluded_boundaries(self) -> None:
        frame, _, truth = make_structure_frame(
            4, 4, breaks=[("v", 1, 0.05, 0.15), ("h", 2, 0.52, 0.62), ("v", 3, 0.6, 0.7)], occluders=[(0.5, 0.5, 0.06)]
        )
        self.assert_structure(_detect(frame), truth)

    def test_missing_segment_is_reported_not_invented(self) -> None:
        frame, _, _ = make_structure_frame(4, 4, breaks=[("h", 2, 0.5, 0.75)])
        result = _detect(frame)
        self.assertFalse(result.topology.regular)
        self.assertEqual(result.topology.state, "UNCERTAIN")
        self.assertEqual(result.topology.face_count, 15)  # two cells merged, no fake edge
        self.assertFalse(result.scene.partition.grid_automatic)

    def test_accumulation_switches_2x2_to_4x4_when_more_boundaries_appear(self) -> None:
        processor = VisionProcessor(_config())
        faded = [("v", 1), ("v", 3), ("h", 1), ("h", 3)]
        weak, _, _ = make_structure_frame(4, 4, faded_lines=faded)
        full, _, _ = make_structure_frame(4, 4)
        shapes = []
        for index, frame in enumerate([weak] * 3 + [full] * 4):
            result = processor.process(frame, index, 30.0, {"source": "TEST"})
            shapes.append((result.topology.rows, result.topology.cols))
        self.assertEqual(shapes[:3], [(2, 2)] * 3)
        self.assertEqual(shapes[-1], (4, 4))
        self.assertLessEqual(shapes.index((4, 4)), 6)

    def test_equal_spacing_fit_is_gone(self) -> None:
        import vision.grid_topology as topology_module

        self.assertFalse(hasattr(topology_module, "_fit_divisions"))
        self.assertFalse(hasattr(topology_module, "_cluster_profile"))

    def test_dense_trace_does_not_depend_on_topology_result(self) -> None:
        frame, _, truth = make_structure_frame(4, 4)
        result = _detect(frame)
        result.topology = GridTopology(rows=2, cols=2, confidence=1.0, automatic=True)  # wrong seed on purpose
        trace = trace_dense_grid(frame, result)
        self.assertEqual((trace.rows, trace.cols), (4, 4))
        self.assertEqual(len(trace.edge_samples), 2 * 4 * 5)
        spacing = np.concatenate([np.linalg.norm(np.diff(p, axis=0), axis=1) for p in trace.edge_samples.values()])
        self.assertLess(float(np.median(spacing)), 4.5)
        self.assertGreater(float(np.median(spacing)), 1.5)
        # published edges run node-to-node; extended search lines are internal only
        for key, path in trace.edge_samples.items():
            axis, i, j = key.split("_")
            i, j = int(i), int(j)
            start = trace.nodes[i, j]
            end = trace.nodes[i, j + 1] if axis == "h" else trace.nodes[i + 1, j]
            np.testing.assert_allclose(path[0], start, atol=1e-3)
            np.testing.assert_allclose(path[-1], end, atol=1e-3)
        self.assertGreater(len(trace.row_lines[0]), len(trace.edge_samples["h_0_0"]) * 4)

    def test_nodes_and_edges_carry_confidence(self) -> None:
        frame, _, _ = make_structure_frame(3, 3)
        trace = trace_partition(frame, _detect(frame).detection.corners)
        self.assertEqual(trace.node_confidence.shape, (4, 4))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in trace.edge_confidence.values()))
        self.assertEqual(trace.node_types[1][1], "X")
        self.assertIn(trace.node_types[0][1], ("T", "X"))


class RealImageRegressionTests(unittest.TestCase):
    """The attached global-camera photo is a 4x4 partition (was detected as 2x2)."""

    @classmethod
    def setUpClass(cls) -> None:
        buffer = np.fromfile(str(REAL_4X4), dtype=np.uint8)
        cls.frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def test_legacy_overlay_photo_is_4x4_not_2x2(self) -> None:
        processor = VisionProcessor(_config())
        states = []
        for index in range(4):
            result = processor.process(self.frame, index, 30.0, {"source": "REAL"})
            self.assertEqual((result.topology.rows, result.topology.cols), (4, 4))
            states.append(result.topology.state)
        # This legacy screenshot already contains thick green/cyan overlays
        # that erase short physical edge segments.  It remains a topology-count
        # regression; confirmation is checked on the clean live capture.
        self.assertIn(states[-1], ("UNCERTAIN", "TENTATIVE", "CONFIRMED"))
        self.assertGreater(result.scene.partition.boundary_points, 1500)
        positions = result.topology.graph.col_positions()
        self.assertTrue(all(0.15 < b - a < 0.35 for a, b in zip(positions, positions[1:])))

    def test_real_photo_without_overlay_marks_does_not_collapse_to_2x2(self) -> None:
        """Remove one old overlay type and ensure topology does not collapse.

        The green polyline lies exactly on real dividers, so removing it also
        damages those dividers.  Exact 4x4 recovery from this synthetic
        inpainting is not required; the clean current-camera frame below is the
        exact-topology regression.
        """
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        masks = {
            "cyan": cv2.inRange(hsv, (85, 120, 150), (105, 255, 255)),
            "green": cv2.inRange(hsv, (35, 80, 80), (85, 255, 255)),
        }
        for name, mask in masks.items():
            with self.subTest(overlay=name):
                clean = cv2.inpaint(self.frame, cv2.dilate(mask, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
                result = _detect(clean)
                self.assertGreaterEqual(result.topology.rows, 3)
                self.assertGreaterEqual(result.topology.cols, 3)

    def test_outer_top_right_corner_is_not_cut(self) -> None:
        result = _detect(self.frame)
        top_right = result.detection.corners[1]
        self.assertGreater(float(top_right[0]), 760.0)  # previous detector returned x=674

    def test_current_raw_frame_uses_raised_rims_not_floor_shadows(self) -> None:
        raw = cv2.imdecode(np.fromfile(str(CURRENT_RAW_4X4), dtype=np.uint8), cv2.IMREAD_COLOR)
        processor = VisionProcessor(_config())
        result = None
        for index in range(4):
            result = processor.process(raw, index, 30.0, {"source": "CURRENT_RAW"})
        assert result is not None and result.dense_trace is not None
        self.assertEqual((result.topology.rows, result.topology.cols), (4, 4))
        self.assertEqual(result.topology.state, "CONFIRMED")
        # The old refiner moved the upper rim down to the floor contact edge
        # (about y=336).  The physical outer rim is near y=312.
        self.assertLess(float(np.median(result.dense_trace.row_lines[0][:, 1])), 325.0)
        positions = result.topology.graph.col_positions()
        self.assertEqual(len(positions), 5)  # four vertical floor shadows must not become dividers


# ============================================================================ stereo height


class RuntimeGridSafetyTests(unittest.TestCase):
    @staticmethod
    def cells(x_lines: list[float], y_lines: list[float]) -> list[HeightCell]:
        result: list[HeightCell] = []
        cols = len(x_lines) - 1
        for row in range(len(y_lines) - 1):
            for col in range(cols):
                polygon = np.asarray(
                    [
                        [x_lines[col], y_lines[row]],
                        [x_lines[col + 1], y_lines[row]],
                        [x_lines[col + 1], y_lines[row + 1]],
                        [x_lines[col], y_lines[row + 1]],
                    ],
                    np.float32,
                )
                result.append(HeightCell(row * cols + col, row, col, polygon))
        return result

    def test_same_cell_count_but_different_dividers_invalidates_height_history(self) -> None:
        first = self.cells([0, 100, 200], [0, 100, 200])
        changed = self.cells([0, 132, 200], [0, 100, 200])
        self.assertNotEqual(cell_geometry_signature(first, 2, 2), cell_geometry_signature(changed, 2, 2))

    def test_subpixel_boundary_jitter_keeps_height_history(self) -> None:
        first = self.cells([0, 100, 200], [0, 100, 200])
        jittered = [HeightCell(cell.id, cell.row, cell.col, cell.polygon + 0.35) for cell in first]
        self.assertEqual(cell_geometry_signature(first, 2, 2), cell_geometry_signature(jittered, 2, 2))


class StereoHeightTests(unittest.TestCase):
    estimator = StereoHeightEstimator({"max_features": 4000})

    def measure(self, heights, **kwargs):
        global_frame, wrist_frame, calibration, cells, _ = render_stereo_cells(np.asarray(heights, float), **kwargs)
        return self.estimator.process(global_frame, wrist_frame, cells, calibration)

    def test_independent_heights_including_centre_cells(self) -> None:
        heights = np.array([[0, 12, 0, 30], [25, 45, 60, 0], [0, 18, 35, 0], [8, 0, 0, 52]], float)
        result = self.measure(heights, cell_mm=70.0)
        for cell in result.cells:
            expected = heights[cell.row, cell.col]
            with self.subTest(cell=cell.id):
                self.assertIsNotNone(cell.height_mm)
                self.assertLess(abs(cell.height_mm - expected), 2.0)
                self.assertGreaterEqual(cell.valid_3d_points, 8)
                self.assertIsNotNone(cell.uncertainty_mm)
                self.assertGreater(cell.confidence, 0.3)
                self.assertEqual(cell.height_state, HeightState.FLOOR if expected == 0 else HeightState.ABOVE_FLOOR)
        centre = {cell.id: cell for cell in result.cells}
        self.assertLess(abs(centre[5].height_mm - 45) + abs(centre[6].height_mm - 60) + abs(centre[10].height_mm - 35), 4.0)

    def test_colour_change_only_gives_equal_heights(self) -> None:
        rng = np.random.default_rng(1)
        tints = [tuple(int(v) for v in rng.integers(40, 255, 3)) for _ in range(9)]
        result = self.measure(np.full((3, 3), 20.0), tints=tints)
        heights = [cell.height_mm for cell in result.cells]
        self.assertTrue(all(h is not None for h in heights))
        self.assertLess(max(heights) - min(heights), 2.0)
        self.assertTrue(all(cell.height_state == HeightState.ABOVE_FLOOR for cell in result.cells))

    def test_same_colour_different_heights_are_separated(self) -> None:
        heights = np.array([[0, 10, 20], [30, 40, 50], [60, 5, 15]], float)
        result = self.measure(heights, tints=[(128, 128, 128)] * 9)
        measured = np.array([cell.height_mm for cell in result.cells])
        np.testing.assert_allclose(measured, heights.ravel(), atol=2.0)
        self.assertEqual(len(np.unique(np.round(measured / 5.0))), 9)

    def test_changing_one_cell_does_not_change_others(self) -> None:
        base = np.array([[0, 20, 0], [10, 30, 0], [0, 0, 40]], float)
        changed = base.copy()
        changed[1, 1] = 55.0
        first = self.measure(base)
        second = self.measure(changed)
        for a, b in zip(first.cells, second.cells):
            if a.id == 4:
                self.assertLess(abs(b.height_mm - 55.0), 2.0)
                continue
            self.assertLess(abs(a.height_mm - b.height_mm), 1.0, f"cell {a.id}")

    def test_textureless_cell_is_unknown_not_guessed(self) -> None:
        result = self.measure(np.array([[0, 20, 0], [0, 35, 0], [0, 0, 0]], float), textureless_cells=[4])
        centre = result.cells[4]
        self.assertEqual(centre.height_state, HeightState.UNKNOWN)
        self.assertIn(centre.reason, ("INSUFFICIENT_3D_POINTS", "NO_DOMINANT_SURFACE", "HIGH_UNCERTAINTY"))
        self.assertEqual(result.cells[1].height_state, HeightState.ABOVE_FLOOR)

    def test_missing_calibration_is_uncalibrated(self) -> None:
        global_frame, wrist_frame, _, cells, _ = render_stereo_cells(np.zeros((2, 2)))
        result = self.estimator.process(global_frame, wrist_frame, cells, None)
        self.assertFalse(result.calibrated)
        self.assertEqual(result.status, "UNCALIBRATED")
        self.assertTrue(all(c.height_state == HeightState.UNCALIBRATED and c.height_mm is None for c in result.cells))

    def test_stereo_calibration_from_checkerboard_views(self) -> None:
        _, _, truth, _, extras = render_stereo_cells(np.zeros((3, 3)))
        K = extras["K"]
        r1, t1 = extras["world_to_global"]
        r2, t2 = extras["world_to_wrist"]
        pattern = np.zeros((9 * 6, 3), np.float32)
        pattern[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 20.0
        rng = np.random.default_rng(4)
        objects, g_list, w_list = [], [], []
        for _ in range(8):
            angle = rng.uniform(-0.35, 0.35, 3)
            rotation, _ = cv2.Rodrigues(angle)
            offset = np.array([rng.uniform(20, 90), rng.uniform(20, 90), rng.uniform(0, 60)])
            world = pattern @ rotation.T + offset
            g = extras["project"](r1, t1, world) + rng.normal(0, 0.1, (len(world), 2))
            w = extras["project"](r2, t2, world) + rng.normal(0, 0.1, (len(world), 2))
            objects.append(pattern)
            g_list.append(g.astype(np.float32))
            w_list.append(w.astype(np.float32))
        floor_world = np.column_stack((rng.uniform(0, 225, 40), rng.uniform(0, 225, 40), np.zeros(40)))
        calibration = StereoCalibration.from_stereo_views(objects, g_list, w_list, K, np.zeros(5), K, np.zeros(5), (1280, 720), (1280, 720))
        np.testing.assert_allclose(calibration.R, truth.R, atol=2e-3)
        np.testing.assert_allclose(calibration.T, truth.T, atol=1.0)
        g_floor = extras["project"](r1, t1, floor_world)
        w_floor = extras["project"](r2, t2, floor_world)
        normal, offset, rms, _ = plane_from_points(calibration.triangulate_pixels(g_floor, w_floor))
        calibration = calibration.with_floor_plane(normal, offset, rms, source="test")
        self.assertLess(abs(calibration.plane_offset - truth.plane_offset), 1.0)
        self.assertGreater(float(calibration.plane_normal @ truth.plane_normal), 0.9999)
        path = ROOT / "tests" / "data" / "_tmp_stereo.json"
        try:
            calibration.save(path)
            loaded = StereoCalibration.load(path)
            np.testing.assert_allclose(loaded.T, calibration.T)
        finally:
            path.unlink(missing_ok=True)

    def test_height_path_never_uses_colour_thresholds(self) -> None:
        for name in ("vision/stereo_height.py", "dual_camera_main.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            for forbidden in ("color_delta_threshold", "COLOR_BGR2HSV", "OccupancyDetector", "COLOR_BGR2LAB"):
                self.assertNotIn(forbidden, source, f"{forbidden} in {name}")


if __name__ == "__main__":
    unittest.main()
