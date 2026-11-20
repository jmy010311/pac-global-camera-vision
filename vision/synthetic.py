from __future__ import annotations

import cv2
import numpy as np


def make_partition_frame(
    rows: int = 3,
    cols: int = 3,
    target_corners: np.ndarray | None = None,
    frame_size: tuple[int, int] = (720, 1080),
    occupied_cells: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic perspective scene for offline pipeline checks."""
    rng = np.random.default_rng(2026)
    frame_height, frame_width = frame_size
    frame = np.full((frame_height, frame_width, 3), (42, 48, 52), dtype=np.uint8)
    cell_size = 180
    board_width = cols * cell_size
    board_height = rows * cell_size
    board = np.full((board_height, board_width, 3), (112, 68, 32), dtype=np.uint8)
    cv2.rectangle(board, (5, 5), (board_width - 6, board_height - 6), (25, 25, 25), 14)
    for col in range(1, cols):
        x = int(round(board_width * col / cols))
        cv2.line(board, (x, 8), (x, board_height - 9), (35, 35, 35), 10, cv2.LINE_AA)
    for row in range(1, rows):
        y = int(round(board_height * row / rows))
        cv2.line(board, (8, y), (board_width - 9, y), (35, 35, 35), 10, cv2.LINE_AA)
    for cell_id in occupied_cells or []:
        row, col = divmod(cell_id, cols)
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        left = int(col * cell_size + cell_size * 0.22)
        right = int((col + 1) * cell_size - cell_size * 0.22)
        top = int(row * cell_size + cell_size * 0.20)
        bottom = int((row + 1) * cell_size - cell_size * 0.16)
        cv2.rectangle(board, (left, top), (right, bottom), (220, 225, 230), -1, cv2.LINE_AA)
        for offset in range(-40, 60, 18):
            cv2.line(board, (left, top + offset), (right, bottom + offset), (175, 180, 185), 3, cv2.LINE_AA)

    source = np.array(
        [[0, 0], [board_width - 1, 0], [board_width - 1, board_height - 1], [0, board_height - 1]],
        dtype=np.float32,
    )
    if target_corners is None:
        target = np.array([[230, 100], [880, 135], [930, 650], [160, 625]], dtype=np.float32)
    else:
        target = np.asarray(target_corners, dtype=np.float32).reshape(4, 2)
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(board, matrix, (frame.shape[1], frame.shape[0]))
    mask = cv2.warpPerspective(
        np.full((board_height, board_width), 255, dtype=np.uint8),
        matrix,
        (frame.shape[1], frame.shape[0]),
    )
    frame[mask > 0] = warped[mask > 0]
    noise = rng.normal(0, 2.5, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame, target


def _fractions(count_or_fractions: int | list[float]) -> np.ndarray:
    if isinstance(count_or_fractions, int):
        return np.full(count_or_fractions, 1.0 / count_or_fractions)
    values = np.asarray(count_or_fractions, dtype=np.float64)
    return values / values.sum()


def make_structure_frame(
    rows: int | list[float] = 3,
    cols: int | list[float] = 3,
    *,
    target_corners: np.ndarray | None = None,
    frame_size: tuple[int, int] = (720, 1080),
    board_px: int = 720,
    divider_px: int = 7,
    divider_color: tuple[int, int, int] = (215, 218, 220),
    background: tuple[int, int, int] = (120, 124, 128),
    cell_colors: str = "uniform",
    uniform_color: tuple[int, int, int] = (60, 62, 66),
    bow: float = 0.0,
    shadow: bool = False,
    breaks: list[tuple[str, int, float, float]] | None = None,
    occluders: list[tuple[float, float, float]] | None = None,
    wall_faces: bool = False,
    faded_lines: list[tuple[str, int]] | None = None,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deterministic partition scene for boundary-detection tests.

    ``rows``/``cols`` are counts or relative cell sizes (unequal cells).
    ``bow`` bends every divider and the outer walls (fraction of the board).
    ``breaks`` = (axis 'h'|'v', interior index 1.., start, stop) erases a
    divider piece; ``occluders`` = (u, v, radius) opaque blobs;
    ``faded_lines`` = (axis, interior index) dividers drawn with near-zero
    contrast (used to emulate a frame where they are not observed).
    Returns (frame, outer corners, ground truth dict with normalized line
    positions).
    """

    rng = np.random.default_rng(seed)
    row_f = _fractions(rows)
    col_f = _fractions(cols)
    n_rows, n_cols = len(row_f), len(col_f)
    aspect = n_cols / max(n_rows, 1) if isinstance(cols, int) and isinstance(rows, int) else 1.0
    width = int(round(board_px * min(1.0, aspect))) if aspect <= 1 else board_px
    height = int(round(board_px / aspect)) if aspect > 1 else board_px
    ys = np.concatenate(([0.0], np.cumsum(row_f)))
    xs = np.concatenate(([0.0], np.cumsum(col_f)))
    board = np.zeros((height, width, 3), np.uint8)

    def bend(axis: str, position: float, s: np.ndarray, index: int) -> np.ndarray:
        amplitude = bow * (1.0 if index % 2 == 0 else -0.8)
        return position + amplitude * np.sin(np.pi * s)

    s = np.linspace(0.0, 1.0, 200)

    def curve(axis: str, index: int, positions: np.ndarray) -> np.ndarray:
        offset = bend(axis, positions[index], s, index)
        if axis == "v":
            points = np.column_stack((offset * (width - 1), s * (height - 1)))
        else:
            points = np.column_stack((s * (width - 1), offset * (height - 1)))
        return np.round(points).astype(np.int32)

    # cell interiors: rasterize every cell region bounded by the curved lines
    yy, xx = np.mgrid[0:height, 0:width]
    un = xx / max(width - 1, 1)
    vn = yy / max(height - 1, 1)
    col_index = np.zeros((height, width), np.int32)
    row_index = np.zeros((height, width), np.int32)
    for j in range(1, n_cols):
        boundary = xs[j] + bow * (1.0 if j % 2 == 0 else -0.8) * np.sin(np.pi * vn)
        col_index += (un >= boundary).astype(np.int32)
    for i in range(1, n_rows):
        boundary = ys[i] + bow * (1.0 if i % 2 == 0 else -0.8) * np.sin(np.pi * un)
        row_index += (vn >= boundary).astype(np.int32)
    palette = []
    for cell in range(n_rows * n_cols):
        if cell_colors == "random":
            while True:
                colour = rng.integers(10, 250, 3)
                lab_a = cv2.cvtColor(np.uint8([[colour]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
                lab_d = cv2.cvtColor(np.uint8([[divider_color]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
                if np.linalg.norm(lab_a - lab_d) > 40:
                    break
            palette.append(tuple(int(c) for c in colour))
        else:
            palette.append(uniform_color)
    palette_array = np.asarray(palette, np.uint8)
    board[:] = palette_array[row_index * n_cols + col_index]
    texture = rng.normal(0, 4.0, board.shape)
    board = np.clip(board.astype(np.float32) + texture, 0, 255).astype(np.uint8)

    faded = set(faded_lines or [])
    for axis, positions, count in (("v", xs, n_cols), ("h", ys, n_rows)):
        for index in range(count + 1):
            points = curve(axis, index, positions)
            thickness = divider_px * (2 if index in (0, count) else 1)
            if wall_faces and 0 < index < count:
                shift = np.array([divider_px * 3, 0]) if axis == "v" else np.array([0, divider_px * 3])
                face = np.vstack((points, (points + shift)[::-1]))
                cv2.fillPoly(board, [face.astype(np.int32)], (150, 152, 156), cv2.LINE_AA)
            if (axis, index) in faded:
                continue
            cv2.polylines(board, [points], False, divider_color, thickness, cv2.LINE_AA)
    for axis, index, start, stop in breaks or []:
        positions = xs if axis == "v" else ys
        points = curve(axis, index, positions)
        segment = points[int(start * (len(points) - 1)) : int(stop * (len(points) - 1)) + 1]
        cell_colour = palette[min(n_rows * n_cols - 1, (index if axis == "h" else 0) * n_cols + (index if axis == "v" else 0))]
        cv2.polylines(board, [segment], False, cell_colour, divider_px + 4, cv2.LINE_AA)
    for u, v, radius in occluders or []:
        cv2.circle(board, (int(u * (width - 1)), int(v * (height - 1))), int(radius * width), (95, 70, 60), -1, cv2.LINE_AA)

    frame_height, frame_width = frame_size
    frame = np.full((frame_height, frame_width, 3), background, np.uint8)
    frame = np.clip(frame.astype(np.float32) + rng.normal(0, 3.0, frame.shape), 0, 255).astype(np.uint8)
    source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    if target_corners is None:
        target = np.array([[250, 90], [850, 120], [880, 640], [215, 615]], np.float32)
    else:
        target = np.asarray(target_corners, np.float32).reshape(4, 2)
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(board, matrix, (frame_width, frame_height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((height, width), 255, np.uint8), matrix, (frame_width, frame_height))
    frame[mask > 0] = warped[mask > 0]
    if shadow:
        gy, gx = np.mgrid[0:frame_height, 0:frame_width].astype(np.float32)
        illumination = 0.45 + 0.75 * (gx / frame_width) * (0.6 + 0.4 * gy / frame_height)
        shade = np.ones((frame_height, frame_width), np.float32)
        polygon = np.array(
            [[frame_width * 0.20, 0], [frame_width * 0.55, 0], [frame_width * 0.35, frame_height], [0, frame_height]],
            np.int32,
        )
        cv2.fillPoly(shade, [polygon], 0.35)
        shade = cv2.GaussianBlur(shade, (0, 0), 18)
        frame = np.clip(frame.astype(np.float32) * (illumination * shade)[:, :, None], 0, 255).astype(np.uint8)
    frame = np.clip(frame.astype(np.int16) + rng.normal(0, 2.0, frame.shape).astype(np.int16), 0, 255).astype(np.uint8)
    truth = {"rows": n_rows, "cols": n_cols, "row_positions": ys.tolist(), "col_positions": xs.tolist()}
    return frame, target, truth


# --------------------------------------------------------------------------- stereo scenes


def look_at(center: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, -1.0, 0.0])) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera at ``center`` looking at ``target`` (z forward)."""
    center = np.asarray(center, np.float64)
    forward = np.asarray(target, np.float64) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    return rotation, -rotation @ center


def _texture(size: tuple[int, int], rng: np.random.Generator, tint: tuple[int, int, int], textured: bool) -> np.ndarray:
    height, width = size
    if not textured:
        return np.full((height, width, 3), tint, np.uint8)
    noise = rng.uniform(0, 1, (height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.6)
    noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-6)
    blobs = cv2.GaussianBlur(rng.uniform(0, 1, (height, width)).astype(np.float32), (0, 0), 6.0)
    blobs = (blobs - blobs.min()) / max(float(np.ptp(blobs)), 1e-6)
    pattern = np.clip(0.25 + 0.55 * noise + 0.35 * blobs, 0, 1.2)
    return np.clip(pattern[:, :, None] * np.asarray(tint, np.float32)[None, None, :], 0, 255).astype(np.uint8)


def render_stereo_cells(
    heights_mm: np.ndarray,
    *,
    tints: list[tuple[int, int, int]] | None = None,
    textureless_cells: list[int] | None = None,
    cell_mm: float = 75.0,
    image_size: tuple[int, int] = (1280, 720),
    focal_px: float = 1400.0,
    global_center: tuple[float, float, float] | None = None,
    wrist_center: tuple[float, float, float] | None = None,
    wall_mm: float | None = None,
    texture_px_per_mm: float = 3.0,
    seed: int = 11,
):
    """Render a global and a wrist view of cells whose floors sit at given heights.

    World frame: floor z = 0 (up), box spans x,y in [0, cols*cell] x [0, rows*cell].
    Returns (global_frame, wrist_frame, StereoCalibration, list[HeightCell]).
    """
    from .stereo_height import HeightCell, StereoCalibration

    heights = np.asarray(heights_mm, np.float64)
    rows, cols = heights.shape
    width_mm, height_mm = cols * cell_mm, rows * cell_mm
    target = np.array([width_mm / 2.0, height_mm / 2.0, 0.0])
    c1 = np.array(global_center if global_center is not None else (width_mm / 2.0, height_mm / 2.0 + 60.0, 720.0))
    c2 = np.array(wrist_center if wrist_center is not None else (width_mm / 2.0 + 170.0, height_mm / 2.0 - 50.0, 560.0))
    width, height = image_size
    K = np.array([[focal_px, 0, width / 2.0], [0, focal_px, height / 2.0], [0, 0, 1.0]])
    r1, t1 = look_at(c1, target)
    r2, t2 = look_at(c2, target)
    rng = np.random.default_rng(seed)
    textureless = set(textureless_cells or [])
    tints = tints or [(200, 200, 200)] * (rows * cols)
    scale = texture_px_per_mm
    margin_mm = 90.0

    def homography(rotation: np.ndarray, translation: np.ndarray, z: float, origin: tuple[float, float]) -> np.ndarray:
        # texture pixel (u, v) -> world (origin + (u, v)/scale, z) -> image
        to_world = np.array([[1.0 / scale, 0, origin[0]], [0, 1.0 / scale, origin[1]], [0, 0, 1.0]])
        plane = np.column_stack((rotation[:, 0], rotation[:, 1], rotation[:, 2] * z + translation))
        return K @ plane @ to_world

    def project(rotation: np.ndarray, translation: np.ndarray, points: np.ndarray) -> np.ndarray:
        camera = points @ rotation.T + translation
        pixels = camera @ K.T
        return (pixels[:, :2] / pixels[:, 2:3]).astype(np.float32)

    floor_tex = _texture(
        (int((height_mm + 2 * margin_mm) * scale), int((width_mm + 2 * margin_mm) * scale)), rng, (150, 150, 150), True
    )
    cell_textures = []
    for index in range(rows * cols):
        cell_textures.append(_texture((int(cell_mm * scale), int(cell_mm * scale)), rng, tints[index], index not in textureless))

    frames = []
    for rotation, translation in ((r1, t1), (r2, t2)):
        frame = cv2.warpPerspective(
            floor_tex, homography(rotation, translation, 0.0, (-margin_mm, -margin_mm)), (width, height), flags=cv2.INTER_LINEAR
        )
        order = np.argsort(heights.ravel(), kind="stable")
        for index in order:
            row, col = divmod(int(index), cols)
            z = float(heights[row, col])
            origin = (col * cell_mm, row * cell_mm)
            matrix = homography(rotation, translation, z, origin)
            warped = cv2.warpPerspective(cell_textures[index], matrix, (width, height), flags=cv2.INTER_LINEAR)
            corners = np.array(
                [[origin[0], origin[1], z], [origin[0] + cell_mm, origin[1], z], [origin[0] + cell_mm, origin[1] + cell_mm, z], [origin[0], origin[1] + cell_mm, z]]
            )
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [np.round(project(rotation, translation, corners)).astype(np.int32)], 255)
            frame[mask > 0] = warped[mask > 0]
        noise = rng.normal(0, 1.5, frame.shape)
        frames.append(np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8))

    rotation = r2 @ r1.T
    translation = t2 - rotation @ t1
    normal_c1 = r1 @ np.array([0.0, 0.0, 1.0])
    offset = float(-normal_c1 @ t1)
    calibration = StereoCalibration(
        global_image_size=image_size,
        wrist_image_size=image_size,
        K_global=K.copy(),
        D_global=np.zeros(5),
        K_wrist=K.copy(),
        D_wrist=np.zeros(5),
        R=rotation,
        T=translation,
        plane_normal=normal_c1,
        plane_offset=offset,
        notes={"floor_plane_measured": True, "synthetic": True},
    )
    calibration.validate()
    top = float(heights.max()) if wall_mm is None else float(wall_mm)
    cells = []
    for row in range(rows):
        for col in range(cols):
            corners = np.array(
                [
                    [col * cell_mm, row * cell_mm, top],
                    [(col + 1) * cell_mm, row * cell_mm, top],
                    [(col + 1) * cell_mm, (row + 1) * cell_mm, top],
                    [col * cell_mm, (row + 1) * cell_mm, top],
                ]
            )
            cells.append(HeightCell(row * cols + col, row, col, project(r1, t1, corners)))
    extras = {"K": K, "world_to_global": (r1, t1), "world_to_wrist": (r2, t2), "project": project}
    return frames[0], frames[1], calibration, cells, extras
