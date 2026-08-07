"""湖蓝色卡纸上的拼图碎片检测与吸取点调试工具。

示例：
    python fragment_vision.py --source 0
    python fragment_vision.py --source test2.jpg
    python fragment_vision.py --source test2.jpg --no-gui --output result.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np


Point = Tuple[int, int]
A4_SHORT_MM = 210.0
A4_LONG_MM = 297.0
PIXELS_PER_MM = 4.0
# 自备图2中存在约 1 cm 的真实短边，只合并明显小于它的毛刺。
MIN_DETECTED_EDGE_MM = 5.0


@dataclass
class DetectConfig:
    hue_tolerance: int = 14
    saturation_min: int = 55
    value_min: int = 35
    open_size: int = 5
    close_size: int = 11
    paper_erode_size: int = 9
    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.15
    min_paper_area_ratio: float = 0.08
    polygon_epsilon_percent: float = 1.5
    safe_radius_px: int = 12


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def learn_blue_hue(hsv: np.ndarray) -> int:
    """从高饱和度区域自动估计湖蓝卡纸的主色相。"""
    hue, saturation, value = cv2.split(hsv)
    colorful = (saturation > 70) & (value > 40)
    if not np.any(colorful):
        raise RuntimeError("画面中没有检测到足够的湖蓝色区域")
    histogram = np.bincount(hue[colorful].ravel(), minlength=180)
    return int(np.argmax(histogram))


def build_blue_background_mask(
    frame: np.ndarray,
    config: DetectConfig,
) -> Tuple[np.ndarray, int]:
    """学习湖蓝背景并返回背景掩膜。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    background_hue = learn_blue_hue(hsv)

    hue_delta = np.abs(hue.astype(np.int16) - background_hue)
    hue_delta = np.minimum(hue_delta, 180 - hue_delta)
    background = (
        (hue_delta <= config.hue_tolerance)
        & (saturation >= config.saturation_min)
        & (value >= config.value_min)
    ).astype(np.uint8) * 255

    background = cv2.morphologyEx(
        background,
        cv2.MORPH_CLOSE,
        np.ones(
            (odd_kernel(config.paper_erode_size),) * 2,
            dtype=np.uint8,
        ),
    )
    return background, background_hue


def order_corners(points: np.ndarray) -> np.ndarray:
    """将四角排序为左上、右上、右下、左下。"""
    points = points.reshape(4, 2).astype(np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    return np.asarray(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def select_a4_quadrilateral(
    candidate_points: np.ndarray,
    paper_contour: np.ndarray,
) -> np.ndarray:
    """从受背景干扰的候选角点中选择最符合A4比例的四边形。"""
    points = candidate_points.reshape(-1, 2)
    if len(points) < 4 or len(points) > 8:
        raise RuntimeError(
            "A4边界候选角数量异常：{}".format(len(points))
        )

    target_aspect_ratio = A4_LONG_MM / A4_SHORT_MM
    contour_area = max(1.0, float(cv2.contourArea(paper_contour)))
    best_score = float("inf")
    best_corners = None

    for selected in combinations(points, 4):
        corners = order_corners(
            np.asarray(selected, dtype=np.float32)
        )
        if len(np.unique(corners, axis=0)) != 4:
            continue
        polygon = corners.reshape(-1, 1, 2)
        if not cv2.isContourConvex(polygon):
            continue

        top = np.linalg.norm(corners[1] - corners[0])
        bottom = np.linalg.norm(corners[2] - corners[3])
        left = np.linalg.norm(corners[3] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])
        mean_width = (top + bottom) * 0.5
        mean_height = (left + right) * 0.5
        if min(mean_width, mean_height) < 1.0:
            continue

        aspect_ratio = max(mean_width, mean_height) / min(
            mean_width,
            mean_height,
        )
        quadrilateral_area = abs(float(cv2.contourArea(polygon)))
        area_coverage = quadrilateral_area / contour_area
        if area_coverage < 0.70 or area_coverage > 1.15:
            continue

        aspect_error = abs(
            np.log(aspect_ratio / target_aspect_ratio)
        )
        coverage_error = abs(1.0 - area_coverage)
        score = 3.0 * aspect_error + coverage_error
        if score < best_score:
            best_score = score
            best_corners = corners

    if best_corners is None or best_score > 0.80:
        raise RuntimeError("没有找到比例合理的A4四角组合")
    return best_corners


def line_intersection(
    first_point: np.ndarray,
    first_direction: np.ndarray,
    second_point: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray:
    matrix = np.column_stack(
        (first_direction, -second_direction)
    ).astype(np.float64)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-6:
        raise RuntimeError("A4相邻边接近平行，无法精确计算角点")
    parameters = np.linalg.solve(
        matrix,
        (second_point - first_point).astype(np.float64),
    )
    return (
        first_point + parameters[0] * first_direction
    ).astype(np.float32)


def fit_contour_side(
    contour_points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 2.0:
        raise RuntimeError("A4候选边过短")

    relative = contour_points - start
    projection = np.sum(relative * direction, axis=1) / (length * length)
    perpendicular = np.abs(
        relative[:, 0] * direction[1]
        - relative[:, 1] * direction[0]
    ) / length
    distance_limit = max(4.0, length * 0.035)
    selected = contour_points[
        (projection >= 0.06)
        & (projection <= 0.94)
        & (perpendicular <= distance_limit)
    ]
    if len(selected) < 8:
        return start.astype(np.float32), (
            direction / length
        ).astype(np.float32)

    fitted = cv2.fitLine(
        selected.astype(np.float32),
        cv2.DIST_HUBER,
        0,
        0.01,
        0.01,
    ).reshape(-1)
    fitted_direction = np.asarray(
        [fitted[0], fitted[1]],
        dtype=np.float32,
    )
    fitted_point = np.asarray(
        [fitted[2], fitted[3]],
        dtype=np.float32,
    )
    return fitted_point, fitted_direction


def refine_a4_corners(
    paper_contour: np.ndarray,
    initial_corners: np.ndarray,
) -> np.ndarray:
    """用纸张四条边的轮廓点拟合直线，再求交点以稳定四角。"""
    initial = order_corners(initial_corners)
    contour_points = paper_contour.reshape(-1, 2).astype(np.float32)
    fitted_lines = []
    for edge_index in range(4):
        fitted_lines.append(
            fit_contour_side(
                contour_points,
                initial[edge_index],
                initial[(edge_index + 1) % 4],
            )
        )

    refined = np.asarray(
        [
            line_intersection(
                fitted_lines[3][0],
                fitted_lines[3][1],
                fitted_lines[0][0],
                fitted_lines[0][1],
            ),
            line_intersection(
                fitted_lines[0][0],
                fitted_lines[0][1],
                fitted_lines[1][0],
                fitted_lines[1][1],
            ),
            line_intersection(
                fitted_lines[1][0],
                fitted_lines[1][1],
                fitted_lines[2][0],
                fitted_lines[2][1],
            ),
            line_intersection(
                fitted_lines[2][0],
                fitted_lines[2][1],
                fitted_lines[3][0],
                fitted_lines[3][1],
            ),
        ],
        dtype=np.float32,
    )

    diagonal = max(
        1.0,
        float(np.linalg.norm(initial[2] - initial[0])),
    )
    maximum_shift = float(
        np.max(np.linalg.norm(refined - initial, axis=1))
    )
    polygon = refined.reshape(-1, 1, 2)
    if (
        maximum_shift > 0.10 * diagonal
        or not cv2.isContourConvex(polygon)
    ):
        return initial
    return refined


def validate_a4_source(
    background: np.ndarray,
    source: np.ndarray,
) -> None:
    """验证候选四边形内部确实主要由湖蓝A4覆盖。"""
    validation_width = 210
    validation_height = 297
    destination = np.asarray(
        [
            [0, 0],
            [validation_width - 1, 0],
            [validation_width - 1, validation_height - 1],
            [0, validation_height - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        source.astype(np.float32),
        destination,
    )
    warped = cv2.warpPerspective(
        background,
        homography,
        (validation_width, validation_height),
        flags=cv2.INTER_NEAREST,
    )
    overall_ratio = float(np.mean(warped > 0))
    band = 12
    inset = 3
    edge_ratios = [
        float(np.mean(warped[inset:band, :] > 0)),
        float(np.mean(warped[-band:-inset, :] > 0)),
        float(np.mean(warped[:, inset:band] > 0)),
        float(np.mean(warped[:, -band:-inset] > 0)),
    ]
    mean_edge_ratio = float(np.mean(edge_ratios))
    # 最外圈会受到透视插值和角点亚像素误差影响，因此边带只用于
    # 排除明显错误的四边形，主体覆盖率仍是主要依据。
    if (
        overall_ratio < 0.72
        or mean_edge_ratio < 0.35
        or min(edge_ratios) < 0.08
    ):
        raise RuntimeError(
            "A4四角不可靠：矫正后纸面覆盖不足 "
            "overall={:.2f}, edges={}".format(
                overall_ratio,
                [round(value, 2) for value in edge_ratios],
            )
        )


def validate_a4_not_clipped(
    paper_contour: np.ndarray,
    frame_shape: Tuple[int, ...],
    margin_ratio: float = 0.006,
) -> None:
    """Reject a paper contour clipped by any camera-frame boundary."""
    frame_height, frame_width = frame_shape[:2]
    margin = max(
        3,
        int(round(min(frame_width, frame_height) * margin_ratio)),
    )
    points = paper_contour.reshape(-1, 2)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    clipped_edges = []
    if minimum[0] <= margin:
        clipped_edges.append("left")
    if maximum[0] >= frame_width - 1 - margin:
        clipped_edges.append("right")
    if minimum[1] <= margin:
        clipped_edges.append("top")
    if maximum[1] >= frame_height - 1 - margin:
        clipped_edges.append("bottom")
    if clipped_edges:
        raise RuntimeError(
            "A4 boundary is outside camera view: "
            + ",".join(clipped_edges)
        )


def rectify_a4(
    frame: np.ndarray,
    config: DetectConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """检测湖蓝 A4 四角并矫正为固定比例的俯视图。"""
    background, background_hue = build_blue_background_mask(frame, config)
    contours, _ = cv2.findContours(
        background,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise RuntimeError("没有找到湖蓝 A4 卡纸")

    paper_contour = max(contours, key=cv2.contourArea)
    validate_a4_not_clipped(paper_contour, frame.shape)
    frame_area = frame.shape[0] * frame.shape[1]
    paper_area_ratio = cv2.contourArea(paper_contour) / frame_area
    if paper_area_ratio < config.min_paper_area_ratio:
        raise RuntimeError(
            "湖蓝 A4 卡纸在画面中占比过小：{:.1%}".format(
                paper_area_ratio
            )
        )

    hull = cv2.convexHull(paper_contour)
    perimeter = cv2.arcLength(hull, True)
    corners = cv2.approxPolyDP(hull, perimeter * 0.015, True)
    if len(corners) == 4:
        source = select_a4_quadrilateral(
            corners,
            paper_contour,
        )
    else:
        candidate_corners = cv2.approxPolyDP(
            hull,
            perimeter * 0.010,
            True,
        )
        source = select_a4_quadrilateral(
            candidate_corners,
            paper_contour,
        )
    source = refine_a4_corners(paper_contour, source)
    validate_a4_source(background, source)

    # 固定纸面坐标系：
    # X始终沿A4短边（0~210 mm），Y始终沿A4长边（0~297 mm）。
    # 横向画面统一顺时针旋转为规范竖向画面。
    image_ordered_corners = source.copy()
    top_width = np.linalg.norm(source[1] - source[0])
    bottom_width = np.linalg.norm(source[2] - source[3])
    left_height = np.linalg.norm(source[3] - source[0])
    right_height = np.linalg.norm(source[2] - source[1])
    portrait = (left_height + right_height) >= (top_width + bottom_width)
    input_orientation = "portrait" if portrait else "landscape"
    if not portrait:
        # 原图左下、左上、右上、右下分别映射到规范图
        # 左上、右上、右下、左下。
        source = source[[3, 0, 1, 2]]

    width_mm = A4_SHORT_MM
    height_mm = A4_LONG_MM
    width_px = int(round(width_mm * PIXELS_PER_MM))
    height_px = int(round(height_mm * PIXELS_PER_MM))
    destination = np.asarray(
        [
            [0, 0],
            [width_px - 1, 0],
            [width_px - 1, height_px - 1],
            [0, height_px - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    rectified = cv2.warpPerspective(
        frame,
        homography,
        (width_px, height_px),
    )
    calibration = {
        "paper_corners_px": np.round(source, 1).tolist(),
        "image_ordered_corners_px": np.round(
            image_ordered_corners,
            1,
        ).tolist(),
        "paper_size_mm": [width_mm, height_mm],
        "pixels_per_mm": PIXELS_PER_MM,
        "homography": np.round(homography, 8).tolist(),
        "background_hue_before_rectify": background_hue,
        "input_orientation": input_orientation,
        "canonical_orientation": "portrait",
        "coordinate_system": {
            "origin": "canonical_top_left",
            "x_axis": "A4_short_edge_0_to_210_mm",
            "y_axis": "A4_long_edge_0_to_297_mm",
        },
    }
    return rectified, calibration


def build_piece_mask(
    frame: np.ndarray,
    config: DetectConfig,
) -> Tuple[np.ndarray, int]:
    """提取矫正后湖蓝卡纸范围内的非湖蓝碎片。"""
    background, background_hue = build_blue_background_mask(frame, config)
    contours, _ = cv2.findContours(
        background,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise RuntimeError("没有找到湖蓝卡纸区域")

    paper_contour = max(contours, key=cv2.contourArea)
    paper_support = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(
        paper_support,
        [paper_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )
    frame_height, frame_width = frame.shape[:2]
    filled_ratio = float(np.mean(paper_support > 0))
    horizontal_inset = max(3, int(round(frame_width * 0.008)))
    vertical_inset = max(3, int(round(frame_height * 0.008)))
    horizontal_band = max(7, int(round(frame_width * 0.025)))
    vertical_band = max(7, int(round(frame_height * 0.025)))
    edge_ratios = [
        float(
            np.mean(
                paper_support[
                    vertical_inset:
                    vertical_inset + vertical_band,
                    :,
                ] > 0
            )
        ),
        float(
            np.mean(
                paper_support[
                    -vertical_inset - vertical_band:
                    -vertical_inset,
                    :,
                ] > 0
            )
        ),
        float(
            np.mean(
                paper_support[
                    :,
                    horizontal_inset:
                    horizontal_inset + horizontal_band,
                ] > 0
            )
        ),
        float(
            np.mean(
                paper_support[
                    :,
                    -horizontal_inset - horizontal_band:
                    -horizontal_inset,
                ] > 0
            )
        ),
    ]
    mean_edge_ratio = float(np.mean(edge_ratios))
    extremely_low_edges = sum(
        value < 0.10 for value in edge_ratios
    )
    if (
        filled_ratio < 0.75
        or mean_edge_ratio < 0.35
        or extremely_low_edges >= 2
    ):
        raise RuntimeError(
            "矫正后的A4边界不完整，拒绝输出坐标 "
            "fill={:.2f}, edges={}".format(
                filled_ratio,
                [round(value, 2) for value in edge_ratios],
            )
        )
    paper_support = cv2.erode(
        paper_support,
        np.ones(
            (odd_kernel(config.paper_erode_size),) * 2,
            dtype=np.uint8,
        ),
    )

    mask = cv2.bitwise_and(
        cv2.bitwise_not(background),
        paper_support,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((odd_kernel(config.open_size),) * 2, dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((odd_kernel(config.close_size),) * 2, dtype=np.uint8),
    )
    return mask, background_hue


def contour_centroid(contour: np.ndarray) -> Tuple[float, float]:
    """将碎片视为厚度和密度均匀的薄片，计算其面积重心。"""
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-8:
        x, y, width, height = cv2.boundingRect(contour)
        return x + width / 2.0, y + height / 2.0
    return (
        moments["m10"] / moments["m00"],
        moments["m01"] / moments["m00"],
    )


def remove_false_short_edges(
    polygon: np.ndarray,
    reference_contour: np.ndarray,
    minimum_length_px: float,
) -> np.ndarray:
    """合并明显短于题目下限的毛刺边，并尽量保持原轮廓面积。"""
    points = polygon.reshape(-1, 2)
    reference_area = abs(float(cv2.contourArea(reference_contour)))
    while len(points) > 3:
        lengths = np.linalg.norm(
            np.roll(points, -1, axis=0) - points,
            axis=1,
        )
        edge_index = int(np.argmin(lengths))
        if float(lengths[edge_index]) >= minimum_length_px:
            break

        next_index = (edge_index + 1) % len(points)
        candidates = [
            np.delete(points, edge_index, axis=0),
            np.delete(points, next_index, axis=0),
        ]
        points = min(
            candidates,
            key=lambda candidate: abs(
                abs(
                    float(
                        cv2.contourArea(
                            candidate.astype(np.float32).reshape(-1, 1, 2)
                        )
                    )
                )
                - reference_area
            ),
        )
    return points.astype(np.int32).reshape(-1, 1, 2)


def remove_nearly_collinear_vertices(
    polygon: np.ndarray,
    minimum_interior_angle_deg: float = 165.0,
) -> np.ndarray:
    """Merge vertices that only split one almost-straight physical edge."""
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    while len(points) > 3:
        angles = []
        for vertex_id, vertex in enumerate(points):
            previous_vector = points[vertex_id - 1] - vertex
            next_vector = points[
                (vertex_id + 1) % len(points)
            ] - vertex
            denominator = max(
                1e-6,
                float(
                    np.linalg.norm(previous_vector)
                    * np.linalg.norm(next_vector)
                ),
            )
            cosine_value = np.clip(
                float(
                    np.dot(previous_vector, next_vector)
                )
                / denominator,
                -1.0,
                1.0,
            )
            angles.append(
                float(np.degrees(np.arccos(cosine_value)))
            )
        vertex_id = int(np.argmax(angles))
        if angles[vertex_id] < minimum_interior_angle_deg:
            break
        points = np.delete(points, vertex_id, axis=0)
    return np.round(points).astype(np.int32).reshape(-1, 1, 2)


def polygon_contour_fit_score(
    polygon: np.ndarray,
    reference_contour: np.ndarray,
) -> float:
    """Score straight-edge fit while penalizing area distortion."""
    vertices = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    contour_points = np.asarray(
        reference_contour,
        dtype=np.float32,
    ).reshape(-1, 2)
    if len(vertices) < 3 or len(contour_points) < 3:
        return float("inf")

    distances = []
    for edge_id in range(len(vertices)):
        start = vertices[edge_id]
        vector = vertices[(edge_id + 1) % len(vertices)] - start
        length_squared = max(float(np.dot(vector, vector)), 1e-6)
        projection = np.clip(
            ((contour_points - start) @ vector) / length_squared,
            0.0,
            1.0,
        )
        nearest = start + projection[:, None] * vector
        distances.append(
            np.linalg.norm(contour_points - nearest, axis=1)
        )
    mean_distance = float(
        np.mean(np.min(np.stack(distances, axis=1), axis=1))
    )
    reference_area = max(
        1.0,
        abs(float(cv2.contourArea(reference_contour))),
    )
    polygon_area = abs(float(cv2.contourArea(polygon)))
    area_error = abs(polygon_area / reference_area - 1.0)
    return mean_distance + area_error * 20.0


def generate_polygon_candidates(
    contour: np.ndarray,
    config: DetectConfig,
    primary_polygon: np.ndarray,
    maximum_candidates: int = 2,
) -> List[Tuple[np.ndarray, float]]:
    """Keep the primary polygon and one strong alternate vertex model."""
    perimeter = cv2.arcLength(contour, True)
    minimum_length_px = MIN_DETECTED_EDGE_MM * PIXELS_PER_MM
    primary = np.asarray(primary_polygon, dtype=np.int32).reshape(-1, 1, 2)
    primary_score = polygon_contour_fit_score(primary, contour)
    best_by_vertex_count: Dict[int, Tuple[np.ndarray, float]] = {}

    for epsilon_percent in np.arange(0.6, 3.01, 0.2):
        candidate = cv2.approxPolyDP(
            contour,
            perimeter * float(epsilon_percent) / 100.0,
            True,
        )
        candidate = remove_false_short_edges(
            candidate,
            contour,
            minimum_length_px,
        )
        candidate = remove_nearly_collinear_vertices(candidate)
        vertex_count = len(candidate)
        if not 3 <= vertex_count <= 6:
            continue
        if not cv2.isContourConvex(candidate):
            continue
        lengths = np.linalg.norm(
            np.roll(
                candidate.reshape(-1, 2),
                -1,
                axis=0,
            )
            - candidate.reshape(-1, 2),
            axis=1,
        )
        if float(np.min(lengths)) < minimum_length_px:
            continue
        score = polygon_contour_fit_score(candidate, contour)
        previous = best_by_vertex_count.get(vertex_count)
        if previous is None or score < previous[1]:
            best_by_vertex_count[vertex_count] = (
                candidate.astype(np.int32),
                score,
            )

    candidates = [(primary, primary_score)]
    alternatives = sorted(
        (
            value
            for vertex_count, value in best_by_vertex_count.items()
            if vertex_count != len(primary)
        ),
        key=lambda item: item[1],
    )
    candidates.extend(
        alternatives[: max(0, maximum_candidates - 1)]
    )
    return candidates


def sharpen_polygon_from_contour(
    polygon: np.ndarray,
    reference_contour: np.ndarray,
    maximum_corner_shift_px: float = 16.0,
) -> np.ndarray:
    """Fit each contour side separately and intersect lines for sharp corners."""
    vertices = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    contour_points = np.asarray(
        reference_contour,
        dtype=np.float32,
    ).reshape(-1, 2)
    edge_count = len(vertices)
    if edge_count < 3 or len(contour_points) < edge_count * 4:
        return polygon

    # Assign every raw contour point to its nearest initial polygon side.
    distances = []
    for edge_id in range(edge_count):
        start = vertices[edge_id]
        vector = vertices[(edge_id + 1) % edge_count] - start
        length_squared = max(float(np.dot(vector, vector)), 1e-6)
        projection = np.clip(
            ((contour_points - start) @ vector) / length_squared,
            0.0,
            1.0,
        )
        nearest = start + projection[:, None] * vector
        distances.append(
            np.sum((contour_points - nearest) ** 2, axis=1)
        )
    assignments = np.argmin(np.stack(distances, axis=1), axis=1)

    fitted_lines = []
    for edge_id in range(edge_count):
        start = vertices[edge_id]
        vector = vertices[(edge_id + 1) % edge_count] - start
        length = float(np.linalg.norm(vector))
        if length < 2.0:
            return polygon
        direction = vector / length
        side_points = contour_points[assignments == edge_id]
        if len(side_points) < 4:
            return polygon

        # Corner pixels are rounded and less reliable; fit the central side.
        along = (side_points - start) @ direction
        central = side_points[
            (along >= length * 0.10) & (along <= length * 0.90)
        ]
        if len(central) >= 4:
            side_points = central
        vx, vy, x0, y0 = cv2.fitLine(
            side_points.reshape(-1, 1, 2),
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        ).reshape(-1)
        fitted_direction = np.asarray([vx, vy], dtype=np.float32)
        if float(np.dot(fitted_direction, direction)) < 0:
            fitted_direction *= -1.0
        fitted_lines.append(
            (
                np.asarray([x0, y0], dtype=np.float32),
                fitted_direction,
            )
        )

    sharp_vertices = []
    for vertex_id in range(edge_count):
        previous_point, previous_direction = fitted_lines[
            (vertex_id - 1) % edge_count
        ]
        current_point, current_direction = fitted_lines[vertex_id]
        denominator = (
            previous_direction[0] * current_direction[1]
            - previous_direction[1] * current_direction[0]
        )
        if abs(float(denominator)) < 0.08:
            return polygon
        delta = current_point - previous_point
        distance = (
            delta[0] * current_direction[1]
            - delta[1] * current_direction[0]
        ) / denominator
        corner = previous_point + distance * previous_direction
        if (
            not np.all(np.isfinite(corner))
            or float(np.linalg.norm(corner - vertices[vertex_id]))
            > maximum_corner_shift_px
        ):
            return polygon
        sharp_vertices.append(corner)

    sharp = np.asarray(sharp_vertices, dtype=np.float32).reshape(-1, 1, 2)
    original_area = abs(float(cv2.contourArea(vertices)))
    sharp_area = abs(float(cv2.contourArea(sharp)))
    if (
        original_area < 1.0
        or not 0.85 <= sharp_area / original_area <= 1.15
        or not cv2.isContourConvex(sharp)
    ):
        return polygon
    return np.round(sharp).astype(np.int32)


def choose_pickup_point(
    contour: np.ndarray,
    centroid: Tuple[float, float],
    safe_radius_px: int,
) -> Tuple[Point, float, str]:
    """优先吸取面积重心；重心不安全时选择最近的安全位置。"""
    x, y, width, height = cv2.boundingRect(contour)
    padding = max(2, safe_radius_px + 2)
    x0 = x - padding
    y0 = y - padding

    local_contour = contour.copy()
    local_contour[:, 0, 0] -= x0
    local_contour[:, 0, 1] -= y0
    local_mask = np.zeros(
        (height + 2 * padding, width + 2 * padding),
        dtype=np.uint8,
    )
    cv2.drawContours(
        local_mask,
        [local_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )
    distance = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)

    local_cx = int(round(centroid[0] - x0))
    local_cy = int(round(centroid[1] - y0))
    centroid_clearance = float(distance[local_cy, local_cx])
    if centroid_clearance >= safe_radius_px:
        return (
            (int(round(centroid[0])), int(round(centroid[1]))),
            centroid_clearance,
            "centroid",
        )

    safe_y, safe_x = np.where(distance >= safe_radius_px)
    if len(safe_x) > 0:
        squared_distance = (
            (safe_x - (centroid[0] - x0)) ** 2
            + (safe_y - (centroid[1] - y0)) ** 2
        )
        best = int(np.argmin(squared_distance))
        px = int(safe_x[best])
        py = int(safe_y[best])
        return (
            (px + x0, py + y0),
            float(distance[py, px]),
            "nearest_safe",
        )

    _, maximum, _, location = cv2.minMaxLoc(distance)
    return (
        (location[0] + x0, location[1] + y0),
        float(maximum),
        "max_clearance",
    )


def detect_pieces(
    frame: np.ndarray,
    config: DetectConfig,
) -> Tuple[np.ndarray, List[Dict[str, object]], int]:
    mask, background_hue = build_piece_mask(frame, config)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width
    min_area = frame_area * config.min_area_ratio
    max_area = frame_area * config.max_area_ratio

    pieces = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        boundary_margin = max(2, config.paper_erode_size)
        boundary_artifact = (
            height > frame_height * 0.75
            or width > frame_width * 0.85
            or width / float(height) < 0.18
            or x <= boundary_margin
            or y <= boundary_margin
            or x + width >= frame_width - boundary_margin
            or y + height >= frame_height - boundary_margin
        )
        if boundary_artifact:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = perimeter * config.polygon_epsilon_percent / 100.0
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        polygon = remove_false_short_edges(
            polygon,
            contour,
            MIN_DETECTED_EDGE_MM * PIXELS_PER_MM,
        )
        polygon = sharpen_polygon_from_contour(
            polygon,
            contour,
        )
        # Corner sharpening can create a tiny extra segment near an
        # otherwise valid tip. Remove those post-sharpen artifacts too.
        polygon = remove_false_short_edges(
            polygon,
            contour,
            MIN_DETECTED_EDGE_MM * PIXELS_PER_MM,
        )
        polygon = remove_nearly_collinear_vertices(polygon)
        polygon_candidates = generate_polygon_candidates(
            contour,
            config,
            polygon,
        )
        centroid = contour_centroid(contour)
        pickup, clearance, pickup_mode = choose_pickup_point(
            contour,
            centroid,
            config.safe_radius_px,
        )
        pieces.append(
            {
                "_contour": contour,
                "_polygon": polygon,
                "_polygon_candidates": polygon_candidates,
                "area_px2": area,
                "centroid_px": centroid,
                "pickup_px": pickup,
                "clearance_px": clearance,
                "pickup_mode": pickup_mode,
            }
        )

    pieces.sort(
        key=lambda piece: (
            piece["centroid_px"][1],
            piece["centroid_px"][0],
        )
    )

    filled_mask = np.zeros(mask.shape, dtype=np.uint8)
    results = []
    for piece_id, piece in enumerate(pieces):
        cv2.drawContours(
            filled_mask,
            [piece["_contour"]],
            -1,
            255,
            thickness=cv2.FILLED,
        )
        results.append(
            {
                "id": piece_id,
                "area_px2": round(piece["area_px2"], 1),
                "centroid_px": [
                    round(piece["centroid_px"][0], 1),
                    round(piece["centroid_px"][1], 1),
                ],
                "pickup_px": list(piece["pickup_px"]),
                "clearance_px": round(piece["clearance_px"], 1),
                "pickup_mode": piece["pickup_mode"],
                "vertices_px": piece["_polygon"].reshape(-1, 2).tolist(),
                "polygon_candidates_px": [
                    candidate.reshape(-1, 2).tolist()
                    for candidate, _ in piece[
                        "_polygon_candidates"
                    ]
                ],
                "polygon_candidate_scores": [
                    round(float(score), 4)
                    for _, score in piece["_polygon_candidates"]
                ],
                "_contour": piece["_contour"],
                "_polygon": piece["_polygon"],
            }
        )
    return filled_mask, results, background_hue


def draw_results(
    frame: np.ndarray,
    pieces: List[Dict[str, object]],
    background_hue: int,
) -> np.ndarray:
    output = frame.copy()
    paper_width_mm = frame.shape[1] / PIXELS_PER_MM
    paper_height_mm = frame.shape[0] / PIXELS_PER_MM
    cv2.putText(
        output,
        "A4 {:.0f}x{:.0f} X=short Y=long O=TL hue={} P={}".format(
            paper_width_mm,
            paper_height_mm,
            background_hue,
            len(pieces),
        ),
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for piece in pieces:
        pickup = tuple(piece["pickup_px"])
        cv2.drawContours(output, [piece["_contour"]], -1, (0, 200, 0), 2)
        cv2.drawContours(output, [piece["_polygon"]], -1, (255, 180, 0), 2)
        cv2.circle(output, pickup, 4, (0, 0, 255), -1)
        cv2.circle(
            output,
            pickup,
            max(1, int(round(piece["clearance_px"]))),
            (0, 0, 255),
            1,
        )
        pickup_mm = (
            piece["pickup_px"][0] / PIXELS_PER_MM,
            piece["pickup_px"][1] / PIXELS_PER_MM,
        )
        label = "P{} S({:.1f},{:.1f})mm".format(
            piece["id"],
            pickup_mm[0],
            pickup_mm[1],
        )
        font_scale = 0.42
        thickness = 1
        label_width = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0][0]
        label_x = min(
            max(4, pickup[0] + 8),
            max(4, output.shape[1] - label_width - 5),
        )
        label_y = min(
            max(18, pickup[1] - 8),
            output.shape[0] - 5,
        )
        position = (label_x, label_y)
        cv2.putText(
            output,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return output


def serializable_pieces(
    pieces: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    return [
        {key: value for key, value in piece.items() if not key.startswith("_")}
        for piece in pieces
    ]


def create_controls(config: DetectConfig) -> None:
    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
    noop = lambda _value: None
    cv2.createTrackbar(
        "hue tolerance",
        "controls",
        config.hue_tolerance,
        40,
        noop,
    )
    cv2.createTrackbar(
        "saturation min",
        "controls",
        config.saturation_min,
        255,
        noop,
    )
    cv2.createTrackbar(
        "value min",
        "controls",
        config.value_min,
        255,
        noop,
    )
    cv2.createTrackbar("open", "controls", config.open_size, 31, noop)
    cv2.createTrackbar("close", "controls", config.close_size, 51, noop)
    cv2.createTrackbar(
        "polygon eps x10",
        "controls",
        int(round(config.polygon_epsilon_percent * 10)),
        80,
        noop,
    )
    cv2.createTrackbar(
        "safe radius",
        "controls",
        config.safe_radius_px,
        100,
        noop,
    )
    cv2.createTrackbar(
        "min area permille",
        "controls",
        int(round(config.min_area_ratio * 1000)),
        50,
        noop,
    )
    cv2.createTrackbar(
        "max area percent",
        "controls",
        int(round(config.max_area_ratio * 100)),
        50,
        noop,
    )


def read_controls() -> DetectConfig:
    return DetectConfig(
        hue_tolerance=max(
            1,
            cv2.getTrackbarPos("hue tolerance", "controls"),
        ),
        saturation_min=cv2.getTrackbarPos("saturation min", "controls"),
        value_min=cv2.getTrackbarPos("value min", "controls"),
        open_size=max(1, cv2.getTrackbarPos("open", "controls")),
        close_size=max(1, cv2.getTrackbarPos("close", "controls")),
        polygon_epsilon_percent=max(
            0.1,
            cv2.getTrackbarPos("polygon eps x10", "controls") / 10.0,
        ),
        safe_radius_px=max(
            1,
            cv2.getTrackbarPos("safe radius", "controls"),
        ),
        min_area_ratio=max(
            0.001,
            cv2.getTrackbarPos("min area permille", "controls") / 1000.0,
        ),
        max_area_ratio=max(
            0.01,
            cv2.getTrackbarPos("max area percent", "controls") / 100.0,
        ),
    )


def make_payload(
    frame: np.ndarray,
    pieces: List[Dict[str, object]],
    background_hue: int,
    calibration: Dict[str, object],
) -> Dict[str, object]:
    output_pieces = serializable_pieces(pieces)
    for piece in output_pieces:
        piece["centroid_mm"] = [
            round(value / PIXELS_PER_MM, 2)
            for value in piece["centroid_px"]
        ]
        piece["pickup_mm"] = [
            round(value / PIXELS_PER_MM, 2)
            for value in piece["pickup_px"]
        ]
        piece["clearance_mm"] = round(
            piece["clearance_px"] / PIXELS_PER_MM,
            2,
        )
        piece["vertices_mm"] = [
            [
                round(vertex[0] / PIXELS_PER_MM, 2),
                round(vertex[1] / PIXELS_PER_MM, 2),
            ]
            for vertex in piece["vertices_px"]
        ]
        piece["polygon_candidates_mm"] = [
            [
                [
                    round(vertex[0] / PIXELS_PER_MM, 2),
                    round(vertex[1] / PIXELS_PER_MM, 2),
                ]
                for vertex in candidate
            ]
            for candidate in piece.get(
                "polygon_candidates_px",
                [piece["vertices_px"]],
            )
        ]
        vertices_mm = np.asarray(piece["vertices_mm"], dtype=np.float32)
        piece["edge_lengths_mm"] = np.round(
            np.linalg.norm(
                np.roll(vertices_mm, -1, axis=0) - vertices_mm,
                axis=1,
            ),
            2,
        ).tolist()
    return {
        "rectified_size_px": [frame.shape[1], frame.shape[0]],
        "background_hue": background_hue,
        "calibration": calibration,
        "pieces": output_pieces,
    }


def process_frame(
    frame: np.ndarray,
    config: DetectConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    rectified, calibration = rectify_a4(frame, config)
    mask, pieces, background_hue = detect_pieces(rectified, config)
    annotated = draw_results(rectified, pieces, background_hue)
    payload = make_payload(
        rectified,
        pieces,
        background_hue,
        calibration,
    )
    return mask, annotated, payload


def resize_for_debug(
    frame: np.ndarray,
    maximum_side: int = 1500,
) -> np.ndarray:
    if max(frame.shape[:2]) <= maximum_side:
        return frame
    scale = maximum_side / float(max(frame.shape[:2]))
    return cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def run_image(
    image_path: str,
    no_gui: bool,
    output_path: str,
) -> int:
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError("无法读取图片：{}".format(image_path))
    frame = resize_for_debug(frame)
    load_ms = (time.perf_counter() - load_start) * 1000.0
    config = DetectConfig()

    if no_gui:
        vision_start = time.perf_counter()
        mask, annotated, payload = process_frame(frame, config)
        vision_ms = (time.perf_counter() - vision_start) * 1000.0
        if output_path:
            cv2.imwrite(output_path, annotated)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        payload["timing_ms"] = {
            "image_load": round(load_ms, 2),
            "vision": round(vision_ms, 2),
            "total": round(total_ms, 2),
        }
        print(
            "耗时：读取 {:.2f} ms，视觉 {:.2f} ms，总计 {:.2f} ms".format(
                load_ms,
                vision_ms,
                total_ms,
            ),
            file=sys.stderr,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    create_controls(config)
    while True:
        vision_start = time.perf_counter()
        mask, annotated, payload = process_frame(frame, read_controls())
        vision_ms = (time.perf_counter() - vision_start) * 1000.0
        payload["timing_ms"] = {"vision": round(vision_ms, 2)}
        cv2.putText(
            annotated,
            "vision {:.1f} ms".format(vision_ms),
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("frame", annotated)
        cv2.imshow("filled mask", mask)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            save_path = output_path or "fragment_debug_result.jpg"
            cv2.imwrite(save_path, annotated)
            cv2.imwrite("fragment_debug_mask.png", mask)
            print("已保存：{}".format(save_path))
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    cv2.destroyAllWindows()
    return 0


def run_camera(camera_index: int, width: int, height: int) -> int:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        raise RuntimeError("无法打开摄像头索引 {}".format(camera_index))

    create_controls(DetectConfig())
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("摄像头读取画面失败")
            vision_start = time.perf_counter()
            mask, annotated, payload = process_frame(frame, read_controls())
            vision_ms = (time.perf_counter() - vision_start) * 1000.0
            payload["timing_ms"] = {"vision": round(vision_ms, 2)}
            cv2.putText(
                annotated,
                "vision {:.1f} ms".format(vision_ms),
                (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("frame", annotated)
            cv2.imshow("filled mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                cv2.imwrite("fragment_debug_result.jpg", annotated)
                cv2.imwrite("fragment_debug_mask.png", mask)
                print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def parse_source(value: str) -> Union[int, str]:
    return int(value) if value.isdigit() else value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="0",
        help="摄像头索引或图片路径，默认 0",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="图片模式下只输出 JSON，不打开窗口",
    )
    parser.add_argument(
        "--output",
        default="",
        help="保存标注图的路径",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    source = parse_source(args.source)
    if isinstance(source, int):
        if args.no_gui:
            raise ValueError("摄像头模式不能使用 --no-gui")
        return run_camera(source, args.width, args.height)
    return run_image(source, args.no_gui, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
