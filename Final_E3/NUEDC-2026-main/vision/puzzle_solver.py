"""最多四块多边形碎片的几何拼接求解器。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from math import atan2, cos, degrees, radians, sin
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class SearchState:
    transforms: Dict[int, Tuple[np.ndarray, np.ndarray]]
    used_edges: frozenset
    match_error: float
    matches: Tuple[Tuple[int, int, int, int, float], ...]


def polygon_area(polygon: np.ndarray) -> float:
    return abs(float(cv2.contourArea(polygon.astype(np.float32))))


def apply_transform(
    polygon: np.ndarray,
    transform: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    rotation, translation = transform
    return polygon @ rotation.T + translation


def edge_lengths(polygon: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)


def edge_alignment_transforms(
    moving_polygon: np.ndarray,
    moving_edge: int,
    fixed_polygon_world: np.ndarray,
    fixed_edge: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """生成反向共线的中点对齐和两种端点对齐。"""
    moving_start = moving_polygon[moving_edge]
    moving_end = moving_polygon[(moving_edge + 1) % len(moving_polygon)]
    fixed_start = fixed_polygon_world[fixed_edge]
    fixed_end = fixed_polygon_world[(fixed_edge + 1) % len(fixed_polygon_world)]

    moving_vector = moving_end - moving_start
    target_vector = fixed_start - fixed_end
    angle = atan2(target_vector[1], target_vector[0]) - atan2(
        moving_vector[1],
        moving_vector[0],
    )
    rotation = np.asarray(
        [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]],
        dtype=np.float32,
    )
    moving_midpoint = (moving_start + moving_end) * 0.5
    fixed_midpoint = (fixed_start + fixed_end) * 0.5
    translations = [
        fixed_midpoint - rotation @ moving_midpoint,
        fixed_start - rotation @ moving_end,
        fixed_end - rotation @ moving_start,
    ]
    unique = []
    for translation in translations:
        if not any(np.linalg.norm(translation - old[1]) < 0.5 for old in unique):
            unique.append((rotation, translation))
    return unique


def partial_edge_alignment_transforms(
    moving_polygon: np.ndarray,
    moving_edge: int,
    fixed_polygon_world: np.ndarray,
    fixed_edge: int,
    offsets_on_long_edge: Sequence[float],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Align a short edge to discrete partition offsets on a longer edge."""
    moving_start = moving_polygon[moving_edge]
    moving_end = moving_polygon[(moving_edge + 1) % len(moving_polygon)]
    fixed_start = fixed_polygon_world[fixed_edge]
    fixed_end = fixed_polygon_world[(fixed_edge + 1) % len(fixed_polygon_world)]
    moving_vector = moving_end - moving_start
    fixed_vector = fixed_start - fixed_end
    moving_length = float(np.linalg.norm(moving_vector))
    fixed_length = float(np.linalg.norm(fixed_vector))
    if moving_length < 1e-6 or fixed_length < 1e-6:
        return []

    angle = atan2(fixed_vector[1], fixed_vector[0]) - atan2(
        moving_vector[1],
        moving_vector[0],
    )
    rotation = np.asarray(
        [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]],
        dtype=np.float32,
    )
    transforms = []
    if fixed_length >= moving_length:
        fixed_direction = fixed_vector / fixed_length
        for offset in offsets_on_long_edge:
            target_start = fixed_end + fixed_direction * float(offset)
            translation = target_start - rotation @ moving_start
            transforms.append((rotation, translation))
    else:
        moving_direction = moving_vector / moving_length
        for offset in offsets_on_long_edge:
            moving_sub_start = (
                moving_start + moving_direction * float(offset)
            )
            translation = fixed_end - rotation @ moving_sub_start
            transforms.append((rotation, translation))

    unique = []
    for transform in transforms:
        if not any(
            np.linalg.norm(transform[1] - old[1]) < 0.5
            for old in unique
        ):
            unique.append(transform)
    return unique


def convex_overlap_area(a: np.ndarray, b: np.ndarray) -> float:
    """当前题目自备碎片为凸多边形；返回两凸多边形相交面积。"""
    try:
        area, _ = cv2.intersectConvexConvex(
            a.astype(np.float32),
            b.astype(np.float32),
        )
        return float(area)
    except cv2.error:
        return float("inf")


def state_polygons(
    polygons: Sequence[np.ndarray],
    state: SearchState,
) -> Dict[int, np.ndarray]:
    return {
        piece_id: apply_transform(polygons[piece_id], transform)
        for piece_id, transform in state.transforms.items()
    }


def partial_score(
    polygons: Sequence[np.ndarray],
    state: SearchState,
) -> float:
    placed = state_polygons(polygons, state)
    all_points = np.concatenate(list(placed.values()), axis=0).astype(np.float32)
    (_, _), (width, height), _ = cv2.minAreaRect(all_points)
    rectangle_area = max(1e-6, float(width * height))
    pieces_area = sum(polygon_area(polygon) for polygon in placed.values())
    compactness = max(0.0, rectangle_area / max(pieces_area, 1e-6) - 1.0)
    return state.match_error + 0.08 * compactness


def raster_union_metrics(
    polygons: Sequence[np.ndarray],
    pixels_per_mm: float = 4.0,
) -> Tuple[int, float]:
    all_points = np.concatenate(polygons, axis=0)
    minimum = np.floor(np.min(all_points, axis=0) - 3.0)
    maximum = np.ceil(np.max(all_points, axis=0) + 3.0)
    size = np.maximum(1, np.ceil((maximum - minimum) * pixels_per_mm)).astype(int)
    mask = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    for polygon in polygons:
        shifted = np.round((polygon - minimum) * pixels_per_mm).astype(np.int32)
        cv2.fillPoly(mask, [shifted], 255)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 99, 0.0
    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    outer = cv2.approxPolyDP(largest, perimeter * 0.015, True)
    connected_ratio = cv2.contourArea(largest) / max(
        1.0,
        sum(cv2.contourArea(contour) for contour in contours),
    )
    return len(outer), float(connected_ratio)


def final_score(
    polygons: Sequence[np.ndarray],
    state: SearchState,
    target_size_mm: Tuple[float, float],
) -> Tuple[float, Dict[str, float]]:
    placed_map = state_polygons(polygons, state)
    placed = list(placed_map.values())
    all_points = np.concatenate(placed, axis=0).astype(np.float32)
    (_, _), (width, height), _ = cv2.minAreaRect(all_points)
    short_side, long_side = sorted((float(width), float(height)))
    target_short, target_long = sorted(target_size_mm)

    rectangle_area = max(1e-6, short_side * long_side)
    pieces_area = sum(polygon_area(polygon) for polygon in placed)
    fill_error = abs(rectangle_area - pieces_area) / rectangle_area
    size_error = (
        abs(short_side - target_short) / target_short
        + abs(long_side - target_long) / target_long
    )
    outer_vertices, connected_ratio = raster_union_metrics(placed)
    outer_error = abs(outer_vertices - 4) * 0.20
    connected_error = (1.0 - connected_ratio) * 2.0
    score = (
        state.match_error * 0.8
        + fill_error * 3.0
        + size_error * 1.8
        + outer_error
        + connected_error
    )
    metrics = {
        "score": score,
        "short_side_mm": short_side,
        "long_side_mm": long_side,
        "fill_ratio": pieces_area / rectangle_area,
        "outer_vertices": float(outer_vertices),
        "connected_ratio": connected_ratio,
        "match_error": state.match_error,
    }
    return score, metrics


def unknown_rectangle_score(
    polygons: Sequence[np.ndarray],
    state: SearchState,
    long_side_range_mm: Tuple[float, float] = (90.0, 120.0),
    short_side_range_mm: Tuple[float, float] = (50.0, 90.0),
) -> Tuple[float, Dict[str, float]]:
    """不预设宽高，只按矩形完整度和赛题尺寸范围评价拼法。"""
    placed_map = state_polygons(polygons, state)
    placed = list(placed_map.values())
    all_points = np.concatenate(placed, axis=0).astype(np.float32)
    (_, _), (width, height), _ = cv2.minAreaRect(all_points)
    short_side, long_side = sorted((float(width), float(height)))
    rectangle_area = max(1e-6, short_side * long_side)
    pieces_area = sum(polygon_area(polygon) for polygon in placed)
    fill_ratio = pieces_area / rectangle_area
    fill_error = abs(1.0 - fill_ratio)
    outer_vertices, connected_ratio = raster_union_metrics(placed)

    def range_error(value, limits):
        lower, upper = limits
        if value < lower:
            return (lower - value) / max(lower, 1e-6)
        if value > upper:
            return (value - upper) / max(upper, 1e-6)
        return 0.0

    dimension_error = (
        range_error(long_side, long_side_range_mm)
        + range_error(short_side, short_side_range_mm)
    )
    outer_error = abs(outer_vertices - 4) * 0.24
    connected_error = (1.0 - connected_ratio) * 2.5
    score = (
        state.match_error * 0.8
        + fill_error * 4.0
        + outer_error
        + connected_error
        + dimension_error * 4.0
    )
    metrics = {
        "score": float(score),
        "short_side_mm": short_side,
        "long_side_mm": long_side,
        "fill_ratio": fill_ratio,
        "outer_vertices": float(outer_vertices),
        "connected_ratio": connected_ratio,
        "match_error": state.match_error,
        "dimension_error": dimension_error,
        "strategy": "unknown_rectangle_edge_search",
    }
    return float(score), metrics


def solve_edge_search(
    polygons: Sequence[np.ndarray],
    target_size_mm: Optional[Tuple[float, float]] = (100.0, 60.0),
    edge_tolerance: float = 0.85,
    beam_width: int = 2000,
    progress_callback=None,
) -> Tuple[SearchState, Dict[str, float]]:
    """返回最接近目标矩形的拼接状态。"""
    if not 2 <= len(polygons) <= 4:
        raise ValueError("求解器只支持2至4块碎片")
    polygons = [np.asarray(polygon, dtype=np.float32) for polygon in polygons]
    lengths = [edge_lengths(polygon) for polygon in polygons]
    anchor = int(np.argmax([polygon_area(polygon) for polygon in polygons]))
    identity = (
        np.eye(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    states = [
        SearchState(
            transforms={anchor: identity},
            used_edges=frozenset(),
            match_error=0.0,
            matches=(),
        )
    ]

    total_stages = max(1, len(polygons) - 1)
    while len(states[0].transforms) < len(polygons):
        stage_index = len(states[0].transforms) - 1
        expanded = []
        for state_index, state in enumerate(states):
            if progress_callback is not None:
                progress_callback(
                    min(
                        0.94,
                        (
                            stage_index
                            + state_index / max(1, len(states))
                        )
                        / total_stages,
                    ),
                    "edge stage {}/{}".format(
                        stage_index + 1,
                        total_stages,
                    ),
                )
            placed_world = state_polygons(polygons, state)
            unplaced = [
                piece_id
                for piece_id in range(len(polygons))
                if piece_id not in state.transforms
            ]
            for moving_id in unplaced:
                for fixed_id, fixed_polygon in placed_world.items():
                    for fixed_edge, fixed_length in enumerate(lengths[fixed_id]):
                        for moving_edge, moving_length in enumerate(
                            lengths[moving_id]
                        ):
                            relative_error = abs(
                                float(fixed_length - moving_length)
                            ) / max(float(fixed_length), float(moving_length))
                            if relative_error > edge_tolerance:
                                continue

                            transforms_for_edge = edge_alignment_transforms(
                                polygons[moving_id],
                                moving_edge,
                                fixed_polygon,
                                fixed_edge,
                            )
                            for transform in transforms_for_edge:
                                moving_world = apply_transform(
                                    polygons[moving_id],
                                    transform,
                                )
                                moving_area = polygon_area(moving_world)
                                overlaps = [
                                    convex_overlap_area(moving_world, other)
                                    for other in placed_world.values()
                                ]
                                if any(
                                    overlap > moving_area * 0.025
                                    for overlap in overlaps
                                ):
                                    continue

                                transforms = dict(state.transforms)
                                transforms[moving_id] = transform
                                used_edges = set(state.used_edges)
                                used_edges.add((moving_id, moving_edge))
                                expanded.append(
                                    SearchState(
                                        transforms=transforms,
                                        used_edges=frozenset(used_edges),
                                        match_error=(
                                            state.match_error
                                            + relative_error * 0.15
                                        ),
                                        matches=state.matches
                                        + (
                                            (
                                                fixed_id,
                                                fixed_edge,
                                                moving_id,
                                                moving_edge,
                                                relative_error,
                                            ),
                                        ),
                                    )
                                )

        if not expanded:
            raise RuntimeError("没有找到可继续拼接的候选边")
        expanded.sort(key=lambda state: partial_score(polygons, state))
        states = expanded[:beam_width]

    if target_size_mm is None:
        scored = [
            (unknown_rectangle_score(polygons, state), state)
            for state in states
        ]
    else:
        scored = [
            (final_score(polygons, state, target_size_mm), state)
            for state in states
        ]
    (score, metrics), best_state = min(scored, key=lambda item: item[0][0])
    metrics["score"] = score
    if progress_callback is not None:
        progress_callback(1.0, "edge search complete")
    return best_state, metrics


def generate_edge_search_states(
    polygons: Sequence[np.ndarray],
    edge_tolerance: float = 0.35,
    beam_width: int = 5000,
    allow_partitioned_edges: bool = False,
    deduplicate_states: bool = False,
    progress_callback=None,
) -> List[SearchState]:
    """Generate complete edge-alignment candidates for texture-aware solvers.

    Unlike ``solve_edge_search`` this function intentionally does not choose a
    winner.  Mode 3 can therefore re-rank the geometric candidates with the
    playing-card artwork along each seam.
    """
    if not 2 <= len(polygons) <= 4:
        raise ValueError("candidate generation supports 2 to 4 pieces")
    polygons = [np.asarray(polygon, dtype=np.float32) for polygon in polygons]
    lengths = [edge_lengths(polygon) for polygon in polygons]
    partition_offsets = {}
    if allow_partitioned_edges:
        edge_records = [
            (piece_id, edge_id, float(length))
            for piece_id, piece_lengths in enumerate(lengths)
            for edge_id, length in enumerate(piece_lengths)
        ]
        for long_piece, long_edge, long_length in edge_records:
            shorter = [
                record
                for record in edge_records
                if record[0] != long_piece
                and record[2] < long_length * (1.0 - edge_tolerance)
                and record[2] >= long_length * 0.12
            ]
            for count in range(2, min(3, len(shorter)) + 1):
                for group in combinations(shorter, count):
                    if len({record[0] for record in group}) != count:
                        continue
                    group_length = sum(record[2] for record in group)
                    relative_error = abs(
                        group_length - long_length
                    ) / max(long_length, 1e-6)
                    if relative_error > max(edge_tolerance, 0.20):
                        continue
                    long_key = (long_piece, long_edge)
                    for ordering in permutations(group):
                        offset = 0.0
                        for short_piece, short_edge, short_length in ordering:
                            short_key = (short_piece, short_edge)
                            partition_offsets.setdefault(
                                (long_key, short_key),
                                set(),
                            ).add(round(offset, 3))
                            offset += short_length
    anchor = int(np.argmax([polygon_area(polygon) for polygon in polygons]))
    identity = (
        np.eye(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    states = [
        SearchState(
            transforms={anchor: identity},
            used_edges=frozenset(),
            match_error=0.0,
            matches=(),
        )
    ]

    total_stages = max(1, len(polygons) - 1)
    while len(states[0].transforms) < len(polygons):
        stage_index = len(states[0].transforms) - 1
        expanded = []
        for state_index, state in enumerate(states):
            if progress_callback is not None:
                progress_callback(
                    min(
                        0.96,
                        (
                            stage_index
                            + state_index / max(1, len(states))
                        )
                        / total_stages,
                    ),
                    "candidate stage {}/{}".format(
                        stage_index + 1,
                        total_stages,
                    ),
                )
            placed_world = state_polygons(polygons, state)
            unplaced = [
                piece_id
                for piece_id in range(len(polygons))
                if piece_id not in state.transforms
            ]
            for moving_id in unplaced:
                for fixed_id, fixed_polygon in placed_world.items():
                    for fixed_edge, fixed_length in enumerate(lengths[fixed_id]):
                        for moving_edge, moving_length in enumerate(
                            lengths[moving_id]
                        ):
                            relative_error = abs(
                                float(fixed_length - moving_length)
                            ) / max(float(fixed_length), float(moving_length))
                            fixed_key = (fixed_id, fixed_edge)
                            moving_key = (moving_id, moving_edge)
                            if fixed_length >= moving_length:
                                partition_key = (fixed_key, moving_key)
                            else:
                                partition_key = (moving_key, fixed_key)
                            is_full_match = relative_error <= edge_tolerance
                            is_partition_match = (
                                allow_partitioned_edges
                                and partition_key in partition_offsets
                            )
                            if not is_full_match and not is_partition_match:
                                continue
                            transforms_for_edge = []
                            if is_full_match:
                                full_transforms = edge_alignment_transforms(
                                    polygons[moving_id],
                                    moving_edge,
                                    fixed_polygon,
                                    fixed_edge,
                                )
                                # Partition fallback already has many offset
                                # hypotheses. Keep only the most stable
                                # midpoint alignment for ordinary full seams.
                                if allow_partitioned_edges:
                                    full_transforms = full_transforms[:1]
                                transforms_for_edge.extend(full_transforms)
                            if is_partition_match:
                                transforms_for_edge.extend(
                                    partial_edge_alignment_transforms(
                                        polygons[moving_id],
                                        moving_edge,
                                        fixed_polygon,
                                        fixed_edge,
                                        sorted(
                                            partition_offsets[partition_key]
                                        ),
                                    )
                                )
                            for transform in transforms_for_edge:
                                moving_world = apply_transform(
                                    polygons[moving_id],
                                    transform,
                                )
                                moving_area = polygon_area(moving_world)
                                if any(
                                    convex_overlap_area(moving_world, other)
                                    > moving_area * 0.025
                                    for other in placed_world.values()
                                ):
                                    continue
                                transforms = dict(state.transforms)
                                transforms[moving_id] = transform
                                expanded.append(
                                    SearchState(
                                        transforms=transforms,
                                        used_edges=state.used_edges
                                        | {(moving_id, moving_edge)},
                                        match_error=state.match_error
                                        + (
                                            relative_error * 0.15
                                            if is_full_match
                                            else 0.02
                                        ),
                                        matches=state.matches
                                        + (
                                            (
                                                fixed_id,
                                                fixed_edge,
                                                moving_id,
                                                moving_edge,
                                                relative_error,
                                            ),
                                        ),
                                    )
                                )
        if not expanded:
            return []
        if deduplicate_states:
            # The same placement is often reached through several attachment
            # orders.  Merge near-identical transforms before sorting and
            # beam pruning so later stages do not repeat the same geometry.
            deduplicated = {}
            for candidate in expanded:
                candidate_score = partial_score(polygons, candidate)
                signature = transform_signature(
                    candidate,
                    angle_step_deg=0.5,
                    translation_step_mm=0.25,
                )
                previous = deduplicated.get(signature)
                if (
                    previous is None
                    or candidate_score < previous[0]
                ):
                    deduplicated[signature] = (
                        candidate_score,
                        candidate,
                    )
            expanded = [
                candidate
                for _, candidate in deduplicated.values()
            ]
        expanded.sort(key=lambda state: partial_score(polygons, state))
        states = expanded[:beam_width]
    if progress_callback is not None:
        progress_callback(1.0, "candidate generation complete")
    return states


def solve_unknown_rectangle_legacy(
    polygons: Sequence[np.ndarray],
    edge_tolerance: float = 0.18,
    beam_width: int = 2500,
    progress_callback=None,
) -> Tuple[SearchState, Dict[str, float]]:
    """求解赛题模式2：未知数量、未知宽高的白色碎片矩形。"""
    state, metrics = solve_edge_search(
        polygons,
        target_size_mm=None,
        edge_tolerance=edge_tolerance,
        beam_width=beam_width,
        progress_callback=progress_callback,
    )
    if metrics["outer_vertices"] != 4.0:
        raise RuntimeError(
            "未知碎片当前最优拼法外轮廓不是矩形：{}个顶点".format(
                int(metrics["outer_vertices"])
            )
        )
    if metrics["connected_ratio"] < 0.98:
        raise RuntimeError("未知碎片当前最优拼法没有完全连通")
    if metrics["fill_ratio"] < 0.86 or metrics["fill_ratio"] > 1.08:
        raise RuntimeError(
            "未知碎片矩形填充率异常：{:.3f}".format(
                metrics["fill_ratio"]
            )
        )
    if metrics["dimension_error"] > 0.08:
        raise RuntimeError(
            "未知碎片拼接尺寸超出赛题范围：{:.1f}x{:.1f}mm".format(
                metrics["long_side_mm"],
                metrics["short_side_mm"],
            )
        )
    return state, metrics


def transform_signature(
    state: SearchState,
    angle_step_deg: float = 1.0,
    translation_step_mm: float = 0.75,
) -> Tuple[int, ...]:
    """Quantize placements so repeated construction paths can be merged."""
    signature = []
    for piece_id in sorted(state.transforms):
        rotation, translation = state.transforms[piece_id]
        angle = degrees(atan2(rotation[1, 0], rotation[0, 0]))
        signature.extend(
            (
                int(piece_id),
                int(round(angle / angle_step_deg)),
                int(round(float(translation[0]) / translation_step_mm)),
                int(round(float(translation[1]) / translation_step_mm)),
            )
        )
    return tuple(signature)


def unknown_partial_is_feasible(
    polygons: Sequence[np.ndarray],
    state: SearchState,
    maximum_long_mm: float = 130.0,
    maximum_short_mm: float = 105.0,
) -> bool:
    """Reject a partial layout whose bounding box cannot become legal."""
    placed = list(state_polygons(polygons, state).values())
    points = np.concatenate(placed, axis=0).astype(np.float32)
    (_, _), (width, height), _ = cv2.minAreaRect(points)
    short_side, long_side = sorted((float(width), float(height)))
    if long_side > maximum_long_mm or short_side > maximum_short_mm:
        return False

    return True


def cheap_unknown_score(
    polygons: Sequence[np.ndarray],
    state: SearchState,
) -> float:
    """Rank candidates without the expensive raster-union operation."""
    placed = list(state_polygons(polygons, state).values())
    points = np.concatenate(placed, axis=0).astype(np.float32)
    (_, _), (width, height), _ = cv2.minAreaRect(points)
    short_side, long_side = sorted((float(width), float(height)))
    rectangle_area = max(1e-6, short_side * long_side)
    pieces_area = sum(polygon_area(polygon) for polygon in placed)
    fill_error = abs(1.0 - pieces_area / rectangle_area)

    dimension_error = 0.0
    if long_side < 90.0:
        dimension_error += (90.0 - long_side) / 90.0
    elif long_side > 120.0:
        dimension_error += (long_side - 120.0) / 120.0
    if short_side < 50.0:
        dimension_error += (50.0 - short_side) / 50.0
    elif short_side > 90.0:
        dimension_error += (short_side - 90.0) / 90.0
    return (
        float(state.match_error)
        + fill_error * 3.5
        + dimension_error * 8.0
    )


def build_partition_edge_offsets(
    lengths: Sequence[np.ndarray],
    edge_tolerance: float,
) -> Dict[
    Tuple[Tuple[int, int], Tuple[int, int]],
    Tuple[float, ...],
]:
    """Find offsets where two or three short edges can cover one long edge."""
    records = [
        (piece_id, edge_id, float(length))
        for piece_id, piece_lengths in enumerate(lengths)
        for edge_id, length in enumerate(piece_lengths)
    ]
    offsets = {}
    for long_piece, long_edge, long_length in records:
        shorter = [
            record
            for record in records
            if record[0] != long_piece
            and record[2] < long_length * (1.0 - edge_tolerance)
            and record[2] >= long_length * 0.12
        ]
        for count in range(2, min(3, len(shorter)) + 1):
            for group in combinations(shorter, count):
                if len({record[0] for record in group}) != count:
                    continue
                combined = sum(record[2] for record in group)
                if (
                    abs(combined - long_length)
                    / max(long_length, 1e-6)
                    > max(edge_tolerance, 0.16)
                ):
                    continue
                for ordering in permutations(group):
                    offset = 0.0
                    for short_piece, short_edge, short_length in ordering:
                        key = (
                            (long_piece, long_edge),
                            (short_piece, short_edge),
                        )
                        offsets.setdefault(key, set()).add(
                            round(offset, 3)
                        )
                        offset += short_length
    return {
        key: tuple(sorted(values))
        for key, values in offsets.items()
    }


def connection_topology_signature(
    state: SearchState,
) -> Tuple[Tuple[int, int], ...]:
    """Represent only which pieces touch, independent of placement order."""
    return tuple(
        sorted(
            {
                tuple(sorted((int(match[0]), int(match[2]))))
                for match in state.matches
            }
        )
    )


def select_diverse_unknown_states(
    polygons: Sequence[np.ndarray],
    states: Sequence[SearchState],
    beam_width: int,
) -> List[SearchState]:
    """Keep several candidates from each connection topology."""
    ordered = sorted(
        states,
        key=lambda state: cheap_unknown_score(polygons, state),
    )
    beam_width = max(20, int(beam_width))
    per_topology = max(3, min(8, beam_width // 8))
    counts = {}
    selected = []
    selected_ids = set()
    for state in ordered:
        topology = connection_topology_signature(state)
        if counts.get(topology, 0) >= per_topology:
            continue
        selected.append(state)
        selected_ids.add(id(state))
        counts[topology] = counts.get(topology, 0) + 1
        if len(selected) >= beam_width:
            return selected
    for state in ordered:
        if id(state) in selected_ids:
            continue
        selected.append(state)
        if len(selected) >= beam_width:
            break
    return selected


def rank_unknown_rectangle_candidates(
    polygons: Sequence[np.ndarray],
    edge_tolerance: float = 0.18,
    beam_width: int = 40,
    time_limit_seconds: float = 20.0,
    maximum_results: int = 24,
    allow_partitioned_edges: bool = False,
    progress_callback=None,
) -> List[Tuple[SearchState, Dict[str, float]]]:
    """Generate a small ranked set of strictly legal Mode-2 candidates.

    Each newly attached seam edge can be used only once. A longer edge on the
    placed assembly may still accept multiple shorter seam segments.
    Equivalent placements from different construction orders are merged.
    """
    if not 2 <= len(polygons) <= 4:
        raise ValueError("Mode 2 supports 2 to 4 pieces")
    polygons = [
        np.asarray(polygon, dtype=np.float32)
        for polygon in polygons
    ]
    lengths = [edge_lengths(polygon) for polygon in polygons]
    partition_offsets = (
        build_partition_edge_offsets(lengths, edge_tolerance)
        if allow_partitioned_edges
        else {}
    )
    anchor = int(np.argmax([polygon_area(polygon) for polygon in polygons]))
    identity = (
        np.eye(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    states = [
        SearchState(
            transforms={anchor: identity},
            used_edges=frozenset(),
            match_error=0.0,
            matches=(),
        )
    ]
    started = time.time()
    total_stages = len(polygons) - 1

    for stage_index in range(total_stages):
        deduplicated = {}
        rejection_counts = {
            "used": 0,
            "length": 0,
            "overlap": 0,
            "bounds": 0,
        }
        state_count = max(1, len(states))
        for state_index, state in enumerate(states):
            elapsed = time.time() - started
            if elapsed > time_limit_seconds:
                raise RuntimeError(
                    "Mode 2 search timeout after {:.1f}s".format(elapsed)
                )
            if progress_callback is not None:
                progress_callback(
                    min(
                        0.82,
                        (
                            stage_index
                            + state_index / state_count
                        )
                        / max(1, total_stages)
                        * 0.82,
                    ),
                    "fast edge stage {}/{}".format(
                        stage_index + 1,
                        total_stages,
                    ),
                )

            placed_world = state_polygons(polygons, state)
            unplaced = [
                piece_id
                for piece_id in range(len(polygons))
                if piece_id not in state.transforms
            ]
            for moving_id in unplaced:
                for fixed_id, fixed_polygon in placed_world.items():
                    for fixed_edge, fixed_length in enumerate(
                        lengths[fixed_id]
                    ):
                        fixed_key = (fixed_id, fixed_edge)
                        for moving_edge, moving_length in enumerate(
                            lengths[moving_id]
                        ):
                            moving_key = (moving_id, moving_edge)
                            if moving_key in state.used_edges:
                                rejection_counts["used"] += 1
                                continue
                            relative_error = abs(
                                float(fixed_length - moving_length)
                            ) / max(
                                float(fixed_length),
                                float(moving_length),
                                1e-6,
                            )
                            if fixed_length >= moving_length:
                                partition_key = (
                                    fixed_key,
                                    moving_key,
                                )
                            else:
                                partition_key = (
                                    moving_key,
                                    fixed_key,
                                )
                            is_full_match = (
                                relative_error <= edge_tolerance
                            )
                            is_partition_match = (
                                partition_key in partition_offsets
                            )
                            if not is_full_match and not is_partition_match:
                                rejection_counts["length"] += 1
                                continue

                            transforms_for_edge = []
                            if is_full_match:
                                transforms_for_edge.extend(
                                    edge_alignment_transforms(
                                        polygons[moving_id],
                                        moving_edge,
                                        fixed_polygon,
                                        fixed_edge,
                                    )
                                )
                            if is_partition_match:
                                transforms_for_edge.extend(
                                    partial_edge_alignment_transforms(
                                        polygons[moving_id],
                                        moving_edge,
                                        fixed_polygon,
                                        fixed_edge,
                                        partition_offsets[partition_key],
                                    )
                                )
                            for transform in transforms_for_edge:
                                moving_world = apply_transform(
                                    polygons[moving_id],
                                    transform,
                                )
                                moving_area = polygon_area(moving_world)
                                if any(
                                    convex_overlap_area(moving_world, other)
                                    > moving_area * 0.025
                                    for other in placed_world.values()
                                ):
                                    rejection_counts["overlap"] += 1
                                    continue

                                transforms = dict(state.transforms)
                                transforms[moving_id] = transform
                                used_edges = set(state.used_edges)
                                if (
                                    is_full_match
                                    or moving_length <= fixed_length
                                ):
                                    used_edges.add(moving_key)
                                candidate = SearchState(
                                    transforms=transforms,
                                    used_edges=frozenset(used_edges),
                                    match_error=(
                                        state.match_error
                                        + (
                                            relative_error * 0.18
                                            if is_full_match
                                            else 0.025
                                        )
                                    ),
                                    matches=state.matches
                                    + (
                                        (
                                            fixed_id,
                                            fixed_edge,
                                            moving_id,
                                            moving_edge,
                                            relative_error,
                                        ),
                                    ),
                                )
                                if not unknown_partial_is_feasible(
                                    polygons,
                                    candidate,
                                ):
                                    rejection_counts["bounds"] += 1
                                    continue
                                signature = transform_signature(candidate)
                                previous = deduplicated.get(signature)
                                if (
                                    previous is None
                                    or candidate.match_error
                                    < previous.match_error
                                ):
                                    deduplicated[signature] = candidate

        if not deduplicated:
            raise RuntimeError(
                "Mode 2 has no feasible edge candidate at stage {}; "
                "rejected={}".format(
                    stage_index + 1,
                    rejection_counts,
                )
            )
        expanded = list(deduplicated.values())
        states = select_diverse_unknown_states(
            polygons,
            expanded,
            beam_width,
        )

    if progress_callback is not None:
        progress_callback(0.86, "validate legal rectangles")

    states.sort(key=lambda item: cheap_unknown_score(polygons, item))
    validated = []
    validation_rejections = {
        "size": 0,
        "outer": 0,
        "connected": 0,
        "fill": 0,
    }
    closest_metrics = None
    validation_limit = min(len(states), max(40, maximum_results * 3))
    for state_id, state in enumerate(states[:validation_limit]):
        if time.time() - started > time_limit_seconds:
            break
        if progress_callback is not None:
            progress_callback(
                0.86 + 0.13 * state_id / max(1, validation_limit),
                "validate rectangle {}/{}".format(
                    state_id + 1,
                    validation_limit,
                ),
            )
        score, metrics = unknown_rectangle_score(polygons, state)
        if (
            closest_metrics is None
            or float(score) < float(closest_metrics["score"])
        ):
            closest_metrics = dict(metrics)
            closest_metrics["score"] = float(score)
        rejected = False
        if metrics["dimension_error"] > 0.04:
            validation_rejections["size"] += 1
            rejected = True
        strict_rectangle = metrics["outer_vertices"] == 4.0
        measured_near_rectangle = (
            metrics["outer_vertices"] <= 8.0
            and metrics["dimension_error"] <= 0.04
            and metrics["fill_ratio"] >= 0.94
            and metrics["connected_ratio"] >= 0.98
            and metrics["match_error"] <= 0.08
        )
        if not strict_rectangle and not measured_near_rectangle:
            validation_rejections["outer"] += 1
            rejected = True
        if metrics["connected_ratio"] < 0.98:
            validation_rejections["connected"] += 1
            rejected = True
        if not 0.86 <= metrics["fill_ratio"] <= 1.08:
            validation_rejections["fill"] += 1
            rejected = True
        if rejected:
            continue
        metrics["score"] = float(score)
        metrics["strategy"] = (
            "fast_unique_edge_search"
            if strict_rectangle
            else "fast_measured_rectangle_search"
        )
        metrics["outer_vertices_relaxed"] = float(
            not strict_rectangle
        )
        metrics["partitioned_edges"] = float(allow_partitioned_edges)
        metrics["candidate_count"] = float(len(states))
        validated.append((state, metrics))

    if not validated:
        if closest_metrics is None:
            closest_text = "none"
        else:
            closest_text = (
                "{:.1f}x{:.1f}mm fill={:.3f} outer={} "
                "connected={:.3f} dim_error={:.3f}"
            ).format(
                closest_metrics["long_side_mm"],
                closest_metrics["short_side_mm"],
                closest_metrics["fill_ratio"],
                int(closest_metrics["outer_vertices"]),
                closest_metrics["connected_ratio"],
                closest_metrics["dimension_error"],
            )
        raise RuntimeError(
            "Mode 2 found no legal rectangular candidate; "
            "checked={} rejected={} closest={}".format(
                validation_limit,
                validation_rejections,
                closest_text,
            )
        )
    validated.sort(key=lambda item: item[1]["score"])
    if progress_callback is not None:
        progress_callback(1.0, "fast edge search complete")
    return validated[:maximum_results]


def solve_unknown_rectangle(
    polygons: Sequence[np.ndarray],
    edge_tolerance: float = 0.18,
    beam_width: int = 40,
    time_limit_seconds: float = 20.0,
    progress_callback=None,
) -> Tuple[SearchState, Dict[str, float]]:
    """Return the highest-ranked strict Mode-2 rectangle candidate."""
    return rank_unknown_rectangle_candidates(
        polygons,
        edge_tolerance=edge_tolerance,
        beam_width=beam_width,
        time_limit_seconds=time_limit_seconds,
        progress_callback=progress_callback,
    )[0]


def interior_angles(polygon: np.ndarray) -> np.ndarray:
    angles = []
    for index, vertex in enumerate(polygon):
        previous_vector = polygon[index - 1] - vertex
        next_vector = polygon[(index + 1) % len(polygon)] - vertex
        denominator = max(
            1e-8,
            float(np.linalg.norm(previous_vector) * np.linalg.norm(next_vector)),
        )
        cosine_value = np.clip(
            float(np.dot(previous_vector, next_vector)) / denominator,
            -1.0,
            1.0,
        )
        angles.append(degrees(np.arccos(cosine_value)))
    return np.asarray(angles, dtype=np.float32)


def rectangle_corner_definitions(
    target_size_mm: Tuple[float, float],
):
    width, height = target_size_mm
    return [
        (
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
        ),
        (
            np.asarray([width, 0.0], dtype=np.float32),
            np.asarray([-1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
        ),
        (
            np.asarray([width, height], dtype=np.float32),
            np.asarray([-1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, -1.0], dtype=np.float32),
        ),
        (
            np.asarray([0.0, height], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, -1.0], dtype=np.float32),
        ),
    ]


def rotation_fitting_two_directions(
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> Tuple[np.ndarray, float]:
    source_a = source_a / max(1e-8, float(np.linalg.norm(source_a)))
    source_b = source_b / max(1e-8, float(np.linalg.norm(source_b)))
    target_a = target_a / max(1e-8, float(np.linalg.norm(target_a)))
    target_b = target_b / max(1e-8, float(np.linalg.norm(target_b)))
    cross_sum = (
        source_a[0] * target_a[1] - source_a[1] * target_a[0]
        + source_b[0] * target_b[1]
        - source_b[1] * target_b[0]
    )
    dot_sum = float(np.dot(source_a, target_a) + np.dot(source_b, target_b))
    angle = atan2(float(cross_sum), dot_sum)
    rotation = np.asarray(
        [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]],
        dtype=np.float32,
    )
    fitted_a = rotation @ source_a
    fitted_b = rotation @ source_b
    residual = (
        degrees(np.arccos(np.clip(np.dot(fitted_a, target_a), -1.0, 1.0)))
        + degrees(np.arccos(np.clip(np.dot(fitted_b, target_b), -1.0, 1.0)))
    ) * 0.5
    return rotation, residual


def corner_options_for_piece(
    polygon: np.ndarray,
    piece_id: int,
    target_size_mm: Tuple[float, float],
    angle_tolerance_deg: float = 18.0,
    outside_tolerance_mm: float = 6.0,
):
    options = []
    angles = interior_angles(polygon)
    target_width, target_height = target_size_mm
    for vertex_id, angle in enumerate(angles):
        if abs(float(angle) - 90.0) > angle_tolerance_deg:
            continue
        vertex = polygon[vertex_id]
        previous_vector = polygon[vertex_id - 1] - vertex
        next_vector = polygon[(vertex_id + 1) % len(polygon)] - vertex
        for corner_id, (corner, axis_a, axis_b) in enumerate(
            rectangle_corner_definitions(target_size_mm)
        ):
            for first_axis, second_axis in (
                (axis_a, axis_b),
                (axis_b, axis_a),
            ):
                rotation, residual = rotation_fitting_two_directions(
                    previous_vector,
                    next_vector,
                    first_axis,
                    second_axis,
                )
                if residual > angle_tolerance_deg:
                    continue
                translation = corner - rotation @ vertex
                transformed = apply_transform(
                    polygon,
                    (rotation, translation),
                )
                minimum = np.min(transformed, axis=0)
                maximum = np.max(transformed, axis=0)
                inside = (
                    minimum[0] >= -outside_tolerance_mm
                    and minimum[1] >= -outside_tolerance_mm
                    and maximum[0] <= target_width + outside_tolerance_mm
                    and maximum[1] <= target_height + outside_tolerance_mm
                )
                if not inside:
                    continue
                options.append(
                    {
                        "piece_id": piece_id,
                        "vertex_id": vertex_id,
                        "corner_id": corner_id,
                        "transform": (rotation, translation),
                        "angle_error": abs(float(angle) - 90.0) / 90.0,
                    }
                )
    return options


def transforms_have_overlap(
    polygons: Sequence[np.ndarray],
    transforms: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> bool:
    placed = {
        piece_id: apply_transform(polygons[piece_id], transform)
        for piece_id, transform in transforms.items()
    }
    for first_id, second_id in combinations(placed.keys(), 2):
        overlap = convex_overlap_area(placed[first_id], placed[second_id])
        smaller_area = min(
            polygon_area(placed[first_id]),
            polygon_area(placed[second_id]),
        )
        if overlap > smaller_area * 0.025:
            return True
    return False


def solve_corner_first(
    polygons: Sequence[np.ndarray],
    target_size_mm: Tuple[float, float],
) -> Tuple[SearchState, Dict[str, float]]:
    polygons = [np.asarray(polygon, dtype=np.float32) for polygon in polygons]
    options_by_piece = {
        piece_id: corner_options_for_piece(
            polygon,
            piece_id,
            target_size_mm,
        )
        for piece_id, polygon in enumerate(polygons)
    }
    smallest_piece = int(
        np.argmin([polygon_area(polygon) for polygon in polygons])
    )
    anchor_options = options_by_piece[smallest_piece]
    if not anchor_options:
        raise RuntimeError("最小碎片没有可靠的直角候选")

    states = []
    for option in anchor_options:
        states.append(
            (
                {smallest_piece: option["transform"]},
                {option["corner_id"]},
                option["angle_error"],
            )
        )

    while len(states[0][0]) < min(3, len(polygons)):
        expanded = []
        for transforms, occupied_corners, angle_error in states:
            for piece_id in range(len(polygons)):
                if piece_id in transforms:
                    continue
                for option in options_by_piece[piece_id]:
                    if option["corner_id"] in occupied_corners:
                        continue
                    next_transforms = dict(transforms)
                    next_transforms[piece_id] = option["transform"]
                    if transforms_have_overlap(polygons, next_transforms):
                        continue
                    expanded.append(
                        (
                            next_transforms,
                            occupied_corners | {option["corner_id"]},
                            angle_error + option["angle_error"],
                        )
                    )
        if not expanded:
            raise RuntimeError("无法由三个直角建立矩形框架")
        expanded.sort(
            key=lambda item: item[2]
            + partial_score(
                polygons,
                SearchState(
                    transforms=item[0],
                    used_edges=frozenset(),
                    match_error=item[2],
                    matches=(),
                ),
            )
        )
        states = expanded[:500]

    full_states = []
    target_width, target_height = target_size_mm
    for transforms, occupied_corners, angle_error in states:
        remaining = [
            piece_id
            for piece_id in range(len(polygons))
            if piece_id not in transforms
        ]
        if not remaining:
            full_states.append(
                SearchState(
                    transforms=transforms,
                    used_edges=frozenset(),
                    match_error=angle_error,
                    matches=(),
                )
            )
            continue

        moving_id = remaining[0]
        candidate_transforms = []
        for option in options_by_piece[moving_id]:
            if option["corner_id"] not in occupied_corners:
                candidate_transforms.append(
                    (option["transform"], option["angle_error"], ())
                )

        placed_world = {
            piece_id: apply_transform(polygons[piece_id], transform)
            for piece_id, transform in transforms.items()
        }
        moving_lengths = edge_lengths(polygons[moving_id])
        for fixed_id, fixed_polygon in placed_world.items():
            fixed_lengths = edge_lengths(polygons[fixed_id])
            for fixed_edge, fixed_length in enumerate(fixed_lengths):
                for moving_edge, moving_length in enumerate(moving_lengths):
                    relative_error = abs(
                        float(fixed_length - moving_length)
                    ) / max(float(fixed_length), float(moving_length))
                    for transform in edge_alignment_transforms(
                        polygons[moving_id],
                        moving_edge,
                        fixed_polygon,
                        fixed_edge,
                    ):
                        candidate_transforms.append(
                            (
                                transform,
                                relative_error * 0.15,
                                (
                                    (
                                        fixed_id,
                                        fixed_edge,
                                        moving_id,
                                        moving_edge,
                                        relative_error,
                                    ),
                                ),
                            )
                        )

        for transform, extra_error, matches in candidate_transforms:
            next_transforms = dict(transforms)
            next_transforms[moving_id] = transform
            if transforms_have_overlap(polygons, next_transforms):
                continue
            moving_world = apply_transform(polygons[moving_id], transform)
            minimum = np.min(moving_world, axis=0)
            maximum = np.max(moving_world, axis=0)
            if (
                minimum[0] < -6.0
                or minimum[1] < -6.0
                or maximum[0] > target_width + 6.0
                or maximum[1] > target_height + 6.0
            ):
                continue
            full_states.append(
                SearchState(
                    transforms=next_transforms,
                    used_edges=frozenset(),
                    match_error=angle_error + extra_error,
                    matches=matches,
                )
            )

    if not full_states:
        raise RuntimeError("三个直角确定后无法放入剩余碎片")
    scored = [
        (final_score(polygons, state, target_size_mm), state)
        for state in full_states
    ]
    (score, metrics), best_state = min(scored, key=lambda item: item[0][0])
    metrics["score"] = score
    metrics["strategy"] = "corner_first"
    return best_state, metrics


def solve_polygons(
    polygons: Sequence[np.ndarray],
    target_size_mm: Tuple[float, float] = (100.0, 60.0),
    edge_tolerance: float = 0.85,
    beam_width: int = 2000,
) -> Tuple[SearchState, Dict[str, float]]:
    """优先使用三个直角求解，评分不合格时回退通用边搜索。"""
    try:
        corner_state, corner_metrics = solve_corner_first(
            polygons,
            target_size_mm,
        )
        if (
            corner_metrics["outer_vertices"] == 4.0
            and corner_metrics["fill_ratio"] >= 0.85
            and corner_metrics["score"] <= 0.9
        ):
            return corner_state, corner_metrics
    except RuntimeError:
        corner_state = None
        corner_metrics = None

    edge_state, edge_metrics = solve_edge_search(
        polygons,
        target_size_mm,
        edge_tolerance,
        beam_width,
    )
    edge_metrics["strategy"] = "edge_fallback"
    if (
        corner_state is not None
        and corner_metrics is not None
        and corner_metrics["score"] < edge_metrics["score"]
    ):
        return corner_state, corner_metrics
    return edge_state, edge_metrics


def mode1_rectangle_is_valid(
    metrics: Dict[str, float],
    minimum_fill_ratio: float = 0.86,
    maximum_dimension_error: float = 0.12,
) -> bool:
    """Strict acceptance gate for the known 100x60 mm Mode-1 rectangle."""
    dimension_error = metrics.get("dimension_error")
    if dimension_error is None:
        dimension_error = (
            abs(float(metrics["short_side_mm"]) - 60.0) / 60.0
            + abs(float(metrics["long_side_mm"]) - 100.0) / 100.0
        )
        metrics["dimension_error"] = float(dimension_error)
    return (
        metrics["outer_vertices"] == 4.0
        and metrics["connected_ratio"] >= 0.98
        and minimum_fill_ratio <= metrics["fill_ratio"] <= 1.08
        and dimension_error <= maximum_dimension_error
    )


def solve_mode1_geometry(
    polygons: Sequence[np.ndarray],
    target_size_mm: Tuple[float, float] = (100.0, 60.0),
    edge_tolerance: float = 0.22,
    partition_beam_width: int = 40,
    progress_callback=None,
) -> Tuple[SearchState, Dict[str, float]]:
    """Solve Mode 1 without a template.

    First try the original smallest-piece/three-corner construction. If that
    does not produce a valid rectangle, search a small candidate set that also
    allows one long cut edge to be assembled from multiple shorter edges.
    """
    polygons = [
        np.asarray(polygon, dtype=np.float32)
        for polygon in polygons
    ]
    if len(polygons) != 4:
        raise RuntimeError("Mode 1 requires exactly four pieces")

    corner_error = None
    if progress_callback is not None:
        progress_callback(0.03, "corner-first search")
    try:
        state, metrics = solve_corner_first(
            polygons,
            target_size_mm,
        )
        if mode1_rectangle_is_valid(metrics):
            metrics["strategy"] = "mode1_corner_first"
            metrics["edge_tolerance"] = edge_tolerance
            metrics["partition_beam_width"] = partition_beam_width
            if progress_callback is not None:
                progress_callback(1.0, "corner solution accepted")
            return state, metrics
        corner_error = (
            "corner result rejected: vertices={} fill={:.3f} "
            "dimension_error={:.3f}".format(
                int(metrics["outer_vertices"]),
                metrics["fill_ratio"],
                metrics["dimension_error"],
            )
        )
    except RuntimeError as error:
        corner_error = str(error)

    if progress_callback is not None:
        progress_callback(0.18, "partitioned-edge fallback")

    def partition_progress(progress, label):
        if progress_callback is not None:
            progress_callback(
                0.18 + float(progress) * 0.68,
                label,
            )

    states = generate_edge_search_states(
        polygons,
        edge_tolerance=edge_tolerance,
        beam_width=partition_beam_width,
        allow_partitioned_edges=True,
        progress_callback=partition_progress,
    )
    candidates = []
    for state_id, state in enumerate(states):
        if progress_callback is not None:
            progress_callback(
                0.86 + 0.12 * state_id / max(1, len(states)),
                "score rectangle candidates",
            )
        score, metrics = final_score(
            polygons,
            state,
            target_size_mm,
        )
        if not mode1_rectangle_is_valid(metrics):
            continue
        candidates.append((float(score), state, metrics))
    if not candidates:
        raise RuntimeError(
            "Mode 1 found no valid 100x60 rectangle; corner={}".format(
                corner_error
            )
        )

    score, state, metrics = min(
        candidates,
        key=lambda item: item[0],
    )
    metrics["score"] = score
    metrics["strategy"] = "mode1_partitioned_edge_search"
    metrics["edge_tolerance"] = edge_tolerance
    metrics["partition_beam_width"] = partition_beam_width
    metrics["valid_partition_candidates"] = len(candidates)
    metrics["corner_stage_error"] = corner_error
    if progress_callback is not None:
        progress_callback(1.0, "Mode 1 solution accepted")
    return state, metrics


def best_rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """尝试循环顶点编号和方向，返回无镜像刚体拟合及RMS误差。"""
    if len(source) != len(target):
        raise ValueError("源多边形与模板顶点数不同")
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    best_transform = None
    best_error = float("inf")
    for candidate in (source, source[::-1]):
        for shift in range(len(source)):
            shifted = np.roll(candidate, shift, axis=0)
            source_center = np.mean(shifted, axis=0)
            target_center = np.mean(target, axis=0)
            source_zero = shifted - source_center
            target_zero = target - target_center
            covariance = source_zero.T @ target_zero
            left, _, right = np.linalg.svd(covariance)
            rotation = right.T @ left.T
            if np.linalg.det(rotation) < 0:
                right[-1, :] *= -1
                rotation = right.T @ left.T
            translation = target_center - rotation @ source_center
            fitted = apply_transform(shifted, (rotation, translation))
            error = float(
                np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1)))
            )
            if error < best_error:
                best_error = error
                best_transform = (rotation.astype(np.float32), translation)
    return best_transform, best_error


def sample_polygon_boundary(
    polygon: np.ndarray,
    sample_count: int = 64,
) -> np.ndarray:
    """Uniformly sample a closed polygon boundary.

    This makes template fitting tolerant of an extra or missing collinear
    corner produced by contour approximation.
    """
    polygon = np.asarray(polygon, dtype=np.float32)
    following = np.roll(polygon, -1, axis=0)
    vectors = following - polygon
    lengths = np.linalg.norm(vectors, axis=1)
    perimeter = float(np.sum(lengths))
    if len(polygon) < 3 or perimeter < 1e-6:
        raise ValueError("polygon boundary is invalid")

    cumulative = np.concatenate(
        (
            np.asarray([0.0], dtype=np.float32),
            np.cumsum(lengths, dtype=np.float32),
        )
    )
    distances = (
        np.arange(sample_count, dtype=np.float32)
        * (perimeter / float(sample_count))
    )
    sampled = []
    edge_id = 0
    for distance in distances:
        while (
            edge_id + 1 < len(polygon)
            and distance >= cumulative[edge_id + 1]
        ):
            edge_id += 1
        edge_length = max(float(lengths[edge_id]), 1e-6)
        ratio = (
            float(distance) - float(cumulative[edge_id])
        ) / edge_length
        sampled.append(
            polygon[edge_id] + vectors[edge_id] * ratio
        )
    return np.asarray(sampled, dtype=np.float32)


def best_template_fit(
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """Fit exact corners when possible, otherwise fit sampled boundaries."""
    if len(source) == len(target):
        return best_rigid_fit(source, target)
    return best_rigid_fit(
        sample_polygon_boundary(source),
        sample_polygon_boundary(target),
    )


def solve_with_template(
    polygons: Sequence[np.ndarray],
    template_polygons: Sequence[np.ndarray],
    maximum_rms_mm: float = 6.0,
) -> Tuple[SearchState, Dict[str, float]]:
    """将同一套固定碎片按形状匹配到已验证的目标模板。"""
    polygons = [np.asarray(polygon, dtype=np.float32) for polygon in polygons]
    templates = [
        np.asarray(polygon, dtype=np.float32)
        for polygon in template_polygons
    ]
    if len(polygons) != len(templates):
        raise RuntimeError("当前碎片数量与固定模板不同")

    pair_fits = {}
    for piece_id, polygon in enumerate(polygons):
        for template_id, template in enumerate(templates):
            if abs(len(polygon) - len(template)) > 1:
                continue
            pair_fits[(piece_id, template_id)] = best_template_fit(
                polygon,
                template,
            )

    best_assignment = None
    best_total = float("inf")
    for assignment in permutations(range(len(templates))):
        if any(
            (piece_id, template_id) not in pair_fits
            for piece_id, template_id in enumerate(assignment)
        ):
            continue
        errors = [
            pair_fits[(piece_id, template_id)][1]
            for piece_id, template_id in enumerate(assignment)
        ]
        total = sum(errors)
        if total < best_total:
            best_total = total
            best_assignment = (assignment, errors)
    if best_assignment is None:
        raise RuntimeError("没有找到与固定模板顶点数一致的分配")

    assignment, errors = best_assignment
    if max(errors) > maximum_rms_mm:
        raise RuntimeError(
            "固定模板匹配误差过大：最大RMS {:.2f} mm".format(max(errors))
        )
    transforms = {
        piece_id: pair_fits[(piece_id, template_id)][0]
        for piece_id, template_id in enumerate(assignment)
    }
    state = SearchState(
        transforms=transforms,
        used_edges=frozenset(),
        match_error=float(np.mean(errors) / 100.0),
        matches=(),
    )
    target_size = (100.0, 60.0)
    assigned_templates = [
        templates[template_id] for template_id in assignment
    ]
    identity = (
        np.eye(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    template_state = SearchState(
        transforms={
            piece_id: identity for piece_id in range(len(assigned_templates))
        },
        used_edges=frozenset(),
        match_error=0.0,
        matches=(),
    )
    _, metrics = final_score(
        assigned_templates,
        template_state,
        target_size,
    )
    metrics["strategy"] = "fixed_template"
    metrics["template_mean_rms_mm"] = float(np.mean(errors))
    metrics["template_max_rms_mm"] = float(max(errors))
    metrics["template_assignment"] = [
        float(template_id) for template_id in assignment
    ]
    return state, metrics


def normalize_solution(
    polygons: Sequence[np.ndarray],
    state: SearchState,
    target_origin_mm: Tuple[float, float] = (55.0, 180.0),
) -> Tuple[List[np.ndarray], List[float]]:
    """将求解结果旋转为横向矩形并移动到A4下半区。"""
    placed_map = state_polygons(polygons, state)
    all_points = np.concatenate(list(placed_map.values()), axis=0).astype(np.float32)
    rectangle = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rectangle)
    vectors = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(vectors, axis=1)
    long_vector = vectors[int(np.argmax(lengths))]
    correction_angle = -atan2(long_vector[1], long_vector[0])
    correction = np.asarray(
        [
            [cos(correction_angle), -sin(correction_angle)],
            [sin(correction_angle), cos(correction_angle)],
        ],
        dtype=np.float32,
    )

    corrected = {
        piece_id: polygon @ correction.T
        for piece_id, polygon in placed_map.items()
    }
    corrected_all = np.concatenate(list(corrected.values()), axis=0)
    minimum = np.min(corrected_all, axis=0)
    offset = np.asarray(target_origin_mm, dtype=np.float32) - minimum

    targets = []
    rotations = []
    for piece_id in range(len(polygons)):
        targets.append(corrected[piece_id] + offset)
        original_rotation = state.transforms[piece_id][0]
        total_rotation = correction @ original_rotation
        rotations.append(
            degrees(atan2(total_rotation[1, 0], total_rotation[0, 0]))
        )
    return targets, rotations
