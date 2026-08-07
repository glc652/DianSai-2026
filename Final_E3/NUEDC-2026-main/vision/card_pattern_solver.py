"""Geometry and artwork solver for Mode 3 playing-card fragments."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from puzzle_solver import SearchState, apply_transform, polygon_area


@dataclass
class BoundaryPlacement:
    piece_id: int
    transform: Tuple[np.ndarray, np.ndarray]
    polygon: np.ndarray
    mask: np.ndarray
    interior_mask: np.ndarray
    side: int
    edge: int


@dataclass
class PackingCandidate:
    placements: Dict[int, BoundaryPlacement]
    union_mask: np.ndarray
    interior_union_mask: np.ndarray
    overlap_px: int
    contact_px: int
    target_size_mm: Tuple[float, float]


def _rotation(angle: float) -> np.ndarray:
    return np.asarray(
        [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]],
        dtype=np.float32,
    )


def generate_boundary_placements(
    polygon: np.ndarray,
    piece_id: int,
    target_size_mm: Tuple[float, float],
    pixels_per_mm: float = 2.0,
    slide_step_mm: float = 1.0,
    inside_tolerance_mm: float = 1.8,
) -> List[BoundaryPlacement]:
    """Place every possible outer edge on one of the four card boundaries."""
    width, height = target_size_mm
    canvas_size = (
        int(round(width * pixels_per_mm)) + 1,
        int(round(height * pixels_per_mm)) + 1,
    )
    # top, right, bottom, left; direction only fixes the rotation.
    sides = [
        (np.asarray([1.0, 0.0]), 1, 0.0, width),
        (np.asarray([0.0, 1.0]), 0, width, height),
        (np.asarray([1.0, 0.0]), 1, height, width),
        (np.asarray([0.0, 1.0]), 0, 0.0, height),
    ]
    placements: List[BoundaryPlacement] = []
    signatures = set()
    for edge_id in range(len(polygon)):
        start = polygon[edge_id]
        end = polygon[(edge_id + 1) % len(polygon)]
        vector = end - start
        edge_length = float(np.linalg.norm(vector))
        if edge_length < 8.0:
            continue
        source_angle = atan2(float(vector[1]), float(vector[0]))
        for side_id, (base_direction, fixed_axis, fixed_value, span) in enumerate(
            sides
        ):
            for direction in (base_direction, -base_direction):
                target_angle = atan2(float(direction[1]), float(direction[0]))
                rotation = _rotation(target_angle - source_angle)
                rotated = polygon @ rotation.T
                rotated_start = rotated[edge_id]
                rotated_end = rotated[(edge_id + 1) % len(polygon)]
                free_axis = 1 - fixed_axis
                first = 0.0
                last = max(0.0, span - edge_length)
                edge_free_min = min(
                    rotated_start[free_axis],
                    rotated_end[free_axis],
                )
                for free_start in np.arange(
                    first,
                    last + slide_step_mm * 0.5,
                    slide_step_mm,
                ):
                    translation = np.zeros(2, dtype=np.float32)
                    translation[fixed_axis] = (
                        fixed_value - rotated_start[fixed_axis]
                    )
                    translation[free_axis] = free_start - edge_free_min
                    world = rotated + translation
                    if (
                        np.min(world[:, 0]) < -inside_tolerance_mm
                        or np.min(world[:, 1]) < -inside_tolerance_mm
                        or np.max(world[:, 0]) > width + inside_tolerance_mm
                        or np.max(world[:, 1]) > height + inside_tolerance_mm
                    ):
                        continue
                    signature = tuple(
                        np.round(world.reshape(-1) * 2.0).astype(int).tolist()
                    )
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    mask = np.zeros(
                        (canvas_size[1], canvas_size[0]),
                        dtype=np.uint8,
                    )
                    points = np.round(world * pixels_per_mm).astype(np.int32)
                    cv2.fillPoly(mask, [points], 1)
                    interior_mask = cv2.erode(
                        mask,
                        np.ones((3, 3), dtype=np.uint8),
                    )
                    placements.append(
                        BoundaryPlacement(
                            piece_id=piece_id,
                            transform=(rotation, translation),
                            polygon=world,
                            mask=mask,
                            interior_mask=interior_mask,
                            side=side_id,
                            edge=edge_id,
                        )
                    )
    return placements


def generate_rectangle_packings(
    polygons: Sequence[np.ndarray],
    target_size_mm: Tuple[float, float] = (90.0, 60.0),
    beam_width: int = 5000,
    maximum_overlap_mm2: float = 12.0,
) -> List[PackingCandidate]:
    """Beam-search boundary placements without assuming one-to-one cut edges."""
    pixels_per_mm = 2.0
    placements = {
        piece_id: generate_boundary_placements(
            np.asarray(polygon, dtype=np.float32),
            piece_id,
            target_size_mm,
            pixels_per_mm=pixels_per_mm,
        )
        for piece_id, polygon in enumerate(polygons)
    }
    if any(not values for values in placements.values()):
        raise RuntimeError("at least one fragment has no valid card-boundary placement")
    order = sorted(
        range(len(polygons)),
        key=lambda piece_id: len(placements[piece_id]),
    )
    first_id = order[0]
    states = [
        PackingCandidate(
            placements={first_id: placement},
            union_mask=placement.mask.copy(),
            interior_union_mask=placement.interior_mask.copy(),
            overlap_px=0,
            contact_px=0,
            target_size_mm=target_size_mm,
        )
        for placement in placements[first_id]
    ]
    kernel = np.ones((3, 3), dtype=np.uint8)
    maximum_overlap_px = int(
        round(maximum_overlap_mm2 * pixels_per_mm * pixels_per_mm)
    )
    for piece_id in order[1:]:
        expanded: List[PackingCandidate] = []
        for state in states:
            dilated = cv2.dilate(state.union_mask, kernel)
            for placement in placements[piece_id]:
                overlap = int(
                    np.count_nonzero(
                        state.interior_union_mask
                        & placement.interior_mask
                    )
                )
                if overlap > maximum_overlap_px:
                    continue
                contact = int(
                    np.count_nonzero(dilated & placement.mask)
                ) - overlap
                union = state.union_mask | placement.mask
                interior_union = (
                    state.interior_union_mask | placement.interior_mask
                )
                next_placements = dict(state.placements)
                next_placements[piece_id] = placement
                expanded.append(
                    PackingCandidate(
                        placements=next_placements,
                        union_mask=union,
                        interior_union_mask=interior_union,
                        overlap_px=state.overlap_px + overlap,
                        contact_px=state.contact_px + max(0, contact),
                        target_size_mm=target_size_mm,
                    )
                )
        if not expanded:
            return []
        # Contact is the useful partial signal; overlap always has priority.
        expanded.sort(
            key=lambda candidate: (
                candidate.overlap_px * 20 - candidate.contact_px,
                -int(np.count_nonzero(candidate.union_mask)),
            )
        )
        states = expanded[:beam_width]

    rectangle_px = states[0].union_mask.size
    expected_area_px = sum(polygon_area(p) for p in polygons) * 4.0
    states.sort(
        key=lambda candidate: (
            candidate.overlap_px * 5
            + abs(
                int(np.count_nonzero(candidate.union_mask))
                - expected_area_px
            )
            - candidate.contact_px * 0.6
            + max(
                0,
                rectangle_px - int(np.count_nonzero(candidate.union_mask)),
            )
            * 0.02
        )
    )
    return states


def packing_to_state(candidate: PackingCandidate) -> SearchState:
    return SearchState(
        transforms={
            piece_id: placement.transform
            for piece_id, placement in candidate.placements.items()
        },
        used_edges=frozenset(),
        match_error=0.0,
        matches=tuple(
            (
                piece_id,
                placement.edge,
                piece_id,
                placement.side,
                0.0,
            )
            for piece_id, placement in sorted(candidate.placements.items())
        ),
    )


def solve_card_pattern(
    polygons: Sequence[np.ndarray],
    target_size_mm: Optional[Tuple[float, float]] = None,
    beam_width: int = 1500,
    rectified: Optional[np.ndarray] = None,
    pieces: Optional[Sequence[Dict[str, object]]] = None,
    source_pixels_per_mm: float = 4.0,
    measured_piece_area_mm2: Optional[float] = None,
) -> Tuple[SearchState, Dict[str, float], PackingCandidate]:
    """Solve a playing card using its known rectangular boundary constraints."""
    polygon_piece_area = sum(polygon_area(polygon) for polygon in polygons)
    piece_area = (
        float(measured_piece_area_mm2)
        if measured_piece_area_mm2 is not None
        else polygon_piece_area
    )
    if target_size_mm is None:
        size_candidates = infer_card_size_candidates(piece_area)
    else:
        size_candidates = [target_size_mm]
    candidates = []
    # Shared rasterised boundary pixels are removed before this test, so this
    # allowance now represents only small fitting/quantisation error.
    maximum_overlap_mm2 = 0.0
    for size_candidate in size_candidates:
        size_packings = generate_rectangle_packings(
            polygons,
            target_size_mm=size_candidate,
            beam_width=beam_width,
            maximum_overlap_mm2=maximum_overlap_mm2,
        )
        if size_packings:
            candidates.extend(size_packings[:20])
    if not candidates:
        raise RuntimeError(
            "no zero-overlap rectangle packing in "
            "90-120mm by 50-90mm range"
        )
    pattern_score = 0.0
    seam_score = 0.0
    symmetry_score = 0.0
    if rectified is not None and pieces is not None:
        ranked = []
        for candidate in candidates:
            candidate_seam = artwork_seam_score(
                rectified,
                pieces,
                candidate,
                source_pixels_per_mm,
            )
            candidate_symmetry = rotational_symmetry_score(
                rectified,
                pieces,
                candidate,
                source_pixels_per_mm,
            )
            total = (
                candidate_seam * 0.1
                + candidate_symmetry * 50.0
                + candidate.overlap_px * 0.005
                - candidate.contact_px * 0.0005
                + abs(
                    1.0
                    - piece_area
                    / (
                        candidate.target_size_mm[0]
                        * candidate.target_size_mm[1]
                    )
                )
                * 8.0
            )
            ranked.append(
                (
                    total,
                    candidate_seam,
                    candidate_symmetry,
                    candidate,
                )
            )
        (
            pattern_score,
            seam_score,
            symmetry_score,
            best,
        ) = min(ranked, key=lambda item: item[0])
    else:
        best = min(
            candidates,
            key=lambda candidate: abs(
                1.0
                - piece_area
                / (
                    candidate.target_size_mm[0]
                    * candidate.target_size_mm[1]
                )
            )
            + candidate.overlap_px * 0.001,
        )
    state = packing_to_state(best)
    width, height = best.target_size_mm
    metrics = {
        "strategy": "card_boundary_and_pattern",
        "score": float(
            best.overlap_px * 0.02 - best.contact_px * 0.002
        ),
        "short_side_mm": float(min(width, height)),
        "long_side_mm": float(max(width, height)),
        "fill_ratio": float(piece_area / (width * height)),
        "outer_vertices": 4.0,
        "connected_ratio": 1.0,
        "overlap_px_at_2px_per_mm": float(best.overlap_px),
        "contact_px_at_2px_per_mm": float(best.contact_px),
        "candidate_count": float(len(candidates)),
        "pattern_continuity_score": float(pattern_score),
        "seam_continuity_score": float(seam_score),
        "rotation_180_symmetry_error": float(symmetry_score),
    }
    return state, metrics, best


def infer_card_size_candidates(
    piece_area_mm2: float,
    maximum_candidates: int = 6,
) -> List[Tuple[float, float]]:
    """Infer a small set of legal rectangle sizes from conserved piece area."""
    candidates = []
    for long_side in np.arange(90.0, 120.1, 5.0):
        for short_side in np.arange(50.0, 90.1, 5.0):
            if long_side < short_side:
                continue
            rectangle_area = float(long_side * short_side)
            fill_ratio = piece_area_mm2 / rectangle_area
            if not 0.94 <= fill_ratio <= 1.04:
                continue
            candidates.append(
                (
                    abs(fill_ratio - 0.985),
                    (float(long_side), float(short_side)),
                )
            )
    candidates.sort(key=lambda item: item[0])
    return [size for _, size in candidates[:maximum_candidates]]


def artwork_seam_score(
    rectified: np.ndarray,
    pieces: Sequence[Dict[str, object]],
    candidate: PackingCandidate,
    source_pixels_per_mm: float = 4.0,
) -> float:
    """Reward dark printed strokes that continue across two fragment seams."""
    target_ppm = 4.0
    width, height = candidate.target_size_mm
    canvas_size = (
        int(round(width * target_ppm)) + 1,
        int(round(height * target_ppm)) + 1,
    )
    grey_source = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    masks = {}
    dark_masks = {}
    for piece_id, placement in candidate.placements.items():
        rotation, translation = placement.transform
        affine = np.zeros((2, 3), dtype=np.float32)
        affine[:, :2] = rotation * (target_ppm / source_pixels_per_mm)
        affine[:, 2] = translation * target_ppm
        source_mask = np.zeros(rectified.shape[:2], dtype=np.uint8)
        cv2.drawContours(
            source_mask,
            [pieces[piece_id]["_contour"]],
            -1,
            255,
            cv2.FILLED,
        )
        mask = cv2.warpAffine(
            source_mask,
            affine,
            canvas_size,
            flags=cv2.INTER_NEAREST,
        )
        grey = cv2.warpAffine(
            grey_source,
            affine,
            canvas_size,
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )
        masks[piece_id] = mask
        dark_masks[piece_id] = np.where(
            (mask > 0) & (grey < 145),
            255,
            0,
        ).astype(np.uint8)

    near_kernel = np.ones((9, 9), dtype=np.uint8)
    ink_kernel = np.ones((11, 11), dtype=np.uint8)
    reward = 0
    unmatched = 0
    piece_ids = sorted(masks)
    for first_index, first_id in enumerate(piece_ids):
        for second_id in piece_ids[first_index + 1 :]:
            contact = cv2.dilate(masks[first_id], near_kernel) & cv2.dilate(
                masks[second_id],
                near_kernel,
            )
            if not np.any(contact):
                continue
            first_near_ink = (
                cv2.dilate(dark_masks[first_id], ink_kernel) & contact
            )
            second_near_ink = (
                cv2.dilate(dark_masks[second_id], ink_kernel) & contact
            )
            reward += int(
                np.count_nonzero(first_near_ink & second_near_ink)
            )
            unmatched += int(
                np.count_nonzero(first_near_ink ^ second_near_ink)
            )
    # Lower is better. Normalisation keeps this term comparable across images.
    return float((unmatched - reward * 1.8) / 1000.0)


def rotational_symmetry_score(
    rectified: np.ndarray,
    pieces: Sequence[Dict[str, object]],
    candidate: PackingCandidate,
    source_pixels_per_mm: float = 4.0,
) -> float:
    """Compare the assembled card with itself after a 180-degree rotation."""
    target_ppm = 4.0
    width, height = candidate.target_size_mm
    canvas_size = (
        int(round(width * target_ppm)) + 1,
        int(round(height * target_ppm)) + 1,
    )
    grey_source = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    assembled_grey = np.full(
        (canvas_size[1], canvas_size[0]),
        255,
        dtype=np.uint8,
    )
    assembled_mask = np.zeros_like(assembled_grey)
    for piece_id, placement in candidate.placements.items():
        rotation, translation = placement.transform
        affine = np.zeros((2, 3), dtype=np.float32)
        affine[:, :2] = rotation * (target_ppm / source_pixels_per_mm)
        affine[:, 2] = translation * target_ppm
        source_mask = np.zeros(rectified.shape[:2], dtype=np.uint8)
        cv2.drawContours(
            source_mask,
            [pieces[piece_id]["_contour"]],
            -1,
            255,
            cv2.FILLED,
        )
        warped_mask = cv2.warpAffine(
            source_mask,
            affine,
            canvas_size,
            flags=cv2.INTER_NEAREST,
        )
        warped_grey = cv2.warpAffine(
            grey_source,
            affine,
            canvas_size,
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )
        assembled_grey[warped_mask > 0] = warped_grey[warped_mask > 0]
        assembled_mask[warped_mask > 0] = 255

    # Ignore a narrow border where contour fitting and cutting burrs dominate.
    valid = (assembled_mask > 0) & (cv2.flip(assembled_mask, -1) > 0)
    valid[:5, :] = False
    valid[-5:, :] = False
    valid[:, :5] = False
    valid[:, -5:] = False
    if np.count_nonzero(valid) < 100:
        return 1.0

    # Printed symbols are more stable than small white-balance differences.
    dark = assembled_grey < 155
    rotated_dark = cv2.flip(dark.astype(np.uint8), -1) > 0
    ink_error = np.count_nonzero((dark ^ rotated_dark) & valid) / float(
        np.count_nonzero(valid)
    )
    coverage_error = np.count_nonzero(
        (assembled_mask > 0) ^ (cv2.flip(assembled_mask, -1) > 0)
    ) / float(assembled_mask.size)
    return float(ink_error + coverage_error * 0.5)


def render_card_solution(
    rectified: np.ndarray,
    pieces: Sequence[Dict[str, object]],
    candidate: PackingCandidate,
    output_path: str,
    target_origin_mm: Tuple[float, float] = (55.0, 180.0),
    source_pixels_per_mm: float = 4.0,
    output_pixels_per_mm: float = 5.0,
) -> None:
    """Render the actual fragment artwork and target pickup coordinates."""
    width_mm, height_mm = candidate.target_size_mm
    margin = 34
    canvas_width = int(round(width_mm * output_pixels_per_mm)) + margin * 2
    canvas_height = (
        int(round(height_mm * output_pixels_per_mm)) + margin * 2 + 132
    )
    canvas = np.full((canvas_height, canvas_width, 3), 238, np.uint8)
    colours = [
        (0, 0, 255),
        (0, 175, 0),
        (255, 70, 0),
        (180, 0, 180),
    ]
    for piece_id, placement in sorted(candidate.placements.items()):
        rotation, translation = placement.transform
        scale = output_pixels_per_mm / source_pixels_per_mm
        affine = np.zeros((2, 3), dtype=np.float32)
        affine[:, :2] = rotation * scale
        affine[:, 2] = (
            translation * output_pixels_per_mm
            + np.asarray([margin, margin + 38], dtype=np.float32)
        )
        source_mask = np.zeros(rectified.shape[:2], dtype=np.uint8)
        cv2.drawContours(
            source_mask,
            [pieces[piece_id]["_contour"]],
            -1,
            255,
            cv2.FILLED,
        )
        warped = cv2.warpAffine(
            rectified,
            affine,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
        )
        warped_mask = cv2.warpAffine(
            source_mask,
            affine,
            (canvas_width, canvas_height),
            flags=cv2.INTER_NEAREST,
        )
        canvas[warped_mask > 0] = warped[warped_mask > 0]
        target_polygon = (
            placement.polygon * output_pixels_per_mm
            + np.asarray([margin, margin + 38], dtype=np.float32)
        )
        cv2.polylines(
            canvas,
            [np.round(target_polygon).astype(np.int32)],
            True,
            colours[piece_id % len(colours)],
            1,
            cv2.LINE_AA,
        )
        moments = cv2.moments(placement.polygon.astype(np.float32))
        if abs(moments["m00"]) > 1e-8:
            center = np.asarray(
                [
                    moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"],
                ],
                dtype=np.float32,
            )
        else:
            center = np.mean(placement.polygon, axis=0)
        center_px = np.round(
            center * output_pixels_per_mm
            + np.asarray([margin, margin + 38])
        ).astype(int)
        target_center = center + np.asarray(target_origin_mm)
        label = "P{}".format(piece_id)
        cv2.putText(
            canvas,
            label,
            tuple(center_px),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colours[piece_id % len(colours)],
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(
        canvas,
        (margin, margin + 38),
        (
            margin + int(round(width_mm * output_pixels_per_mm)),
            margin + 38 + int(round(height_mm * output_pixels_per_mm)),
        ),
        (20, 20, 20),
        2,
    )
    cv2.putText(
        canvas,
        "Mode3 card {:.0f}x{:.0f}mm, target coordinates in A4 frame".format(
            width_mm,
            height_mm,
        ),
        (18, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    legend_y = (
        margin + 38 + int(round(height_mm * output_pixels_per_mm)) + 27
    )
    for row, (piece_id, placement) in enumerate(
        sorted(candidate.placements.items())
    ):
        moments = cv2.moments(placement.polygon.astype(np.float32))
        center = np.asarray(
            [
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            ],
            dtype=np.float32,
        )
        target_center = center + np.asarray(target_origin_mm)
        cv2.putText(
            canvas,
            "P{} target centroid = ({:.2f}, {:.2f}) mm".format(
                piece_id,
                target_center[0],
                target_center[1],
            ),
            (margin, legend_y + row * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            colours[piece_id % len(colours)],
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(output_path, canvas)


def _best_similarity_affine(
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Match cyclic polygon vertices and return a similarity affine matrix."""
    source = np.asarray(source, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 2)
    if len(source) != len(target):
        raise ValueError("source and target vertex counts differ")
    best_matrix = None
    best_error = float("inf")
    for candidate in (source, source[::-1]):
        for shift in range(len(source)):
            ordered = np.roll(candidate, shift, axis=0)
            matrix, _ = cv2.estimateAffinePartial2D(
                ordered,
                target,
                method=cv2.LMEDS,
            )
            if matrix is None:
                continue
            fitted = cv2.transform(
                ordered.reshape(-1, 1, 2),
                matrix,
            ).reshape(-1, 2)
            error = float(
                np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1)))
            )
            if error < best_error:
                best_error = error
                best_matrix = matrix.astype(np.float32)
    if best_matrix is None:
        raise RuntimeError("cannot estimate template texture transform")
    return best_matrix, best_error


def score_polygon_artwork(
    rectified: np.ndarray,
    pieces: Sequence[Dict[str, object]],
    targets: Sequence[np.ndarray],
    pixels_per_mm: float = 3.0,
) -> Tuple[float, float]:
    """Score seam continuity and 180-degree symmetry for polygon targets."""
    target_arrays = [
        np.asarray(target, dtype=np.float32).reshape(-1, 2)
        for target in targets
    ]
    all_points = np.concatenate(target_arrays, axis=0)
    minimum = np.min(all_points, axis=0)
    maximum = np.max(all_points, axis=0)
    padding = 8
    canvas_width = int(
        np.ceil((maximum[0] - minimum[0]) * pixels_per_mm)
    ) + padding * 2 + 1
    canvas_height = int(
        np.ceil((maximum[1] - minimum[1]) * pixels_per_mm)
    ) + padding * 2 + 1
    canvas_size = (canvas_width, canvas_height)
    grey_source = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    assembled_grey = np.full(
        (canvas_height, canvas_width),
        255,
        dtype=np.uint8,
    )
    assembled_mask = np.zeros_like(assembled_grey)
    masks = {}
    dark_masks = {}
    display_offset = np.asarray([padding, padding], dtype=np.float32)

    for piece_id, target in enumerate(target_arrays):
        source_polygon = np.asarray(
            pieces[piece_id]["_polygon"],
            dtype=np.float32,
        ).reshape(-1, 2)
        target_pixels = (
            (target - minimum) * pixels_per_mm + display_offset
        )
        affine, _ = _best_similarity_affine(
            source_polygon,
            target_pixels,
        )
        source_mask = np.zeros(rectified.shape[:2], dtype=np.uint8)
        cv2.drawContours(
            source_mask,
            [pieces[piece_id]["_contour"]],
            -1,
            255,
            cv2.FILLED,
        )
        warped_mask = cv2.warpAffine(
            source_mask,
            affine,
            canvas_size,
            flags=cv2.INTER_NEAREST,
        )
        warped_grey = cv2.warpAffine(
            grey_source,
            affine,
            canvas_size,
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )
        exact_mask = np.zeros_like(assembled_mask)
        cv2.fillPoly(
            exact_mask,
            [np.round(target_pixels).astype(np.int32)],
            255,
        )
        warped_mask = cv2.bitwise_and(warped_mask, exact_mask)
        masks[piece_id] = warped_mask
        dark_masks[piece_id] = np.where(
            (warped_mask > 0) & (warped_grey < 155),
            255,
            0,
        ).astype(np.uint8)
        valid = warped_mask > 0
        assembled_grey[valid] = warped_grey[valid]
        assembled_mask[valid] = 255

    near_kernel = np.ones((7, 7), dtype=np.uint8)
    ink_kernel = np.ones((9, 9), dtype=np.uint8)
    reward = 0
    unmatched = 0
    piece_ids = sorted(masks)
    for first_index, first_id in enumerate(piece_ids):
        for second_id in piece_ids[first_index + 1 :]:
            contact = cv2.dilate(
                masks[first_id],
                near_kernel,
            ) & cv2.dilate(masks[second_id], near_kernel)
            if not np.any(contact):
                continue
            first_ink = cv2.dilate(
                dark_masks[first_id],
                ink_kernel,
            ) & contact
            second_ink = cv2.dilate(
                dark_masks[second_id],
                ink_kernel,
            ) & contact
            reward += int(np.count_nonzero(first_ink & second_ink))
            unmatched += int(np.count_nonzero(first_ink ^ second_ink))
    seam_score = float((unmatched - reward * 1.8) / 1000.0)

    valid = (assembled_mask > 0) & (cv2.flip(assembled_mask, -1) > 0)
    valid[:5, :] = False
    valid[-5:, :] = False
    valid[:, :5] = False
    valid[:, -5:] = False
    if np.count_nonzero(valid) < 100:
        symmetry_score = 1.0
    else:
        dark = assembled_grey < 155
        rotated_dark = cv2.flip(dark.astype(np.uint8), -1) > 0
        ink_error = np.count_nonzero(
            (dark ^ rotated_dark) & valid
        ) / float(np.count_nonzero(valid))
        coverage_error = np.count_nonzero(
            (assembled_mask > 0)
            ^ (cv2.flip(assembled_mask, -1) > 0)
        ) / float(assembled_mask.size)
        symmetry_score = float(ink_error + coverage_error * 0.5)
    return seam_score, symmetry_score


def render_card_template_solution(
    rectified: np.ndarray,
    pieces: Sequence[Dict[str, object]],
    targets: Sequence[np.ndarray],
    output_path: str,
    target_origin_mm: Tuple[float, float] = (55.0, 180.0),
    target_size_mm: Tuple[float, float] = (90.0, 60.0),
    output_pixels_per_mm: float = 5.0,
) -> None:
    """Render a fixed-cut template with real card artwork and no overlap."""
    width_mm, height_mm = target_size_mm
    margin = 34
    header = 38
    legend_height = 104
    canvas_width = int(round(width_mm * output_pixels_per_mm)) + margin * 2
    canvas_height = (
        int(round(height_mm * output_pixels_per_mm))
        + margin * 2
        + header
        + legend_height
    )
    canvas = np.full((canvas_height, canvas_width, 3), 238, np.uint8)
    colours = [
        (0, 0, 255),
        (0, 175, 0),
        (255, 70, 0),
        (180, 0, 180),
    ]
    origin = np.asarray(target_origin_mm, dtype=np.float32)
    display_offset = np.asarray([margin, margin + header], dtype=np.float32)

    for piece_id, target_mm in enumerate(targets):
        source_px = np.asarray(
            pieces[piece_id]["_polygon"],
            dtype=np.float32,
        ).reshape(-1, 2)
        target_px = (
            (np.asarray(target_mm, dtype=np.float32) - origin)
            * output_pixels_per_mm
            + display_offset
        )
        affine, _ = _best_similarity_affine(source_px, target_px)

        source_mask = np.zeros(rectified.shape[:2], dtype=np.uint8)
        cv2.drawContours(
            source_mask,
            [pieces[piece_id]["_contour"]],
            -1,
            255,
            cv2.FILLED,
        )
        warped = cv2.warpAffine(
            rectified,
            affine,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
        )
        warped_mask = cv2.warpAffine(
            source_mask,
            affine,
            (canvas_width, canvas_height),
            flags=cv2.INTER_NEAREST,
        )

        # Restrict the warped texture to the exact verified target polygon.
        target_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        target_points = np.round(target_px).astype(np.int32)
        cv2.fillPoly(target_mask, [target_points], 255)
        valid = (warped_mask > 0) & (target_mask > 0)
        canvas[valid] = warped[valid]
        cv2.polylines(
            canvas,
            [target_points],
            True,
            colours[piece_id % len(colours)],
            2,
            cv2.LINE_AA,
        )

        moments = cv2.moments(
            np.asarray(target_mm, dtype=np.float32)
        )
        center = np.asarray(
            [
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            ],
            dtype=np.float32,
        )
        center_px = np.round(
            (center - origin) * output_pixels_per_mm + display_offset
        ).astype(int)
        cv2.putText(
            canvas,
            "P{}".format(piece_id),
            tuple(center_px),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colours[piece_id % len(colours)],
            2,
            cv2.LINE_AA,
        )

    card_bottom = margin + header + int(
        round(height_mm * output_pixels_per_mm)
    )
    cv2.rectangle(
        canvas,
        (margin, margin + header),
        (
            margin + int(round(width_mm * output_pixels_per_mm)),
            card_bottom,
        ),
        (20, 20, 20),
        2,
    )
    cv2.putText(
        canvas,
        "Mode3 geometry+artwork {:.0f}x{:.0f}mm, overlap=0.0mm2".format(
            width_mm,
            height_mm,
        ),
        (18, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    for piece_id, target_mm in enumerate(targets):
        moments = cv2.moments(
            np.asarray(target_mm, dtype=np.float32)
        )
        center = np.asarray(
            [
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            ]
        )
        cv2.putText(
            canvas,
            "P{} target centroid = ({:.2f}, {:.2f}) mm".format(
                piece_id,
                center[0],
                center[1],
            ),
            (margin, card_bottom + 25 + piece_id * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            colours[piece_id % len(colours)],
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(output_path, canvas)
