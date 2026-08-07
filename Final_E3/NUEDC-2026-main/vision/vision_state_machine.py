r"""三模式拼图视觉状态机及控制联调接口。

标准输入协议示例：
    MODE,1
    START,C:\Users\77019\Desktop\test4.jpg
    STATUS
    NEXT
    ACK,0,OK
    RESET
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from fragment_vision import (
    DetectConfig,
    PIXELS_PER_MM,
    choose_pickup_point,
    contour_centroid,
    detect_pieces,
    draw_results,
    make_payload,
    process_frame,
    remove_false_short_edges,
    rectify_a4,
    resize_for_debug,
    sharpen_polygon_from_contour,
)
from card_pattern_solver import (
    render_card_template_solution,
    score_polygon_artwork,
)
from puzzle_solver import (
    generate_edge_search_states,
    normalize_solution,
    rank_unknown_rectangle_candidates,
    solve_with_template,
    unknown_rectangle_score,
)
from solve_puzzle_image import polygon_centroid, render_solution


class TaskMode(IntEnum):
    FIXED_COLOR = 1
    UNKNOWN_WHITE = 2
    CARD_PATTERN = 3


class VisionState(str, Enum):
    IDLE = "idle"
    MODE_SELECTED = "mode_selected"
    CAPTURING = "capturing"
    CALIBRATING = "calibrating"
    DETECTING = "detecting"
    SOLVING = "solving"
    PLAN_READY = "plan_ready"
    WAITING_ACK = "waiting_ack"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class MoveAction:
    sequence: int
    piece_id: int
    source_pickup_mm: List[float]
    source_centroid_mm: List[float]
    target_centroid_mm: List[float]
    rotation_delta_deg: float
    status: str = "pending"
    retries: int = 0


TaskHandler = Callable[[str], Dict[str, object]]


class VisionStateMachine:
    def __init__(
        self,
        template_path: Optional[str] = None,
        output_directory: Optional[str] = None,
        maximum_retries: int = 2,
        progress_callback: Optional[
            Callable[[float, str], None]
        ] = None,
    ) -> None:
        workspace = Path(__file__).resolve().parent
        self.template_path = Path(
            template_path
            or workspace / "fixed_puzzle_template.json"
        )
        self.output_directory = Path(
            output_directory or workspace / "tmp"
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.maximum_retries = maximum_retries
        self.progress_callback = progress_callback
        self.handlers: Dict[TaskMode, TaskHandler] = {
            TaskMode.FIXED_COLOR: self._run_fixed_color,
            TaskMode.UNKNOWN_WHITE: self._run_unknown_white,
            TaskMode.CARD_PATTERN: self._run_card_pattern,
        }
        self.reset()

    def _report_progress(
        self,
        progress: float,
        message: str,
    ) -> None:
        if self.progress_callback is not None:
            self.progress_callback(
                max(0.0, min(1.0, float(progress))),
                str(message),
            )

    def reset(self) -> Dict[str, object]:
        self.state = VisionState.IDLE
        self.mode: Optional[TaskMode] = None
        self.plan_id: Optional[str] = None
        self.actions: List[MoveAction] = []
        self.active_action: Optional[MoveAction] = None
        self.metrics: Dict[str, object] = {}
        self.timing_ms: Dict[str, float] = {}
        self.last_error: Optional[str] = None
        self.detection_image: Optional[str] = None
        self.solution_image: Optional[str] = None
        return self.status()

    def register_handler(
        self,
        mode: TaskMode,
        handler: TaskHandler,
    ) -> None:
        """后续为模式2、模式3注册对应视觉处理函数。"""
        self.handlers[mode] = handler

    def select_mode(self, mode: int) -> Dict[str, object]:
        selected = TaskMode(mode)
        self.mode = selected
        self.plan_id = None
        self.actions = []
        self.active_action = None
        self.metrics = {}
        self.last_error = None
        self.state = VisionState.MODE_SELECTED
        return self.status()

    def start(self, image_path: str) -> Dict[str, object]:
        if self.mode is None:
            raise RuntimeError("请先通过 MODE 选择赛题模式")
        handler = self.handlers.get(self.mode)
        if handler is None:
            raise NotImplementedError(
                "模式{}尚未接入视觉处理器".format(int(self.mode))
            )

        self.plan_id = uuid.uuid4().hex[:12]
        self.actions = []
        self.active_action = None
        self.last_error = None
        try:
            result = handler(image_path)
            self.actions = result["actions"]
            self.metrics = result["metrics"]
            self.timing_ms = result["timing_ms"]
            self.detection_image = result["detection_image"]
            self.solution_image = result["solution_image"]
            self.state = VisionState.PLAN_READY
            return {
                "plan_id": self.plan_id,
                "state": self.state.value,
                "action_count": len(self.actions),
                "metrics": self.metrics,
                "timing_ms": self.timing_ms,
                "detection_image": self.detection_image,
                "solution_image": self.solution_image,
            }
        except Exception as error:
            self.state = VisionState.ERROR
            self.last_error = str(error)
            raise

    def next_action(self) -> Dict[str, object]:
        if self.state == VisionState.WAITING_ACK and self.active_action:
            return asdict(self.active_action)
        if self.state != VisionState.PLAN_READY:
            raise RuntimeError(
                "当前状态 {} 不能获取下一动作".format(self.state.value)
            )

        pending = [
            action
            for action in self.actions
            if action.status in ("pending", "retry")
        ]
        if not pending:
            self.state = VisionState.COMPLETED
            return {"completed": True}

        self.active_action = pending[0]
        self.active_action.status = "sent"
        self.state = VisionState.WAITING_ACK
        return asdict(self.active_action)

    def acknowledge(
        self,
        piece_id: int,
        success: bool,
    ) -> Dict[str, object]:
        if self.state != VisionState.WAITING_ACK or self.active_action is None:
            raise RuntimeError("当前没有等待确认的动作")
        if self.active_action.piece_id != piece_id:
            raise RuntimeError(
                "确认碎片{}与当前碎片{}不一致".format(
                    piece_id,
                    self.active_action.piece_id,
                )
            )

        if success:
            self.active_action.status = "done"
            self.active_action = None
            if all(action.status == "done" for action in self.actions):
                self.state = VisionState.COMPLETED
            else:
                self.state = VisionState.PLAN_READY
        else:
            self.active_action.retries += 1
            if self.active_action.retries > self.maximum_retries:
                self.active_action.status = "failed"
                self.state = VisionState.ERROR
                self.last_error = "碎片{}移动连续失败".format(piece_id)
            else:
                self.active_action.status = "retry"
                self.active_action = None
                self.state = VisionState.PLAN_READY
        return self.status()

    def get_plan(self) -> Dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "mode": int(self.mode) if self.mode is not None else None,
            "state": self.state.value,
            "actions": [asdict(action) for action in self.actions],
            "metrics": self.metrics,
            "timing_ms": self.timing_ms,
        }

    def status(self) -> Dict[str, object]:
        completed = sum(
            action.status == "done" for action in self.actions
        )
        return {
            "state": self.state.value,
            "mode": int(self.mode) if self.mode is not None else None,
            "plan_id": self.plan_id,
            "completed_actions": completed,
            "total_actions": len(self.actions),
            "active_piece_id": (
                self.active_action.piece_id
                if self.active_action is not None
                else None
            ),
            "last_error": self.last_error,
        }

    def handle_command(self, command: str) -> Dict[str, object]:
        """处理一行控制命令，适合后续串口或TCP转发。"""
        command = command.strip()
        if not command:
            raise ValueError("空命令")
        operation, separator, arguments = command.partition(",")
        operation = operation.strip().upper()

        if operation == "PING":
            return {"pong": True, "state": self.state.value}
        if operation == "RESET":
            return self.reset()
        if operation == "STATUS":
            return self.status()
        if operation == "PLAN":
            return self.get_plan()
        if operation == "MODE":
            if not separator:
                raise ValueError("MODE 命令缺少模式编号")
            return self.select_mode(int(arguments.strip()))
        if operation == "START":
            if not separator:
                raise ValueError("START 命令缺少图片路径")
            return self.start(arguments.strip())
        if operation == "NEXT":
            return self.next_action()
        if operation == "ACK":
            parts = [part.strip() for part in arguments.split(",")]
            if len(parts) != 2:
                raise ValueError("ACK 格式应为 ACK,piece_id,OK|FAIL")
            success = parts[1].upper() == "OK"
            if parts[1].upper() not in ("OK", "FAIL"):
                raise ValueError("ACK 结果只能是 OK 或 FAIL")
            return self.acknowledge(int(parts[0]), success)
        raise ValueError("未知命令：{}".format(operation))

    def _run_fixed_color(self, image_path: str) -> Dict[str, object]:
        total_start = time.perf_counter()
        self.state = VisionState.CAPTURING
        load_start = time.perf_counter()
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError("无法读取图片：{}".format(image_path))
        frame = resize_for_debug(frame)
        load_ms = (time.perf_counter() - load_start) * 1000.0

        self.state = VisionState.CALIBRATING
        vision_start = time.perf_counter()
        _, detection_annotated, detection = process_frame(
            frame,
            DetectConfig(),
        )
        self.state = VisionState.DETECTING
        if len(detection["pieces"]) != 4:
            raise RuntimeError(
                "模式1要求4块碎片，当前识别到{}块".format(
                    len(detection["pieces"])
                )
            )
        vision_ms = (time.perf_counter() - vision_start) * 1000.0

        self.state = VisionState.SOLVING
        solve_start = time.perf_counter()
        self._report_progress(0.05, "match fixed template")
        polygons = [
            np.asarray(piece["vertices_mm"], dtype=np.float32)
            for piece in detection["pieces"]
        ]
        targets, rotations, metrics, state = self._solve_fixed_polygons(
            polygons
        )
        self._report_progress(0.75, "add assembly clearance")
        matches = self._infer_adjacency_matches(
            targets,
            existing_matches=state.matches,
        )
        metrics["fill_ratio_before_clearance"] = float(
            metrics["fill_ratio"]
        )
        targets, clearance_metrics = self._apply_target_clearance(
            targets,
            matches,
            target_origin=np.asarray([55.0, 180.0], dtype=np.float32),
            clearance_mm=3.0,
        )
        metrics.update(clearance_metrics)
        solve_ms = (time.perf_counter() - solve_start) * 1000.0

        detection_image = self.output_directory / "fsm_detection.jpg"
        solution_image = self.output_directory / "fsm_solution.jpg"
        cv2.imwrite(str(detection_image), detection_annotated)
        render_solution(
            targets,
            rotations,
            metrics,
            str(solution_image),
        )
        self._report_progress(1.0, "Mode 1 solution ready")

        # 大片优先放置，最后放最小片，减少已放碎片被碰动的风险。
        order = sorted(
            range(len(polygons)),
            key=lambda piece_id: detection["pieces"][piece_id]["area_px2"],
            reverse=True,
        )
        actions = []
        for sequence, piece_id in enumerate(order):
            piece = detection["pieces"][piece_id]
            target_center = polygon_centroid(targets[piece_id])
            actions.append(
                MoveAction(
                    sequence=sequence,
                    piece_id=piece_id,
                    source_pickup_mm=[
                        round(float(value), 2)
                        for value in piece["pickup_mm"]
                    ],
                    source_centroid_mm=[
                        round(float(value), 2)
                        for value in piece["centroid_mm"]
                    ],
                    target_centroid_mm=[
                        round(float(value), 2)
                        for value in target_center
                    ],
                    rotation_delta_deg=round(
                        self._normalize_angle(rotations[piece_id]),
                        2,
                    ),
                )
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "actions": actions,
            "metrics": metrics,
            "timing_ms": {
                "image_load": round(load_ms, 2),
                "vision": round(vision_ms, 2),
                "puzzle_solve": round(solve_ms, 2),
                "total": round(total_ms, 2),
            },
            "detection_image": str(detection_image),
            "solution_image": str(solution_image),
        }

    def _solve_fixed_polygons(self, polygons):
        if not self.template_path.is_file():
            raise RuntimeError(
                "missing Mode 1 template: {}".format(
                    self.template_path
                )
            )
        with self.template_path.open(
            "r",
            encoding="utf-8",
        ) as template_file:
            template_data = json.load(template_file)
        template_polygons = [
            np.asarray(
                piece["target_vertices_mm"],
                dtype=np.float32,
            )
            for piece in template_data.get("pieces", [])
        ]
        if len(template_polygons) != 4:
            raise RuntimeError(
                "Mode 1 template must contain four pieces"
            )

        state, metrics = solve_with_template(
            polygons,
            template_polygons,
            maximum_rms_mm=8.0,
        )
        assignment = [
            int(template_id)
            for template_id in metrics["template_assignment"]
        ]
        targets = [
            template_polygons[assignment[piece_id]].copy()
            for piece_id in range(len(polygons))
        ]
        rotations = [
            float(
                np.degrees(
                    np.arctan2(
                        state.transforms[piece_id][0][1, 0],
                        state.transforms[piece_id][0][0, 0],
                    )
                )
            )
            for piece_id in range(len(polygons))
        ]
        metrics["template_match_mode"] = "fixed_shape"
        return targets, rotations, metrics, state

    def _run_unknown_white(self, image_path: str) -> Dict[str, object]:
        """模式2：现场未知白色碎片，自动推断目标矩形宽高。"""
        total_start = time.perf_counter()
        self.state = VisionState.CAPTURING
        load_start = time.perf_counter()
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError("无法读取图片：{}".format(image_path))
        frame = resize_for_debug(frame)
        load_ms = (time.perf_counter() - load_start) * 1000.0

        self.state = VisionState.CALIBRATING
        vision_start = time.perf_counter()
        _, detection_annotated, detection = process_frame(
            frame,
            DetectConfig(),
        )
        self.state = VisionState.DETECTING
        piece_count = len(detection["pieces"])
        if not 2 <= piece_count <= 4:
            raise RuntimeError(
                "模式2要求2至4块碎片，当前识别到{}块".format(
                    piece_count
                )
            )
        for piece in detection["pieces"]:
            vertex_count = len(piece["vertices_mm"])
            if vertex_count > 6:
                raise RuntimeError(
                    "P{}检测到{}条边，超过模式2上限5条".format(
                        piece["id"],
                        vertex_count,
                    )
                )
        vision_ms = (time.perf_counter() - vision_start) * 1000.0

        self.state = VisionState.SOLVING
        solve_start = time.perf_counter()
        self._report_progress(0.02, "prepare Mode 2 search")
        primary_polygons = [
            np.asarray(piece["vertices_mm"], dtype=np.float32)
            for piece in detection["pieces"]
        ]
        polygon_models = [(-1, primary_polygons)]
        for piece_id, piece in enumerate(detection["pieces"]):
            alternatives = piece.get("polygon_candidates_mm", [])
            if len(alternatives) < 2:
                continue
            alternate_polygons = list(primary_polygons)
            alternate_polygons[piece_id] = np.asarray(
                alternatives[1],
                dtype=np.float32,
            )
            polygon_models.append((piece_id, alternate_polygons))

        candidates = None
        polygons = primary_polygons
        full_edge_error = None
        selected_alternate_piece = -1
        model_errors = []
        model_count = len(polygon_models)
        for model_id, (
            alternate_piece_id,
            candidate_polygons,
        ) in enumerate(polygon_models):
            def model_progress(local_value, message):
                self._report_progress(
                    0.03
                    + 0.88
                    * (model_id + local_value)
                    / max(1, model_count),
                    "model {}/{}: {}".format(
                        model_id + 1,
                        model_count,
                        message,
                    ),
                )

            current_full_error = None
            try:
                model_progress(0.02, "full-edge search")
                current_candidates = (
                    rank_unknown_rectangle_candidates(
                        candidate_polygons,
                        edge_tolerance=0.18,
                        beam_width=80,
                        time_limit_seconds=30.0,
                        allow_partitioned_edges=False,
                        progress_callback=lambda value, message: (
                            model_progress(
                                0.02 + 0.55 * value,
                                "full edges: " + message,
                            )
                        ),
                    )
                )
            except RuntimeError as error:
                current_full_error = str(error)
                try:
                    model_progress(
                        0.58,
                        "partitioned-edge fallback",
                    )
                    current_candidates = (
                        rank_unknown_rectangle_candidates(
                            candidate_polygons,
                            edge_tolerance=0.18,
                            beam_width=80,
                            time_limit_seconds=30.0,
                            allow_partitioned_edges=True,
                            progress_callback=lambda value, message: (
                                model_progress(
                                    0.58 + 0.40 * value,
                                    "partitioned edges: " + message,
                                )
                            ),
                        )
                    )
                except RuntimeError as partition_error:
                    model_errors.append(
                        "model {}: {}".format(
                            model_id + 1,
                            partition_error,
                        )
                    )
                    continue
            candidates = current_candidates
            polygons = candidate_polygons
            full_edge_error = current_full_error
            selected_alternate_piece = alternate_piece_id
            break

        if candidates is None:
            detail = model_errors[-1] if model_errors else "no model"
            raise RuntimeError(
                "Mode 2 found no solution across {} contour models; {}"
                .format(model_count, detail)
            )
        rejected = []
        selected = None
        for candidate_id, (state, metrics) in enumerate(candidates):
            self._report_progress(
                0.92
                + 0.05
                * candidate_id
                / max(1, len(candidates)),
                "validate candidate {}/{}".format(
                    candidate_id + 1,
                    len(candidates),
                ),
            )
            candidate_targets, candidate_rotations = normalize_solution(
                polygons,
                state,
                target_origin_mm=(55.0, 180.0),
            )
            matches = self._infer_adjacency_matches(
                candidate_targets,
                existing_matches=state.matches,
            )
            try:
                candidate_targets, clearance_metrics = (
                    self._apply_target_clearance(
                        candidate_targets,
                        matches,
                        target_origin=np.asarray(
                            [55.0, 180.0],
                            dtype=np.float32,
                        ),
                        clearance_mm=2.0,
                        size_tolerance_mm=7.0,
                    )
                )
            except RuntimeError as error:
                rejected.append(str(error))
                continue
            metrics.update(clearance_metrics)
            metrics["beam_width"] = 80.0
            metrics["partition_beam_width"] = 80.0
            metrics["contour_model_count"] = float(model_count)
            metrics["alternate_contour_piece_id"] = float(
                selected_alternate_piece
            )
            metrics["clearance_candidate_rank"] = float(candidate_id)
            if full_edge_error is not None:
                metrics["full_edge_fallback_reason"] = full_edge_error
            selected = (
                state,
                metrics,
                candidate_targets,
                candidate_rotations,
            )
            break
        if selected is None:
            detail = rejected[0] if rejected else "no candidate"
            raise RuntimeError(
                "Mode 2 candidates failed final clearance: " + detail
            )
        state, metrics, targets, rotations = selected
        solve_ms = (time.perf_counter() - solve_start) * 1000.0

        detection_image = (
            self.output_directory / "fsm_mode2_detection.jpg"
        )
        solution_image = (
            self.output_directory / "fsm_mode2_solution.jpg"
        )
        cv2.imwrite(str(detection_image), detection_annotated)
        render_solution(
            targets,
            rotations,
            metrics,
            str(solution_image),
        )
        self._report_progress(1.0, "Mode 2 solution ready")

        order = sorted(
            range(len(polygons)),
            key=lambda piece_id: detection["pieces"][piece_id]["area_px2"],
            reverse=True,
        )
        actions = []
        for sequence, piece_id in enumerate(order):
            piece = detection["pieces"][piece_id]
            target_center = polygon_centroid(targets[piece_id])
            actions.append(
                MoveAction(
                    sequence=sequence,
                    piece_id=piece_id,
                    source_pickup_mm=[
                        round(float(value), 2)
                        for value in piece["pickup_mm"]
                    ],
                    source_centroid_mm=[
                        round(float(value), 2)
                        for value in piece["centroid_mm"]
                    ],
                    target_centroid_mm=[
                        round(float(value), 2)
                        for value in target_center
                    ],
                    rotation_delta_deg=round(
                        self._normalize_angle(rotations[piece_id]),
                        2,
                    ),
                )
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "actions": actions,
            "metrics": metrics,
            "timing_ms": {
                "image_load": round(load_ms, 2),
                "vision": round(vision_ms, 2),
                "puzzle_solve": round(solve_ms, 2),
                "total": round(total_ms, 2),
            },
            "detection_image": str(detection_image),
            "solution_image": str(solution_image),
        }

    def _run_card_pattern(self, image_path: str) -> Dict[str, object]:
        """Mode 3: rectangular playing card with artwork continuity."""
        total_start = time.perf_counter()
        self.state = VisionState.CAPTURING
        load_start = time.perf_counter()
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError("cannot read image: {}".format(image_path))
        frame = resize_for_debug(frame)
        load_ms = (time.perf_counter() - load_start) * 1000.0

        self.state = VisionState.CALIBRATING
        vision_start = time.perf_counter()
        config = DetectConfig()
        rectified, calibration = rectify_a4(frame, config)
        _, pieces, background_hue = detect_pieces(rectified, config)
        self._regularize_card_pieces(pieces, config)
        detection_annotated = draw_results(
            rectified,
            pieces,
            background_hue,
        )
        detection = make_payload(
            rectified,
            pieces,
            background_hue,
            calibration,
        )
        self.state = VisionState.DETECTING
        piece_count = len(pieces)
        if not 2 <= piece_count <= 4:
            raise RuntimeError(
                "Mode 3 requires 2 to 4 pieces; detected {}".format(
                    piece_count
                )
            )
        if any(len(piece["vertices_mm"]) > 5 for piece in detection["pieces"]):
            raise RuntimeError("Mode 3 supports at most five edges per piece")
        vision_ms = (time.perf_counter() - vision_start) * 1000.0

        self.state = VisionState.SOLVING
        solve_start = time.perf_counter()
        self._report_progress(0.03, "prepare Mode 3 geometry")
        polygons = [
            np.asarray(piece["vertices_mm"], dtype=np.float32)
            for piece in detection["pieces"]
        ]
        target_origin = np.asarray([55.0, 180.0], dtype=np.float32)
        geometry_search_start = time.perf_counter()
        geometry_filter_ms = 0.0
        geometry_candidates = []
        raw_states = []
        selected_beam_width = 80
        # Most frames are solved quickly with beam=80. Difficult card
        # layouts can lose the correct partial assembly during beam pruning,
        # so retry once with a wider beam only when all first-pass states
        # fail the rectangular-geometry checks.
        for search_attempt, beam_width in enumerate((80, 160)):
            selected_beam_width = beam_width
            raw_states = generate_edge_search_states(
                polygons,
                edge_tolerance=0.18,
                beam_width=beam_width,
                allow_partitioned_edges=True,
                deduplicate_states=True,
                progress_callback=lambda value, message, attempt=search_attempt: (
                    self._report_progress(
                        0.05 + 0.54 * (attempt + value) / 2.0,
                        "geometry beam={}: {}".format(
                            (80, 160)[attempt],
                            message,
                        ),
                    )
                ),
            )
            if not raw_states:
                continue

            filter_start = time.perf_counter()
            signatures = set()
            geometry_candidates = []
            for state_id, candidate_state in enumerate(raw_states):
                if state_id % max(1, len(raw_states) // 20) == 0:
                    self._report_progress(
                        0.59
                        + 0.18
                        * (search_attempt + state_id / max(1, len(raw_states)))
                        / 2.0,
                        "filter beam={} {}/{}".format(
                            beam_width,
                            state_id + 1,
                            len(raw_states),
                        ),
                    )
                geometry_score, candidate_metrics = unknown_rectangle_score(
                    polygons,
                    candidate_state,
                )
                if (
                    candidate_metrics["connected_ratio"] < 0.97
                    or not 0.88 <= candidate_metrics["fill_ratio"] <= 1.08
                    or candidate_metrics["dimension_error"] > 0.06
                    or candidate_metrics["outer_vertices"] > 8.0
                ):
                    continue
                candidate_targets, candidate_rotations = normalize_solution(
                    polygons,
                    candidate_state,
                    target_origin_mm=tuple(target_origin),
                )
                overlap_area_mm2 = self._total_overlap_area(
                    candidate_targets
                )
                # Perspective and contour quantization can create a few
                # square millimetres of provisional overlap. The selected
                # layout is separated later and still must pass the final
                # zero-overlap validation.
                if overlap_area_mm2 > 10.0:
                    continue
                normalized_points = np.concatenate(
                    candidate_targets,
                    axis=0,
                )
                signature = tuple(
                    np.round(
                        (normalized_points - target_origin) * 2.0
                    )
                    .astype(np.int32)
                    .reshape(-1)
                    .tolist()
                )
                if signature in signatures:
                    continue
                signatures.add(signature)
                geometry_candidates.append(
                    (
                        float(geometry_score),
                        candidate_state,
                        candidate_metrics,
                        candidate_targets,
                        candidate_rotations,
                        overlap_area_mm2,
                    )
                )
            geometry_filter_ms += (
                time.perf_counter() - filter_start
            ) * 1000.0
            if geometry_candidates:
                break

        geometry_search_ms = (
            time.perf_counter() - geometry_search_start
        ) * 1000.0 - geometry_filter_ms

        if not geometry_candidates:
            raise RuntimeError(
                "Mode 3 found no zero-overlap rectangular geometry "
                "after beam widths 80 and 160"
            )
        geometry_candidates.sort(key=lambda item: item[0])
        artwork_score_start = time.perf_counter()
        artwork_candidates = []
        artwork_limit = min(60, len(geometry_candidates))
        for artwork_id, (
            geometry_score,
            candidate_state,
            candidate_metrics,
            candidate_targets,
            candidate_rotations,
            overlap_area_mm2,
        ) in enumerate(geometry_candidates[:artwork_limit]):
            self._report_progress(
                0.80
                + 0.16 * artwork_id / max(1, artwork_limit),
                "score artwork {}/{}".format(
                    artwork_id + 1,
                    artwork_limit,
                ),
            )
            seam_score, symmetry_score = score_polygon_artwork(
                rectified,
                pieces,
                candidate_targets,
            )
            combined_score = (
                geometry_score * 3.0
                + seam_score * 0.15
                + symmetry_score * 25.0
                + overlap_area_mm2 * 2.0
            )
            artwork_candidates.append(
                (
                    combined_score,
                    geometry_score,
                    seam_score,
                    symmetry_score,
                    candidate_state,
                    candidate_metrics,
                    candidate_targets,
                    candidate_rotations,
                    overlap_area_mm2,
                )
            )
        artwork_score_ms = (
            time.perf_counter() - artwork_score_start
        ) * 1000.0
        (
            combined_score,
            geometry_score,
            seam_score,
            symmetry_score,
            state,
            metrics,
            targets,
            rotations,
            overlap_before_adjust_mm2,
        ) = min(artwork_candidates, key=lambda item: item[0])
        targets = self._separate_small_overlaps(
            targets,
            target_origin=target_origin,
        )
        targets, adjacency_gaps = self._add_assembly_clearance(
            targets,
            matches=self._infer_adjacency_matches(
                targets,
                existing_matches=state.matches,
            ),
            target_origin=target_origin,
            clearance_mm=3.0,
        )
        targets = self._separate_small_overlaps(
            targets,
            target_origin=target_origin,
        )
        overlap_area_mm2 = self._validate_zero_overlap(
            targets,
            tolerance_mm2=0.001,
        )
        if adjacency_gaps and max(adjacency_gaps) > 20.0:
            raise RuntimeError(
                "Mode 3 adjacent-piece gap exceeds 20mm: {:.2f}mm".format(
                    max(adjacency_gaps)
                )
            )
        all_target_points = np.concatenate(targets, axis=0).astype(
            np.float32
        )
        (_, _), (target_width, target_height), _ = cv2.minAreaRect(
            all_target_points
        )
        target_short, target_long = sorted(
            (float(target_width), float(target_height))
        )
        if not (
            90.0 <= target_long <= 120.0
            and 50.0 <= target_short <= 90.0
        ):
            raise RuntimeError(
                "Mode 3 clearance layout exceeds legal size: "
                "{:.2f}x{:.2f}mm".format(
                    target_long,
                    target_short,
                )
            )
        pieces_area = sum(
            abs(float(cv2.contourArea(target)))
            for target in targets
        )
        metrics["short_side_mm"] = target_short
        metrics["long_side_mm"] = target_long
        metrics["fill_ratio"] = pieces_area / max(
            1e-6,
            target_short * target_long,
        )
        metrics["strategy"] = "mode2_geometry_plus_card_artwork"
        metrics["score"] = float(combined_score)
        metrics["geometry_score"] = float(geometry_score)
        metrics["seam_continuity_score"] = float(seam_score)
        metrics["rotation_180_symmetry_error"] = float(symmetry_score)
        metrics["raw_geometry_candidates"] = float(len(raw_states))
        metrics["geometry_beam_width"] = float(selected_beam_width)
        metrics["valid_geometry_candidates"] = float(
            len(geometry_candidates)
        )
        metrics["artwork_candidates_scored"] = float(
            len(artwork_candidates)
        )
        metrics["overlap_before_adjust_mm2"] = round(
            float(overlap_before_adjust_mm2),
            4,
        )
        metrics["overlap_area_mm2"] = overlap_area_mm2
        metrics["target_clearance_mm"] = 3.0
        metrics["adjacency_gap_min_mm"] = round(
            min(adjacency_gaps) if adjacency_gaps else 0.0,
            3,
        )
        metrics["adjacency_gap_max_mm"] = round(
            max(adjacency_gaps) if adjacency_gaps else 0.0,
            3,
        )
        target_size = (
            float(metrics["long_side_mm"]),
            float(metrics["short_side_mm"]),
        )
        solve_ms = (time.perf_counter() - solve_start) * 1000.0

        detection_image = self.output_directory / "fsm_mode3_detection.jpg"
        solution_image = self.output_directory / "fsm_mode3_solution.jpg"
        cv2.imwrite(str(detection_image), detection_annotated)
        render_card_template_solution(
            rectified,
            pieces,
            targets,
            str(solution_image),
            target_origin_mm=tuple(target_origin),
            target_size_mm=target_size,
        )
        self._report_progress(1.0, "Mode 3 solution ready")

        order = sorted(
            range(piece_count),
            key=lambda piece_id: detection["pieces"][piece_id]["area_px2"],
            reverse=True,
        )
        actions = []
        for sequence, piece_id in enumerate(order):
            piece = detection["pieces"][piece_id]
            target_center = polygon_centroid(targets[piece_id])
            actions.append(
                MoveAction(
                    sequence=sequence,
                    piece_id=piece_id,
                    source_pickup_mm=[
                        round(float(value), 2)
                        for value in piece["pickup_mm"]
                    ],
                    source_centroid_mm=[
                        round(float(value), 2)
                        for value in piece["centroid_mm"]
                    ],
                    target_centroid_mm=[
                        round(float(value), 2)
                        for value in target_center
                    ],
                    rotation_delta_deg=round(
                        self._normalize_angle(rotations[piece_id]),
                        2,
                    ),
                )
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "actions": actions,
            "metrics": metrics,
            "timing_ms": {
                "image_load": round(load_ms, 2),
                "vision": round(vision_ms, 2),
                "geometry_search": round(geometry_search_ms, 2),
                "geometry_filter": round(geometry_filter_ms, 2),
                "artwork_score": round(artwork_score_ms, 2),
                "puzzle_solve": round(solve_ms, 2),
                "total": round(total_ms, 2),
            },
            "detection_image": str(detection_image),
            "solution_image": str(solution_image),
        }

    @staticmethod
    def _regularize_card_pieces(
        pieces: List[Dict[str, object]],
        config: DetectConfig,
    ) -> None:
        """Remove false concavities caused by blue ink touching a cut edge."""
        for piece in pieces:
            hull = cv2.convexHull(piece["_contour"])
            perimeter = cv2.arcLength(hull, True)
            polygon = None
            for epsilon_percent in np.arange(
                config.polygon_epsilon_percent,
                5.01,
                0.25,
            ):
                candidate = cv2.approxPolyDP(
                    hull,
                    perimeter * float(epsilon_percent) / 100.0,
                    True,
                )
                polygon = candidate
                if len(candidate) <= 5:
                    break
            if polygon is None or len(polygon) > 5:
                raise RuntimeError(
                    "P{} cannot be regularized to at most five edges".format(
                        piece["id"]
                    )
                )
            polygon = remove_false_short_edges(
                polygon,
                hull,
                10.0 * PIXELS_PER_MM,
            )
            polygon = sharpen_polygon_from_contour(
                polygon,
                hull,
            )
            centroid = contour_centroid(hull)
            pickup, clearance, pickup_mode = choose_pickup_point(
                hull,
                centroid,
                config.safe_radius_px,
            )
            piece["_contour"] = hull
            piece["_polygon"] = polygon
            piece["area_px2"] = round(float(cv2.contourArea(hull)), 1)
            piece["centroid_px"] = [
                round(float(centroid[0]), 1),
                round(float(centroid[1]), 1),
            ]
            piece["pickup_px"] = list(pickup)
            piece["clearance_px"] = round(float(clearance), 1)
            piece["pickup_mode"] = pickup_mode
            piece["vertices_px"] = polygon.reshape(-1, 2).tolist()

    @staticmethod
    def _validate_zero_overlap(
        polygons: List[np.ndarray],
        tolerance_mm2: float = 0.01,
    ) -> float:
        """Reject a motion plan if any two target pieces overlap."""
        total_overlap = 0.0
        for first_id in range(len(polygons)):
            first = cv2.convexHull(
                np.asarray(polygons[first_id], dtype=np.float32)
            )
            for second_id in range(first_id + 1, len(polygons)):
                second = cv2.convexHull(
                    np.asarray(polygons[second_id], dtype=np.float32)
                )
                overlap, _ = cv2.intersectConvexConvex(first, second)
                overlap = max(0.0, float(overlap))
                if overlap > tolerance_mm2:
                    raise RuntimeError(
                        "Mode 3 target overlap: P{} and P{} = "
                        "{:.3f} mm^2".format(
                            first_id,
                            second_id,
                            overlap,
                        )
                    )
                # Values below tolerance are floating-point contact noise,
                # not a physical overlap, and are reported as zero.
        return round(total_overlap, 4)

    @staticmethod
    def _total_overlap_area(polygons: List[np.ndarray]) -> float:
        total_overlap = 0.0
        for first_id in range(len(polygons)):
            first = cv2.convexHull(
                np.asarray(polygons[first_id], dtype=np.float32)
            )
            for second_id in range(first_id + 1, len(polygons)):
                second = cv2.convexHull(
                    np.asarray(polygons[second_id], dtype=np.float32)
                )
                overlap, _ = cv2.intersectConvexConvex(first, second)
                total_overlap += max(0.0, float(overlap))
        return total_overlap

    @classmethod
    def _separate_small_overlaps(
        cls,
        polygons: List[np.ndarray],
        target_origin: np.ndarray,
        clearance_mm: float = 0.02,
    ) -> List[np.ndarray]:
        """Resolve small fitting overlaps using convex minimum translations."""
        adjusted = [
            np.asarray(polygon, dtype=np.float32).copy()
            for polygon in polygons
        ]
        for _ in range(12):
            changed = False
            for first_id in range(len(adjusted)):
                for second_id in range(first_id + 1, len(adjusted)):
                    first = adjusted[first_id]
                    second = adjusted[second_id]
                    overlap, _ = cv2.intersectConvexConvex(first, second)
                    if float(overlap) <= 1e-5:
                        continue

                    axes = []
                    for polygon in (first, second):
                        edges = np.roll(polygon, -1, axis=0) - polygon
                        for edge in edges:
                            length = float(np.linalg.norm(edge))
                            if length < 1e-6:
                                continue
                            axes.append(
                                np.asarray(
                                    [-edge[1], edge[0]],
                                    dtype=np.float32,
                                )
                                / length
                            )
                    best_axis = None
                    best_depth = float("inf")
                    for axis in axes:
                        first_projection = first @ axis
                        second_projection = second @ axis
                        depth = min(
                            float(np.max(first_projection)),
                            float(np.max(second_projection)),
                        ) - max(
                            float(np.min(first_projection)),
                            float(np.min(second_projection)),
                        )
                        if depth < best_depth:
                            best_depth = depth
                            best_axis = axis
                    if best_axis is None or best_depth <= 0:
                        continue
                    first_center = np.mean(first, axis=0)
                    second_center = np.mean(second, axis=0)
                    if (
                        float(
                            np.dot(
                                second_center - first_center,
                                best_axis,
                            )
                        )
                        < 0
                    ):
                        best_axis = -best_axis
                    shift = (
                        best_axis
                        * (best_depth + clearance_mm)
                        * 0.5
                    )
                    adjusted[first_id] -= shift
                    adjusted[second_id] += shift
                    changed = True
            if not changed:
                break

        remaining = cls._total_overlap_area(adjusted)
        if remaining > 0.001:
            raise RuntimeError(
                "Mode 3 cannot remove residual overlap: "
                "{:.4f} mm^2".format(remaining)
            )
        all_points = np.concatenate(adjusted, axis=0)
        offset = (
            np.asarray(target_origin, dtype=np.float32)
            - np.min(all_points, axis=0)
        )
        return [polygon + offset for polygon in adjusted]

    @staticmethod
    def _polygon_distance(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        """Return the minimum point-to-edge distance between two polygons."""
        def points_to_edges(points, polygon):
            best = float("inf")
            for edge_id in range(len(polygon)):
                start = polygon[edge_id]
                end = polygon[(edge_id + 1) % len(polygon)]
                vector = end - start
                length_squared = max(
                    float(np.dot(vector, vector)),
                    1e-8,
                )
                projection = np.clip(
                    ((points - start) @ vector) / length_squared,
                    0.0,
                    1.0,
                )
                nearest = start + projection[:, None] * vector
                best = min(
                    best,
                    float(
                        np.min(
                            np.linalg.norm(points - nearest, axis=1)
                        )
                    ),
                )
            return best

        return min(
            points_to_edges(first, second),
            points_to_edges(second, first),
        )

    @classmethod
    def _add_assembly_clearance(
        cls,
        polygons: List[np.ndarray],
        matches,
        target_origin: np.ndarray,
        clearance_mm: float,
    ):
        """Open a small mechanical gap along the geometry adjacency tree."""
        adjusted = [
            np.asarray(polygon, dtype=np.float32).copy()
            for polygon in polygons
        ]
        adjacency = sorted(
            {
                tuple(sorted((int(match[0]), int(match[2]))))
                for match in matches
                if int(match[0]) != int(match[2])
            }
        )
        for _ in range(10):
            changed = False
            for first_id, second_id in adjacency:
                distance = cls._polygon_distance(
                    adjusted[first_id],
                    adjusted[second_id],
                )
                if distance >= clearance_mm - 0.02:
                    continue
                first_center = np.mean(adjusted[first_id], axis=0)
                second_center = np.mean(adjusted[second_id], axis=0)
                direction = second_center - first_center
                length = float(np.linalg.norm(direction))
                if length < 1e-6:
                    continue
                direction /= length
                shift = (
                    direction
                    * (clearance_mm - distance)
                    * 0.5
                )
                adjusted[first_id] -= shift
                adjusted[second_id] += shift
                changed = True
            if not changed:
                break

        all_points = np.concatenate(adjusted, axis=0)
        offset = (
            np.asarray(target_origin, dtype=np.float32)
            - np.min(all_points, axis=0)
        )
        adjusted = [polygon + offset for polygon in adjusted]
        gaps = [
            cls._polygon_distance(
                adjusted[first_id],
                adjusted[second_id],
            )
            for first_id, second_id in adjacency
        ]
        return adjusted, gaps

    @classmethod
    def _infer_adjacency_matches(
        cls,
        polygons: List[np.ndarray],
        existing_matches=(),
        contact_threshold_mm: float = 5.0,
    ):
        """Keep all nearby pairs, then complete a minimum-distance tree."""
        edges = []
        for first_id in range(len(polygons)):
            for second_id in range(first_id + 1, len(polygons)):
                edges.append(
                    (
                        cls._polygon_distance(
                            np.asarray(polygons[first_id], dtype=np.float32),
                            np.asarray(polygons[second_id], dtype=np.float32),
                        ),
                        first_id,
                        second_id,
                    )
                )
        edges.sort(key=lambda item: item[0])
        parent = list(range(len(polygons)))

        def find(piece_id):
            while parent[piece_id] != piece_id:
                parent[piece_id] = parent[parent[piece_id]]
                piece_id = parent[piece_id]
            return piece_id

        adjacency = {
            tuple(sorted((int(match[0]), int(match[2]))))
            for match in existing_matches
            if int(match[0]) != int(match[2])
        }
        adjacency.update(
            (first_id, second_id)
            for distance, first_id, second_id in edges
            if distance <= contact_threshold_mm
        )
        for first_id, second_id in sorted(adjacency):
            first_root = find(first_id)
            second_root = find(second_id)
            if first_root != second_root:
                parent[second_root] = first_root

        for _, first_id, second_id in edges:
            first_root = find(first_id)
            second_root = find(second_id)
            if first_root == second_root:
                continue
            parent[second_root] = first_root
            adjacency.add((first_id, second_id))
            if len({find(piece_id) for piece_id in range(len(polygons))}) == 1:
                break
        return tuple(
            (first_id, 0, second_id, 0, 0.0)
            for first_id, second_id in sorted(adjacency)
        )

    @classmethod
    def _apply_target_clearance(
        cls,
        polygons: List[np.ndarray],
        matches,
        target_origin: np.ndarray,
        clearance_mm: float,
        size_tolerance_mm: float = 0.0,
    ):
        adjusted = cls._separate_small_overlaps(
            polygons,
            target_origin=target_origin,
        )
        adjusted, gaps = cls._add_assembly_clearance(
            adjusted,
            matches=matches,
            target_origin=target_origin,
            clearance_mm=clearance_mm,
        )
        adjusted = cls._separate_small_overlaps(
            adjusted,
            target_origin=target_origin,
        )
        overlap = cls._validate_zero_overlap(
            adjusted,
            tolerance_mm2=0.001,
        )
        if gaps and max(gaps) > 20.0:
            raise RuntimeError(
                "adjacent-piece gap exceeds 20mm: {:.2f}mm".format(
                    max(gaps)
                )
            )
        all_points = np.concatenate(adjusted, axis=0).astype(np.float32)
        (_, _), (width, height), _ = cv2.minAreaRect(all_points)
        short_side, long_side = sorted((float(width), float(height)))
        size_tolerance_mm = max(0.0, float(size_tolerance_mm))
        if not (
            90.0 - size_tolerance_mm
            <= long_side
            <= 120.0 + size_tolerance_mm
            and 50.0 - size_tolerance_mm
            <= short_side
            <= 90.0 + size_tolerance_mm
        ):
            raise RuntimeError(
                "clearance layout exceeds legal size: "
                "{:.2f}x{:.2f}mm".format(long_side, short_side)
            )
        area = sum(
            abs(float(cv2.contourArea(polygon)))
            for polygon in adjusted
        )
        return adjusted, {
            "short_side_mm": short_side,
            "long_side_mm": long_side,
            "fill_ratio": area / max(1e-6, short_side * long_side),
            "target_clearance_mm": float(clearance_mm),
            "size_tolerance_mm": size_tolerance_mm,
            "adjacency_gap_min_mm": round(
                min(gaps) if gaps else 0.0,
                3,
            ),
            "adjacency_gap_max_mm": round(
                max(gaps) if gaps else 0.0,
                3,
            ),
            "overlap_area_mm2": overlap,
        }

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return (angle + 180.0) % 360.0 - 180.0


def run_stdio(machine: VisionStateMachine) -> int:
    """每读入一行命令，输出一行JSON，便于控制端联调。"""
    for line in sys.stdin:
        try:
            payload = {
                "ok": True,
                "data": machine.handle_command(line),
            }
        except Exception as error:
            payload = {
                "ok": False,
                "error": str(error),
                "state": machine.state.value,
            }
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


def demo(
    machine: VisionStateMachine,
    image_path: str,
    mode: int = 1,
) -> int:
    commands = [
        "MODE,{}".format(mode),
        "START,{}".format(image_path),
        "PLAN",
    ]
    for command in commands:
        response = machine.handle_command(command)
        print(command)
        print(json.dumps(response, ensure_ascii=False, indent=2))
    while machine.state != VisionState.COMPLETED:
        action = machine.handle_command("NEXT")
        print("NEXT")
        print(json.dumps(action, ensure_ascii=False, indent=2))
        machine.handle_command(
            "ACK,{},OK".format(action["piece_id"])
        )
    print(json.dumps(machine.status(), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", default="", help="使用图片执行完整联调演示")
    parser.add_argument(
        "--demo-mode",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="演示模式编号，默认1",
    )
    parser.add_argument(
        "--output-directory",
        default="",
        help="状态机调试图片输出目录",
    )
    args = parser.parse_args()
    machine = VisionStateMachine(
        output_directory=args.output_directory or None
    )
    if args.demo:
        return demo(machine, args.demo, args.demo_mode)
    return run_stdio(machine)


if __name__ == "__main__":
    raise SystemExit(main())
