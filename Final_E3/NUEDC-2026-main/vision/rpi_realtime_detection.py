"""Raspberry Pi USB-camera live A4 and puzzle-piece detector."""

import argparse
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import fragment_vision as vision
from esp32_puzzle_link import (
    Link,
    PIECE_COUNT_MESSAGE,
    PIECE_DATA_MESSAGE,
    PIECE_FORMAT,
    ReliablePuzzleSender,
    build_result_messages,
    parse_mode_request,
)
from vision_state_machine import VisionStateMachine


class PreviewState:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg = None
        self.frame_id = 0

    def update(self, frame):
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        if not success:
            return
        with self.condition:
            self.jpeg = encoded.tobytes()
            self.frame_id += 1
            self.condition.notify_all()

    def wait_next(self, previous_id):
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_id != previous_id,
                timeout=1.0,
            )
            return self.frame_id, self.jpeg


def make_preview_handler(preview_state):
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                page = (
                    "<!doctype html><html><head>"
                    "<meta charset='utf-8'>"
                    "<meta name='viewport' "
                    "content='width=device-width,initial-scale=1'>"
                    "<title>Raspberry Pi Vision</title>"
                    "<style>body{margin:0;background:#111;color:#eee;"
                    "font-family:sans-serif;text-align:center}"
                    "img{max-width:100vw;max-height:92vh}"
                    "</style></head><body>"
                    "<h3>Raspberry Pi puzzle vision</h3>"
                    "<img src='/stream.mjpg'></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return

            if self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                frame_id = -1
                try:
                    while True:
                        frame_id, jpeg = preview_state.wait_next(
                            frame_id
                        )
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(
                            b"Content-Type: image/jpeg\r\n"
                        )
                        self.wfile.write(
                            "Content-Length: {}\r\n\r\n".format(
                                len(jpeg)
                            ).encode("ascii")
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            self.send_error(404)

        def log_message(self, *_):
            return

    return PreviewHandler


def start_preview_server(preview_state, port):
    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        make_preview_handler(preview_state),
    )
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()
    return server


def open_camera(index, width, height, fps):
    camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError(
            "Cannot open /dev/video{}".format(index)
        )
    return camera


def rotate_frame(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(
            frame,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )
    return frame


def fit_window(frame, maximum_width, maximum_height):
    height, width = frame.shape[:2]
    scale = min(
        1.0,
        maximum_width / float(width),
        maximum_height / float(height),
    )
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def paste_fit(canvas, frame, left, top, width, height):
    if frame is None:
        return
    source_height, source_width = frame.shape[:2]
    scale = min(
        width / float(source_width),
        height / float(source_height),
    )
    resized = cv2.resize(
        frame,
        (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        ),
        interpolation=(
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        ),
    )
    y = top + (height - resized.shape[0]) // 2
    x = left + (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized


def make_landscape_dashboard(
    raw_frame,
    detection,
    solution,
    mode,
    stable_frames,
    required_frames,
    solve_status,
):
    canvas = np.full((720, 1280, 3), 20, dtype=np.uint8)
    right_frame = solution if solution is not None else detection
    paste_fit(canvas, raw_frame, 10, 62, 620, 648)
    paste_fit(canvas, right_frame, 650, 62, 620, 648)
    cv2.putText(
        canvas,
        "MODE {}  AUTO {}/{}  {}".format(
            mode,
            min(stable_frames, required_frames),
            required_frames,
            solve_status,
        ),
        (18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "CAMERA",
        (18, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        "SOLUTION" if solution is not None else "DETECTION",
        (658, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return canvas


def make_terminal_progress(mode):
    started = time.perf_counter()
    width = 30

    def update(progress, message):
        progress = max(0.0, min(1.0, float(progress)))
        filled = min(width, int(round(progress * width)))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.perf_counter() - started
        print(
            "\rSOLVING mode={} [{}] {:3d}%  {:<42} {:6.1f}s".format(
                mode,
                bar,
                int(round(progress * 100.0)),
                str(message)[:42],
                elapsed,
            ),
            end=("\n" if progress >= 1.0 else ""),
            flush=True,
        )

    return update


def send_plan_reliably(link, sender, actions):
    """Freeze vision and deliver count/pieces with strict stop-and-wait."""
    messages = build_result_messages(actions)

    def describe(message):
        message_type, payload = message
        if message_type == PIECE_COUNT_MESSAGE:
            return "COUNT={}".format(payload[0])
        if message_type == PIECE_DATA_MESSAGE:
            values = struct.unpack(PIECE_FORMAT, payload)
            return (
                "PIECE id={} S=({},{})mm T=({},{})mm "
                "R={:+.2f}deg"
            ).format(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5] / 100.0,
            )
        return "TYPE=0x{:02X} length={}".format(
            message_type,
            len(payload),
        )

    sender.start(actions)
    last_delivered = 0
    ignored_modes = 0
    packet_started = time.perf_counter()
    retry_started = link.retransmission_count
    print("TX 1/{} {}".format(len(messages), describe(messages[0])))
    while not sender.complete:
        event = link.poll()
        if parse_mode_request(event) is not None:
            ignored_modes += 1
        sender.step(event)
        if sender.delivered_count != last_delivered:
            now = time.perf_counter()
            ack_ms = (now - packet_started) * 1000.0
            retries = link.retransmission_count - retry_started
            last_delivered = sender.delivered_count
            print(
                "ACK {}/{} {:.1f}ms retries={}".format(
                    sender.delivered_count,
                    sender.total_count,
                    ack_ms,
                    retries,
                ),
                flush=True,
            )
            if sender.delivered_count < len(messages):
                print(
                    "TX {}/{} {}".format(
                        sender.delivered_count + 1,
                        len(messages),
                        describe(messages[sender.delivered_count]),
                    )
                )
                packet_started = now
                retry_started = link.retransmission_count
        # poll() must run much faster than the 50 ms retry interval.
        time.sleep(0.001)
    print(
        "ESP32 TX [{}/{} ACK] complete".format(
            sender.delivered_count,
            sender.total_count,
        )
    )
    sender.clear()
    return ignored_modes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=90,
        help="clockwise image rotation before detection",
    )
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument(
        "--mode",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="puzzle mode",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="disable automatic solve and wait for P",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="show preview through a browser instead of cv2.imshow",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--serial-device",
        help=(
            "ESP32 serial device, normally /dev/serial0; "
            "when set, modes are selected by reliable type 0x03 messages"
        ),
    )
    parser.add_argument(
        "--serial-baud",
        type=int,
        default=115200,
    )
    args = parser.parse_args()

    serial_connection = None
    esp_link = None
    result_sender = None
    pending_mode = None
    ignored_busy_modes = 0
    mode_input_armed = True
    mode_quiet_started = time.monotonic()
    # Prevent stale retransmissions from the completed task from being
    # interpreted as a new user command.
    mode_rearm_quiet_seconds = 1.0
    current_mode = args.mode
    command_active = args.serial_device is None
    if args.serial_device:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "pyserial is required: pip install pyserial"
            ) from error
        serial_connection = serial.Serial(
            args.serial_device,
            args.serial_baud,
            timeout=0,
        )
        esp_link = Link(serial_connection)
        result_sender = ReliablePuzzleSender(esp_link)

    camera = open_camera(
        args.camera,
        args.width,
        args.height,
        args.fps,
    )
    config = vision.DetectConfig()

    # No lens-undistortion step is used for this USB camera.
    last_detection = None
    last_payload = None
    last_error = ""
    solution_preview = None
    stable_piece_count = 0
    stable_frames = 0
    solve_attempted = False
    solve_status = (
        "WAIT_CMD" if args.serial_device else "WAIT"
    )
    frame_id = 0
    preview_state = PreviewState() if args.web else None
    preview_server = None
    if args.web:
        preview_server = start_preview_server(
            preview_state,
            args.port,
        )
        print(
            "Web preview: http://<raspberry-pi-ip>:{}".format(
                args.port
            )
        )

    for _ in range(30):
        camera.read()

    print("Realtime detection started")
    print(
        "Mode {}: auto_solve={}, P=solve, R=restart, "
        "S=save, Q/ESC=quit".format(current_mode, not args.manual)
    )
    if args.serial_device:
        print(
            "ESP32 control: {} at {} baud; waiting for "
            "reliable TYPE=0x03 PAYLOAD=1|2|3".format(
                args.serial_device,
                args.serial_baud,
            )
        )
    if not args.web:
        cv2.namedWindow("Puzzle Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Puzzle Vision", 1280, 720)

    try:
        while True:
            auto_solve_requested = False
            if esp_link is not None and result_sender is not None:
                link_event = esp_link.poll()
                requested_mode = parse_mode_request(link_event)
                if requested_mode is not None:
                    mode_quiet_started = time.monotonic()
                    if command_active or (
                        result_sender.active
                        and not result_sender.complete
                    ) or not mode_input_armed:
                        ignored_busy_modes += 1
                        if ignored_busy_modes == 1:
                            print(
                                "\nESP32 mode {} ignored while "
                                "busy/not armed".format(
                                    requested_mode
                                )
                            )
                    else:
                        pending_mode = requested_mode
                        mode_input_armed = False

                result_sender.step(link_event)
                if result_sender.active and not result_sender.complete:
                    solve_status = "SENDING {}/{}".format(
                        result_sender.delivered_count,
                        result_sender.total_count,
                    )
                if result_sender.complete:
                    print(
                        "\nESP32 delivered {}/{} result packets".format(
                            result_sender.delivered_count,
                            result_sender.total_count,
                        )
                    )
                    if ignored_busy_modes:
                        print(
                            "Ignored {} repeated mode request(s) "
                            "during this task".format(
                                ignored_busy_modes
                            )
                        )
                        ignored_busy_modes = 0
                    result_sender.clear()
                    command_active = False
                    solve_attempted = False
                    stable_piece_count = 0
                    stable_frames = 0
                    solve_status = "WAIT_CMD"
                    mode_input_armed = False
                    mode_quiet_started = time.monotonic()
                    print(
                        "WAITING FOR MODE LINE TO BECOME QUIET"
                    )

                if (
                    not command_active
                    and not result_sender.active
                    and not mode_input_armed
                    and time.monotonic() - mode_quiet_started
                    >= mode_rearm_quiet_seconds
                ):
                    mode_input_armed = True
                    ignored_busy_modes = 0
                    print("\nREADY FOR NEXT ESP32 MODE")

                if (
                    pending_mode is not None
                    and not command_active
                    and not result_sender.active
                ):
                    current_mode = pending_mode
                    pending_mode = None
                    command_active = True
                    solution_preview = None
                    last_detection = None
                    last_payload = None
                    stable_piece_count = 0
                    stable_frames = 0
                    solve_attempted = False
                    solve_status = "DETECTING"
                    last_error = ""
                    print(
                        "\nESP32 requested MODE {}".format(
                            current_mode
                        )
                    )

            success, frame = camera.read()
            if not success:
                print("Camera frame read failed")
                continue

            frame = rotate_frame(frame, args.rotate)
            frame_id += 1
            raw_preview = frame.copy()

            should_detect = (
                args.serial_device is None or command_active
            )
            if (
                should_detect
                and frame_id % max(1, args.detect_every) == 0
            ):
                start = time.perf_counter()
                try:
                    _, annotated, payload = vision.process_frame(
                        frame,
                        config,
                    )
                    elapsed_ms = (
                        time.perf_counter() - start
                    ) * 1000.0
                    piece_count = len(payload["pieces"])
                    cv2.putText(
                        annotated,
                        "Pi USB P={} vision={:.0f}ms".format(
                            piece_count,
                            elapsed_ms,
                        ),
                        (16, 62),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    last_detection = annotated
                    last_payload = payload
                    last_error = ""
                    valid_count = (
                        piece_count == 4
                        if current_mode == 1
                        else 2 <= piece_count <= 4
                    )
                    if not command_active:
                        stable_piece_count = 0
                        stable_frames = 0
                    elif (
                        valid_count
                        and piece_count == stable_piece_count
                    ):
                        stable_frames += 1
                    elif valid_count:
                        stable_piece_count = piece_count
                        stable_frames = 1
                    else:
                        stable_piece_count = 0
                        stable_frames = 0
                    if (
                        not args.manual
                        and command_active
                        and not solve_attempted
                        and stable_frames >= 2
                    ):
                        auto_solve_requested = True
                    print(
                        "\rpieces={} stable={}/2 "
                        "vision={:.1f}ms          ".format(
                            piece_count,
                            min(stable_frames, 2),
                            elapsed_ms,
                        ),
                        end="",
                        flush=True,
                    )
                except Exception as error:
                    stable_piece_count = 0
                    stable_frames = 0
                    last_error = str(error)
                    print(
                        "\rvision_error: {}          ".format(
                            last_error
                        ),
                        end="",
                        flush=True,
                    )

            if last_error:
                cv2.putText(
                    raw_preview,
                    last_error[:85],
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            dashboard = make_landscape_dashboard(
                raw_preview,
                last_detection,
                solution_preview,
                current_mode,
                stable_frames,
                2,
                solve_status,
            )

            if args.web:
                preview_state.update(dashboard)
                key = 255
            else:
                cv2.imshow("Puzzle Vision", dashboard)
                key = cv2.waitKey(1) & 0xFF
            if auto_solve_requested and key not in (
                27,
                ord("q"),
                ord("Q"),
            ):
                key = ord("p")
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                solution_preview = None
                stable_piece_count = 0
                stable_frames = 0
                solve_attempted = False
                solve_status = "WAIT"
                if args.serial_device and not command_active:
                    solve_status = "WAIT_CMD"
                last_error = ""
            if key in (ord("p"), ord("P")):
                solve_attempted = True
                solve_status = "SOLVING"
                solve_input = "rpi_solve_input.jpg"
                solve_detection = "rpi_solve_detection.jpg"
                solve_json = "rpi_solve_detection.json"
                cv2.imwrite(solve_input, frame)
                if last_detection is not None:
                    cv2.imwrite(
                        solve_detection,
                        last_detection,
                    )
                if last_payload is not None:
                    with open(
                        solve_json,
                        "w",
                        encoding="utf-8",
                    ) as output_file:
                        json.dump(
                            last_payload,
                            output_file,
                            ensure_ascii=False,
                            indent=2,
                        )
                solving_screen = raw_preview.copy()
                cv2.putText(
                    solving_screen,
                    "SOLVING MODE {} ...".format(current_mode),
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
                if not args.web:
                    cv2.imshow(
                        "Puzzle Vision",
                        make_landscape_dashboard(
                            solving_screen,
                            last_detection,
                            None,
                            current_mode,
                            stable_frames,
                            2,
                            "SOLVING",
                        ),
                    )
                    cv2.waitKey(1)
                else:
                    preview_state.update(solving_screen)

                print(
                    "\nSOLVING mode={} input={} detection={} json={}".format(
                        current_mode,
                        solve_input,
                        solve_detection,
                        solve_json,
                    )
                )
                try:
                    progress_callback = make_terminal_progress(
                        current_mode
                    )
                    state_machine = VisionStateMachine(
                        output_directory="rpi_tmp",
                        progress_callback=progress_callback,
                    )
                    state_machine.select_mode(current_mode)
                    solve_result = state_machine.start(solve_input)
                    solution_preview = cv2.imread(
                        solve_result["solution_image"]
                    )
                    print(
                        json.dumps(
                            solve_result,
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    print(
                        json.dumps(
                            state_machine.get_plan(),
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    if args.web and solution_preview is not None:
                        preview_state.update(solution_preview)
                    if result_sender is not None:
                        plan = state_machine.get_plan()
                        packet_count = len(plan["actions"]) + 1
                        solve_status = "SENDING 0/{}".format(packet_count)
                        print(
                            "ESP32 sending {} packets "
                            "(count + {} pieces)".format(
                                packet_count,
                                len(plan["actions"]),
                            )
                        )
                        ignored_busy_modes += send_plan_reliably(
                            esp_link,
                            result_sender,
                            plan["actions"],
                        )
                        esp_link.discard_received()
                        if ignored_busy_modes:
                            print(
                                "Ignored {} repeated mode request(s) "
                                "during this task".format(
                                    ignored_busy_modes
                                )
                            )
                            ignored_busy_modes = 0
                        command_active = False
                        solve_attempted = False
                        stable_piece_count = 0
                        stable_frames = 0
                        solve_status = "WAIT_CMD"
                        mode_input_armed = False
                        mode_quiet_started = time.monotonic()
                        print(
                            "WAITING FOR MODE LINE TO BECOME QUIET"
                        )
                    else:
                        solve_status = "DONE"
                except Exception as error:
                    solve_status = "ERROR"
                    last_error = "solve failed: {}".format(error)
                    print("\n" + last_error)
                    if args.serial_device:
                        command_active = False
                        mode_input_armed = False
                        mode_quiet_started = time.monotonic()
                        solve_attempted = False
                        stable_piece_count = 0
                        stable_frames = 0
            if key in (ord("s"), ord("S")):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                raw_path = "rpi_raw_{}.jpg".format(stamp)
                detection_path = "rpi_detection_{}.jpg".format(
                    stamp
                )
                json_path = "rpi_detection_{}.json".format(stamp)
                cv2.imwrite(raw_path, frame)
                if last_detection is not None:
                    cv2.imwrite(detection_path, last_detection)
                if last_payload is not None:
                    with open(
                        json_path,
                        "w",
                        encoding="utf-8",
                    ) as output_file:
                        json.dump(
                            last_payload,
                            output_file,
                            ensure_ascii=False,
                            indent=2,
                        )
                print(
                    "\nSaved: {}, {}, {}".format(
                        raw_path,
                        detection_path,
                        json_path,
                    )
                )
    except KeyboardInterrupt:
        pass
    finally:
        if preview_server is not None:
            preview_server.shutdown()
            preview_server.server_close()
        camera.release()
        if result_sender is not None:
            result_sender.clear()
        if serial_connection is not None:
            serial_connection.close()
        cv2.destroyAllWindows()
        print("\nStopped")


if __name__ == "__main__":
    main()
