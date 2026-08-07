#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
fi

CAMERA_CONTROLS_FILE="${CAMERA_CONTROLS_FILE:-camera_profiles/old_usb_camera.json}"

exec "$PYTHON_BIN" rpi_realtime_detection.py \
    --camera 0 \
    --width 1280 \
    --height 720 \
    --fps 30 \
    --rotate 180 \
    --detect-every 3 \
    --solver-workers 3 \
    --camera-controls-file "$CAMERA_CONTROLS_FILE" \
    --web \
    --port 8080 \
    --serial-device /dev/serial0 \
    --serial-baud 115200
