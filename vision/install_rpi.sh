#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
sudo apt update
sudo apt install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    v4l-utils

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup_fast.py build_ext --inplace --force
.venv/bin/python -c "import vision_fast; print('Native extension:', vision_fast.__file__)"

chmod +x run_rpi.sh
echo "Install complete. Run: ./run_rpi.sh"
echo "Use Mode0 after changing camera or mount."

