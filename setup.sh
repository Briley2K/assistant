#!/usr/bin/env bash
set -e

# ---------------------------------------------------------------
# STEP 1 — System packages (run this in your own terminal first):
#   sudo apt-get install -y python3.14-venv
# ---------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Creating virtual environment ==="
python3 -m venv "$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/.venv/bin/activate"

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing Python packages ==="

# llama-cpp-python with CUDA support (RTX 5080, CUDA 12.8)
echo "--- Building llama-cpp-python with CUDA (takes a few minutes) ---"
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir

# Audio pipeline (uses system arecord/aplay — no portaudio or webrtcvad needed)
pip install numpy

# Speech-to-text
pip install faster-whisper

# Wake word detection
pip install openwakeword

# Text-to-speech
pip install piper-tts

echo ""
echo "=== Setup complete! ==="
echo "Next: source .venv/bin/activate && bash download_models.sh"
