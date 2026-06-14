#!/usr/bin/env bash
# Rebuild the isolated NeuTTS Air environment used by the voice assistant.
#
# NeuTTS needs torch, which won't install in the assistant's main Python 3.14
# venv — so it runs in ITS OWN Python 3.12 venv here (neutts_test/.venv) and the
# assistant talks to it over localhost HTTP (see modules/neutts_tts.py +
# neutts_server.py). This script recreates that venv from scratch.
#
#     bash neutts_test/setup_neutts.sh
#
# Notes that bit us before, baked in here:
#   - The `neutts` package (pip name of neuttsair) needs Python 3.10–3.13; 3.14
#     is too new. We fetch 3.12 via `uv` (no sudo, no apt).
#   - Install torch AND torchaudio from the CPU index TOGETHER first. If you let
#     `neutts` pull torchaudio from default PyPI it grabs a CUDA build whose
#     compiled _torchaudio.so won't load against the CPU torch -> import crash.
#   - The helper runs the backbone on CPU, so a CPU build is all we need.
set -e
cd "$(dirname "$0")/.."                       # repo root
VENV="$PWD/neutts_test/.venv"
PYVER=3.12

# --- locate or install uv (manages the standalone Python 3.12) ---
UV="$(command -v uv || true)"
[ -z "$UV" ] && UV="$(find "$HOME" -maxdepth 6 -type f -name uv -path '*bin*' 2>/dev/null | head -1)"
if [ -z "$UV" ]; then
    echo "Installing uv (user-local, no sudo)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$(find "$HOME" -maxdepth 6 -type f -name uv -path '*bin*' 2>/dev/null | head -1)"
fi
echo "uv: $UV"

"$UV" python install "$PYVER"
echo "Creating venv at $VENV ..."
"$UV" venv --python "$PYVER" "$VENV"

PY="$VENV/bin/python"
echo "=== torch + torchaudio (CPU, matched pair) ==="
"$UV" pip install --python "$PY" torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "=== neutts[llama] + espeak (builds llama-cpp + the neutts wheel; a few min) ==="
CMAKE_ARGS="-DGGML_CUDA=off" "$UV" pip install --python "$PY" \
    "neutts[llama] @ git+https://github.com/neuphonic/neutts-air.git" espeakng-loader

echo "=== verify ==="
"$PY" - <<'PYEOF'
import torch, torchaudio, neucodec, llama_cpp, espeakng_loader
from neuttsair.neutts import NeuTTSAir
print(f"OK: torch {torch.__version__}, torchaudio {torchaudio.__version__}")
PYEOF
echo ""
echo "NeuTTS env ready. The assistant/panel will spawn neutts_server.py on demand."
