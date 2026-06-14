#!/usr/bin/env bash
# Rebuild the isolated image-generation environment used by the assistant.
#
# Stable Diffusion (diffusers + torch) won't install in the assistant's main
# Python 3.14 venv, so it runs in ITS OWN Python 3.12 venv here (imagegen/.venv)
# and the assistant talks to it over localhost HTTP (see modules/imagegen.py +
# imagegen_server.py). This script recreates that venv from scratch.
#
#     bash imagegen/setup_imagegen.sh            # CUDA build (default)
#     IMAGEGEN_CPU=1 bash imagegen/setup_imagegen.sh   # CPU-only torch
#
# Notes:
#   - diffusers needs Python 3.10–3.12; we fetch 3.12 via `uv` (no sudo/apt).
#   - Default is a CUDA torch build (the helper runs on the GPU). Set IMAGEGEN_CPU=1
#     to install the CPU index instead (much slower generation, but no VRAM use).
set -e
cd "$(dirname "$0")/.."                       # repo root
VENV="$PWD/imagegen/.venv"
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
if [ -n "$IMAGEGEN_CPU" ]; then
    echo "=== torch (CPU build) ==="
    "$UV" pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu
else
    echo "=== torch (CUDA build) ==="
    "$UV" pip install --python "$PY" torch
fi

echo "=== diffusers + transformers + accelerate + safetensors ==="
"$UV" pip install --python "$PY" diffusers transformers accelerate safetensors

echo "=== verify ==="
"$PY" - <<'PYEOF'
import torch, diffusers, transformers
print(f"OK: torch {torch.__version__}, diffusers {diffusers.__version__}, "
      f"cuda_available={torch.cuda.is_available()}")
PYEOF
echo ""
echo "Image-gen env ready. The assistant/panel will spawn imagegen_server.py on"
echo "demand; the model itself downloads from HuggingFace on first generation."
