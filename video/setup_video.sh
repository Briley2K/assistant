#!/usr/bin/env bash
# Rebuild the isolated text-to-video environment used by the assistant.
#
# Video diffusion (diffusers + torch) runs in ITS OWN Python venv here
# (video/.venv) and the assistant talks to it over localhost HTTP (see
# modules/videogen.py + video_server.py). This script recreates that venv.
#
#     bash video/setup_video.sh               # CUDA build (default)
#     VIDEO_CPU=1 bash video/setup_video.sh    # CPU-only torch (very slow)
#
# Notes:
#   - diffusers/transformers need Python 3.10–3.12; we fetch 3.12 via `uv`.
#   - imageio-ffmpeg provides the ffmpeg used to encode frames into an mp4.
#   - Newer video models (Wan, LTX) need a recent diffusers, so we install from
#     git main to be safe.
set -e
cd "$(dirname "$0")/.."                       # repo root
VENV="$PWD/video/.venv"
PYVER=3.12

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
if [ -n "$VIDEO_CPU" ]; then
    echo "=== torch (CPU build) ==="
    "$UV" pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu
else
    echo "=== torch (CUDA build) ==="
    "$UV" pip install --python "$PY" torch
fi

echo "=== diffusers (git main) + transformers + accelerate + ftfy + imageio-ffmpeg ==="
"$UV" pip install --python "$PY" \
    "git+https://github.com/huggingface/diffusers.git" \
    transformers accelerate safetensors ftfy imageio imageio-ffmpeg

echo "=== verify ==="
"$PY" - <<'PYEOF'
import torch, diffusers, imageio_ffmpeg
print(f"OK: torch {torch.__version__}, diffusers {diffusers.__version__}, "
      f"cuda_available={torch.cuda.is_available()}")
PYEOF
echo ""
echo "Video env ready. The panel will spawn video_server.py on demand; the model"
echo "downloads from HuggingFace on first generation (or use the Download button)."
