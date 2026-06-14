#!/usr/bin/env bash
# Rebuild the isolated Chatterbox (Resemble AI) TTS environment.
#
# Chatterbox needs torch, which won't install in the assistant's main Python 3.14
# venv — so it runs in ITS OWN Python 3.12 venv here (chatterbox/.venv) and the
# assistant talks to it over localhost HTTP (see modules/chatterbox_tts.py +
# chatterbox_server.py). This recreates that venv from scratch, on the GPU.
#
#     bash chatterbox/setup_chatterbox.sh
#
# Notes that bit us, baked in here:
#   - chatterbox-tts pins torch==2.6.0, which has NO Blackwell (sm_120 / RTX 50xx)
#     kernels. So we install a cu128 torch FIRST, then install chatterbox-tts with
#     an --override that loosens the torch pin so it keeps the cu128 build.
#   - Python 3.12 (not 3.14): chatterbox's deps (librosa->numba) need a mature
#     ecosystem. uv fetches 3.12 (no sudo, no apt).
set -e
cd "$(dirname "$0")/.."                       # repo root
VENV="$PWD/chatterbox/.venv"
PYVER=3.12
# CUDA wheel index for Blackwell (sm_120). cu128 = CUDA 12.8; your driver (13.x)
# is backward-compatible. Bump if a newer cu wheel is needed.
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

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

echo "=== torch + torchaudio (CUDA 12.8 / Blackwell) ==="
"$UV" pip install --python "$PY" torch torchaudio --index-url "$TORCH_INDEX"

echo "=== chatterbox-tts (torch pin overridden to keep the cu128 build) ==="
OVR="$(mktemp)"; printf 'torch>=2.9\ntorchaudio>=2.9\n' > "$OVR"
"$UV" pip install --python "$PY" --extra-index-url "$TORCH_INDEX" --override "$OVR" chatterbox-tts
rm -f "$OVR"

echo "=== verify (loads on GPU, downloads weights first run) ==="
"$PY" - <<'PYEOF'
import torch
from chatterbox.tts import ChatterboxTTS
assert torch.cuda.is_available(), "CUDA not available in the chatterbox venv!"
print(f"OK: torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}, "
      f"capability {torch.cuda.get_device_capability(0)}")
PYEOF
echo ""
echo "Chatterbox env ready. The assistant/panel will spawn chatterbox_server.py on demand."
