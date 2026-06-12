#!/usr/bin/env bash
# Downloads every model the assistant needs. Idempotent — skips anything
# already present (including the Gemma GGUF on the LM Studio drive, if mounted).
# Called by setup.sh; safe to re-run standalone (inside the venv).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Run inside the project venv if we aren't already.
if [ -z "$VIRTUAL_ENV" ] && [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo "=== Downloading Piper TTS voice (en_US-lessac-medium) ==="
PIPER_DIR="$SCRIPT_DIR/models/piper"
mkdir -p "$PIPER_DIR"

if [ ! -f "$PIPER_DIR/en_US-lessac-medium.onnx" ] || [ "$(wc -c < "$PIPER_DIR/en_US-lessac-medium.onnx")" -lt 1000 ]; then
    python3 -c "
from piper.download_voices import download_voice
from pathlib import Path
download_voice('en_US-lessac-medium', Path('$PIPER_DIR'))
print('Piper voice downloaded.')
"
else
    echo "Piper voice already present."
fi

echo ""
echo "=== Downloading Kokoro TTS model + voices (from HuggingFace) ==="
python3 "$SCRIPT_DIR/download_kokoro.py"

echo ""
echo "=== Downloading Gemma 4 12B LLM (GGUF + audio/vision mmproj, ~8 GB) ==="
python3 - "$SCRIPT_DIR" <<'EOF'
import os, sys
sys.path.insert(0, sys.argv[1])
import config
from huggingface_hub import hf_hub_download

REPO = "lmstudio-community/gemma-4-12B-it-GGUF"
DEST = os.path.join(sys.argv[1], "models", "gemma")

# Always fetch the Gemma GGUF (the bundled local model) regardless of which
# model is currently selected. Other models (e.g. Nemotron) run via Ollama and
# are pulled with `ollama pull`, not downloaded here.
_gemma = config.LLM_MODELS["gemma4-12b"]
for filename in (_gemma["gguf"], _gemma["mmproj"]):
    current_path = os.path.join(DEST, filename)
    if os.path.exists(current_path):
        print(f"already present: {current_path}")
    else:
        print(f"downloading {filename} (resumes if interrupted)...")
        hf_hub_download(REPO, filename, local_dir=DEST)
        print(f"saved to {os.path.join(DEST, filename)}")
EOF

echo ""
echo "=== Downloading Whisper STT model ==="
python3 - "$SCRIPT_DIR" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import config
from faster_whisper import download_model

print(f"fetching faster-whisper '{config.WHISPER_MODEL}' (cached if present)...")
download_model(config.WHISPER_MODEL)
print("Whisper model ready.")
EOF

echo ""
echo "NOTE: openWakeWord models are bundled with the package — no download needed."
echo ""
echo "=== All models ready! Run: $SCRIPT_DIR/.venv/bin/python3 assistant.py ==="
