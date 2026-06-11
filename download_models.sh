#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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
echo "NOTE: The Whisper 'small' STT model (~240MB) auto-downloads on first run (whisper mode only)."
echo "NOTE: openWakeWord models are bundled with the package — no download needed."
echo ""
echo "=== Ready! Run: python3 $SCRIPT_DIR/assistant.py ==="
