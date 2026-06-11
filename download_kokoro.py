#!/usr/bin/env python3
"""
Fetch the Kokoro TTS model + all voices from HuggingFace (GitHub releases are
blocked in some environments) and assemble the combined voices .npz that
kokoro-onnx expects. Idempotent.
"""
import os, sys
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
DST = os.path.join(os.path.dirname(__file__), "models", "kokoro")
MODEL = os.path.join(DST, "kokoro-v1.0.onnx")
VOICES = os.path.join(DST, "voices-v1.0.npz")


def main():
    os.makedirs(DST, exist_ok=True)

    if not os.path.exists(MODEL) or os.path.getsize(MODEL) < 1_000_000:
        print("Downloading Kokoro model...")
        import shutil
        shutil.copy(hf_hub_download(REPO, "onnx/model.onnx"), MODEL)
    print(f"model: {os.path.getsize(MODEL)//1024//1024} MB")

    if not os.path.exists(VOICES):
        voice_files = [f for f in list_repo_files(REPO) if f.startswith("voices/") and f.endswith(".bin")]
        print(f"Downloading {len(voice_files)} voices...")
        voices = {}
        for vf in voice_files:
            name = os.path.splitext(os.path.basename(vf))[0]
            arr = np.fromfile(hf_hub_download(REPO, vf), dtype=np.float32).reshape(-1, 1, 256)
            voices[name] = arr
        np.savez(VOICES, **voices)
        print(f"voices: {len(voices)} -> {VOICES}")
    else:
        print(f"voices: present ({VOICES})")

    print("\nDone. Available voices:")
    names = sorted(np.load(VOICES).keys())
    print(", ".join(names))


if __name__ == "__main__":
    main()
