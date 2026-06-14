#!/usr/bin/env python3
"""
Pre-download a video model's weights from HuggingFace into the shared HF cache,
which the video helper venv reads at load time. Driven by VIDEOGEN_MODELS.

    python3 download_video_model.py <video_model_key>

Launched detached by the control panel's video "Download" button; progress goes
to logs/video_download.log. A sentinel file (logs/.video_downloading, contents =
repo id) marks an in-progress download; removed on success or failure.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

_MARKER = os.path.join(config._BASE, "logs", ".video_downloading")
_IGNORE = ["*.bin", "*.ckpt", "*.pth", "*.onnx", "*.msgpack", "*.h5"]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in config.VIDEOGEN_MODELS:
        print(f"usage: download_video_model.py <{ '|'.join(config.VIDEOGEN_MODELS) }>")
        return 2

    key = sys.argv[1]
    repo = config.VIDEOGEN_MODELS[key]["repo"]
    os.makedirs(os.path.dirname(_MARKER), exist_ok=True)
    try:
        with open(_MARKER, "w") as f:
            f.write(repo)
        from huggingface_hub import snapshot_download
        print(f"Downloading video model '{key}' ({repo}) ...", flush=True)
        snapshot_download(repo_id=repo, ignore_patterns=_IGNORE)
        print(f"Done — '{key}' is cached and ready.", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR downloading {repo}: {type(e).__name__}: {e}", flush=True)
        return 1
    finally:
        try:
            os.remove(_MARKER)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
