#!/usr/bin/env python3
"""
Pre-download an image model's weights from HuggingFace into the shared HF cache
(~/.cache/huggingface), which the image helper venv reads at load time. Driven by
the IMAGEGEN_MODELS registry in config.py.

    python3 download_image_model.py <image_model_key>

Launched detached by the control panel's image "Download" button; progress is
written to logs/image_download.log. Safe to re-run — snapshot_download resumes
and skips already-present files. A sentinel file (logs/.image_downloading, whose
contents are the repo id) marks an in-progress download so the panel can show
status; it is removed on success or failure.

We skip *.bin / *.ckpt / *.pth duplicates: diffusers prefers .safetensors, so the
fp32 safetensors we actually load are fetched while the redundant pickle copies
(roughly half the repo) are not.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

_MARKER = os.path.join(config._BASE, "logs", ".image_downloading")
_IGNORE = ["*.bin", "*.ckpt", "*.pth", "*.onnx", "*.msgpack", "*.h5"]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in config.IMAGEGEN_MODELS:
        print(f"usage: download_image_model.py <{ '|'.join(config.IMAGEGEN_MODELS) }>")
        return 2

    key = sys.argv[1]
    repo = config.IMAGEGEN_MODELS[key]["repo"]
    os.makedirs(os.path.dirname(_MARKER), exist_ok=True)
    try:
        with open(_MARKER, "w") as f:
            f.write(repo)
        from huggingface_hub import snapshot_download
        print(f"Downloading image model '{key}' ({repo}) ...", flush=True)
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
