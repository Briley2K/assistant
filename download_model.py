#!/usr/bin/env python3
"""
Download a single model's GGUF (and mmproj, if any) from its HuggingFace repo
into the model's local dir. Driven by the LLM_MODELS registry in config.py.

    python3 download_model.py <model_key>

Launched detached by the control panel's "Download" button; progress is written
to logs/model_download.log. Safe to re-run — hf_hub_download resumes/skips.
A sentinel file (<dir>/.downloading) marks an in-progress download so the panel
can show status; it is removed on success or failure.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in config.LLM_MODELS:
        print(f"usage: download_model.py <{ '|'.join(config.LLM_MODELS) }>")
        return 2

    key = sys.argv[1]
    m = config.LLM_MODELS[key]
    repo = m.get("hf_repo")
    dest = m["dirs"][0]
    if not repo:
        print(f"No hf_repo configured for '{key}' — nothing to download.")
        return 2

    os.makedirs(dest, exist_ok=True)
    sentinel = os.path.join(dest, ".downloading")
    open(sentinel, "w").close()
    try:
        from huggingface_hub import hf_hub_download
        files = [m["gguf"]] + ([m["mmproj"]] if m.get("mmproj") else [])
        for fn in files:
            target = os.path.join(dest, fn)
            if os.path.exists(target):
                print(f"already present: {target}", flush=True)
                continue
            print(f"downloading {fn} from {repo} (resumes if interrupted)...", flush=True)
            hf_hub_download(repo, fn, local_dir=dest)
            print(f"saved: {target}", flush=True)
        print(f"DONE: {key} ready in {dest}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR downloading {key}: {e}", flush=True)
        return 1
    finally:
        try:
            os.remove(sentinel)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
