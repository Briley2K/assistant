"""
Side-panel content for the overlay. When a reply contains something the user
would rather read or copy than hear — code, a story, a list — the assistant
writes it here and the overlay (overlay.py) pops a side panel showing it with a
Copy button, above a live view of the conversation. One small JSON object,
overwritten in place:
  {"ts": <unix>, "title": "Python code", "kind": "code"|"text", "content": "..."}
"""
import os
import json
import time

import config

PANEL_PATH = config.PANEL_PATH


def show(title: str, kind: str, content: str) -> None:
    """Publish an artifact for the overlay to display (no-op if disabled/empty)."""
    if not config.SIDE_PANEL_ENABLED or not content.strip():
        return
    try:
        os.makedirs(os.path.dirname(PANEL_PATH), exist_ok=True)
        tmp = PANEL_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"ts": time.time(), "title": title, "kind": kind,
                       "content": content}, f)
        os.replace(tmp, PANEL_PATH)   # atomic — the overlay never reads a half file
    except OSError:
        pass


def clear() -> None:
    try:
        os.remove(PANEL_PATH)
    except OSError:
        pass
