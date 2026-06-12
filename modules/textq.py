"""
File-backed inbox for typed messages from the control panel. The panel appends
one JSON object per line; the running assistant drains new lines in order and
answers them through its normal pipeline (reusing the already-loaded model, so
no second copy is loaded). One object per line:
  {"text": "..."}
The assistant tracks a byte offset into the file; `drain()` resets to 0 if the
file shrinks (the panel cleared the conversation), so old messages never replay.
"""
import os
import json

import config

INBOX_PATH = os.path.join(os.path.dirname(config.CHATLOG_PATH), "text_inbox.jsonl")


def send(text: str) -> None:
    """Append a typed message for the assistant to pick up (no-op if empty)."""
    text = (text or "").strip()
    if not text:
        return
    os.makedirs(os.path.dirname(INBOX_PATH), exist_ok=True)
    with open(INBOX_PATH, "a") as f:
        f.write(json.dumps({"text": text}) + "\n")


def size() -> int:
    """Current inbox size in bytes (0 if it doesn't exist yet)."""
    try:
        return os.path.getsize(INBOX_PATH)
    except OSError:
        return 0


def drain(offset: int) -> tuple[list[str], int]:
    """Return (new messages after `offset`, new offset). Resets to 0 if the file
    was truncated/cleared since `offset` was taken."""
    try:
        if size() < offset:        # cleared or rotated — start over
            offset = 0
        with open(INBOX_PATH) as f:
            f.seek(offset)
            msgs = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msgs.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
            return msgs, f.tell()
    except FileNotFoundError:
        return [], offset


def clear() -> None:
    try:
        open(INBOX_PATH, "w").close()
    except OSError:
        pass
