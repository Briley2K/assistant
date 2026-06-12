"""
Append-only conversation log (JSONL). The assistant writes turns here; the
control panel tails the file to show the chat live. One JSON object per line:
{"ts": <unix seconds>, "role": "user"|"assistant", "text": "...",
 "meta": {"ttft_ms": ..., "tok_per_sec": ..., "voice_ms": ..., "tokens": ...}}
The optional "meta" object (assistant turns only) carries timing measurements.
"""
import os
import json
import time

import config


def log(role: str, text: str, meta: dict | None = None) -> None:
    """Append one turn. No-ops on empty text or if logging is disabled.

    `meta` holds optional per-turn measurements (e.g. time-to-first-token,
    tokens/sec) that the chat view renders alongside the message."""
    if not config.CHATLOG_ENABLED:
        return
    text = (text or "").strip()
    if not text:
        return
    try:
        os.makedirs(os.path.dirname(config.CHATLOG_PATH), exist_ok=True)
        entry = {"ts": time.time(), "role": role, "text": text}
        if meta:
            entry["meta"] = meta
        with open(config.CHATLOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def tail(limit: int = 300) -> list[dict]:
    """Return the most recent `limit` turns (oldest first)."""
    try:
        with open(config.CHATLOG_PATH) as f:
            lines = f.readlines()[-limit:]
    except FileNotFoundError:
        return []
    turns = []
    for line in lines:
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return turns


def clear() -> None:
    try:
        open(config.CHATLOG_PATH, "w").close()
    except OSError:
        pass
