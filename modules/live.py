"""
Live in-progress turn for the chat view. The assistant writes the current turn
here as it unfolds — what it's hearing/thinking and what it's saying, token by
token — so the control panel can stream the conversation in instead of only
showing completed turns. One small JSON object, overwritten in place and cleared
back to idle when the turn ends:
  {"phase": "listening"|"thinking"|"speaking"|"idle",
   "user": "<transcript or null>", "assistant": "<partial reply>", "ts": <unix>}
"""
import os
import json
import time

import config

LIVE_PATH = os.path.join(os.path.dirname(config.CHATLOG_PATH), "live.json")

_state = {"phase": "idle", "user": None, "assistant": "", "ts": 0.0}
_last_write = 0.0   # monotonic; throttles the fast token stream


def _flush() -> None:
    if not config.CHATLOG_ENABLED:
        return
    _state["ts"] = time.time()
    try:
        os.makedirs(os.path.dirname(LIVE_PATH), exist_ok=True)
        tmp = LIVE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_state, f)
        os.replace(tmp, LIVE_PATH)   # atomic — readers never see a half file
    except OSError:
        pass


def begin(phase: str = "listening", user=None) -> None:
    """Start a fresh live turn."""
    global _last_write
    _last_write = 0.0
    _state.update(phase=phase, user=user, assistant="")
    _flush()


def phase(p: str) -> None:
    _state["phase"] = p
    _flush()


def set_user(text: str) -> None:
    _state["user"] = text
    _flush()


def assistant_delta(text: str, throttle: float = 0.12) -> None:
    """Append a streamed delta to the in-progress reply, writing at most every
    `throttle` seconds so a fast stream doesn't hammer the disk. Pair with
    flush() so the final text always lands."""
    global _last_write
    _state["assistant"] += text
    now = time.monotonic()
    if now - _last_write >= throttle:
        _last_write = now
        _flush()


def flush() -> None:
    """Force-write the current state (e.g. after the last token)."""
    global _last_write
    _last_write = time.monotonic()
    _flush()


def clear() -> None:
    """Mark the turn finished — the completed turn now lives in the chat log."""
    _state.update(phase="idle", user=None, assistant="")
    _flush()
