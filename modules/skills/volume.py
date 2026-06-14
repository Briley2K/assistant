"""
Per-application volume control (PipeWire / WirePlumber).

Lets the user say things like "turn down chrome" (halve that app's current
volume) or "turn the volume back up" (restore lowered apps to 100%) WITHOUT
touching the overall system volume — each playing app is its own audio stream.

How it works:
  * `pw-dump` lists every audio output stream as JSON; we pick the ones whose
    media.class is Stream/Output/Audio and fuzzy-match the spoken app name
    against their application/node/media names.
  * `wpctl get-volume` / `set-volume` / `set-mute` read and change the volume of
    that specific stream node (not the sink), so other apps are unaffected.

We remember which apps we lowered (`_lowered`) so a bare "turn the volume back
up" can restore them even when no app is named.
"""
import os
import re
import json
import subprocess

from modules import phrases
from modules.skills import skill

# Normalized names of apps we've turned down this session, so "turn it back up"
# (no app named) knows what to restore.
_lowered: set[str] = set()

_DOWN_WORDS = {"down", "lower", "quieter", "reduce", "soften", "half", "low"}
_UP_WORDS = {"up", "restore", "back", "reset", "normal", "louder", "raise",
             "full", "100", "max"}
_MUTE_WORDS = {"mute", "silence", "off"}
_UNMUTE_WORDS = {"unmute", "on"}


def _clean_env() -> dict:
    """Env minus snap-injected library paths that can break system binaries
    (wpctl/pw-dump) when the assistant is started from a snap terminal.
    Mirrors assistant._clean_env / apps._launch_env."""
    env = {}
    for k, v in os.environ.items():
        if k.startswith("SNAP") or k.endswith("_VSCODE_SNAP_ORIG"):
            continue
        if k == "PATH":
            env[k] = ":".join(p for p in v.split(":") if "/snap/" not in p)
        elif "/snap/" in v:
            orig = os.environ.get(f"{k}_VSCODE_SNAP_ORIG", "")
            if orig:
                env[k] = orig
        else:
            env[k] = v
    return env


def _run(argv: list[str]) -> str:
    return subprocess.run(
        argv, env=_clean_env(), capture_output=True, text=True, timeout=5,
    ).stdout


def _output_streams() -> list[dict]:
    """Active audio output streams: [{id, names:[...]}], newest-looking first."""
    try:
        data = json.loads(_run(["pw-dump"]) or "[]")
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError):
        return []
    streams = []
    for obj in data:
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        props = (obj.get("info") or {}).get("props") or {}
        if props.get("media.class") != "Stream/Output/Audio":
            continue
        names = [props.get(k) for k in (
            "application.name", "node.name", "media.name",
            "application.process.binary")]
        names = [n for n in names if n]
        if names:
            streams.append({"id": obj["id"], "names": names})
    return streams


def _norm_names(stream: dict) -> set[str]:
    return {phrases.normalize(n) for n in stream["names"]}


def _match(query: str, streams: list[dict]) -> list[dict]:
    """Streams whose name matches the spoken app query (substring or fuzzy)."""
    q = phrases.normalize(query)
    if not q:
        return []
    hits = []
    for s in streams:
        for nn in _norm_names(s):
            if not nn:
                continue
            if (q == nn or q in nn.split() or q in nn or nn in q
                    or phrases.similarity(q, nn) >= 0.6):
                hits.append(s)
                break
    return hits


def _get_vol(node_id) -> float | None:
    m = re.search(r"Volume:\s*([\d.]+)", _run(["wpctl", "get-volume", str(node_id)]))
    return float(m.group(1)) if m else None


def _set_vol(node_id, vol: float) -> None:
    # Cap at 1.0 (100%) so "back up" never overshoots into amplification.
    _run(["wpctl", "set-volume", str(node_id), f"{max(0.0, min(vol, 1.0)):.2f}"])


def _set_mute(node_id, muted: bool) -> None:
    _run(["wpctl", "set-mute", str(node_id), "1" if muted else "0"])


@skill("set_app_volume",
       "Raise, lower, restore, mute, or set the audio volume of a specific "
       "running app (Chrome, Spotify, ...) WITHOUT changing the overall system "
       "volume. ALWAYS call this for ANY app-volume request, including turning "
       "the volume back UP or restoring it — never just say you did it. "
       "action 'down' halves the app's current volume, 'up' restores it to 100%, "
       "'mute'/'unmute' toggle it, or pass a number 0-100 for an exact percent. "
       "A bare 'up' with no app restores every app that was previously turned down.",
       {"app": "the app's name as spoken, e.g. 'chrome' (omit only for a bare 'up' to restore all lowered apps)",
        "action": "down, up, mute, unmute, or a number 0-100"})
def _set_app_volume(args):
    app = (args.get("app") or args.get("name") or "").strip()
    action = str(args.get("action") or args.get("level")
                 or args.get("direction") or "down").strip().lower().rstrip("%")

    streams = _output_streams()
    if not streams:
        return {"error": "no apps are currently playing audio"}

    # "Turn the volume back up" with no app named → restore everything we lowered.
    if not app and action in _UP_WORDS:
        restored = []
        for s in streams:
            if _lowered & _norm_names(s):
                _set_vol(s["id"], 1.0)
                _lowered.difference_update(_norm_names(s))
                restored.append(s["names"][0])
        if not restored:  # nothing tracked — restore all output streams to 100%
            for s in streams:
                _set_vol(s["id"], 1.0)
                restored.append(s["names"][0])
        return {"ok": True, "action": "up", "apps": restored, "volume": "100%"}

    if not app:
        playing = ", ".join(sorted({s["names"][0] for s in streams}))
        return {"error": f"which app? currently playing: {playing or 'nothing'}"}

    hits = _match(app, streams)
    if not hits:
        playing = ", ".join(sorted({s["names"][0] for s in streams}))
        return {"error": f"'{app}' isn't playing audio. Currently playing: {playing or 'nothing'}"}

    results = []
    for s in hits:
        name = s["names"][0]
        if action in _DOWN_WORDS:
            cur = _get_vol(s["id"])
            new = (cur if cur is not None else 1.0) * 0.5
            _set_vol(s["id"], new)
            _lowered.update(_norm_names(s))
            results.append({"app": name, "volume": f"{round(new * 100)}%"})
        elif action in _UP_WORDS:
            _set_vol(s["id"], 1.0)
            _lowered.difference_update(_norm_names(s))
            results.append({"app": name, "volume": "100%"})
        elif action in _MUTE_WORDS:
            _set_mute(s["id"], True)
            results.append({"app": name, "volume": "muted"})
        elif action in _UNMUTE_WORDS:
            _set_mute(s["id"], False)
            results.append({"app": name, "volume": "unmuted"})
        elif re.fullmatch(r"\d+(\.\d+)?", action):
            pct = max(0.0, min(float(action), 100.0))
            _set_vol(s["id"], pct / 100.0)
            if pct < 100:
                _lowered.update(_norm_names(s))
            else:
                _lowered.difference_update(_norm_names(s))
            results.append({"app": name, "volume": f"{round(pct)}%"})
        else:
            return {"error": f"unknown action '{action}' (use down, up, mute, unmute, or 0-100)"}

    return {"ok": True, "changed": results}
