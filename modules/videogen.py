"""
Video generation (client side). A diffusers text-to-video pipeline runs in
video/.venv (CUDA torch + diffusers + imageio-ffmpeg) as a persistent localhost
HTTP helper that the main assistant calls — same isolation pattern as imagegen.
See video/video_server.py.

The generate_video skill (modules/skills/video.py) calls generate_to_file(),
which writes an MP4 into config.VIDEOGEN_OUT_DIR and records its name as the
turn's "pending" video. After the reply finishes streaming, assistant.py picks
up take_pending() and appends a [[VIDEO:name]] marker to the logged turn so the
control panel renders it inline (and it's never spoken aloud).
"""
import os
import time
import json
import threading
import subprocess
import urllib.request
import urllib.error

import config

_proc: subprocess.Popen | None = None
_BASE = f"http://127.0.0.1:{config.VIDEOGEN_PORT}"
_LOG = os.path.join(config._BASE, "logs", "video_helper.log")

_pending: list[str] = []
_pending_lock = threading.Lock()


def _health() -> dict | None:
    try:
        with urllib.request.urlopen(f"{_BASE}/health", timeout=2) as r:
            if r.status != 200:
                return None
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def ensure_server(wait_s: float = 600.0) -> bool:
    """Make sure the video helper is up, loaded, and serving the CONFIGURED model,
    spawning it in the isolated venv if needed (first run downloads the model).
    Respawns if the selected model changed. Returns True once ready. Video models
    are large and slow to load, hence the long timeout."""
    global _proc
    if not config.VIDEOGEN_ENABLED:
        print("[VideoGen] No video model selected (set one in the panel).")
        return False
    h = _health()
    if h and h.get("ready"):
        if h.get("model") == config.VIDEOGEN_REPO:
            return True
        print(f"[VideoGen] Model changed → restarting helper "
              f"({h.get('model')} → {config.VIDEOGEN_REPO}).")
        stop()
        for _ in range(20):
            if _health() is None:
                break
            time.sleep(0.25)

    if _proc is None or _proc.poll() is not None:
        py = config.VIDEOGEN_PYTHON
        server = config.VIDEOGEN_SERVER
        if not (os.path.exists(py) and os.path.exists(server)):
            print(f"[VideoGen] Not installed (missing {py} or {server}). "
                  "Run video/setup_video.sh first.")
            return False
        env = dict(os.environ,
                   VIDEOGEN_PORT=str(config.VIDEOGEN_PORT),
                   VIDEOGEN_MODEL=config.VIDEOGEN_REPO,
                   VIDEOGEN_DEVICE=config.VIDEOGEN_DEVICE,
                   VIDEOGEN_STEPS=str(config.VIDEOGEN_STEPS),
                   VIDEOGEN_FRAMES=str(config.VIDEOGEN_FRAMES),
                   VIDEOGEN_FPS=str(config.VIDEOGEN_FPS))
        print(f"[VideoGen] Starting helper "
              f"(model: {config.VIDEOGEN_REPO}, {config.VIDEOGEN_DEVICE})...")
        os.makedirs(os.path.dirname(_LOG), exist_ok=True)
        logf = open(_LOG, "a")
        _proc = subprocess.Popen([py, server], env=env, stdout=logf, stderr=logf)

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            print(f"[VideoGen] Helper exited unexpectedly — see {_LOG}.")
            return False
        h = _health()
        if h and h.get("ready"):
            print(f"[VideoGen] Helper ready ({h.get('device')}).")
            return True
        if h and h.get("error"):
            print(f"[VideoGen] Helper failed to load: {h['error']}")
            return False
        time.sleep(1.0)
    print("[VideoGen] Helper did not become ready in time.")
    return False


def warmup() -> None:
    ensure_server()


def generate(prompt: str, frames: int | None = None, steps: int | None = None) -> bytes:
    """Generate a video (MP4 bytes) for `prompt` via the helper, or raise."""
    if not ensure_server():
        raise RuntimeError("video helper unavailable")
    payload: dict = {"prompt": prompt}
    if frames:
        payload["frames"] = frames
    if steps:
        payload["steps"] = steps
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{_BASE}/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:   # video gen is slow
        if r.status != 200:
            raise RuntimeError(f"helper error: {r.read().decode(errors='replace')}")
        return r.read()


def _slug(text: str, n: int = 32) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in text.lower()]
    s = "".join(keep).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return (s[:n].strip("-")) or "video"


def save_video(prompt: str, **kw) -> str:
    """Generate a video and save it under config.VIDEOGEN_OUT_DIR. Returns the bare
    filename. Does NOT register it as a pending chat video (used by /video page)."""
    data = generate(prompt, **kw)
    out_dir = config.VIDEOGEN_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    name = f"{int(time.time())}-{_slug(prompt)}.mp4"
    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(data)
    return name


def generate_to_file(prompt: str, **kw) -> str:
    """Like save_video, but also registers the video as pending for the current
    turn so the chat view can render it (used by the generate_video skill)."""
    name = save_video(prompt, **kw)
    with _pending_lock:
        _pending.append(name)
    return name


def take_pending() -> list[str]:
    """Return and clear the videos generated since the last call (one turn's worth)."""
    with _pending_lock:
        out = list(_pending)
        _pending.clear()
    return out


def stop() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
    _proc = None
