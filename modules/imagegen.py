"""
Image generation (client side). A Stable Diffusion pipeline runs in
imagegen/.venv (CUDA torch + diffusers) as a persistent localhost HTTP helper
that the main assistant calls — same isolation pattern as NeuTTS/Chatterbox.
See imagegen/imagegen_server.py.

The generate_image skill (modules/skills/imagegen.py) calls generate_to_file(),
which writes a PNG into config.IMAGEGEN_OUT_DIR and records its name as the
turn's "pending" image. After the reply finishes streaming, assistant.py picks
up take_pending() and appends an [[IMAGE:name]] marker to the logged turn so the
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
_BASE = f"http://127.0.0.1:{config.IMAGEGEN_PORT}"
_LOG = os.path.join(config._BASE, "logs", "imagegen_helper.log")

# Image filenames generated during the current turn, awaiting attachment to the
# logged reply. Guarded by a lock since the skill runs on the LLM thread.
_pending: list[str] = []
_pending_lock = threading.Lock()


def _health() -> dict | None:
    """The helper's /health payload, or None if it's not reachable/ready."""
    try:
        with urllib.request.urlopen(f"{_BASE}/health", timeout=2) as r:
            if r.status != 200:
                return None
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def ensure_server(wait_s: float = 300.0) -> bool:
    """Make sure the image helper is up, loaded, and serving the CONFIGURED model,
    spawning it in the isolated venv if needed (first run also downloads the
    model). Respawns it if the selected model changed. Returns True once ready.
    Model load + first-run download can take a while, hence the long timeout."""
    global _proc
    if not config.IMAGEGEN_ENABLED:
        print("[ImageGen] No image model selected (set one in the panel).")
        return False
    h = _health()
    if h and h.get("ready"):
        if h.get("model") == config.IMAGEGEN_REPO:
            return True
        # Selected model changed — tear the helper down and respawn on the new one.
        print(f"[ImageGen] Model changed → restarting helper "
              f"({h.get('model')} → {config.IMAGEGEN_REPO}).")
        stop()
        for _ in range(20):
            if _health() is None:
                break
            time.sleep(0.25)

    if _proc is None or _proc.poll() is not None:
        py = config.IMAGEGEN_PYTHON
        server = config.IMAGEGEN_SERVER
        if not (os.path.exists(py) and os.path.exists(server)):
            print(f"[ImageGen] Not installed (missing {py} or {server}). "
                  "Run imagegen/setup_imagegen.sh first.")
            return False
        env = dict(os.environ,
                   IMAGEGEN_PORT=str(config.IMAGEGEN_PORT),
                   IMAGEGEN_MODEL=config.IMAGEGEN_REPO,
                   IMAGEGEN_DEVICE=config.IMAGEGEN_DEVICE,
                   IMAGEGEN_STEPS=str(config.IMAGEGEN_STEPS),
                   IMAGEGEN_SIZE=str(config.IMAGEGEN_IMG_SIZE))
        print(f"[ImageGen] Starting helper "
              f"(model: {config.IMAGEGEN_REPO}, {config.IMAGEGEN_DEVICE})...")
        os.makedirs(os.path.dirname(_LOG), exist_ok=True)
        logf = open(_LOG, "a")
        _proc = subprocess.Popen([py, server], env=env, stdout=logf, stderr=logf)

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            print(f"[ImageGen] Helper exited unexpectedly — see {_LOG}.")
            return False
        h = _health()
        if h and h.get("ready"):
            print(f"[ImageGen] Helper ready ({h.get('device')}).")
            return True
        if h and h.get("error"):
            print(f"[ImageGen] Helper failed to load: {h['error']}")
            return False
        time.sleep(1.0)
    print("[ImageGen] Helper did not become ready in time.")
    return False


def warmup() -> None:
    ensure_server()


def generate(prompt: str, steps: int | None = None,
             width: int | None = None, height: int | None = None,
             guidance: float | None = None, sampler: str | None = None) -> bytes:
    """Generate an image (PNG bytes) for `prompt` via the helper, or raise."""
    if not ensure_server():
        raise RuntimeError("image helper unavailable")
    payload: dict = {"prompt": prompt}
    if steps:
        payload["steps"] = steps
    if width:
        payload["width"] = width
    if height:
        payload["height"] = height
    if guidance is not None:
        payload["guidance"] = guidance
    if sampler:
        payload["sampler"] = sampler
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{_BASE}/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if r.status != 200:
            raise RuntimeError(f"helper error: {r.read().decode(errors='replace')}")
        return r.read()


def _slug(text: str, n: int = 32) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in text.lower()]
    s = "".join(keep).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return (s[:n].strip("-")) or "image"


def save_image(prompt: str, **kw) -> str:
    """Generate an image and save it under config.IMAGEGEN_OUT_DIR. Returns the
    bare filename. Does NOT register it as a pending chat image (used by the
    standalone /image page). Raises on failure."""
    data = generate(prompt, **kw)
    out_dir = config.IMAGEGEN_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    name = f"{int(time.time())}-{_slug(prompt)}.png"
    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(data)
    return name


def generate_to_file(prompt: str, **kw) -> str:
    """Like save_image, but also registers the image as pending for the current
    turn so the chat view can render it (used by the generate_image skill)."""
    name = save_image(prompt, **kw)
    with _pending_lock:
        _pending.append(name)
    return name


def take_pending() -> list[str]:
    """Return and clear the images generated since the last call (one turn's
    worth). assistant.py calls this after a reply finishes streaming."""
    with _pending_lock:
        out = list(_pending)
        _pending.clear()
    return out


def stop() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
    _proc = None
