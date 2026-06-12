"""
NeuTTS Air TTS engine (client side). The model itself runs in an isolated
Python 3.12 venv (it needs torch, which the main 3.14 venv can't install), so
this module spawns that venv's `neutts_server.py` as a persistent localhost HTTP
helper and calls it to synthesize speech. See neutts_test/neutts_server.py.
"""
import os
import time
import json
import subprocess
import urllib.request
import urllib.error

import config

_proc: subprocess.Popen | None = None
_BASE = f"http://127.0.0.1:{config.NEUTTS_PORT}"


def _health() -> dict | None:
    """The helper's /health payload, or None if it's not reachable/ready."""
    try:
        with urllib.request.urlopen(f"{_BASE}/health", timeout=2) as r:
            if r.status != 200:
                return None
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _matches_config(h: dict) -> bool:
    return (h.get("voice") == config.NEUTTS_VOICE
            and h.get("backbone") == config.NEUTTS_BACKBONE)


def ensure_server(wait_s: float = 180.0) -> bool:
    """Make sure the NeuTTS helper is up, loaded, and serving the CONFIGURED
    voice/model. Spawns it in the isolated venv if needed (model load ~15-30s).
    If a helper is already running with a different voice/backbone, it's killed
    and respawned so settings changes take effect. Returns True once ready."""
    global _proc
    h = _health()
    if h and h.get("ready"):
        if _matches_config(h):
            return True
        # Wrong voice/model — stop it (even if it's not our child) and respawn.
        pid = h.get("pid")
        print(f"[TTS] NeuTTS voice/model changed — restarting helper (pid {pid}).")
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
        elif pid:
            try:
                os.kill(int(pid), 15)
            except (OSError, ValueError):
                pass
        _proc = None
        for _ in range(20):                 # wait for the port to free
            if _health() is None:
                break
            time.sleep(0.25)

    if _proc is None or _proc.poll() is not None:
        py = config.NEUTTS_PYTHON
        server = config.NEUTTS_SERVER
        if not (os.path.exists(py) and os.path.exists(server)):
            print(f"[TTS] NeuTTS not installed (missing {py} or {server}). "
                  "Run the neutts_test setup first.")
            return False
        ref_wav, ref_txt = config.neutts_ref_paths()
        env = dict(os.environ,
                   NEUTTS_PORT=str(config.NEUTTS_PORT),
                   NEUTTS_BACKBONE=config.NEUTTS_BACKBONE,
                   NEUTTS_REF_WAV=ref_wav,
                   NEUTTS_REF_TXT=ref_txt)
        print(f"[TTS] Starting NeuTTS helper (voice: {config.NEUTTS_VOICE})...")
        _proc = subprocess.Popen([py, server], env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            print("[TTS] NeuTTS helper exited unexpectedly.")
            return False
        h = _health()
        if h and h.get("ready"):
            print("[TTS] NeuTTS helper ready.")
            return True
        time.sleep(1.0)
    print("[TTS] NeuTTS helper did not become ready in time.")
    return False


def warmup() -> None:
    ensure_server()


def synth_wav(text: str, voice: str | None = None) -> bytes:
    """Synthesize text → WAV bytes (24 kHz mono int16) via the helper. An
    optional voice name overrides the configured reference (used by the panel's
    voice preview); the assistant leaves it None to use the configured voice."""
    if not ensure_server():
        raise RuntimeError("NeuTTS helper unavailable")
    payload = {"text": text}
    if voice:
        payload["voice"] = voice
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{_BASE}/tts", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def stop() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
    _proc = None
