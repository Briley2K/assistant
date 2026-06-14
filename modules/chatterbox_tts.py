"""
Chatterbox TTS engine (client side). The model runs in chatterbox/.venv (CUDA
torch + chatterbox-tts) as a persistent localhost HTTP helper that the main
assistant calls — same isolation pattern as NeuTTS. See chatterbox/chatterbox_server.py.

Unlike NeuTTS, the helper's output is sent to logs/chatterbox_helper.log (not
/dev/null) so a synthesis failure is actually visible instead of silently falling
back to Kokoro forever.
"""
import os
import time
import json
import subprocess
import urllib.request
import urllib.error

import config

_proc: subprocess.Popen | None = None
_BASE = f"http://127.0.0.1:{config.CHATTERBOX_PORT}"
_LOG = os.path.join(config._BASE, "logs", "chatterbox_helper.log")


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
    return h.get("voice") == config.CHATTERBOX_VOICE


def ensure_server(wait_s: float = 240.0) -> bool:
    """Make sure the Chatterbox helper is up, loaded, and serving the CONFIGURED
    voice. Spawns it in the isolated venv if needed (model load + first-run weight
    download can take a while). Respawns it if the configured voice changed.
    Returns True once ready."""
    global _proc
    h = _health()
    if h and h.get("ready"):
        if _matches_config(h):
            return True
        pid = h.get("pid")
        print(f"[TTS] Chatterbox voice changed — restarting helper (pid {pid}).")
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
        py = config.CHATTERBOX_PYTHON
        server = config.CHATTERBOX_SERVER
        if not (os.path.exists(py) and os.path.exists(server)):
            print(f"[TTS] Chatterbox not installed (missing {py} or {server}). "
                  "Run chatterbox/setup_chatterbox.sh first.")
            return False
        ref_wav = config.chatterbox_ref_wav()
        env = dict(os.environ,
                   CB_PORT=str(config.CHATTERBOX_PORT),
                   CB_DEVICE=config.CHATTERBOX_DEVICE,
                   CB_REF_WAV=ref_wav,
                   CB_SAMPLES_DIR=os.path.dirname(ref_wav),
                   CB_EXAGGERATION=str(config.CHATTERBOX_EXAGGERATION),
                   CB_CFG=str(config.CHATTERBOX_CFG),
                   CB_TEMPERATURE=str(config.CHATTERBOX_TEMPERATURE))
        print(f"[TTS] Starting Chatterbox helper "
              f"(voice: {config.CHATTERBOX_VOICE}, {config.CHATTERBOX_DEVICE})...")
        os.makedirs(os.path.dirname(_LOG), exist_ok=True)
        logf = open(_LOG, "a")
        _proc = subprocess.Popen([py, server], env=env, stdout=logf, stderr=logf)

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            print(f"[TTS] Chatterbox helper exited unexpectedly — see {_LOG}.")
            return False
        h = _health()
        if h and h.get("ready"):
            print(f"[TTS] Chatterbox helper ready ({h.get('device')}).")
            return True
        time.sleep(1.0)
    print("[TTS] Chatterbox helper did not become ready in time.")
    return False


def warmup() -> None:
    ensure_server()


def synth_wav(text: str, voice: str | None = None) -> bytes:
    """Synthesize text → WAV bytes (24 kHz mono int16) via the helper. An optional
    voice name overrides the configured reference (used by the panel preview)."""
    if not ensure_server():
        raise RuntimeError("Chatterbox helper unavailable")
    payload = {"text": text}
    if voice:
        payload["voice"] = voice
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{_BASE}/tts", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def stop() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
    _proc = None
