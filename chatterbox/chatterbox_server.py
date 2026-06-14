#!/usr/bin/env python3
"""
Persistent Chatterbox (Resemble AI) TTS helper. Runs in this directory's isolated
Python 3.12 venv (CUDA torch + chatterbox-tts) so it can use the GPU, loads the
model + reference voice ONCE, and serves synthesis over localhost HTTP — the main
assistant (Python 3.14, which can't install torch) calls it as a TTS engine.

Chatterbox clones from an audio prompt alone (no transcript needed), so a voice is
just a <name>.wav. Reference conditionals are encoded once per voice and cached.

  GET  /health  -> 200 once the model + reference voice are loaded
  POST /tts     -> body {"text": "...", "voice": "<optional name/abs path>"}
                   returns 24 kHz mono int16 WAV bytes

Configured via env vars (set by the assistant when it spawns this):
  CB_PORT          (default 5009)
  CB_DEVICE        cuda | cpu  (default cuda; falls back to cpu if unavailable)
  CB_REF_WAV       reference voice wav to clone (default ../neutts_test/samples/sophon_2.wav)
  CB_SAMPLES_DIR   dir of <name>.wav clips, for per-request voice override
  CB_EXAGGERATION  expressiveness 0..1   (default 0.5)
  CB_CFG           cfg/pacing weight 0..1 (default 0.5; lower = slower/calmer)
  CB_TEMPERATURE   sampling temperature   (default 0.8)
"""
import io
import os
import sys
import json
import wave
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("CB_PORT", "5009"))
DEVICE = os.environ.get("CB_DEVICE", "cuda")
REF_WAV = os.environ.get(
    "CB_REF_WAV", os.path.join(HERE, "..", "neutts_test", "samples", "sophon_2.wav"))
SAMPLES_DIR = os.environ.get("CB_SAMPLES_DIR", os.path.dirname(os.path.abspath(REF_WAV)))
EXAGGERATION = float(os.environ.get("CB_EXAGGERATION", "0.5"))
CFG = float(os.environ.get("CB_CFG", "0.5"))
TEMPERATURE = float(os.environ.get("CB_TEMPERATURE", "0.8"))

_model = None
_sr = 24000
_default_voice = os.path.splitext(os.path.basename(REF_WAV))[0]
_conds_cache: dict[str, object] = {}     # voice name -> conditionals object
_lock = threading.Lock()                  # serialize synthesis (one model instance)
_ready = threading.Event()


def _resolve_device(want: str) -> str:
    import torch
    if want != "cpu" and torch.cuda.is_available():
        return "cuda"
    if want != "cpu":
        print("[chatterbox] CUDA unavailable — falling back to CPU.", flush=True)
    return "cpu"


def _ref_path(voice: str) -> str:
    if os.path.isabs(voice):
        return voice if voice.endswith(".wav") else voice + ".wav"
    return os.path.join(SAMPLES_DIR, f"{voice}.wav")


def _load():
    global _model, _sr, DEVICE
    from chatterbox.tts import ChatterboxTTS
    DEVICE = _resolve_device(DEVICE)
    print(f"[chatterbox] loading model on {DEVICE} (ref={os.path.basename(REF_WAV)}) ...", flush=True)
    _model = ChatterboxTTS.from_pretrained(device=DEVICE)
    _sr = int(getattr(_model, "sr", 24000))
    _model.prepare_conditionals(REF_WAV, exaggeration=EXAGGERATION)
    _conds_cache[_default_voice] = _model.conds
    _ready.set()
    print(f"[chatterbox] ready on port {PORT} (sr={_sr}, device={DEVICE})", flush=True)


def _select_voice(voice):
    """Point the model at a voice's conditionals, encoding+caching on first use.
    Falls back to the default voice if the named clip is missing."""
    name = voice or _default_voice
    if name not in _conds_cache:
        path = _ref_path(name)
        if not os.path.exists(path):
            name = _default_voice
        else:
            _model.prepare_conditionals(path, exaggeration=EXAGGERATION)
            _conds_cache[name] = _model.conds
    _model.conds = _conds_cache[name]


def _to_wav_bytes(wav) -> bytes:
    """Chatterbox output tensor -> 24 kHz mono int16 WAV bytes."""
    arr = np.asarray(wav.squeeze().detach().cpu().numpy(), dtype=np.float32)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_sr)
        wf.writeframes(pcm)
    return buf.getvalue()


def _synth(text: str, voice=None) -> bytes:
    with _lock:
        _select_voice(voice)
        wav = _model.generate(text, exaggeration=EXAGGERATION,
                              cfg_weight=CFG, temperature=TEMPERATURE)
    return _to_wav_bytes(wav)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):   # quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            ok = _ready.is_set()
            body = json.dumps({"ready": ok, "voice": _default_voice,
                               "device": DEVICE, "pid": os.getpid()}).encode()
            self.send_response(200 if ok else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/tts":
            self.send_response(404); self.end_headers(); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            text = (body.get("text") or "").strip()
            if not text:
                self.send_response(400); self.end_headers(); return
            data = _synth(text, body.get("voice"))
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            print(f"[chatterbox] synth error: {e}", flush=True)
            self.send_response(500); self.end_headers()


def main():
    # Load the model in the background so /health answers immediately.
    threading.Thread(target=_load, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[chatterbox] http server listening on 127.0.0.1:{PORT}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
