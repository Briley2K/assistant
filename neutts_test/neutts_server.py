#!/usr/bin/env python3
"""
Persistent NeuTTS Air synthesis helper. Runs in this directory's isolated
Python 3.12 venv (which has torch + neutts), loads the model + reference voice
ONCE, and serves synthesis over localhost HTTP so the main assistant (Python
3.14, which can't import neutts) can use it as a TTS engine.

  GET  /health  -> 200 "ready" once the model + reference voice are loaded
  POST /tts     -> body {"text": "..."} ; returns 24 kHz mono int16 WAV bytes

Configured via env vars (set by the assistant when it spawns this):
  NEUTTS_PORT      (default 5008)
  NEUTTS_BACKBONE  HF repo of the GGUF backbone (default neuphonic/neutts-air-q4-gguf)
  NEUTTS_REF_WAV   reference voice wav  (default samples/jo.wav)
  NEUTTS_REF_TXT   reference voice transcript (default samples/jo.txt)
"""
import io
import os
import sys
import json
import wave
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("NEUTTS_PORT", "5008"))
BACKBONE = os.environ.get("NEUTTS_BACKBONE", "neuphonic/neutts-air-q4-gguf")
REF_WAV = os.environ.get("NEUTTS_REF_WAV", os.path.join(HERE, "samples", "jo.wav"))
REF_TXT = os.environ.get("NEUTTS_REF_TXT", os.path.join(HERE, "samples", "jo.txt"))
SAMPLE_RATE = 24000

# --- bundled espeak-ng (no system install needed) ---
try:
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    if hasattr(EspeakWrapper, "set_data_path"):
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
except Exception as e:
    print(f"[neutts] espeak setup warning: {e}", flush=True)

import numpy as np

SAMPLES = os.path.join(HERE, "samples")
_tts = None
_refs: dict[str, tuple] = {}   # voice name -> (encoded_ref, ref_text), cached
_default_voice = os.path.splitext(os.path.basename(REF_WAV))[0]
_lock = threading.Lock()   # serialize synthesis (one model instance)
_ready = threading.Event()


def _encode_voice(name: str):
    """Return (ref_codes, ref_text) for a voice name, encoding+caching on first
    use. Falls back to the default voice if the named one is missing."""
    if name in _refs:
        return _refs[name]
    wav = os.path.join(SAMPLES, f"{name}.wav")
    txt = os.path.join(SAMPLES, f"{name}.txt")
    if not (os.path.exists(wav) and os.path.exists(txt)):
        return _refs[_default_voice]
    pair = (_tts.encode_reference(wav), open(txt).read().strip())
    _refs[name] = pair
    return pair


def _load():
    global _tts
    print(f"[neutts] loading backbone={BACKBONE} ref={os.path.basename(REF_WAV)} ...", flush=True)
    try:
        from neuttsair.neutts import NeuTTSAir
    except ImportError:
        from neutts import NeuTTS as NeuTTSAir
    _tts = NeuTTSAir(backbone_repo=BACKBONE, backbone_device="cpu",
                     codec_repo="neuphonic/neucodec", codec_device="cpu")
    _refs[_default_voice] = (_tts.encode_reference(REF_WAV), open(REF_TXT).read().strip())
    _ready.set()
    print(f"[neutts] ready on port {PORT}", flush=True)


def _to_wav_bytes(wav) -> bytes:
    """float array -> 24 kHz mono int16 WAV bytes."""
    arr = np.asarray(wav, dtype=np.float32).squeeze()
    pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def _synth(text: str, voice: str | None = None) -> bytes:
    ref, ref_text = _encode_voice(voice or _default_voice)
    with _lock:
        wav = _tts.infer(text, ref, ref_text)
    return _to_wav_bytes(wav)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):   # quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            ok = _ready.is_set()
            body = json.dumps({"ready": ok, "voice": _default_voice,
                               "backbone": BACKBONE, "pid": os.getpid()}).encode()
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
            print(f"[neutts] synth error: {e}", flush=True)
            self.send_response(500); self.end_headers()


def main():
    # Load the model in the background so /health answers immediately.
    threading.Thread(target=_load, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[neutts] http server listening on 127.0.0.1:{PORT}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
