#!/usr/bin/env python3
"""
Persistent local text-to-video helper. Runs in this directory's isolated Python
venv (CUDA torch + diffusers + imageio-ffmpeg), loads a video diffusion pipeline
ONCE, and serves generation over localhost HTTP so the main assistant (which
can't import torch) can use it.

  GET  /health    -> 200 {"ready": bool, "model": ..., "device": ..., "pid": ...}
  POST /generate  -> body {"prompt": "...", "frames": <int?>, "steps": <int?>,
                           "fps": <int?>} ; returns MP4 bytes

Configured via env vars (set by the assistant when it spawns this):
  VIDEOGEN_PORT    (default 5011)
  VIDEOGEN_MODEL   diffusers repo id, local diffusers folder, or single-file path
  VIDEOGEN_DEVICE  "cuda" or "cpu" (default cuda)
  VIDEOGEN_STEPS   override inference steps (default 0 = model-appropriate)
  VIDEOGEN_FRAMES  number of frames to generate (default 81)
  VIDEOGEN_FPS     frames per second for the output mp4 (default 16)
"""
import io
import os
import sys
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT   = int(os.environ.get("VIDEOGEN_PORT", "5011"))
MODEL  = os.environ.get("VIDEOGEN_MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
DEVICE = os.environ.get("VIDEOGEN_DEVICE", "cuda")
STEPS  = int(os.environ.get("VIDEOGEN_STEPS", "0"))
FRAMES = int(os.environ.get("VIDEOGEN_FRAMES", "81"))
FPS    = int(os.environ.get("VIDEOGEN_FPS", "16"))

_pipe = None
_lock = threading.Lock()        # serialize generation (one pipeline instance)
_ready = threading.Event()
_load_error = None


def _default_steps() -> int:
    return STEPS if STEPS > 0 else 40


def _load():
    global _pipe, _load_error
    print(f"[videogen] loading model={MODEL} on {DEVICE} ...", flush=True)
    try:
        import torch
        from diffusers import DiffusionPipeline
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        is_file = MODEL.lower().endswith((".safetensors", ".ckpt")) and os.path.exists(MODEL)
        if is_file:
            print("[videogen] loading custom checkpoint via from_single_file ...", flush=True)
            pipe = DiffusionPipeline.from_single_file(MODEL, torch_dtype=dtype)
        else:
            # DiffusionPipeline auto-resolves the right class (WanPipeline, LTXPipeline, …)
            # from the repo's model_index.json.
            pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=dtype)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        if DEVICE == "cuda":
            # Offload submodules between CPU/GPU so large video models fit in VRAM.
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe = pipe.to("cuda")
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
        else:
            pipe = pipe.to("cpu")
        global _pipe
        _pipe = pipe
        _ready.set()
        print(f"[videogen] ready on port {PORT}", flush=True)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        print(f"[videogen] load failed: {_load_error}", flush=True)


def _generate(prompt: str, frames=None, steps=None, fps=None) -> bytes:
    from diffusers.utils import export_to_video
    n_frames = int(frames) if frames else FRAMES
    n_steps = int(steps) if steps else _default_steps()
    out_fps = int(fps) if fps else FPS
    kwargs = {"prompt": prompt, "num_frames": n_frames, "num_inference_steps": n_steps}
    with _lock:
        result = _pipe(**kwargs)
    video_frames = result.frames[0]            # list of PIL images
    tmp = os.path.join(tempfile.gettempdir(), f"videogen_{os.getpid()}.mp4")
    export_to_video(video_frames, tmp, fps=out_fps)
    with open(tmp, "rb") as f:
        data = f.read()
    try:
        os.remove(tmp)
    except OSError:
        pass
    return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):   # quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            ok = _ready.is_set()
            body = json.dumps({"ready": ok, "model": MODEL, "device": DEVICE,
                               "error": _load_error, "pid": os.getpid()}).encode()
            self.send_response(200 if ok else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/generate":
            self.send_response(404); self.end_headers(); return
        if not _ready.is_set():
            self.send_response(503); self.end_headers()
            self.wfile.write((_load_error or "model not ready").encode())
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                self.send_response(400); self.end_headers(); return
            data = _generate(prompt, frames=body.get("frames"),
                             steps=body.get("steps"), fps=body.get("fps"))
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            print(f"[videogen] generate error: {e}", flush=True)
            self.send_response(500); self.end_headers()
            try:
                self.wfile.write(f"{type(e).__name__}: {e}".encode())
            except Exception:
                pass


def main():
    threading.Thread(target=_load, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[videogen] http server listening on 127.0.0.1:{PORT}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
