#!/usr/bin/env python3
"""
Persistent local image-generation helper. Runs in this directory's isolated
Python 3.12 venv (which has CUDA torch + diffusers), loads a Stable Diffusion
pipeline ONCE, and serves text-to-image over localhost HTTP so the main
assistant (Python 3.14, which can't import torch) can use it.

  GET  /health    -> 200 {"ready": bool, "model": ..., "device": ..., "pid": ...}
  POST /generate  -> body {"prompt": "...", "steps": <int?>, "width": <int?>,
                           "height": <int?>, "seed": <int?>} ; returns PNG bytes

Configured via env vars (set by the assistant when it spawns this):
  IMAGEGEN_PORT    (default 5010)
  IMAGEGEN_MODEL   diffusers model id / path (default stabilityai/sd-turbo)
  IMAGEGEN_DEVICE  "cuda" or "cpu" (default cuda)
  IMAGEGEN_STEPS   override inference steps (default 0 = model-appropriate)
"""
import io
import os
import sys
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT   = int(os.environ.get("IMAGEGEN_PORT", "5010"))
MODEL  = os.environ.get("IMAGEGEN_MODEL", "stabilityai/sd-turbo")
DEVICE = os.environ.get("IMAGEGEN_DEVICE", "cuda")
STEPS  = int(os.environ.get("IMAGEGEN_STEPS", "0"))
SIZE   = int(os.environ.get("IMAGEGEN_SIZE", "512"))   # native resolution default

_pipe = None
_lock = threading.Lock()        # serialize generation (one pipeline instance)
_ready = threading.Event()
_load_error: str | None = None


_orig_sched = None          # the pipeline's default scheduler, for "default"/restore
_gen_size = SIZE            # native generation resolution (refined at load time)

# Sampler name -> (diffusers scheduler class name, use_karras_sigmas).
_SAMPLERS = {
    "default":         None,
    "euler":           ("EulerDiscreteScheduler", False),
    "euler_a":         ("EulerAncestralDiscreteScheduler", False),
    "dpmpp_2m":        ("DPMSolverMultistepScheduler", False),
    "dpmpp_2m_karras": ("DPMSolverMultistepScheduler", True),
    "dpmpp_sde":       ("DPMSolverSinglestepScheduler", False),
    "unipc":           ("UniPCMultistepScheduler", False),
    "ddim":            ("DDIMScheduler", False),
    "lms":             ("LMSDiscreteScheduler", False),
    "heun":            ("HeunDiscreteScheduler", False),
}


def _apply_sampler(name: str):
    """Swap the pipeline scheduler to the requested sampler (called under _lock).
    'default'/unknown restores the model's original scheduler."""
    spec = _SAMPLERS.get(name or "default")
    if not spec:
        if _orig_sched is not None:
            _pipe.scheduler = _orig_sched
        return
    cls_name, karras = spec
    import diffusers
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        return
    kw = {"use_karras_sigmas": True} if karras else {}
    try:
        _pipe.scheduler = cls.from_config(_pipe.scheduler.config, **kw)
    except Exception as e:
        print(f"[imagegen] sampler '{name}' failed ({e}); keeping current.", flush=True)


def _detect_single_file_class(path: str):
    """Pick StableDiffusionXLPipeline vs StableDiffusionPipeline for a single-file
    checkpoint by inspecting its tensor keys (SDXL has a second text encoder and
    UNet add_embedding); falls back to the configured size if inspection fails."""
    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
    is_xl = SIZE >= 1024
    try:
        if path.lower().endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                keys = list(f.keys())
        else:
            import torch
            sd = torch.load(path, map_location="cpu", weights_only=False)
            keys = list((sd.get("state_dict", sd)).keys())
        # SDXL markers: dual text encoders (conditioner.embedders.1 in original
        # format, or text_encoder_2 in diffusers format) and UNet add_embedding.
        is_xl = any(k.startswith("conditioner.embedders.1") for k in keys) \
            or any(k.startswith("text_encoder_2") for k in keys) \
            or any("add_embedding" in k for k in keys)
        print(f"[imagegen] checkpoint inspected: {'SDXL' if is_xl else 'SD1.x'} "
              f"({len(keys)} tensors).", flush=True)
    except Exception as e:
        print(f"[imagegen] could not inspect checkpoint ({e}); "
              f"guessing by size ({'SDXL' if is_xl else 'SD1.x'}).", flush=True)
    return StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline


def _is_turbo() -> bool:
    """Turbo/LCM-style models are distilled for very few steps and no guidance."""
    return "turbo" in MODEL.lower() or "lcm" in MODEL.lower()


def _default_steps() -> int:
    if STEPS > 0:
        return STEPS
    return 2 if _is_turbo() else 25


def _load():
    global _pipe, _load_error
    print(f"[imagegen] loading model={MODEL} on {DEVICE} ...", flush=True)
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        is_file = MODEL.lower().endswith((".safetensors", ".ckpt")) and os.path.exists(MODEL)
        if is_file:
            # A custom uploaded checkpoint (single file). Detect SD vs SDXL from the
            # checkpoint's own tensor keys (filename/size are unreliable), then load
            # with the matching pipeline class via from_single_file.
            from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
            cls = _detect_single_file_class(MODEL)
            print(f"[imagegen] loading custom checkpoint as {cls.__name__} ...", flush=True)
            pipe = cls.from_single_file(MODEL, torch_dtype=dtype)
            # SDXL-based checkpoints (incl. Illustrious/Pony) render at 1024; SD1.x at 512.
            # Trust the detected architecture over the filename-based config size.
            global _gen_size
            _gen_size = 1024 if cls is StableDiffusionXLPipeline else 512
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(MODEL, torch_dtype=dtype)
        pipe = pipe.to(DEVICE)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        # Trade a little speed for a lot less VRAM so we can coexist with the LLM.
        if DEVICE == "cuda":
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        global _pipe, _orig_sched
        _pipe = pipe
        _orig_sched = pipe.scheduler           # remember default for the "default" sampler
        _ready.set()
        print(f"[imagegen] ready on port {PORT}", flush=True)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        print(f"[imagegen] load failed: {_load_error}", flush=True)


def _generate(prompt: str, steps=None, width=None, height=None, seed=None,
              guidance=None, sampler=None) -> bytes:
    import torch
    n = int(steps) if steps else _default_steps()
    kwargs = {"prompt": prompt, "num_inference_steps": n}
    if guidance is not None and str(guidance) != "":
        kwargs["guidance_scale"] = float(guidance)   # explicit CFG scale wins
    elif _is_turbo():
        kwargs["guidance_scale"] = 0.0               # turbo models are trained guidance-free
    kwargs["width"] = int(width) if width else _gen_size
    kwargs["height"] = int(height) if height else _gen_size
    if seed is not None:
        kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(int(seed))
    with _lock:
        _apply_sampler(sampler)
        image = _pipe(**kwargs).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


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
            data = _generate(prompt, steps=body.get("steps"),
                             width=body.get("width"), height=body.get("height"),
                             seed=body.get("seed"), guidance=body.get("guidance"),
                             sampler=body.get("sampler"))
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            print(f"[imagegen] generate error: {e}", flush=True)
            self.send_response(500); self.end_headers()
            try:
                self.wfile.write(f"{type(e).__name__}: {e}".encode())
            except Exception:
                pass


def main():
    # Load the model in the background so /health answers immediately.
    threading.Thread(target=_load, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[imagegen] http server listening on 127.0.0.1:{PORT}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
