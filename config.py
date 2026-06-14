"""
Configuration. Hard paths + defaults live here; user-editable settings are
loaded from settings.json (managed via the control panel) and override the
defaults below. Every module imports the resulting module-level names.
"""
import os
import json
import openwakeword as _oww

_BASE = os.path.dirname(__file__)
SETTINGS_PATH = os.path.join(_BASE, "settings.json")

# Where browser-uploaded custom models land. A model selection of "custom:<file>"
# (LLM .gguf or image .safetensors/.ckpt) resolves to a file in these dirs.
CUSTOM_GGUF_DIR  = os.path.join(_BASE, "models", "custom-gguf")
CUSTOM_IMAGE_DIR = os.path.join(_BASE, "models", "custom-image")
CUSTOM_VIDEO_DIR = os.path.join(_BASE, "models", "custom-video")

# Bundled openWakeWord models live here; a wake-word key maps to "<key>_v0.1".
_OWW_DIR = os.path.join(os.path.dirname(_oww.__file__), "resources", "models")
# Phrases with a pretrained openWakeWord model (instant, near-zero CPU). Any
# other phrase falls back to the trained Whisper listener (modules/wake_word.py).
BUNDLED_WAKE_WORDS = {
    "hey jarvis":  "hey_jarvis",
    "alexa":       "alexa",
    "hey mycroft": "hey_mycroft",
    "hey marvin":  "hey_marvin",
    "timer":       "timer",
}


def _load_settings() -> dict:
    """Read settings.json, tolerating a missing/corrupt file."""
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_S = _load_settings()


# --- LLM models (switchable from the control panel) ---
# A catalog of selectable models. The panel's "Model" dropdown stores a key in
# settings.json ("llm_model"); everything below — GGUF paths, the Ollama tag,
# and whether the model can hear audio natively — derives from the chosen entry.
# Add a new model by adding an entry here; it then appears in the dropdown.
#
#   ollama:  default Ollama tag (the panel's "Ollama model" field overrides it).
#   gguf:    GGUF filename for the local llama-cpp backend.
#   mmproj:  audio/vision projector filename, or None for text-only models. Only
#            a model WITH an mmproj can use native-audio STT; text-only models
#            fall back to Whisper automatically (see STT_MODE below).
#   dirs:    directories searched for the GGUF/mmproj, in order. If nothing is
#            found, paths point at dirs[0] so download_models.sh knows where to
#            put them.
LLM_MODELS = {
    "gemma4-12b": {
        "label":  "Gemma 4 12B (local · native audio)",
        "ollama": "gemma4-12b",
        "gguf":   "gemma-4-12B-it-Q4_K_M.gguf",
        "mmproj": "mmproj-gemma-4-12B-it-BF16.gguf",   # audio+vision
        "hf_repo": "lmstudio-community/gemma-4-12B-it-GGUF",
        "dirs":   [
            os.path.join(_BASE, "models", "gemma"),
            ("/run/media/briley/AE24D19024D15C41/Users/Briley/.lmstudio/models/"
             "lmstudio-community/gemma-4-12B-it-GGUF"),
        ],
    },
    "gemma4-12b-coder": {
        # Community coding fine-tune of Gemma 4 12B (text-only — the GGUF repo has
        # no mmproj, so speech uses Whisper, not native audio). Q4_K_M is the
        # balanced quant; the repo also ships Q2_K / Q6_K / Q8_0 if you swap the
        # filename. A bigger token budget than the base chat model so longer code
        # answers aren't cut off (these go to the side panel, not spoken aloud).
        "label":  "Gemma 4 12B Coder (local · text · coding)",
        "ollama": "gemma4-12b-coder",
        "gguf":   "gemma4-coding-Q4_K_M.gguf",   # default; overridden by selected quant
        "mmproj": None,
        "max_tokens": 1024,
        "hf_repo": "yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF",
        "dirs":   [os.path.join(_BASE, "models", "gemma-coder")],
        # Selectable quants (the panel shows a picker + a VRAM-fit bar). size_gb is
        # the on-disk GGUF size; the loader needs that plus ~2GB headroom on the GPU.
        "quants": [
            {"name": "Q2_K",   "gguf": "gemma4-coding-Q2_K.gguf",   "size_gb": 4.83},
            {"name": "Q4_K_M", "gguf": "gemma4-coding-Q4_K_M.gguf", "size_gb": 7.38, "default": True},
            {"name": "Q6_K",   "gguf": "gemma4-coding-Q6_K.gguf",   "size_gb": 9.79},
            {"name": "Q8_0",   "gguf": "gemma4-coding-Q8_0.gguf",   "size_gb": 12.67},
        ],
    },
    "gemma4-12b-obliterated": {
        # Abliterated (uncensored) Gemma 4 12B — safety guardrails removed via
        # weight surgery, text-only GGUF (no mmproj, so speech uses Whisper, not
        # native audio). Q4_K_M is the balanced quant; the repo also ships
        # Q5_K_M / Q6_K / Q8_0 / BF16.
        "label":  "Gemma 4 12B Obliterated (local · text · uncensored)",
        "ollama": "gemma4-12b-obliterated",
        "gguf":   "Gemma-4-12B-OBLITERATED-Q4_K_M.gguf",   # default; overridden by selected quant
        "mmproj": None,
        "max_tokens": 1024,
        "hf_repo": "OBLITERATUS/Gemma-4-12B-OBLITERATED",
        "dirs":   [os.path.join(_BASE, "models", "gemma-obliterated")],
        # Selectable quants (the panel shows a picker + a VRAM-fit bar). size_gb is
        # the on-disk GGUF size; the loader needs that plus ~2GB headroom on the GPU.
        "quants": [
            {"name": "Q4_K_M", "gguf": "Gemma-4-12B-OBLITERATED-Q4_K_M.gguf", "size_gb": 6.9, "default": True},
            {"name": "Q5_K_M", "gguf": "Gemma-4-12B-OBLITERATED-Q5_K_M.gguf", "size_gb": 8.0},
            {"name": "Q6_K",   "gguf": "Gemma-4-12B-OBLITERATED-Q6_K.gguf",   "size_gb": 9.1},
            {"name": "Q8_0",   "gguf": "Gemma-4-12B-OBLITERATED-Q8_0.gguf",   "size_gb": 12.7},
            {"name": "BF16",   "gguf": "Gemma-4-12B-OBLITERATED-BF16.gguf",   "size_gb": 22.0},
        ],
    },
    "nemotron-3-nano-omni-30b": {
        "label":  "NVIDIA Nemotron 3 Nano Omni 30B-A3B",
        # Multimodal MoE (30B total / ~3B active). Run options:
        #   • Ollama (text-only):  ollama pull nemotron-3-nano:30b
        #   • Local GGUF (llama-cpp), repo:
        #     unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF
        #   • Remote API (OpenRouter, OpenAI-compatible), set the API backend to:
        #     base_url https://openrouter.ai/api/v1
        #     model    nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
        # mmproj is left None: the native-audio handler is Gemma-specific, so this
        # model runs text-only here (Whisper STT) regardless of backend.
        "ollama": "nemotron-3-nano:30b",
        # UD-IQ4_XS (~19.5GB) is the best-quality quant that fits ~22GB of RAM
        # (after Gemma unloads) on this 30GB CPU box. For a 24GB+ GPU use
        # UD-Q4_K_XL (23.9GB); to save RAM, UD-Q2_K_XL (18.5GB). Whatever you set
        # here is the file the Download button fetches and the loader expects.
        "gguf":   "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-IQ4_XS.gguf",
        "mmproj": None,
        # Reasoning model: the chat template opens the turn in a <think> block, so
        # generation starts with chain-of-thought and the real answer only follows
        # the closing </think>. The reasoning is stripped from spoken output (see
        # modules/llm.py). It needs a larger token budget than a chat model so the
        # answer isn't cut off by a long think.
        "reasoning":  True,
        "max_tokens": 2048,
        "hf_repo": "unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF",
        "dirs":   [os.path.join(_BASE, "models", "nemotron")],
    },
}

# Selected model (key into LLM_MODELS); "none" disables the language model
# entirely (image-only mode — use the panel's /image page). Falls back to Gemma
# if set to something unknown.
LLM_MODEL = _S.get("llm_model", "gemma4-12b")
_IS_CUSTOM_LLM = isinstance(LLM_MODEL, str) and LLM_MODEL.startswith("custom:")
if LLM_MODEL != "none" and not _IS_CUSTOM_LLM and LLM_MODEL not in LLM_MODELS:
    LLM_MODEL = "gemma4-12b"
LLM_ENABLED = LLM_MODEL != "none"
# A custom uploaded GGUF has no registry entry (text-only, llama-cpp).
_MODEL = LLM_MODELS.get(LLM_MODEL) if LLM_ENABLED else None


# --- Quant variants ---
# A model may ship several quantizations (Q4_K_M, Q8_0, …); the panel exposes a
# picker and stores the choice per-model in settings ("model_quants": {key: name}).
# Single-file models (no "quants") behave exactly as before.
def model_quants(key: str) -> list:
    """The quant variants for a model key, or [] if it ships a single file."""
    return (LLM_MODELS.get(key) or {}).get("quants") or []


def _default_quant(quants: list) -> dict | None:
    if not quants:
        return None
    return next((q for q in quants if q.get("default")), quants[0])


def selected_quant(key: str) -> dict | None:
    """The quant chosen for a model in settings, else its default, else None for a
    single-file model."""
    quants = model_quants(key)
    if not quants:
        return None
    want = (_S.get("model_quants") or {}).get(key)
    return next((q for q in quants if q["name"] == want), None) or _default_quant(quants)


def gguf_for(key: str) -> str:
    """GGUF filename for a model key, honoring its selected quant."""
    q = selected_quant(key)
    return q["gguf"] if q else LLM_MODELS[key]["gguf"]


def _find_gguf(filename: str, dirs: list[str]) -> str:
    for d in dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return os.path.join(dirs[0], filename)


if not LLM_ENABLED:
    LLM_MODEL_PATH = None
elif _IS_CUSTOM_LLM:
    LLM_MODEL_PATH = os.path.join(CUSTOM_GGUF_DIR, LLM_MODEL.split(":", 1)[1])
else:
    LLM_MODEL_PATH = _find_gguf(gguf_for(LLM_MODEL), _MODEL["dirs"])
LLM_MMPROJ_PATH = (_find_gguf(_MODEL["mmproj"], _MODEL["dirs"])
                   if LLM_ENABLED and _MODEL and _MODEL["mmproj"] else None)

# True if the selected model can hear audio directly (has an mmproj projector).
NATIVE_AUDIO_OK = LLM_MMPROJ_PATH is not None

# Speech input: "native" feeds audio straight to the model (no Whisper);
# "whisper" transcribes first, then sends text. Native needs an audio-capable
# model — text-only models (e.g. Nemotron) fall back to Whisper automatically.
STT_MODE = _S.get("stt_mode", "native")
if STT_MODE == "native" and not NATIVE_AUDIO_OK:
    STT_MODE = "whisper"

# Backend: "auto" (Ollama→llama-cpp), "ollama", "llamacpp", or "api".
LLM_BACKEND  = _S.get("llm_backend", "auto")
OLLAMA_HOST  = _S.get("ollama_host", "http://localhost:11434")
# Blank "ollama_model" → use the selected model's default tag from LLM_MODELS.
OLLAMA_MODEL = (_S.get("ollama_model") or "").strip() or (_MODEL["ollama"] if _MODEL else "")

# Remote OpenAI-compatible API (used when LLM_BACKEND == "api").
_API         = _S.get("api", {})
API_BASE_URL = _API.get("base_url", "")
API_KEY      = _API.get("api_key", "")
API_MODEL    = _API.get("model", "")

LLM_N_CTX        = 4096
LLM_TEMPERATURE  = 0.7
CONTEXT_TURNS    = 10       # max conversation turns to keep in memory

# Reasoning models wrap chain-of-thought in <think>…</think> that must never be
# spoken; LLM_MAX_TOKENS gives them room to think and still answer in one pass.
LLM_REASONING    = bool(_MODEL.get("reasoning", False)) if _MODEL else False
LLM_MAX_TOKENS   = int(_MODEL.get("max_tokens", 512)) if _MODEL else 512

# Where to run the LLM: "auto" (GPU if it has enough free VRAM, else CPU),
# "gpu" (force full GPU offload), or "cpu". "auto" is the default.
LLM_DEVICE = _S.get("llm_device", "auto")

# Partial GPU offload: how many of the model's layers to place on the GPU (the
# rest run on CPU + RAM). For big models that don't fully fit in VRAM — e.g.
# Nemotron 30B on a 16GB card — put as many layers on the GPU as fit and the
# rest on the CPU. Accepts: "" / "auto" (follow LLM_DEVICE), "all", "none", or a
# number like 24. When set to a number/all/none it OVERRIDES LLM_DEVICE.
LLM_GPU_LAYERS = str(_S.get("llm_gpu_layers", "")).strip().lower()

# Cap on the share of GPU *compute* (SMs) Cleo may use, 10-100. 100 = no limit.
# Enforced via NVIDIA MPS (modules/gpu.py); does not affect VRAM. Applied at
# startup; the adaptive throttle below can tighten it live at wake.
GPU_COMPUTE_PERCENT = max(10, min(100, int(_S.get("gpu_compute_percent", 100))))

# --- Adaptive GPU throttle (decided live, when waking) ---
# On wake, sample the GPU's recent load and adapt to whatever ELSE is using the
# card (Cleo is fully unloaded while asleep, so the load measured is other apps'):
#   • cap Cleo's GPU compute to the free headroom via MPS, so she time-shares the
#     card instead of fighting a game/app for it, and
#   • drop to a smaller, locally-present quant that fits the free VRAM.
GPU_ADAPTIVE_THROTTLE    = bool(_S.get("gpu_adaptive_throttle", True))
# Seconds to average GPU utilization over before deciding the cap (the user-facing
# "usage over the past N seconds"). Bigger = steadier reading but a slower wake.
GPU_LOAD_SAMPLE_SECONDS  = max(1.0, float(_S.get("gpu_load_sample_seconds", 5.0)))
# Never throttle below this — a floor so Cleo stays usable even under a heavy game.
GPU_MIN_COMPUTE_PERCENT  = max(5, min(100, int(_S.get("gpu_min_compute_percent", 10))))
# Only treat the GPU as "busy with another app" once its recent average
# utilization reaches this; below it Cleo runs at her normal baseline cap.
GPU_INTERFERENCE_PERCENT = max(0, min(100, int(_S.get("gpu_interference_percent", 50))))

# Headroom on top of the model file(s) for the KV cache (n_ctx) and compute
# buffers — the model needs more VRAM than its on-disk size.
_VRAM_OVERHEAD_MB = 2000


def _gpu_free_mb() -> int | None:
    """Free VRAM (MB) on the largest CUDA device, or None if there's no usable
    NVIDIA GPU. Uses nvidia-smi, which ships with any working CUDA driver."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    vals = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    return max(vals) if vals else None


def gpu_total_mb() -> int | None:
    """Total VRAM (MB) on the largest CUDA device, or None if there's no usable
    NVIDIA GPU. Used by the control panel's VRAM-fit bar."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    vals = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    return max(vals) if vals else None


# VRAM headroom (GB) the panel adds on top of the model file for its fit estimate:
# the KV cache (n_ctx) and compute buffers the model needs beyond its on-disk size.
VRAM_OVERHEAD_GB = _VRAM_OVERHEAD_MB / 1024


def _llm_vram_need_mb() -> int:
    """Estimated VRAM to fit the LLM fully on the GPU: model file(s) + overhead."""
    need = _VRAM_OVERHEAD_MB
    paths = [LLM_MODEL_PATH]
    if STT_MODE == "native" and LLM_MMPROJ_PATH:
        paths.append(LLM_MMPROJ_PATH)     # mmproj is co-loaded for native audio
    for p in paths:
        if p and os.path.exists(p):
            need += os.path.getsize(p) // (1024 * 1024)
    return need


def _present_quants(key: str) -> list[tuple[dict, str]]:
    """(quant, resolved_path) for each quant of `key` whose GGUF is on disk."""
    out = []
    for q in model_quants(key):
        p = _find_gguf(q["gguf"], LLM_MODELS[key]["dirs"])
        if os.path.exists(p):
            out.append((q, p))
    return out


def wake_load_plan(free_mb: int) -> dict | None:
    """Pick how to load the selected LLM into `free_mb` of VRAM. Pass the VRAM
    free *right now*, measured AFTER Whisper/anything else GPU-resident has loaded,
    so `free_mb` already reflects their footprint (no estimating needed). Returns
    a dict, or None when there's nothing to adapt (no language model, or a custom
    single-file GGUF):
        quant       chosen quant name (None for a single-file model)
        gguf        GGUF path to load
        gpu_layers  -1 if the choice fits fully on the GPU, else a conservative
                    partial-offload count so the load can't OOM
        downgraded  True only when the chosen quant is SMALLER than the selected
                    one (i.e. an actual memory-driven downgrade)
    Only ever downgrades — it won't auto-upgrade to a bigger quant on a free GPU.
    """
    if not LLM_ENABLED or _IS_CUSTOM_LLM or not _MODEL:
        return None
    avail = free_mb

    def need(file_mb: int) -> int:
        return file_mb + _VRAM_OVERHEAD_MB

    def layers_for(file_mb: int) -> int:
        # Fits fully → all layers on GPU. Otherwise offload as many as fit, using
        # a conservative per-layer estimate so we under-commit rather than OOM.
        if need(file_mb) <= avail:
            return -1
        n = int((_MODEL or {}).get("n_layers", 48))
        per_layer = max(1, file_mb // max(1, n))
        return max(0, int((avail - _VRAM_OVERHEAD_MB) // per_layer))

    present = _present_quants(LLM_MODEL)
    if not present:
        # Single-file model (or no quant downloaded yet): keep the configured
        # file, only adjust offload so it still loads under tight VRAM.
        if LLM_MODEL_PATH and os.path.exists(LLM_MODEL_PATH):
            file_mb = os.path.getsize(LLM_MODEL_PATH) // (1024 * 1024)
            return {"quant": None, "gguf": LLM_MODEL_PATH,
                    "gpu_layers": layers_for(file_mb), "downgraded": False}
        return None

    def file_mb(q: dict) -> int:
        return int(q["size_gb"] * 1024)

    fits = [(q, p) for q, p in present if need(file_mb(q)) <= avail]
    if fits:
        q, p = max(fits, key=lambda x: x[0]["size_gb"])      # largest that fits
    else:
        q, p = min(present, key=lambda x: x[0]["size_gb"])    # smallest we have

    cur = selected_quant(LLM_MODEL)
    downgraded = bool(cur and q["size_gb"] < cur["size_gb"])
    return {"quant": q["name"], "gguf": p,
            "gpu_layers": layers_for(file_mb(q)), "downgraded": downgraded}


def _resolve_gpu_layers() -> int:
    """Layers to offload to GPU: -1 = all, 0 = none (CPU), N = first N layers.
    An explicit LLM_GPU_LAYERS wins; otherwise honor LLM_DEVICE."""
    if not LLM_ENABLED:
        return 0                            # no language model (image-only mode)
    # Explicit partial-offload override.
    if LLM_GPU_LAYERS not in ("", "auto"):
        if LLM_GPU_LAYERS in ("all", "max", "gpu"):
            return -1
        if LLM_GPU_LAYERS in ("none", "cpu", "0"):
            return 0
        try:
            return max(0, int(LLM_GPU_LAYERS))
        except ValueError:
            pass                            # malformed → fall through to device
    if LLM_DEVICE == "cpu":
        return 0
    if LLM_DEVICE == "gpu":
        return -1
    free = _gpu_free_mb()                  # auto
    if free is None:
        return 0                            # no GPU → CPU + RAM
    return -1 if free >= _llm_vram_need_mb() else 0


# llama-cpp only: number of model layers to put on the GPU (-1 all, 0 CPU, N=partial).
LLM_N_GPU_LAYERS = _resolve_gpu_layers()

# --- CPU compute cap ---
# When a big model runs on the CPU, every physical core pins to 100% the instant
# a question starts processing. That all-core saturation can starve the desktop
# of CPU and hard-lock the machine (it froze hard enough to need a power-off —
# no OOM, nothing logged). The mirror of GPU_COMPUTE_PERCENT: cap the share of
# CPU cores Cleo may use, 10-100. 100 = no limit. Applied on the next restart.
CPU_COMPUTE_PERCENT = max(10, min(100, int(_S.get("cpu_compute_percent", 100))))


def _physical_cores() -> int:
    """Physical CPU cores (what llama-cpp itself defaults n_threads to), so a CPU
    percentage maps to a sensible thread cap instead of oversubscribing SMT
    siblings. Falls back to the logical count if /proc/cpuinfo is unreadable."""
    try:
        seen, phys, core = set(), None, None
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                if key == "physical id":
                    phys = val
                elif key == "core id":
                    core = val
                elif key == "" and phys is not None and core is not None:
                    seen.add((phys, core)); phys = core = None
        if phys is not None and core is not None:
            seen.add((phys, core))
        return len(seen) or (os.cpu_count() or 8)
    except OSError:
        return os.cpu_count() or 8


CPU_PHYSICAL_CORES = _physical_cores()

# CPU threads for the local llama-cpp backend. An explicit "llm_threads" wins
# (advanced override); otherwise the CPU compute cap drives it — N% of the
# physical cores, rounded, leaving the rest free for the desktop. "" / "auto"
# with a 100% cap means the llama-cpp default (all physical cores).
_threads_raw = str(_S.get("llm_threads", "")).strip().lower()
if _threads_raw not in ("", "auto"):
    try:
        LLM_N_THREADS = max(1, int(_threads_raw))
    except ValueError:
        LLM_N_THREADS = None
elif CPU_COMPUTE_PERCENT < 100:
    LLM_N_THREADS = max(1, round(CPU_PHYSICAL_CORES * CPU_COMPUTE_PERCENT / 100))
else:
    LLM_N_THREADS = None

# Cap OpenMP-based native libraries (Whisper on CPU, numpy/BLAS, ctranslate2) to
# the same thread budget so the limit covers *anything* Cleo runs on the CPU, not
# just llama-cpp. Must be set before those libraries are imported — config is
# imported before any of them, so this is early enough to take effect.
if LLM_N_THREADS is not None:
    os.environ["OMP_NUM_THREADS"] = str(LLM_N_THREADS)

SYSTEM_PROMPT = _S.get(
    "system_prompt",
    "You are a helpful, concise voice assistant. Keep responses short.",
)

# --- Sleep mode ---
# Saying "<wake word>, <sleep command>" unloads the LLM (frees GPU memory) but
# keeps wake-word detection running. "<wake word>, <wake command>" reloads it.
SLEEP_COMMAND = _S.get("sleep_command", "go to sleep")
WAKE_COMMAND  = _S.get("wake_command", "wake up")
WAKE_REPLY    = _S.get("wake_reply", "Awake and ready.")
SLEEP_REPLY   = _S.get("sleep_reply", "Going to sleep.")

# Quick spoken acknowledgements: if the user says just the wake word and then
# pauses, Cleo replies with one of these and keeps listening for the request.
WAKE_ACKS = _S.get("wake_acks") or ["Yes?", "What's up?", "Yeah?"]

# Conversational follow-up: after a reply, keep listening for more without
# needing the wake word again. When the user pauses, Cleo asks if there's
# anything else; the conversation only ends (speech input closes until the next
# wake word) when the user declines or stays quiet.
FOLLOWUP_ENABLED = bool(_S.get("followup_mode", True))
FOLLOWUP_PROMPT  = _S.get("followup_prompt", "Is there anything else I can help with?")
FOLLOWUP_SIGNOFF = _S.get("followup_signoff", "Okay, I'll be here if you need me.")

# Screen viewing: let the user say "look at my screen" and Cleo captures a
# monitor (via the ScreenCast portal) and looks at it. Native audio only — the
# model needs the audio+vision mmproj. One-time GNOME share permission applies.
SCREEN_VIEW_ENABLED = bool(_S.get("screen_view", True)) and STT_MODE == "native"
SCREEN_PICK_PROMPT  = _S.get("screen_pick_prompt", "Which screen would you like me to look at?")

# Side panel: when a reply contains read/copy content (code, a story, a poem, a
# list), Cleo shows it in a side overlay with a Copy button instead of reading it
# aloud. The overlay (overlay.py) polls logs/panel.json, written by modules/panel.
SIDE_PANEL_ENABLED = bool(_S.get("side_panel", True))
PANEL_PATH = os.path.join(_BASE, "logs", "panel.json")

# Whisper model used only while asleep to spot the wake command. Runs on the
# CPU so the GPU stays completely free for other work during sleep.
SLEEP_WHISPER_MODEL = _S.get("sleep_whisper_model", "base")

# --- Status orb overlay (Siri-style, top-right of screen) ---
OVERLAY_ENABLED = bool(_S.get("overlay", True))
OVERLAY_PORT    = int(_S.get("overlay_port", 5006))

# --- Conversation log (shown live in the control panel) ---
CHATLOG_ENABLED = bool(_S.get("chatlog", True))
CHATLOG_PATH    = os.path.join(_BASE, "logs", "chat.jsonl")

# --- Skills / tools ---
SKILLS_ENABLED  = bool(_S.get("skills", True))
SKILLS_DISABLED = set(_S.get("skills_disabled", []))   # skill names to turn off

# --- File access (read-only) ---
# Cleo can read files on this machine, but ONLY inside the folders you grant
# (FILE_ACCESS_ROOTS) and never anything matching the blacklist. Both are managed
# from the control panel. Disabled by default — and even when enabled, nothing is
# readable until you add at least one allowed folder. Reads are size-capped
# (FILE_ACCESS_MAX_KB) and binary files are refused, so a stray read can't flood
# the model or dump a blob. See modules/skills/files.py for the enforcement.
#
# Blacklist entries are matched against each candidate's resolved (realpath) path:
# an entry containing * ? or [ is treated as a glob (tested against the full path
# AND the bare filename); anything else is a literal path or parent folder to
# block. The defaults below shield common secrets even inside an allowed folder.
DEFAULT_FILE_BLACKLIST = [
    "*.env", ".env", "*.pem", "*.key", "id_rsa*", "id_ed25519*", "*.kdbx",
    "*/.ssh/*", "*/.aws/*", "*/.gnupg/*", "*secret*", "*password*",
    "*credentials*", "*.sqlite", "*.sqlite3",
    os.path.join(_BASE, "settings.json"),   # holds Cleo's own API key
]


def _expand_paths(items) -> list[str]:
    """Normalize a list of user-entered paths/patterns: drop blanks, expand ~ and
    $VARS. Globs pass through unchanged (expanduser/expandvars only touch ~ / $)."""
    out = []
    for p in items or []:
        p = str(p).strip()
        if p:
            out.append(os.path.expanduser(os.path.expandvars(p)))
    return out


FILE_ACCESS_ENABLED   = bool(_S.get("file_access", False))
FILE_ACCESS_ROOTS     = _expand_paths(_S.get("file_access_roots", []))
FILE_ACCESS_BLACKLIST = _expand_paths(_S.get("file_access_blacklist", DEFAULT_FILE_BLACKLIST))
FILE_ACCESS_MAX_KB    = max(1, int(_S.get("file_access_max_kb", 256)))

# Weather skill default ("home") location. Set a place name in the control
# panel; lat/lon are optional and take precedence if both are given.
WEATHER_PLACE = _S.get("weather_place", "")
WEATHER_LAT   = _S.get("weather_lat", None)
WEATHER_LON   = _S.get("weather_lon", None)
WEATHER_UNITS = _S.get("weather_units", "fahrenheit")   # or "celsius"

# --- Wake word ---
# Free-text phrase. Old settings stored e.g. "hey_jarvis", so underscores are
# treated as spaces. Phrases in BUNDLED_WAKE_WORDS use the pretrained model;
# anything else is trained once at startup (variants file below).
WAKE_PHRASE = " ".join(str(_S.get("wake_word", "hey cleo")).lower().replace("_", " ").split()) \
              or "hey cleo"

_WAKE_KEY = BUNDLED_WAKE_WORDS.get(WAKE_PHRASE)
WAKE_WORD_MODEL = os.path.join(_OWW_DIR, f"{_WAKE_KEY}_v0.1.onnx") if _WAKE_KEY else None
WAKE_WORD_LABEL = f"{_WAKE_KEY}_v0.1" if _WAKE_KEY else None
WAKE_WORD_THRESHOLD = float(_S.get("wake_word_threshold", 0.5))

# Custom-phrase training output: accepted transcription variants of the phrase.
WAKE_VARIANTS_PATH = os.path.join(
    _BASE, "models", "wake", WAKE_PHRASE.replace(" ", "_") + ".json")

# --- Speech-to-text (Whisper) ---
WHISPER_MODEL    = _S.get("whisper_model", "small")
WHISPER_DEVICE   = "cuda"    # runs on GPU (ctranslate2 ships cuDNN); auto-falls back to CPU
WHISPER_COMPUTE  = "float16" # "int8" on CPU fallback
WHISPER_LANGUAGE = "en"      # None for auto-detect

# --- Text-to-speech ---
# "kokoro" (natural) | "piper" (fast/robotic) | "neutts" (NeuTTS Air, voice-cloning,
# runs in the isolated neutts_test venv via a helper — see modules/neutts_tts.py).
TTS_ENGINE = _S.get("tts_engine", "kokoro")

# NeuTTS Air (on-device voice cloning). The model runs in neutts_test/.venv
# (Python 3.12 + torch); the main app talks to a localhost helper it spawns.
_NEUTTS_DIR     = os.path.join(_BASE, "neutts_test")
NEUTTS_PYTHON   = os.path.join(_NEUTTS_DIR, ".venv", "bin", "python")
NEUTTS_SERVER   = os.path.join(_NEUTTS_DIR, "neutts_server.py")
_NEUTTS_SAMPLES = os.path.join(_NEUTTS_DIR, "samples")
NEUTTS_PORT     = int(_S.get("neutts_port", 5008))
# q4 (fast) or q8 (higher quality) GGUF backbone.
NEUTTS_BACKBONE = _S.get("neutts_backbone", "neuphonic/neutts-air-q4-gguf")
# Reference voice to clone: a name in neutts_test/samples (jo=female EN default,
# dave=male EN, …) or an absolute path to a custom <name>.wav (+ <name>.txt).
NEUTTS_VOICE    = _S.get("neutts_voice", "jo")


def neutts_ref_paths() -> tuple[str, str]:
    """(wav, txt) for the configured NeuTTS reference voice. A bare name maps to
    neutts_test/samples/<name>.{wav,txt}; an absolute .wav path is used as-is
    (its transcript is the sibling .txt)."""
    v = NEUTTS_VOICE
    if os.path.isabs(v):
        wav = v if v.endswith(".wav") else v + ".wav"
        return wav, os.path.splitext(wav)[0] + ".txt"
    return (os.path.join(_NEUTTS_SAMPLES, f"{v}.wav"),
            os.path.join(_NEUTTS_SAMPLES, f"{v}.txt"))


def neutts_voices() -> list[str]:
    """Bundled reference-voice names available in neutts_test/samples."""
    try:
        return sorted(f[:-4] for f in os.listdir(_NEUTTS_SAMPLES)
                      if f.endswith(".wav")
                      and os.path.exists(os.path.join(_NEUTTS_SAMPLES, f[:-4] + ".txt")))
    except OSError:
        return ["jo"]


# Chatterbox (Resemble AI) — high-quality voice cloning that runs on the GPU. Like
# NeuTTS it lives in its own venv (chatterbox/.venv, CUDA torch + chatterbox-tts)
# and is driven via a localhost helper (see modules/chatterbox_tts.py). It clones
# from an audio prompt alone — no transcript — so it reuses the NeuTTS sample clips.
_CB_DIR           = os.path.join(_BASE, "chatterbox")
CHATTERBOX_PYTHON = os.path.join(_CB_DIR, ".venv", "bin", "python")
CHATTERBOX_SERVER = os.path.join(_CB_DIR, "chatterbox_server.py")
CHATTERBOX_PORT   = int(_S.get("chatterbox_port", 5009))
CHATTERBOX_DEVICE = _S.get("chatterbox_device", "cuda")          # "cuda" or "cpu"
# Reference voice: a name in neutts_test/samples or an absolute path to a .wav.
# Defaults to whatever NeuTTS voice is set so switching engines keeps the voice.
CHATTERBOX_VOICE  = _S.get("chatterbox_voice") or _S.get("neutts_voice", "sophon_2")
# Expressiveness / pacing / sampling knobs passed to generate().
CHATTERBOX_EXAGGERATION = float(_S.get("chatterbox_exaggeration", 0.5))
CHATTERBOX_CFG          = float(_S.get("chatterbox_cfg", 0.5))
CHATTERBOX_TEMPERATURE  = float(_S.get("chatterbox_temperature", 0.8))
# Approx VRAM (GB) the Chatterbox model holds on the GPU — shown in the control
# panel's VRAM bar, stacked on top of the LLM, so you can see if both fit.
CHATTERBOX_VRAM_GB      = float(_S.get("chatterbox_vram_gb", 4.75))


def chatterbox_ref_wav() -> str:
    """Reference .wav for the configured Chatterbox voice (reuses the NeuTTS
    sample clips; a bare name maps into neutts_test/samples)."""
    v = CHATTERBOX_VOICE
    if os.path.isabs(v):
        return v if v.endswith(".wav") else v + ".wav"
    return os.path.join(_NEUTTS_SAMPLES, f"{v}.wav")


# Image generation (Stable Diffusion via diffusers). Same isolation pattern as
# NeuTTS/Chatterbox: it lives in its own venv (imagegen/.venv, CUDA torch +
# diffusers) and is driven via a localhost helper (modules/imagegen.py +
# imagegen/imagegen_server.py). The model can call it through the generate_image
# skill; generated PNGs land in logs/images and render inline in the chat window.
_IMAGEGEN_DIR    = os.path.join(_BASE, "imagegen")
IMAGEGEN_PYTHON  = os.path.join(_IMAGEGEN_DIR, ".venv", "bin", "python")
IMAGEGEN_SERVER  = os.path.join(_IMAGEGEN_DIR, "imagegen_server.py")
IMAGEGEN_PORT    = int(_S.get("imagegen_port", 5010))

# Image-model catalog (diffusers model ids). Like LLM_MODELS, picking an entry in
# the panel is all that's needed; the helper downloads the weights from
# HuggingFace on first generation. "vram_gb" is a rough fp16 working-set estimate
# (for the fit hint); "size" is the model's native resolution.
IMAGEGEN_MODELS = {
    "sd-turbo": {
        "label": "SD-Turbo — fast, ~4 GB VRAM (512px)",
        "repo":  "stabilityai/sd-turbo", "vram_gb": 4.0, "size": 512, "dl_gb": 5.0,
    },
    "sdxl-turbo": {
        "label": "SDXL-Turbo — sharper, ~8 GB VRAM (512px)",
        "repo":  "stabilityai/sdxl-turbo", "vram_gb": 8.0, "size": 512, "dl_gb": 7.0,
    },
    "sdxl": {
        "label": "SDXL 1.0 — best quality, ~10 GB VRAM (1024px)",
        "repo":  "stabilityai/stable-diffusion-xl-base-1.0", "vram_gb": 10.0, "size": 1024, "dl_gb": 7.0,
    },
    # Flux is text-to-IMAGE (not video) — it lives here, not in the video section.
    # FLUX.2 [klein] is the small/open variant; verify the repo id if it 404s.
    "flux2-klein": {
        "label": "FLUX.2 Klein — high quality (1024px, heavy)",
        "repo":  "black-forest-labs/FLUX.2-klein", "vram_gb": 14.0, "size": 1024, "dl_gb": 24.0,
    },
}

# Selected image model (key into IMAGEGEN_MODELS); "none" disables generation.
IMAGE_MODEL = _S.get("image_model", "none")
_IS_CUSTOM_IMG = isinstance(IMAGE_MODEL, str) and IMAGE_MODEL.startswith("custom:")
if IMAGE_MODEL != "none" and not _IS_CUSTOM_IMG and IMAGE_MODEL not in IMAGEGEN_MODELS:
    IMAGE_MODEL = "none"
IMAGEGEN_ENABLED  = IMAGE_MODEL != "none"
if not IMAGEGEN_ENABLED:
    IMAGEGEN_REPO, IMAGEGEN_IMG_SIZE = None, 512
elif _IS_CUSTOM_IMG:
    # A custom uploaded checkpoint: REPO is its local file path; the helper loads
    # it with from_single_file. Resolution guessed from the name (SDXL → 1024px).
    _fname = IMAGE_MODEL.split(":", 1)[1]
    IMAGEGEN_REPO = os.path.join(CUSTOM_IMAGE_DIR, _fname)
    IMAGEGEN_IMG_SIZE = 1024 if "xl" in _fname.lower() else 512
else:
    _IMG = IMAGEGEN_MODELS[IMAGE_MODEL]
    IMAGEGEN_REPO = _IMG["repo"]          # diffusers id the helper loads
    IMAGEGEN_IMG_SIZE = _IMG["size"]
IMAGEGEN_DEVICE   = _S.get("image_device", "cuda")     # "cuda" or "cpu"
IMAGEGEN_STEPS    = int(_S.get("image_steps", 0))      # 0 = model-appropriate default
# Where generated PNGs are written (and served from by the control panel).
IMAGEGEN_OUT_DIR  = os.path.join(_BASE, "logs", "images")


# Video generation (diffusers text-to-video). Same isolation pattern as image gen:
# its own venv (video/.venv, CUDA torch + diffusers + imageio-ffmpeg), driven via
# a localhost helper (modules/videogen.py + video/video_server.py). Selecting a
# model in the panel is all that's needed; weights download on first generation.
# These repos are new/fast-moving — verify/adjust the repo ids if one 404s, or
# just upload your own checkpoint.
_VIDEO_DIR       = os.path.join(_BASE, "video")
VIDEOGEN_PYTHON  = os.path.join(_VIDEO_DIR, ".venv", "bin", "python")
VIDEOGEN_SERVER  = os.path.join(_VIDEO_DIR, "video_server.py")
VIDEOGEN_PORT    = int(_S.get("videogen_port", 5011))

VIDEOGEN_MODELS = {
    "wan-1.3b": {
        "label": "Wan 2.1 T2V 1.3B — lightest, fits 16 GB (480p)",
        "repo":  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "vram_gb": 12.0, "size": 480, "frames": 81, "fps": 16, "dl_gb": 17.0,
    },
    "ltx-2": {
        "label": "LTX-2 (LTX-Video) — fast, longer clips (704px)",
        "repo":  "Lightricks/LTX-Video",
        "vram_gb": 12.0, "size": 704, "frames": 97, "fps": 24, "dl_gb": 18.0,
    },
    "wan-14b": {
        "label": "Wan 2.1 T2V 14B — best quality (needs 24 GB+)",
        "repo":  "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "vram_gb": 40.0, "size": 720, "frames": 81, "fps": 16, "dl_gb": 33.0,
    },
}

VIDEO_MODEL = _S.get("video_model", "none")
_IS_CUSTOM_VID = isinstance(VIDEO_MODEL, str) and VIDEO_MODEL.startswith("custom:")
if VIDEO_MODEL != "none" and not _IS_CUSTOM_VID and VIDEO_MODEL not in VIDEOGEN_MODELS:
    VIDEO_MODEL = "none"
VIDEOGEN_ENABLED = VIDEO_MODEL != "none"
if not VIDEOGEN_ENABLED:
    VIDEOGEN_REPO, VIDEOGEN_SIZE, VIDEOGEN_FRAMES, VIDEOGEN_FPS = None, 480, 81, 16
elif _IS_CUSTOM_VID:
    _vfname = VIDEO_MODEL.split(":", 1)[1]
    VIDEOGEN_REPO = os.path.join(CUSTOM_VIDEO_DIR, _vfname)
    VIDEOGEN_SIZE, VIDEOGEN_FRAMES, VIDEOGEN_FPS = 480, 81, 16
else:
    _VID = VIDEOGEN_MODELS[VIDEO_MODEL]
    VIDEOGEN_REPO   = _VID["repo"]
    VIDEOGEN_SIZE   = _VID["size"]
    VIDEOGEN_FRAMES = _VID["frames"]
    VIDEOGEN_FPS    = _VID["fps"]
VIDEOGEN_DEVICE  = _S.get("video_device", "cuda")     # "cuda" or "cpu"
VIDEOGEN_STEPS   = int(_S.get("video_steps", 0))      # 0 = model-appropriate default
VIDEOGEN_OUT_DIR = os.path.join(_BASE, "logs", "videos")


# Piper
PIPER_VOICE   = os.path.join(_BASE, "models", "piper", "en_US-lessac-medium.onnx")
PIPER_SPEAKER = 0

# Kokoro
KOKORO_MODEL  = os.path.join(_BASE, "models", "kokoro", "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(_BASE, "models", "kokoro", "voices-v1.0.npz")
KOKORO_VOICE  = _S.get("kokoro_voice", "af_heart")
KOKORO_SPEED  = float(_S.get("kokoro_speed", 1.0))

# --- Audio ---
SAMPLE_RATE  = 16000
CHANNELS     = 1
MIC_DEVICE   = None   # None = follow Ubuntu's selected mic (PipeWire default source).
                      # Override with a pw-record --target node name/serial if needed.
SILENCE_MS   = 800    # ms of silence before ending recording

# Cleo's speaking volume, 0.0-1.0 (1.0 = full scale). Applied per playback via
# pw-play --volume, independent of the system volume. Read live from settings on
# each utterance (see modules/audio), so a change takes effect without a restart.
TTS_VOLUME   = max(0.0, min(1.0, float(_S.get("tts_volume", 1.0))))


def tts_volume() -> float:
    """Current speaking volume, re-read from settings.json so the control-panel
    slider takes effect on the next utterance without restarting the assistant.
    Falls back to the value loaded at startup if the file can't be read."""
    try:
        return max(0.0, min(1.0, float(_load_settings().get("tts_volume", TTS_VOLUME))))
    except (TypeError, ValueError):
        return TTS_VOLUME
