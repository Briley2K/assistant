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
        "dirs":   [
            os.path.join(_BASE, "models", "gemma"),
            ("/run/media/briley/AE24D19024D15C41/Users/Briley/.lmstudio/models/"
             "lmstudio-community/gemma-4-12B-it-GGUF"),
        ],
    },
    "nemotron-nano-30b": {
        "label":  "NVIDIA Nemotron Nano 3 30B (Ollama · text-only)",
        # NOTE: verify this tag matches what `ollama list` shows. Pull it first:
        #       ollama pull nemotron-3-nano-30b
        # Adjust the tag here, or override it in the control panel's Ollama field.
        "ollama": "nemotron-3-nano-30b",
        "gguf":   "nemotron-3-nano-30b-Q4_K_M.gguf",
        "mmproj": None,            # text-only — no native audio, uses Whisper
        "dirs":   [os.path.join(_BASE, "models", "nemotron")],
    },
}

# Selected model (key into LLM_MODELS); falls back to Gemma if unknown.
LLM_MODEL = _S.get("llm_model", "gemma4-12b")
if LLM_MODEL not in LLM_MODELS:
    LLM_MODEL = "gemma4-12b"
_MODEL = LLM_MODELS[LLM_MODEL]


def _find_gguf(filename: str, dirs: list[str]) -> str:
    for d in dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return os.path.join(dirs[0], filename)


LLM_MODEL_PATH  = _find_gguf(_MODEL["gguf"], _MODEL["dirs"])
LLM_MMPROJ_PATH = _find_gguf(_MODEL["mmproj"], _MODEL["dirs"]) if _MODEL["mmproj"] else None

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
OLLAMA_MODEL = (_S.get("ollama_model") or "").strip() or _MODEL["ollama"]

# Remote OpenAI-compatible API (used when LLM_BACKEND == "api").
_API         = _S.get("api", {})
API_BASE_URL = _API.get("base_url", "")
API_KEY      = _API.get("api_key", "")
API_MODEL    = _API.get("model", "")

LLM_N_CTX        = 4096
LLM_TEMPERATURE  = 0.7
CONTEXT_TURNS    = 10       # max conversation turns to keep in memory

# Where to run the LLM: "auto" (GPU if it has enough free VRAM, else CPU),
# "gpu" (force full GPU offload), or "cpu". "auto" is the default.
LLM_DEVICE = _S.get("llm_device", "auto")

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


def _llm_vram_need_mb() -> int:
    """Estimated VRAM to fit the LLM fully on the GPU: model file(s) + overhead."""
    need = _VRAM_OVERHEAD_MB
    paths = [LLM_MODEL_PATH]
    if STT_MODE == "native" and LLM_MMPROJ_PATH:
        paths.append(LLM_MMPROJ_PATH)     # mmproj is co-loaded for native audio
    for p in paths:
        if os.path.exists(p):
            need += os.path.getsize(p) // (1024 * 1024)
    return need


def _resolve_gpu_layers() -> int:
    """Layers to offload to GPU: -1 = all, 0 = none (CPU). Honors LLM_DEVICE."""
    if LLM_DEVICE == "cpu":
        return 0
    if LLM_DEVICE == "gpu":
        return -1
    free = _gpu_free_mb()                  # auto
    if free is None:
        return 0                            # no GPU → CPU + RAM
    return -1 if free >= _llm_vram_need_mb() else 0


# llama-cpp only: number of model layers to put on the GPU (-1 all, 0 CPU).
LLM_N_GPU_LAYERS = _resolve_gpu_layers()

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
TTS_ENGINE = _S.get("tts_engine", "kokoro")   # "kokoro" (natural) | "piper" (fast/robotic)

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
