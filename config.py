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
WAKE_WORDS = ["hey_jarvis", "alexa", "hey_mycroft", "hey_marvin", "timer"]


def _load_settings() -> dict:
    """Read settings.json, tolerating a missing/corrupt file."""
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_S = _load_settings()


# --- LLM ---
_GGUF_DIR = ("/run/media/briley/AE24D19024D15C41/Users/Briley/.lmstudio/models/"
             "lmstudio-community/gemma-4-12B-it-GGUF")
GEMMA_MODEL_PATH  = os.path.join(_GGUF_DIR, "gemma-4-12B-it-Q4_K_M.gguf")
GEMMA_MMPROJ_PATH = os.path.join(_GGUF_DIR, "mmproj-gemma-4-12B-it-BF16.gguf")  # audio+vision

# Speech input: "native" feeds audio straight to Gemma 4 (no Whisper);
# "whisper" transcribes first, then sends text.
STT_MODE = _S.get("stt_mode", "native")

# Backend: "auto" (Ollama→llama-cpp), "ollama", "llamacpp", or "api".
LLM_BACKEND  = _S.get("llm_backend", "auto")
OLLAMA_HOST  = _S.get("ollama_host", "http://localhost:11434")
OLLAMA_MODEL = _S.get("ollama_model", "gemma4-12b")

# Remote OpenAI-compatible API (used when LLM_BACKEND == "api").
_API         = _S.get("api", {})
API_BASE_URL = _API.get("base_url", "")
API_KEY      = _API.get("api_key", "")
API_MODEL    = _API.get("model", "")

LLM_N_GPU_LAYERS = -1        # llama-cpp only: 0 = CPU. (GPU build blocked by gcc15/CUDA13 here)
LLM_N_CTX        = 4096
LLM_TEMPERATURE  = 0.7
CONTEXT_TURNS    = 10       # max conversation turns to keep in memory

SYSTEM_PROMPT = _S.get(
    "system_prompt",
    "You are a helpful, concise voice assistant. Keep responses short.",
)

# --- Wake word ---
_WAKE = _S.get("wake_word", "hey_jarvis")
if _WAKE not in WAKE_WORDS:
    _WAKE = "hey_jarvis"
WAKE_WORD_MODEL     = os.path.join(_OWW_DIR, f"{_WAKE}_v0.1.onnx")
WAKE_WORD_LABEL     = f"{_WAKE}_v0.1"
WAKE_WORD_THRESHOLD = float(_S.get("wake_word_threshold", 0.5))

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
