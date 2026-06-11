"""
LLM wrapper with two interchangeable backends:

  - "ollama":   talks to a local Ollama server (GPU accelerated). Preferred.
  - "llamacpp": loads the GGUF directly via llama-cpp-python (CPU here).

config.LLM_BACKEND selects one, or "auto" probes Ollama and falls back to
llama-cpp. Conversation history handling is shared across both.
"""
import json
import urllib.request
import urllib.error

import config

_history: list[dict] = []
_backend: str | None = None      # resolved backend: "ollama" | "llamacpp"
_llm = None                       # lazy llama_cpp.Llama instance (text)
_audio_llm = None                 # lazy llama_cpp.Llama instance (native audio + mmproj)


# --------------------------------------------------------------------------
# Backend resolution
# --------------------------------------------------------------------------
def _ollama_available() -> bool:
    """True if the Ollama server is up and the configured model is present."""
    try:
        req = urllib.request.Request(f"{config.OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            tags = json.loads(resp.read())
        names = {m["name"] for m in tags.get("models", [])}
        # match "gemma4-12b" against "gemma4-12b:latest" etc.
        return any(n == config.OLLAMA_MODEL or n.startswith(config.OLLAMA_MODEL + ":")
                   for n in names)
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return False


def _resolve_backend() -> str:
    global _backend
    if _backend is not None:
        return _backend

    choice = config.LLM_BACKEND
    if choice in ("ollama", "llamacpp", "api"):
        _backend = choice
    else:  # auto
        _backend = "ollama" if _ollama_available() else "llamacpp"

    _cpp = "llama-cpp (GPU)" if config.LLM_N_GPU_LAYERS != 0 else "llama-cpp (CPU)"
    labels = {"ollama": "Ollama (GPU)", "llamacpp": _cpp, "api": "Remote API"}
    print(f"[LLM] Backend: {labels[_backend]}")
    return _backend


# --------------------------------------------------------------------------
# Ollama backend
# --------------------------------------------------------------------------
def _call_ollama(messages: list[dict]) -> str:
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_ctx": config.LLM_N_CTX,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["message"]["content"].strip()


# --------------------------------------------------------------------------
# Remote API backend (OpenAI-compatible /chat/completions)
# --------------------------------------------------------------------------
def _call_api(messages: list[dict]) -> str:
    base = config.API_BASE_URL.rstrip("/")
    payload = {
        "model": config.API_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,
    }
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if config.API_KEY:
        headers["Authorization"] = f"Bearer {config.API_KEY}"
    req = urllib.request.Request(f"{base}/chat/completions", data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------
# llama-cpp backend
# --------------------------------------------------------------------------
def _get_llamacpp():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        print(f"[LLM] Loading GGUF: {config.GEMMA_MODEL_PATH}")
        _llm = Llama(
            model_path=config.GEMMA_MODEL_PATH,
            n_gpu_layers=config.LLM_N_GPU_LAYERS,
            n_ctx=config.LLM_N_CTX,
            verbose=False,
        )
        print("[LLM] GGUF loaded.")
    return _llm


def _call_llamacpp(messages: list[dict]) -> str:
    llm = _get_llamacpp()
    response = llm.create_chat_completion(
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=512,
        stop=["<end_of_turn>"],
    )
    return response["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------
# Native audio backend (Gemma 4 hears speech directly via mtmd)
# --------------------------------------------------------------------------
def _get_audio_llm():
    global _audio_llm
    if _audio_llm is None:
        from llama_cpp import Llama
        from modules.gemma_audio import AudioGemma4Handler
        print("[LLM] Loading Gemma 4 + audio mmproj (native voice) on GPU...")
        handler = AudioGemma4Handler(clip_model_path=config.GEMMA_MMPROJ_PATH, verbose=False)
        _audio_llm = Llama(
            model_path=config.GEMMA_MODEL_PATH,
            chat_handler=handler,
            n_gpu_layers=config.LLM_N_GPU_LAYERS,
            n_ctx=config.LLM_N_CTX,
            verbose=False,
        )
        print("[LLM] Native-audio model loaded.")
    return _audio_llm


def chat_audio(wav_bytes: bytes) -> str:
    """Send raw speech audio to Gemma 4 and get a reply (no Whisper)."""
    from modules.gemma_audio import audio_part
    llm = _get_audio_llm()

    _history.append({"role": "user", "content": [audio_part(wav_bytes)]})

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    messages.extend(_history[-config.CONTEXT_TURNS * 2:])

    out = llm.create_chat_completion(
        messages=messages, temperature=config.LLM_TEMPERATURE, max_tokens=512,
    )
    reply = out["choices"][0]["message"]["content"].strip()
    _history.append({"role": "assistant", "content": reply})
    return reply


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def warmup() -> None:
    """Preload whichever model the configured mode needs."""
    if config.STT_MODE == "native":
        _get_audio_llm()
    elif _resolve_backend() == "llamacpp":
        _get_llamacpp()


def chat(user_message: str) -> str:
    """Send a message and get a response, maintaining conversation history."""
    backend = _resolve_backend()

    _history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    turn_limit = config.CONTEXT_TURNS * 2   # user + assistant pairs
    messages.extend(_history[-turn_limit:])

    if backend == "ollama":
        reply = _call_ollama(messages)
    elif backend == "api":
        reply = _call_api(messages)
    else:
        reply = _call_llamacpp(messages)

    _history.append({"role": "assistant", "content": reply})
    return reply


def reset_history() -> None:
    """Clear conversation history."""
    _history.clear()
