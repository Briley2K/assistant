"""
LLM wrapper with two interchangeable backends:

  - "ollama":   talks to a local Ollama server (GPU accelerated). Preferred.
  - "llamacpp": loads the GGUF directly via llama-cpp-python (CPU here).

config.LLM_BACKEND selects one, or "auto" probes Ollama and falls back to
llama-cpp. Conversation history handling is shared across both.
"""
import re
import json
import urllib.request
import urllib.error

import config

_history: list[dict] = []
_backend: str | None = None      # resolved backend: "ollama" | "llamacpp"
_llm = None                       # lazy llama_cpp.Llama instance (text)
_audio_llm = None                 # lazy llama_cpp.Llama instance (native audio + mmproj)

# In native-audio mode there is no transcript to string-match against, so the
# model itself flags the sleep command by replying with this token (see
# chat_audio). assistant.py checks for it and unloads the model.
SLEEP_TOKEN = "[SLEEP]"


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

    _cpp = f"llama-cpp ({_device_label()})"
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
def _device_label() -> str:
    """How llama-cpp will actually run: GPU only if layers are offloaded AND the
    installed build supports it (a CPU-only build silently ignores n_gpu_layers)."""
    if config.LLM_N_GPU_LAYERS == 0:
        return "CPU"
    try:
        from llama_cpp import llama_cpp as _C
        if not _C.llama_supports_gpu_offload():
            return "CPU (this llama-cpp build has no GPU support — rebuild with CUDA)"
    except Exception:
        pass
    return "GPU"


def _get_llamacpp():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        print(f"[LLM] Loading GGUF on {_device_label()}: {config.GEMMA_MODEL_PATH}")
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
        print(f"[LLM] Loading Gemma 4 + audio mmproj (native voice) on {_device_label()}...")
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

    system = config.SYSTEM_PROMPT
    if config.SLEEP_COMMAND:
        system += (
            f'\nIf the user says "{config.SLEEP_COMMAND}" or otherwise clearly tells you '
            f"to go to sleep, reply with exactly {SLEEP_TOKEN} and nothing else."
        )

    messages = [{"role": "system", "content": system}]
    messages.extend(_history[-config.CONTEXT_TURNS * 2:])

    out = llm.create_chat_completion(
        messages=messages, temperature=config.LLM_TEMPERATURE, max_tokens=512,
    )
    reply = out["choices"][0]["message"]["content"].strip()
    _history.append({"role": "assistant", "content": reply})
    if SLEEP_TOKEN in reply:
        del _history[-2:]   # keep the sleep exchange out of future context
    return reply


# --------------------------------------------------------------------------
# Tool-calling orchestration (shared by the streaming generators)
# --------------------------------------------------------------------------
def _full_system(native: bool) -> str:
    """System prompt = persona + (sleep instruction if native) + tool catalog."""
    parts = [config.SYSTEM_PROMPT]
    if native and config.SLEEP_COMMAND:
        parts.append(
            f'If the user says "{config.SLEEP_COMMAND}" or otherwise clearly tells you '
            f"to go to sleep, reply with exactly {SLEEP_TOKEN} and nothing else."
        )
    if config.SKILLS_ENABLED:
        from modules import skills
        tp = skills.tools_prompt()
        if tp:
            parts.append(tp)
    return "\n".join(parts)


def _stream_completion(messages: list[dict], audio: bool):
    """Yield content deltas for one generation pass over `messages`."""
    m = _get_audio_llm() if audio else _get_llamacpp()
    kw = dict(messages=messages, temperature=config.LLM_TEMPERATURE,
              max_tokens=512, stream=True)
    if not audio:
        kw["stop"] = ["<end_of_turn>"]
    for ch in m.create_chat_completion(**kw):
        delta = ch["choices"][0].get("delta", {}).get("content")
        if delta:
            yield delta


# Gemma 4 leaks control tokens into the *streamed* output (the non-streaming
# parser would strip them): an often-empty thinking block, channel markers, and
# turn delimiters. We filter them on the fly so nothing is spoken or parsed wrong.
_THOUGHT_RE = re.compile(r"<\|channel>thought.*?<channel\|>", re.S)
_MARK_RE = re.compile(r"<\|[^>]*>|<[^>]*\|>|<(?:end_of_turn|start_of_turn|eos|bos)>")


def _strip_stream(raw):
    """Yield cleaned text from a raw delta stream: removes Gemma thinking blocks
    and control tokens, holding back only a possible partial token at the tail."""
    buf = ""
    for delta in raw:
        buf += delta
        buf = _THOUGHT_RE.sub("", buf)            # drop complete thinking blocks
        # A channel marker is open but not yet closed — could be a thinking
        # block forming, so hold everything until the close arrives.
        if "<|channel>" in buf and "<channel|>" not in buf:
            continue
        buf = _MARK_RE.sub("", buf)
        lt = buf.rfind("<")                       # possible partial marker at the end
        if lt != -1 and ">" not in buf[lt:]:
            emit, buf = buf[:lt], buf[lt:]
        else:
            emit, buf = buf, ""
        if emit:
            yield emit
    buf = _MARK_RE.sub("", _THOUGHT_RE.sub("", buf))
    if buf:
        yield buf


def _run_with_tools(messages: list[dict], audio: bool):
    """Run a possibly-multi-step tool turn, yielding ONLY the final answer's
    text deltas. Each pass is filtered, then peeked: if it starts with the tool
    token it's executed and fed back; otherwise it's streamed out to be spoken."""
    from modules import skills
    for _step in range(skills.MAX_TOOL_STEPS + 1):
        clean = _strip_stream(_stream_completion(messages, audio=audio))
        head, is_tool = "", None
        for chunk in clean:
            head += chunk
            if is_tool is None and len(head.lstrip()) >= len(skills.TOOL_TOKEN):
                is_tool = head.lstrip().startswith(skills.TOOL_TOKEN)
                if not is_tool:                   # ordinary answer — speak it
                    yield head
                    yield from clean
                    return
            # while is_tool is True we keep buffering the call, speaking nothing
        if not is_tool:                           # reply shorter than the token
            if head.strip():
                yield head
            return

        call = skills.parse_tool_call(head)
        if not call:                              # malformed call — surface it
            yield head
            return
        name, args = call
        print(f"\n[Skill] {name}({args})", flush=True)
        result = skills.dispatch(name, args)
        print(f"[Skill] -> {result}", flush=True)
        messages.append({"role": "assistant", "content": head.strip()})
        messages.append({"role": "user", "content": (
            f"[[TOOL_RESULT]] {name}: {json.dumps(result)}\n"
            "Answer the user conversationally and briefly using this result. "
            "Don't mention the tool or the raw data. Only call another tool if "
            "truly necessary.")})
    yield "Sorry, I couldn't complete that."


def chat_audio_stream(wav_bytes: bytes):
    """Tool-aware streaming reply to speech audio. Yields final-answer text
    deltas (tool calls handled silently). History updates when consumed."""
    from modules.gemma_audio import audio_part
    _get_audio_llm()

    _history.append({"role": "user", "content": [audio_part(wav_bytes)]})

    messages = [{"role": "system", "content": _full_system(native=True)}]
    messages.extend(_history[-config.CONTEXT_TURNS * 2:])

    parts: list[str] = []
    for delta in _run_with_tools(messages, audio=True):
        parts.append(delta)
        yield delta

    reply = "".join(parts).strip()
    _history.append({"role": "assistant", "content": reply})
    if SLEEP_TOKEN in reply:
        del _history[-2:]


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


def _tool_loop_nonstream(messages: list[dict], backend: str) -> str:
    """Non-streaming tool loop for Ollama / remote API backends."""
    from modules import skills
    for _step in range(skills.MAX_TOOL_STEPS + 1):
        reply = _call_ollama(messages) if backend == "ollama" else _call_api(messages)
        call = skills.parse_tool_call(reply)
        if not call:
            return reply
        name, args = call
        print(f"[Skill] {name}({args})", flush=True)
        result = skills.dispatch(name, args)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": (
            f"[[TOOL_RESULT]] {name}: {json.dumps(result)}\n"
            "Answer the user conversationally and briefly using this result.")})
    return "Sorry, I couldn't complete that."


def chat_stream(user_message: str):
    """Tool-aware streaming reply to text. llama-cpp streams token-by-token;
    Ollama/API emit the full reply at once. History updates when consumed."""
    backend = _resolve_backend()

    _history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": _full_system(native=False)}]
    messages.extend(_history[-config.CONTEXT_TURNS * 2:])

    parts: list[str] = []
    if backend == "llamacpp":
        for delta in _run_with_tools(messages, audio=False):
            parts.append(delta)
            yield delta
    else:
        text = _tool_loop_nonstream(messages, backend)
        parts.append(text)
        yield text

    _history.append({"role": "assistant", "content": "".join(parts).strip()})


def reset_history() -> None:
    """Clear conversation history."""
    _history.clear()


def unload() -> None:
    """Free the main LLM (GPU/CPU memory). warmup() reloads it later."""
    global _llm, _audio_llm
    import gc

    if _audio_llm is not None:
        handler = getattr(_audio_llm, "chat_handler", None)
        _audio_llm.close()
        _audio_llm = None
        # The mtmd (mmproj) context is only freed via the handler's exit stack —
        # there is no __del__, so close it explicitly or it leaks GPU memory.
        if handler is not None and hasattr(handler, "_exit_stack"):
            handler._exit_stack.close()

    if _llm is not None:
        _llm.close()
        _llm = None

    if _backend == "ollama":
        # Ask the Ollama server to evict the model from VRAM immediately.
        try:
            payload = json.dumps({"model": config.OLLAMA_MODEL, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                f"{config.OLLAMA_HOST}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10).read()
        except (urllib.error.URLError, OSError):
            pass

    gc.collect()
    print("[LLM] Model unloaded — sleeping.")
