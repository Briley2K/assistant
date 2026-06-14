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

# Wake-time load overrides (set by assistant._adapt_to_gpu before warmup) so a
# reload can pick a quant / offload that fits the VRAM free right now. None = use
# the configured defaults. Only consulted while the model is unloaded.
_gguf_override: str | None = None
_gpu_layers_override: int | None = None


def set_load_override(gguf_path: str | None, gpu_layers: int | None) -> None:
    """Override the GGUF file and GPU-layer count used by the NEXT model load.
    Takes effect only while the model is unloaded — call before warmup()."""
    global _gguf_override, _gpu_layers_override
    _gguf_override = gguf_path
    _gpu_layers_override = gpu_layers


def _active_model_path() -> str:
    return _gguf_override or config.LLM_MODEL_PATH


def _active_gpu_layers() -> int:
    return _gpu_layers_override if _gpu_layers_override is not None else config.LLM_N_GPU_LAYERS

# In native-audio mode there is no transcript to string-match against, so the
# model itself flags the sleep command by replying with this token (see
# chat_audio). assistant.py checks for it and unloads the model.
SLEEP_TOKEN = "[SLEEP]"


def is_sleep(reply: str) -> bool:
    """True if a model reply is the sleep signal — tolerant of the case and
    punctuation the audio model sometimes adds (e.g. '[sleep]', 'Sleep.'), but
    not a normal sentence that merely mentions sleeping."""
    r = (reply or "").strip()
    if SLEEP_TOKEN in r.upper():
        return True
    return re.sub(r"[^a-z]", "", r.lower()) == "sleep"   # bare token, nothing else


# Like the sleep token, the model flags "look at my screen" requests with a
# token (native mode has no transcript to match), which assistant.py turns into
# a screenshot fed back to the model.
VIEW_SCREEN_TOKEN = "[VIEWSCREEN]"


def is_view_screen(reply: str) -> bool:
    """True if the model is asking to look at the user's screen."""
    return VIEW_SCREEN_TOKEN in (reply or "").upper()


# Side-panel visibility — "show the panel" / "hide the panel" — same token
# pattern; assistant.py forwards the command to the overlay.
SHOW_PANEL_TOKEN = "[SHOWPANEL]"
HIDE_PANEL_TOKEN = "[HIDEPANEL]"


def panel_command(reply: str):
    """'show' / 'hide' if the reply is a side-panel command, else None."""
    r = (reply or "").upper()
    if SHOW_PANEL_TOKEN in r:
        return "show"
    if HIDE_PANEL_TOKEN in r:
        return "hide"
    return None


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
        "max_tokens": config.LLM_MAX_TOKENS,
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
        print(f"[LLM] Loading GGUF on {_device_label()}: {_active_model_path()}")
        _llm = Llama(
            model_path=_active_model_path(),
            n_gpu_layers=_active_gpu_layers(),
            n_ctx=config.LLM_N_CTX,
            n_threads=config.LLM_N_THREADS,
            n_threads_batch=config.LLM_N_THREADS,
            verbose=False,
        )
        print("[LLM] GGUF loaded.")
    return _llm


def _call_llamacpp(messages: list[dict]) -> str:
    llm = _get_llamacpp()
    response = llm.create_chat_completion(
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
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
        handler = AudioGemma4Handler(clip_model_path=config.LLM_MMPROJ_PATH, verbose=False)
        _audio_llm = Llama(
            model_path=_active_model_path(),
            chat_handler=handler,
            n_gpu_layers=_active_gpu_layers(),
            n_ctx=config.LLM_N_CTX,
            n_threads=config.LLM_N_THREADS,
            n_threads_batch=config.LLM_N_THREADS,
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
        messages=messages, temperature=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS,
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
    if native and config.SCREEN_VIEW_ENABLED:
        parts.append(
            "If the user asks you to look at, check out, or see their screen, reply "
            f"with exactly {VIEW_SCREEN_TOKEN} and nothing else — a screenshot will "
            "then be given to you to look at."
        )
    if config.SIDE_PANEL_ENABLED:
        parts.append(
            "When you produce something the user would want to read or copy rather "
            "than just hear — source code, a story, a poem, an essay, a long list, "
            "or structured data — put that content inside a triple-backtick fenced "
            "block (``` with a language tag for code, or ```text for prose). It is "
            "shown on a side panel with a Copy button, not read aloud, so keep what "
            "you say out loud to a short spoken introduction of one sentence."
        )
        if native:
            parts.append(
                "If the user asks you to show, open, or bring up the side panel, "
                f"reply with exactly {SHOW_PANEL_TOKEN} and nothing else. If they ask "
                "you to hide, close, or dismiss the side panel, reply with exactly "
                f"{HIDE_PANEL_TOKEN} and nothing else."
            )
    if config.SKILLS_ENABLED:
        from modules import skills
        tp = skills.tools_prompt()
        if tp:
            parts.append(tp)
        if config.FILE_ACCESS_ENABLED:
            from modules.skills import files
            hint = files.locations_hint()
            if hint:
                parts.append(hint)
    return "\n".join(parts)


def _stream_completion(messages: list[dict], audio: bool):
    """Yield content deltas for one generation pass over `messages`."""
    m = _get_audio_llm() if audio else _get_llamacpp()
    kw = dict(messages=messages, temperature=config.LLM_TEMPERATURE,
              max_tokens=config.LLM_MAX_TOKENS, stream=True)
    if not audio:
        kw["stop"] = ["<end_of_turn>"]
    for ch in m.create_chat_completion(**kw):
        delta = ch["choices"][0].get("delta", {}).get("content")
        if delta:
            yield delta


# The local Gemma 4 build wraps its reasoning in a channel block:
#   <|channel>thought\n<channel|> ...reasoning/answer... <turn|>
# The angle-bracket markers are *special tokens*, which llama-cpp strips from the
# streamed text — so the literal "<|channel>" never reaches us and the regexes
# below only ever fire for a backend that surfaces the brackets (e.g. a remote
# API). What DOES leak is the channel *name* between them: an ordinary "thought"
# text token on its own line. Normally the chat handler swallows it, but when the
# model gets stuck looping the header it emits "thought\n" over and over with no
# real answer — that wall of "thought" is what the TTS was reading aloud.
_THOUGHT_RE = re.compile(r"<\|channel>thought.*?<channel\|>", re.S)
_MARK_RE = re.compile(r"<\|[^>]*>|<[^>]*\|>|<(?:end_of_turn|start_of_turn|eos|bos)>")

# Bare reasoning-channel labels that leak as a word alone on a line once the
# surrounding special tokens are stripped. Kept tight to what this model emits so
# real prose ("I gave it some thought.", "Think it over") is never touched — only
# a line that is *exactly* one of these is dropped.
_CHANNEL_LABELS = frozenset({"thought", "thoughts", "think", "thinking"})
# A healthy reply leaks zero such lines; this many means the model is looping on
# the thinking header, so we cut the generation off instead of spinning to
# max_tokens (and never speak the repeats).
_MAX_CHANNEL_LINES = 3


def _is_channel_label(line: str) -> bool:
    return line.strip().lower() in _CHANNEL_LABELS


def _label_prefix(s: str) -> bool:
    """True if `s` (a partial, still-growing line) could become a bare channel
    label — so we hold it back until a space or newline reveals which it is."""
    t = s.strip().lower()
    return bool(t) and not any(c.isspace() for c in t) \
        and any(lbl.startswith(t) for lbl in _CHANNEL_LABELS)


def _strip_stream(raw):
    """Yield speakable text from a raw delta stream. Strips any literal control
    markers and the leaked reasoning-channel header (a bare "thought" line), and
    cuts the stream off if the model is stuck looping that header so the wall of
    "thought" never reaches the speaker. Holds back only a possible partial
    marker or label forming at the tail."""
    buf = ""
    channel_lines = 0
    line_dirty = False          # already emitted text on the current (unfinished) line?
    for delta in raw:
        buf += delta
        buf = _MARK_RE.sub("", _THOUGHT_RE.sub("", buf))
        # A literal channel marker is open but not yet closed (only happens on a
        # bracket-surfacing backend) — hold until the close arrives.
        if "<|channel>" in buf and "<channel|>" not in buf:
            continue
        # Emit completed lines; drop a line that is *only* a leaked channel label
        # (but never one we've already started speaking — that's real prose).
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            if not line_dirty and _is_channel_label(line):
                channel_lines += 1
                if channel_lines >= _MAX_CHANNEL_LINES:
                    return                       # degenerate reasoning loop — stop
                line_dirty = False
                continue
            yield line + "\n"
            line_dirty = False
        # buf is now a partial line. Hold it if it might still grow into a bare
        # label; otherwise emit, keeping back a possible partial marker at the end.
        if not line_dirty and _label_prefix(buf):
            continue
        lt = buf.rfind("<")
        if lt != -1 and ">" not in buf[lt:]:
            emit, buf = buf[:lt], buf[lt:]
        else:
            emit, buf = buf, ""
        if emit:
            yield emit
            line_dirty = True
    buf = _MARK_RE.sub("", _THOUGHT_RE.sub("", buf))
    if buf:                     # trailing text with no closing newline is real prose
        yield buf


# --------------------------------------------------------------------------
# Reasoning models (Nemotron): the chat template opens the assistant turn inside
# a <think> block, so generation begins with chain-of-thought and the real answer
# only starts after the closing </think>. None of the reasoning may be spoken.
# --------------------------------------------------------------------------
_THINK_CLOSE = "</think>"


def _strip_reasoning(raw):
    """Gate a raw delta stream for a reasoning model: swallow everything up to and
    including </think>, then stream the answer that follows. If the tag never
    arrives (the model reasoned past the token budget without answering), speak
    nothing rather than reading the chain-of-thought aloud."""
    buf, thinking = "", True
    for delta in raw:
        if not thinking:
            yield delta
            continue
        buf += delta
        idx = buf.find(_THINK_CLOSE)
        if idx != -1:
            print(f"[LLM] (thought {len(buf[:idx].strip())} chars, not spoken)", flush=True)
            thinking = False
            rest = buf[idx + len(_THINK_CLOSE):]
            buf = ""
            if rest:
                yield rest
    if thinking:
        print("[LLM] reasoning didn't finish within the token budget — no answer "
              "to speak. Raise the model's max_tokens if this recurs.", flush=True)


def _strip_reasoning_text(text: str) -> str:
    """Drop a leading <think>…</think> block from a complete reply (the
    non-streaming Ollama/API path). No closing tag → leave the text untouched."""
    i = text.find(_THINK_CLOSE)
    return text[i + len(_THINK_CLOSE):].lstrip() if i != -1 else text


def _run_with_tools(messages: list[dict], audio: bool):
    """Run a possibly-multi-step tool turn, yielding ONLY the final answer's
    text deltas. Each pass is filtered, then peeked: if it starts with the tool
    token it's executed and fed back; otherwise it's streamed out to be spoken."""
    from modules import skills
    for _step in range(skills.MAX_TOOL_STEPS + 1):
        raw = _stream_completion(messages, audio=audio)
        if config.LLM_REASONING:
            raw = _strip_reasoning(raw)   # drop <think>…</think> before anything else
        clean = _strip_stream(raw)
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


def chat_image_stream(image_bytes: bytes, instruction: str = ""):
    """Tool-aware streaming reply about a screenshot: feeds the image plus an
    instruction (and the prior conversation as context), yielding final-answer
    text deltas. The raw image is dropped from history afterward so later turns
    don't re-encode it — a short text trace is kept in its place."""
    from modules.gemma_audio import image_part
    _get_audio_llm()

    content = [image_part(image_bytes)]
    if instruction:
        content.append({"type": "text", "text": instruction})
    user_idx = len(_history)
    _history.append({"role": "user", "content": content})

    messages = [{"role": "system", "content": _full_system(native=True)}]
    messages.extend(_history[-config.CONTEXT_TURNS * 2:])

    parts: list[str] = []
    for delta in _run_with_tools(messages, audio=True):
        parts.append(delta)
        yield delta

    reply = "".join(parts).strip()
    _history.append({"role": "assistant", "content": reply})
    _history[user_idx] = {"role": "user",
                          "content": (f"[Looked at the user's screen] {instruction}".strip())}


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

    if config.LLM_REASONING:
        reply = _strip_reasoning_text(reply)
    _history.append({"role": "assistant", "content": reply})
    return reply


def _tool_loop_nonstream(messages: list[dict], backend: str) -> str:
    """Non-streaming tool loop for Ollama / remote API backends."""
    from modules import skills
    for _step in range(skills.MAX_TOOL_STEPS + 1):
        reply = _call_ollama(messages) if backend == "ollama" else _call_api(messages)
        if config.LLM_REASONING:
            reply = _strip_reasoning_text(reply)
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


def chat_text_stream(user_message: str):
    """Tool-aware streaming reply to a TYPED message (from the control panel).

    Like chat_stream, but in native-audio mode it generates on the model that is
    already loaded for voice (the audio instance) instead of loading a second
    text-only copy of the model. Shares conversation history with the voice turns
    so typed and spoken messages keep one context. History updates when consumed."""
    backend = _resolve_backend()

    _history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": _full_system(native=False)}]
    messages.extend(_history[-config.CONTEXT_TURNS * 2:])

    parts: list[str] = []
    if backend == "llamacpp":
        # Reuse the already-loaded audio model for text when running native, so a
        # typed message never triggers a second model load.
        audio = config.STT_MODE == "native"
        for delta in _run_with_tools(messages, audio=audio):
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
