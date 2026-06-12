#!/usr/bin/env python3
"""
Local voice assistant — Gemma 4 12B + Whisper + Piper TTS
Say "Hey Jarvis" (or whatever WAKE_WORD is set to in config.py) to activate.
"""
import re
import sys
import time
import atexit
import random
import signal
import subprocess

# Add project root to path so 'config' and 'modules' are importable
import os
sys.path.insert(0, os.path.dirname(__file__))

import config
from modules.wake_word import listen_for_wake_word
from modules import audio, stt, llm, tts, status, phrases, wake_word, chatlog, live, gpu, screen, panel

NATIVE_AUDIO = config.STT_MODE == "native"

# Wake/sleep commands are matched fuzzily (phrases.matches) so a misheard
# "cleo" doesn't drop the command. Sleep uses a strict threshold (a false match
# mid-chat is costly); waking uses a lenient one (missing it is worse).
_WAKE_MATCH_THRESHOLD  = 0.6
_SLEEP_MATCH_THRESHOLD = 0.8


def _spotter_text(wav_bytes: bytes) -> str:
    """Best-effort CPU-Whisper transcript of the user's audio ('' on error)."""
    try:
        return stt.transcribe_spotter(wav_bytes)
    except Exception:
        return ""


# Ways a user signals they're done, ending the follow-up conversation. Multi-word
# phrases are matched anywhere in the utterance; the short ones must be the whole
# reply (so "no" closes, but "no, tell me about dogs" stays in the conversation).
_DECLINE_PHRASES = [
    "no thank you", "no thanks", "nothing else", "nothing thanks", "that is all",
    "that's all", "that is it", "that's it", "that is everything", "that's everything",
    "that will be all", "that'll be all", "i'm all set", "im all set", "all set",
    "i'm good", "im good", "i'm done", "im done", "we're done", "we are done",
    "i'm fine thanks", "that's everything thanks", "no i'm good", "no im good",
]
_DECLINE_WHOLE = {"no", "nope", "nah", "done", "goodbye", "bye", "no thanks"}


def _user_declines(text: str) -> bool:
    """True if the user indicated they need nothing more."""
    norm = phrases.normalize(text)
    if not norm:
        return False
    if norm in _DECLINE_WHOLE:
        return True
    return any(phrases.contains(text, p) for p in _DECLINE_PHRASES)


# Spoken ways of asking Cleo to look at the screen — a fallback for when the
# model doesn't emit the [VIEWSCREEN] token on its own.
_SCREEN_TRIGGERS = [
    "look at my screen", "look at the screen", "check out my screen", "check my screen",
    "see my screen", "view my screen", "take a look at my screen", "look at my monitor",
    "look at my display", "can you see my screen", "watch my screen",
]


def _wants_screen(text: str) -> bool:
    return any(phrases.matches(text, p, 0.78) for p in _SCREEN_TRIGGERS)


# Spoken side-panel commands. "show the panel" and "hide the panel" differ by a
# single word, so phrase-level fuzzy matching can't tell them apart — instead we
# require the word "panel" (fuzzy, for mishearings) plus an intent verb.
_PANEL_SHOW_VERBS = {"show", "open", "display", "view", "bring", "pop", "unhide"}
_PANEL_HIDE_VERBS = {"hide", "close", "dismiss", "remove", "away", "rid", "collapse"}


def _panel_intent(text: str):
    """'show' / 'hide' if the user asked to show or hide the side panel."""
    words = phrases.normalize(text).split()
    if not any(phrases.similarity(w, "panel") >= 0.8 for w in words):
        return None
    ws = set(words)
    if ws & _PANEL_HIDE_VERBS:
        return "hide"
    if ws & _PANEL_SHOW_VERBS:
        return "show"
    return None


def _process_utterance(wav_bytes: bytes):
    """Generate and speak a reply to one utterance, streaming it live. Returns
    (user_text, response, metrics, action) where action is None, "sleep", or
    "view" (look at the screen). user_text may be None if no transcript yet."""
    if NATIVE_AUDIO:
        print("Assistant: ", end="", flush=True)
        response, metrics = _speak_stream(llm.chat_audio_stream(wav_bytes))
        if llm.is_sleep(response):            # model flagged the sleep command
            return "", response, metrics, "sleep"
        if config.SCREEN_VIEW_ENABLED and llm.is_view_screen(response):
            return _spotter_text(wav_bytes), response, metrics, "view"
        cmd = config.SIDE_PANEL_ENABLED and llm.panel_command(response)
        if cmd:
            return _spotter_text(wav_bytes), response, metrics, f"panel_{cmd}"
        # The audio model doesn't reliably emit those tokens, so also check what
        # was actually said (we transcribe for the chat log regardless).
        user_text = _spotter_text(wav_bytes)
        if phrases.matches(user_text, config.SLEEP_COMMAND, _SLEEP_MATCH_THRESHOLD):
            return user_text, response, metrics, "sleep"
        if config.SCREEN_VIEW_ENABLED and _wants_screen(user_text):
            return user_text, response, metrics, "view"
        cmd = config.SIDE_PANEL_ENABLED and _panel_intent(user_text)
        if cmd:
            return user_text, response, metrics, f"panel_{cmd}"
        return user_text, response, metrics, None

    user_text = stt.transcribe(wav_bytes)
    if not user_text:
        return None, None, {}, None
    print(f"You: {user_text}")
    live.set_user(user_text)              # whisper mode: transcript is ready now
    if phrases.matches(user_text, config.SLEEP_COMMAND, _SLEEP_MATCH_THRESHOLD):
        return user_text, None, {}, "sleep"
    cmd = config.SIDE_PANEL_ENABLED and _panel_intent(user_text)
    if cmd:                               # no LLM round-trip needed for this
        return user_text, None, {}, f"panel_{cmd}"
    if user_text.lower().strip() in {"clear history", "reset", "forget everything"}:
        llm.reset_history()
        response = "Conversation history cleared."
        print(f"Assistant: {response}\n")
        status.set_state("speaking")
        tts.speak(response)
        return user_text, response, {}, None
    print("Assistant: ", end="", flush=True)
    response, metrics = _speak_stream(llm.chat_stream(user_text))
    return user_text, response, metrics, None


def _spoken_screen_list(labels: list[str]) -> str:
    """Render monitor labels as natural speech, e.g. 'left, center, or right screen'."""
    if len(labels) == 1:
        return f"{labels[0]} screen"
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]} screen"
    return ", ".join(labels[:-1]) + f", or {labels[-1]} screen"


def _view_screen(request_text: str) -> None:
    """Ask which monitor (when several are shared), capture it, and let the model
    look at it and respond — using the conversation so far as context. Speaks a
    short apology and returns quietly if capture isn't possible."""
    mons = screen.shared_monitors()
    if not mons:
        status.set_state("speaking")
        tts.speak("I can't see your screen yet. Screen sharing needs to be set up first.")
        return

    if len(mons) == 1:
        chosen = mons[0]
    else:
        status.set_state("speaking")
        tts.speak(f"{config.SCREEN_PICK_PROMPT} I can see your "
                  f"{_spoken_screen_list([m['label'] for m in mons])}.")
        status.set_state("listening")
        live.begin("listening")
        print("\n[Which screen?]", flush=True)
        ans = audio.record_until_silence()
        if not audio.last_stats.get("triggered"):
            return
        chosen = screen.resolve(_spotter_text(ans), mons)
        if not chosen:
            status.set_state("speaking")
            tts.speak("Sorry, I wasn't sure which screen you meant.")
            return

    status.set_state("thinking")
    live.phase("thinking")
    print(f"[Capturing {chosen['label']} screen...]", flush=True)
    png, err = screen.capture(chosen)
    if err or not png:
        status.set_state("speaking")
        tts.speak("Sorry, I couldn't capture your screen.")
        return

    instruction = (
        f"This is the user's {chosen['label']} screen. If they already asked you to "
        "do something with it, do that now using what you see. Otherwise, briefly say "
        "what's on the screen and ask what they'd like you to do."
    )
    print("Assistant: ", end="", flush=True)
    response, metrics = _speak_stream(llm.chat_image_stream(png, instruction))
    if config.CHATLOG_ENABLED and response:
        chatlog.log("assistant", response, metrics)


def _run_conversation(initial_wav: bytes | None) -> bool:
    """Handle a back-and-forth conversation after waking. Replies to the first
    utterance, then keeps listening for follow-ups without the wake word: while
    the user keeps talking it keeps answering; on a pause it asks if there's
    anything else; it ends when the user declines or stays quiet a second time.
    Returns True if the user asked Cleo to go to sleep."""
    pending = initial_wav
    prompted = False   # have we already asked "anything else?" since the last reply

    while True:
        if pending is not None:
            wav_bytes, pending = pending, None
        else:
            status.set_state("listening")
            live.begin("listening")
            print("\n[Listening for follow-up...]", flush=True)
            wav_bytes = audio.record_until_silence()
            if not audio.last_stats.get("triggered"):
                if prompted:                          # asked already, still silence
                    status.set_state("speaking")
                    tts.speak(config.FOLLOWUP_SIGNOFF)
                    return False
                status.set_state("speaking")          # nudge once, then keep listening
                tts.speak(config.FOLLOWUP_PROMPT)
                prompted = True
                continue

        status.set_state("thinking")
        live.phase("thinking")
        prompted = False

        user_text, response, metrics, action = _process_utterance(wav_bytes)

        if action == "sleep":
            if user_text and config.CHATLOG_ENABLED:
                chatlog.log("user", user_text)
            _go_to_sleep()
            live.clear()
            return True

        if action == "view":      # "look at my screen" — capture and look
            if user_text and config.CHATLOG_ENABLED:
                chatlog.log("user", user_text)
            _view_screen(user_text or "")
            live.clear()
            if not config.FOLLOWUP_ENABLED:
                return False
            continue              # stay in the conversation after looking

        if action in ("panel_show", "panel_hide"):   # side-panel visibility
            cmd = action.split("_")[1]
            if user_text and config.CHATLOG_ENABLED:
                chatlog.log("user", user_text)
            status.panel(cmd)
            print(f"[Panel] {cmd}", flush=True)
            status.set_state("speaking")
            tts.speak("Okay.")
            live.clear()
            if not config.FOLLOWUP_ENABLED:
                return False
            continue              # stay in the conversation

        if response is None:        # whisper mode heard nothing intelligible
            live.clear()
            continue                # stay in the conversation and keep listening

        if config.CHATLOG_ENABLED:
            if user_text is None:
                user_text = _spotter_text(wav_bytes)
            chatlog.log("user", user_text)
            if response:
                chatlog.log("assistant", response, metrics)
        live.clear()

        # User said they're done — the reply just spoken serves as the goodbye.
        if _user_declines(user_text or ""):
            return False
        # Single-turn mode: stop after one exchange (no follow-up listening).
        if not config.FOLLOWUP_ENABLED:
            return False

# A sentence ends at .!?… (or a newline) followed by whitespace — the lookahead
# avoids splitting "3.5" and waits for the boundary to actually arrive mid-stream.
_SENTENCE_END = re.compile(r"[.!?…]+(?=\s)|\n")
_SOFT_FLUSH_CHARS = 240   # speak a long, punctuation-less run rather than stalling

_FENCE_RE = re.compile(r"```([^\n`]*)\n?(.*?)```", re.S)
# Fence languages that mean "prose", not source code — affects the panel title.
_PROSE_LANGS = {"", "text", "txt", "plain", "prose", "story", "md", "markdown"}


class _FenceSplitter:
    """Incrementally separates spoken text from fenced (```...```) blocks across a
    token stream, so code/stories shown on the side panel aren't read aloud. feed()
    returns only the text outside fences; the full reply is captured separately."""

    def __init__(self):
        self.in_fence = False
        self._tail = ""        # held-back trailing backticks (a fence may straddle deltas)

    def feed(self, text: str) -> str:
        s, self._tail = self._tail + text, ""
        out, i = [], 0
        while True:
            idx = s.find("```", i)
            if idx == -1:
                rest = s[i:]
                hold = 0
                while hold < 2 and hold < len(rest) and rest[-1 - hold] == "`":
                    hold += 1   # keep up to 2 trailing backticks in case ``` is forming
                emit = rest[:len(rest) - hold] if hold else rest
                self._tail = rest[len(rest) - hold:] if hold else ""
                if emit and not self.in_fence:
                    out.append(emit)
                break
            seg = s[i:idx]
            if seg and not self.in_fence:
                out.append(seg)
            self.in_fence = not self.in_fence
            i = idx + 3
        return "".join(out)

    def flush(self) -> str:
        t, self._tail = self._tail, ""
        return t if (t and not self.in_fence) else ""


def _extract_artifacts(text: str):
    """Pull fenced blocks out of a reply: returns (title, kind, content) or None."""
    blocks = [(m.group(1).strip(), m.group(2).rstrip("\n")) for m in _FENCE_RE.finditer(text)]
    blocks = [(lang, body) for lang, body in blocks if body.strip()]
    if not blocks:
        return None
    content = "\n\n".join(body for _lang, body in blocks)
    lang = blocks[0][0]
    code = bool(lang) and lang.lower() not in _PROSE_LANGS
    title = f"{lang} code" if code else (lang.capitalize() if lang else "Text")
    return title, ("code" if code else "text"), content


def _speak_stream(delta_iter):
    """Consume an LLM text-delta stream, speaking each sentence as soon as it
    completes (while later sentences are still being generated). Prints the text
    live and returns (full_response, metrics). Never speaks a chunk containing
    the sleep token.

    `metrics` carries this turn's timing measurements (see _stream_metrics):
    time-to-first-token, generation tokens/sec, and first-token-to-speech delay."""
    speaker = tts.StreamSpeaker()
    buf, full, speaking = "", [], False
    fence = _FenceSplitter()     # keep fenced code/stories out of the spoken audio

    t_start = time.monotonic()   # the moment we begin pulling from the model
    t_first = None               # first token's arrival
    tokens = 0

    def _emit(chunk: str):
        nonlocal speaking
        chunk = chunk.strip()
        if (not chunk or llm.SLEEP_TOKEN in chunk or llm.VIEW_SCREEN_TOKEN in chunk
                or llm.SHOW_PANEL_TOKEN in chunk or llm.HIDE_PANEL_TOKEN in chunk):
            return
        if not speaking:
            status.set_state("speaking")
            live.phase("speaking")
            speaking = True
        speaker.say(chunk)

    for delta in delta_iter:
        if t_first is None:
            t_first = time.monotonic()
        tokens += 1
        full.append(delta)
        live.assistant_delta(delta)   # stream the FULL reply (code included) to the chat view
        print(delta, end="", flush=True)
        buf += fence.feed(delta)      # but only speak text outside fenced blocks
        while True:
            m = _SENTENCE_END.search(buf)
            if m:
                cut = m.end()
            elif len(buf) > _SOFT_FLUSH_CHARS and " " in buf:
                cut = buf.rfind(" ")   # no sentence end yet — break at a word
            else:
                break
            _emit(buf[:cut])
            buf = buf[cut:]

    t_gen_end = time.monotonic()   # generation done (before audio drains)
    _emit(buf + fence.flush())     # whatever's left after the stream ends

    response = "".join(full).strip()
    artifact = _extract_artifacts(response) if config.SIDE_PANEL_ENABLED else None
    if artifact:
        if not speaking:           # reply was only the artifact — say something
            _emit("I've put it on your screen.")
        panel.show(*artifact)

    live.flush()          # make sure the final reply text reaches the view
    print(flush=True)
    speaker.close()       # blocks until all audio has finished playing

    metrics = _stream_metrics(t_start, t_first, t_gen_end, tokens,
                              speaker.first_play_ts)
    return response, metrics


def _stream_metrics(t_start, t_first, t_gen_end, tokens, first_play_ts):
    """Build a metrics dict from the timestamps gathered during streaming.
    Streamed deltas stand in for tokens — close enough for a tokens/sec read."""
    metrics = {}
    if t_first is None:        # empty stream — nothing to measure
        return metrics
    metrics["ttft_ms"] = round((t_first - t_start) * 1000)
    metrics["tokens"] = tokens
    gen = t_gen_end - t_first
    if gen > 0:
        metrics["tok_per_sec"] = round(tokens / gen, 1)
    if first_play_ts is not None:
        metrics["voice_ms"] = round((first_play_ts - t_first) * 1000)
    return metrics


def _go_to_sleep() -> bool:
    print(f"[Sleep] Unloading LLM — say '{config.WAKE_COMMAND}' after the wake word to wake.")
    status.set_state("speaking")
    chatlog.log("assistant", config.SLEEP_REPLY)
    tts.speak(config.SLEEP_REPLY)
    llm.unload()
    stt.unload()           # whisper mode: frees the GPU Whisper too
    stt.warmup_spotter()   # CPU-only wake-command listener — GPU is now fully free
    return True


def _wake_up() -> bool:
    print("[Sleep] Wake command heard — reloading model...")
    llm.warmup()
    if not NATIVE_AUDIO:
        stt._get_model()
    if config.WAKE_WORD_MODEL:
        stt.unload_spotter()   # custom wake phrases keep using the spotter
    print(f"Assistant: {config.WAKE_REPLY}\n")
    status.set_state("speaking")
    chatlog.log("assistant", config.WAKE_REPLY)
    tts.speak(config.WAKE_REPLY)
    return False


def _clean_env() -> dict:
    """Environment for the overlay, minus snap-injected library paths (set when
    launched from e.g. the VSCode snap's terminal) that break system GTK."""
    env = {}
    for k, v in os.environ.items():
        if k.startswith("SNAP") or k.endswith("_VSCODE_SNAP_ORIG"):
            continue
        if k == "PATH":
            env[k] = ":".join(p for p in v.split(":") if "/snap/" not in p)
        elif "/snap/" in v:
            orig = os.environ.get(f"{k}_VSCODE_SNAP_ORIG", "")
            if orig:
                env[k] = orig
        else:
            env[k] = v
    return env


def _start_overlay():
    """Launch the status orb (overlay.py) as a separate process, if possible."""
    if not config.OVERLAY_ENABLED:
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return  # headless session — nothing to draw on
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlay.py")
    env = dict(_clean_env(), OVERLAY_PORT=str(config.OVERLAY_PORT))
    try:
        # System python3: the overlay needs GTK (python3-gi), which the venv lacks.
        # stdin=PIPE doubles as a liveness tether — the overlay exits on hangup,
        # so it never outlives the assistant even if we're killed outright.
        proc = subprocess.Popen(
            ["/usr/bin/python3", script], env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        print(f"[Overlay] Not started: {e}")
        return
    atexit.register(proc.terminate)
    print("[Overlay] Status orb started (top-right).")


def _check_model_file():
    # The local GGUF is only needed when llama-cpp actually loads it: native
    # audio always does, and so does the explicit "llamacpp" text backend.
    # Ollama / remote-API models (e.g. Nemotron) have no local file to check.
    needs_gguf = NATIVE_AUDIO or config.LLM_BACKEND == "llamacpp"
    if needs_gguf and not os.path.exists(config.LLM_MODEL_PATH):
        print(f"ERROR: model not found at:\n  {config.LLM_MODEL_PATH}")
        print("Run 'bash setup.sh' (or 'bash download_models.sh') to download it,")
        print("or switch the backend to Ollama / remote API in the control panel.")
        sys.exit(1)

    if NATIVE_AUDIO and not os.path.exists(config.LLM_MMPROJ_PATH):
        print(f"ERROR: audio mmproj not found at:\n  {config.LLM_MMPROJ_PATH}")
        print("Run 'bash setup.sh' (or 'bash download_models.sh') to download it.")
        sys.exit(1)

    if not os.path.exists(config.PIPER_VOICE):
        print(f"ERROR: Piper voice not found at:\n  {config.PIPER_VOICE}")
        print("Run 'bash setup.sh' (or 'bash download_models.sh') to download it.")
        sys.exit(1)


def main():
    _check_model_file()

    # Apply the GPU compute cap BEFORE any model touches CUDA, or it won't bind.
    gpu.apply_compute_limit()

    print(f"Loading models (mode: {'native audio' if NATIVE_AUDIO else 'whisper'})...")

    # Pre-load only what this mode needs.
    if not NATIVE_AUDIO:
        stt._get_model()
    llm.warmup()
    tts.warmup()

    # Custom wake phrases are trained once (new phrase → ~30s), then cached.
    if wake_word.ensure_ready():
        confirmation = f"Your new wake word, {config.WAKE_PHRASE}, is ready."
        print(f"Assistant: {confirmation}")
        status.set_state("speaking")
        tts.speak(confirmation)
        status.set_state("idle")

    print("\n=== Assistant ready ===")
    print("Say the wake word to begin. Press Ctrl+C to quit.\n")

    def _quit(sig, frame):
        print("\nGoodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _quit)
    signal.signal(signal.SIGTERM, _quit)   # systemd stop — sys.exit runs atexit

    _start_overlay()
    asleep = False

    while True:
        # 1. Wait for the wake word, then record the first utterance.
        status.set_state("idle")
        live.clear()
        listen_for_wake_word()
        status.set_state("listening")
        live.begin("listening")
        print("\n[Listening...]" if not asleep else "\n[Asleep — listening for wake command...]",
              flush=True)

        # The request usually follows the wake word in one breath, so listen with
        # a short onset window first. If the user said ONLY the wake word, give a
        # quick spoken ack ("Yes?") and listen again with the full window.
        wav_bytes = audio.record_until_silence(max_wait_ms=1800)
        if not audio.last_stats.get("triggered"):
            ack = random.choice(config.WAKE_ACKS)
            print(f"Assistant: {ack}", flush=True)
            status.set_state("speaking")
            tts.speak(ack)
            status.set_state("listening")
            live.begin("listening")
            wav_bytes = audio.record_until_silence()
            if not audio.last_stats.get("triggered"):
                print("[No speech detected, going back to idle]")
                continue
        status.set_state("thinking")
        live.phase("thinking")

        # 2. Asleep: everything heavy is unloaded; a small CPU-only Whisper just
        # listens for the wake command, leaving the GPU free for other work.
        if asleep:
            heard = _spotter_text(wav_bytes)
            if phrases.matches(heard, config.WAKE_COMMAND, _WAKE_MATCH_THRESHOLD):
                asleep = _wake_up()
                # Just woke — listen fresh for the actual request (no wake word).
                asleep = _run_conversation(None)
            else:
                print(f"[Sleep] Heard {heard!r} — ignoring (say '{config.WAKE_COMMAND}' to wake).")
            continue

        # 3. Reply to the first utterance, then stay in a follow-up conversation
        # until the user is done. _run_conversation handles the chat log, live
        # view, sleep, and "anything else?" prompting.
        asleep = _run_conversation(wav_bytes)


if __name__ == "__main__":
    main()
