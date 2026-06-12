#!/usr/bin/env python3
"""
Local voice assistant — Gemma 4 12B + Whisper + Piper TTS
Say "Hey Jarvis" (or whatever WAKE_WORD is set to in config.py) to activate.
"""
import re
import sys
import atexit
import signal
import subprocess

# Add project root to path so 'config' and 'modules' are importable
import os
sys.path.insert(0, os.path.dirname(__file__))

import config
from modules.wake_word import listen_for_wake_word
from modules import audio, stt, llm, tts, status, phrases, wake_word, chatlog

NATIVE_AUDIO = config.STT_MODE == "native"

_matches_command = phrases.contains   # spoken text contains the command phrase

# A sentence ends at .!?… (or a newline) followed by whitespace — the lookahead
# avoids splitting "3.5" and waits for the boundary to actually arrive mid-stream.
_SENTENCE_END = re.compile(r"[.!?…]+(?=\s)|\n")
_SOFT_FLUSH_CHARS = 240   # speak a long, punctuation-less run rather than stalling


def _speak_stream(delta_iter) -> str:
    """Consume an LLM text-delta stream, speaking each sentence as soon as it
    completes (while later sentences are still being generated). Prints the text
    live and returns the full response. Never speaks a chunk containing the
    sleep token."""
    speaker = tts.StreamSpeaker()
    buf, full, speaking = "", [], False

    def _emit(chunk: str):
        nonlocal speaking
        chunk = chunk.strip()
        if not chunk or llm.SLEEP_TOKEN in chunk:
            return
        if not speaking:
            status.set_state("speaking")
            speaking = True
        speaker.say(chunk)

    for delta in delta_iter:
        full.append(delta)
        print(delta, end="", flush=True)
        buf += delta
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

    _emit(buf)            # whatever's left after the stream ends
    print(flush=True)
    speaker.close()       # blocks until all audio has finished playing
    return "".join(full).strip()


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
        # 1. Wait for wake word
        status.set_state("idle")
        listen_for_wake_word()
        status.set_state("listening")
        print("\n[Listening...]" if not asleep else "\n[Asleep — listening for wake command...]",
              flush=True)

        # 2. Record speech until silence
        wav_bytes = audio.record_until_silence()
        if not audio.last_stats.get("triggered"):
            print("[No speech detected, going back to idle]")
            continue
        status.set_state("thinking")

        # Asleep: everything heavy is unloaded; a small CPU-only Whisper just
        # listens for the wake command, leaving the GPU free for other work.
        if asleep:
            heard = stt.transcribe_spotter(wav_bytes)
            if _matches_command(heard, config.WAKE_COMMAND):
                asleep = _wake_up()
            else:
                print(f"[Sleep] Heard {heard!r} — ignoring (say '{config.WAKE_COMMAND}' to wake).")
            continue

        # 3. Get a reply and speak it sentence-by-sentence as it streams in.
        user_text = None
        if NATIVE_AUDIO:
            print("Assistant: ", end="", flush=True)
            response = _speak_stream(llm.chat_audio_stream(wav_bytes))
            if llm.SLEEP_TOKEN in response:   # model heard the sleep command
                asleep = _go_to_sleep()
                continue
        else:
            user_text = stt.transcribe(wav_bytes)
            if not user_text:
                print("[No speech detected, going back to idle]")
                continue
            print(f"You: {user_text}")
            if _matches_command(user_text, config.SLEEP_COMMAND):
                chatlog.log("user", user_text)
                asleep = _go_to_sleep()
                continue
            if user_text.lower().strip() in {"clear history", "reset", "forget everything"}:
                llm.reset_history()
                response = "Conversation history cleared."
                print(f"Assistant: {response}\n")
                status.set_state("speaking")
                tts.speak(response)
            else:
                print("Assistant: ", end="", flush=True)
                response = _speak_stream(llm.chat_stream(user_text))

        # 4. Record the turn for the control-panel chat view. In native mode
        # there's no transcript, so transcribe the user's audio now (after the
        # reply, so it adds no latency) with the already-loaded CPU Whisper.
        if config.CHATLOG_ENABLED:
            if user_text is None:
                try:
                    user_text = stt.transcribe_spotter(wav_bytes)
                except Exception:
                    user_text = ""
            chatlog.log("user", user_text)
            chatlog.log("assistant", response)


if __name__ == "__main__":
    main()
