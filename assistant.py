#!/usr/bin/env python3
"""
Local voice assistant — Gemma 4 12B + Whisper + Piper TTS
Say "Hey Jarvis" (or whatever WAKE_WORD is set to in config.py) to activate.
"""
import sys
import signal

# Add project root to path so 'config' and 'modules' are importable
import os
sys.path.insert(0, os.path.dirname(__file__))

import config
from modules.wake_word import listen_for_wake_word
from modules import audio, stt, llm, tts

NATIVE_AUDIO = config.STT_MODE == "native"


def _check_model_file():
    if not os.path.exists(config.GEMMA_MODEL_PATH):
        print(f"ERROR: Gemma model not found at:\n  {config.GEMMA_MODEL_PATH}")
        print("Make sure the Windows drive is mounted and the path in config.py is correct.")
        sys.exit(1)

    if NATIVE_AUDIO and not os.path.exists(config.GEMMA_MMPROJ_PATH):
        print(f"ERROR: audio mmproj not found at:\n  {config.GEMMA_MMPROJ_PATH}")
        sys.exit(1)

    if not os.path.exists(config.PIPER_VOICE):
        print(f"ERROR: Piper voice not found at:\n  {config.PIPER_VOICE}")
        print("Run ./download_models.sh first.")
        sys.exit(1)


def main():
    _check_model_file()

    print(f"Loading models (mode: {'native audio' if NATIVE_AUDIO else 'whisper'})...")

    # Pre-load only what this mode needs.
    if not NATIVE_AUDIO:
        stt._get_model()
    llm.warmup()
    tts.warmup()

    print("\n=== Assistant ready ===")
    print("Say the wake word to begin. Press Ctrl+C to quit.\n")

    def _quit(sig, frame):
        print("\nGoodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _quit)

    while True:
        # 1. Wait for wake word
        listen_for_wake_word()
        print("\n[Listening...]", flush=True)

        # 2. Record speech until silence
        wav_bytes = audio.record_until_silence()
        if not audio.last_stats.get("triggered"):
            print("[No speech detected, going back to sleep]")
            continue

        # 3. Get a reply — either feed audio straight to Gemma, or transcribe first.
        if NATIVE_AUDIO:
            response = llm.chat_audio(wav_bytes)
        else:
            user_text = stt.transcribe(wav_bytes)
            if not user_text:
                print("[No speech detected, going back to sleep]")
                continue
            print(f"You: {user_text}")
            if user_text.lower().strip() in {"clear history", "reset", "forget everything"}:
                llm.reset_history()
                response = "Conversation history cleared."
            else:
                response = llm.chat(user_text)

        print(f"Assistant: {response}\n")

        # 4. Speak response
        tts.speak(response)


if __name__ == "__main__":
    main()
