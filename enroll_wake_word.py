#!/usr/bin/env python3
"""
Interactive wake-word voice enrollment.

Teaches the recognizer YOUR pronunciation of the wake phrase: Cleo asks you to
say it a few times, transcribes each one, and merges how your voice actually gets
heard into the phrase's accepted-variant set (on top of the synthetic variants
trained at first run). This sharpens detection for your accent/mic.

Usage:
    python enroll_wake_word.py            # 5 samples (default)
    python enroll_wake_word.py 8          # 8 samples

Run it in a quiet spot. Only applies to custom phrases (e.g. "hey cleo"); the
bundled phrases like "hey jarvis" use a fixed neural model and aren't trainable
this way. If the assistant service is running it can stay up — PipeWire allows a
second mic capture — but a quiet room matters more than anything.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules import tts, wake_word


def _say(text: str) -> None:
    """Speak a prompt aloud and echo it to the console."""
    print(f"  Cleo: {text}", flush=True)
    try:
        tts.speak(text)
    except Exception:
        pass            # no audio out? the printed prompt still guides the user


def _on_event(name: str, data: dict) -> None:
    phrase = config.WAKE_PHRASE
    if name == "intro":
        _say(f"Let's teach me your voice. When I prompt you, say your wake "
             f"phrase, {phrase}. We'll do this {data['num_samples']} times.")
    elif name == "prompt":
        retry = " again" if data.get("attempt", 1) > data["index"] else ""
        _say(f"Say{retry} {phrase} now. Number {data['index']} of {data['total']}.")
    elif name == "accepted":
        print(f"    ✓ heard {data['heard']!r}  ({data['index']}/{data['total']})", flush=True)
    elif name == "rejected":
        _say("Hmm, that didn't sound right. Let's try that one again.")
        print(f"    ✗ rejected {data['heard']!r}", flush=True)
    elif name == "no_speech":
        _say("I didn't hear anything. Let's try again.")
    elif name == "unsupported":
        _say(f"{phrase} uses a built-in model, so there's nothing to train to your "
             "voice. Pick a custom phrase to use voice enrollment.")
    elif name == "failed":
        _say("Sorry, I couldn't capture any good samples. Let's try again later.")
    elif name == "done":
        _say("All set. I've learned how you say it.")


def main() -> int:
    num = 5
    if len(sys.argv) > 1:
        try:
            num = max(1, min(20, int(sys.argv[1])))
        except ValueError:
            print(f"Ignoring invalid sample count {sys.argv[1]!r}; using {num}.")

    print(f"=== Wake-word voice enrollment — phrase: '{config.WAKE_PHRASE}' ===\n")
    print("Loading speech models...", flush=True)
    tts.warmup()

    summary = wake_word.enroll_voice(num_samples=num, on_event=_on_event)

    print()
    if not summary.get("ok"):
        print(f"Enrollment did not complete: {summary.get('error', 'unknown error')}")
        return 1
    print(f"Enrolled {summary['accepted']} sample(s) in {summary['attempts']} attempt(s).")
    if summary["samples"]:
        print(f"New variants learned from your voice: {summary['samples']}")
    else:
        print("Your voice matched the existing variants — no new ones needed.")
    print(f"Now accepting {len(summary['variants'])} total variants.")
    print(f"Saved to {config.WAKE_VARIANTS_PATH}")
    print("\nRestart the assistant for the updated wake word to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
