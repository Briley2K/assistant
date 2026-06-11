#!/usr/bin/env python3
"""
Mic + transcription test. Records one utterance from your selected mic, prints
the audio levels it measured, and shows what Whisper transcribed.

    python3 mic_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from modules import audio, stt
import config


def main():
    print(f"Mic: {'Ubuntu default source' if config.MIC_DEVICE is None else config.MIC_DEVICE}")
    print("Loading Whisper...")
    stt._get_model()

    print("\n>>> Speak now (say a sentence). It stops after you go quiet.\n")
    wav = audio.record_until_silence()

    s = audio.last_stats
    print("--- levels ---")
    print(f"  noise floor : {s.get('noise_floor', 0):.0f} RMS")
    print(f"  threshold   : {s.get('threshold', 0):.0f} RMS")
    print(f"  peak heard  : {s.get('peak_rms', 0):.0f} RMS")
    print(f"  captured    : {s.get('seconds', 0):.1f}s  (speech detected: {s.get('triggered')})")

    if not s.get("triggered"):
        print("\n⚠ No speech detected — your voice never exceeded the threshold.")
        print("  Check the mic is selected/unmuted in Ubuntu Sound settings and try louder,")
        print("  or lower _MIN_THRESHOLD / raise mic volume.")
        return

    print("\nTranscribing...")
    text = stt.transcribe(wav)
    print(f"\n>>> Whisper heard: {text!r}\n")
    print("✓ Mic + STT working." if text else "⚠ Audio captured but transcript empty.")


if __name__ == "__main__":
    main()
