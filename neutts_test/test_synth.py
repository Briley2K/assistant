#!/usr/bin/env python3
"""
Standalone NeuTTS Air smoke test — synthesizes one sample WAV on CPU and reports
timing, so we can judge quality/speed before any assistant integration.

Runs in the isolated py3.12 venv (neutts_test/.venv). espeak-ng is provided by
the pip package `espeakng-loader` (no system install / sudo needed).
"""
import os
import sys
import time

# --- Point phonemizer at the pip-bundled espeak-ng (no system espeak needed) ---
try:
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    if hasattr(EspeakWrapper, "set_data_path"):
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    print(f"[espeak] using bundled lib: {espeakng_loader.get_library_path()}")
except Exception as e:
    print(f"[espeak] WARNING: could not configure bundled espeak-ng: {e}")

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
REF_WAV = os.path.join(HERE, "samples", "dave.wav")
REF_TXT = os.path.join(HERE, "samples", "dave.txt")
OUT_WAV = os.path.join(HERE, "out.wav")
TEXT = ("Hi, this is a quick test of NeuTTS Air running entirely on the CPU. "
        "If you can hear me clearly, the on-device voice works.")


def load_tts():
    """Construct the TTS object, tolerating either package layout."""
    kw = dict(backbone_repo="neuphonic/neutts-air-q4-gguf", backbone_device="cpu",
              codec_repo="neuphonic/neucodec", codec_device="cpu")
    try:
        from neuttsair.neutts import NeuTTSAir
        return NeuTTSAir(**kw)
    except ImportError:
        from neutts import NeuTTS
        return NeuTTS(**kw)


def main() -> int:
    print("[1/3] loading NeuTTS Air (downloads backbone GGUF + neucodec on first run)...")
    t0 = time.monotonic()
    tts = load_tts()
    print(f"      loaded in {time.monotonic()-t0:.1f}s")

    print("[2/3] encoding reference voice (dave)...")
    ref_text = open(REF_TXT).read().strip()
    ref_codes = tts.encode_reference(REF_WAV)

    print(f"[3/3] synthesizing {len(TEXT)} chars...")
    t1 = time.monotonic()
    wav = tts.infer(TEXT, ref_codes, ref_text)
    synth_s = time.monotonic() - t1

    sr = 24000
    sf.write(OUT_WAV, wav, sr)
    dur = len(wav) / sr
    print(f"\nDONE -> {OUT_WAV}")
    print(f"  audio: {dur:.1f}s | synth: {synth_s:.1f}s | realtime factor: {synth_s/dur:.2f}x "
          f"({'faster than realtime' if synth_s < dur else 'slower than realtime'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
