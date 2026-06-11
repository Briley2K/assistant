"""
Text-to-speech with a selectable engine (config.TTS_ENGINE):
  - "kokoro": natural neural voice (Kokoro-82M, CPU). Default.
  - "piper":  fast, lighter, more robotic.
Both synthesize a WAV that is played via modules.audio (PipeWire default sink).
"""
import io
import wave
import re

import config
from modules import audio

_piper = None


def _get_voice():
    """Lazy-load the Piper voice (also used by the standalone test scripts)."""
    from piper.voice import PiperVoice
    global _piper
    if _piper is None:
        print(f"[TTS] Loading Piper voice from {config.PIPER_VOICE}...")
        _piper = PiperVoice.load(config.PIPER_VOICE)
        print("[TTS] Piper voice loaded.")
    return _piper


def warmup() -> None:
    """Preload the active TTS engine."""
    if config.TTS_ENGINE == "kokoro":
        from modules import kokoro_tts
        kokoro_tts._get()
    else:
        _get_voice()


def speak(text: str) -> None:
    """Synthesize text with the active engine and play it."""
    cleaned = _clean_for_speech(text)
    if not cleaned:
        return

    if config.TTS_ENGINE == "kokoro":
        from modules import kokoro_tts
        wav = kokoro_tts.synth_wav(cleaned)
    else:
        piper = _get_voice()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            piper.synthesize_wav(cleaned, wf)
        wav = buf.getvalue()

    audio.play_audio(wav)


def _clean_for_speech(text: str) -> str:
    """Strip markdown artifacts that sound bad when spoken aloud."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)     # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)           # italic
    text = re.sub(r"`(.+?)`", r"\1", text)             # inline code
    text = re.sub(r"#{1,6}\s", "", text)               # headers
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)   # links
    text = re.sub(r"\n+", " ", text)                   # newlines → space
    return text.strip()
