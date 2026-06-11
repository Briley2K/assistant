"""
Text-to-speech with a selectable engine (config.TTS_ENGINE):
  - "kokoro": natural neural voice (Kokoro-82M, CPU). Default.
  - "piper":  fast, lighter, more robotic.
Both synthesize a WAV that is played via modules.audio (PipeWire default sink).
"""
import io
import wave
import re
import queue
import threading

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


def _synth_wav(cleaned: str) -> bytes:
    """Synthesize already-cleaned text to WAV bytes with the active engine."""
    if config.TTS_ENGINE == "kokoro":
        from modules import kokoro_tts
        return kokoro_tts.synth_wav(cleaned)
    piper = _get_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        piper.synthesize_wav(cleaned, wf)
    return buf.getvalue()


def speak(text: str) -> None:
    """Synthesize text with the active engine and play it."""
    cleaned = _clean_for_speech(text)
    if cleaned:
        audio.play_audio(_synth_wav(cleaned))


class StreamSpeaker:
    """Speaks text chunks in order on background threads. Two stages run
    concurrently — one synthesizes the next chunk while another plays the
    current one — so sentences flow with minimal gaps. Feed it whole sentences
    via say(), then call close() to wait for playback to finish.

    The point: the caller (LLM stream) keeps generating later sentences while
    earlier ones are already being spoken, cutting time-to-first-audio."""

    def __init__(self):
        self._text_q: queue.Queue = queue.Queue()
        self._wav_q: queue.Queue = queue.Queue()
        self._synth_t = threading.Thread(target=self._synth_loop, daemon=True)
        self._play_t = threading.Thread(target=self._play_loop, daemon=True)
        self._synth_t.start()
        self._play_t.start()

    def _synth_loop(self):
        while True:
            text = self._text_q.get()
            if text is None:
                self._wav_q.put(None)
                return
            cleaned = _clean_for_speech(text)
            if not cleaned:
                continue
            try:
                self._wav_q.put(_synth_wav(cleaned))
            except Exception as e:
                print(f"[TTS] stream synth error: {e}")

    def _play_loop(self):
        while True:
            wav = self._wav_q.get()
            if wav is None:
                return
            try:
                audio.play_audio(wav)
            except Exception as e:
                print(f"[TTS] stream play error: {e}")

    def say(self, text: str) -> None:
        self._text_q.put(text)

    def close(self) -> None:
        """Signal end and block until all queued audio has finished playing."""
        self._text_q.put(None)
        self._synth_t.join()
        self._play_t.join()


def _clean_for_speech(text: str) -> str:
    """Strip markdown artifacts that sound bad when spoken aloud."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)     # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)           # italic
    text = re.sub(r"`(.+?)`", r"\1", text)             # inline code
    text = re.sub(r"#{1,6}\s", "", text)               # headers
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)   # links
    text = re.sub(r"\n+", " ", text)                   # newlines → space
    return text.strip()
