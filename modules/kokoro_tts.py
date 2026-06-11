"""
Kokoro TTS backend (more natural than Piper). Runs the Kokoro-82M ONNX model on
CPU (fast, real-time, and avoids competing for VRAM with the Gemma audio model).

We drive the onnxruntime session directly rather than via kokoro_onnx.create(),
to sidestep a dtype bug in 0.4.7 (it sends `speed` as int32; the model wants
float) and to control sentence chunking.
"""
import io
import re
import wave
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro
from kokoro_onnx.config import MAX_PHONEME_LENGTH

import config

SAMPLE_RATE = 24000
_kok: Kokoro | None = None


def _get() -> Kokoro:
    global _kok
    if _kok is None:
        print("[TTS] Loading Kokoro (CPU)...")
        sess = ort.InferenceSession(config.KOKORO_MODEL, providers=["CPUExecutionProvider"])
        _kok = Kokoro.from_session(sess, config.KOKORO_VOICES)
        print(f"[TTS] Kokoro loaded ({len(_kok.get_voices())} voices).")
    return _kok


def list_voices() -> list[str]:
    return _get().get_voices()


def _infer(kok: Kokoro, phonemes: str, style_all: np.ndarray) -> np.ndarray:
    phonemes = phonemes[:MAX_PHONEME_LENGTH]
    tokens = kok.tokenizer.tokenize(phonemes)
    if not tokens:
        return np.zeros(0, dtype=np.float32)
    style = style_all[len(tokens)].astype(np.float32)
    names = [i.name for i in kok.sess.get_inputs()]
    key = "input_ids" if "input_ids" in names else "tokens"
    inputs = {
        key: [[0, *tokens, 0]],
        "style": style,
        "speed": np.array([config.KOKORO_SPEED], dtype=np.float32),
    }
    out = kok.sess.run(None, inputs)[0]
    return np.asarray(out).squeeze()


def synth_wav(text: str, voice: str | None = None) -> bytes:
    """Synthesize text → WAV bytes (24 kHz mono int16)."""
    kok = _get()
    voice = voice or config.KOKORO_VOICE
    if voice not in kok.voices:
        voice = "af_heart"
    style_all = np.asarray(kok.voices[voice])

    pieces = []
    for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        phonemes = kok.tokenizer.phonemize(sentence, lang="en-us")
        audio = _infer(kok, phonemes, style_all)
        if audio.size:
            pieces.append(audio)

    samples = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()
