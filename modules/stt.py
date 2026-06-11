import io
import numpy as np
from faster_whisper import WhisperModel

import config

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        try:
            print(f"[STT] Loading Whisper '{config.WHISPER_MODEL}' on {config.WHISPER_DEVICE}...")
            _model = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE,
            )
        except Exception as e:
            if config.WHISPER_DEVICE != "cpu":
                print(f"[STT] GPU load failed ({e}); falling back to CPU.")
                _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
            else:
                raise
    return _model


def transcribe(wav_bytes: bytes) -> str:
    """Transcribe WAV audio bytes to text. Returns empty string if nothing heard."""
    model = _get_model()

    audio = _wav_bytes_to_float32(wav_bytes)

    segments, _ = model.transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        beam_size=5,
        vad_filter=True,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    import wave
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0
