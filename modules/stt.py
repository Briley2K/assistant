import io
import numpy as np
from faster_whisper import WhisperModel

import config

_model: WhisperModel | None = None
_spotter: WhisperModel | None = None   # small CPU-only model for sleep mode


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


def _get_spotter() -> WhisperModel:
    global _spotter
    if _spotter is None:
        print(f"[STT] Loading Whisper '{config.SLEEP_WHISPER_MODEL}' on CPU (sleep-mode spotter)...")
        _spotter = WhisperModel(config.SLEEP_WHISPER_MODEL, device="cpu", compute_type="int8")
    return _spotter


def transcribe_spotter(wav_bytes: bytes) -> str:
    """Sleep-mode transcription: a small Whisper on the CPU, so the GPU stays
    free while the assistant is asleep. Only used to spot the wake command."""
    audio = _wav_bytes_to_float32(wav_bytes)
    segments, _ = _get_spotter().transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        beam_size=1,
        vad_filter=True,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def warmup_spotter() -> None:
    """Preload the spotter (called when entering sleep, so waking is snappy)."""
    _get_spotter()


def unload_spotter() -> None:
    """Drop the spotter once awake again."""
    global _spotter
    _spotter = None


def unload() -> None:
    """Free the main Whisper model (and any GPU memory it holds) during sleep."""
    global _model
    _model = None


def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    import wave
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    if rate != 16000:   # Whisper expects 16 kHz (e.g. Kokoro WAVs are 24 kHz)
        from scipy.signal import resample_poly
        audio = resample_poly(audio, 16000, rate).astype(np.float32)
    return audio
