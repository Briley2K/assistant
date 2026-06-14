import io
import numpy as np
from faster_whisper import WhisperModel

import config

_model: WhisperModel | None = None
_spotter: WhisperModel | None = None   # small CPU-only model for sleep mode


def _load_whisper(device: str, compute: str) -> WhisperModel:
    """Load Whisper on `device`, falling back to CPU on any failure. CTranslate2
    loads its CUDA libraries (libcublas/libcudnn) *lazily on the first inference*,
    so a broken/mismatched CUDA install (e.g. only CUDA 13 present but ct2 wants
    libcublas.so.12) doesn't surface at construction — it throws mid-transcribe.
    We force a tiny warmup inference here so that failure is caught and we drop to
    CPU, instead of dying when the user actually transcribes something."""
    print(f"[STT] Loading Whisper '{config.WHISPER_MODEL}' on {device}...")
    try:
        m = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute)
        if device != "cpu":
            # Iterate the generator so CTranslate2 actually runs on the GPU now.
            list(m.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)[0])
        return m
    except Exception as e:
        if device != "cpu":
            print(f"[STT] GPU Whisper unavailable ({e}); falling back to CPU.")
            return _load_whisper("cpu", "int8")
        raise


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = _load_whisper(config.WHISPER_DEVICE, config.WHISPER_COMPUTE)
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
    """Decode WAV bytes to mono float32 at 16 kHz (what Whisper expects). Handles
    stereo (downmixed) and 8/16/32-bit PCM so an arbitrary uploaded reference clip
    works — not just our 16-bit mono 16 kHz mic capture (e.g. dave.wav is stereo
    44.1 kHz)."""
    import wave
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:   # 8-bit WAV is unsigned, centered on 128
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width * 8}-bit (use 16-bit PCM)")
    if channels > 1:                       # downmix to mono
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:   # Whisper expects 16 kHz (e.g. Kokoro WAVs are 24 kHz)
        from scipy.signal import resample_poly
        audio = resample_poly(audio, 16000, rate).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)
