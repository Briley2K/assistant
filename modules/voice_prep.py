"""
Prepare an uploaded reference clip so NeuTTS can actually use it.

NeuTTS Air's backbone has a ~2048-token context: a reference longer than ~18s
overflows it and synthesis fails (which the assistant hides by falling back to
Kokoro). Too-short clips (< ~13s) hit a codec padding error. So a usable
reference is ~13-18s of clean mono speech whose transcript matches the audio.

prepare_reference() takes an uploaded file of any channel count / bit depth (and
most container formats), downmixes to mono, trims it to a clean word boundary at
or under the target length, and returns a transcript generated from the FINAL
audio so the two always line up.
"""
import io
import wave

import numpy as np

import config

# Target reference length. NeuTTS' usable window tops out ~18s, but the longer the
# reference the less context is left to actually speak the reply, so we aim a bit
# under that to keep generation headroom (verified to produce full-length speech).
REF_TARGET_SECS = 16.0
# Below this the neucodec padding step fails outright; surfaced as a warning since
# trimming can't lengthen a clip.
REF_MIN_SECS = 13.0


def _decode_mono(audio_bytes):
    """(mono float32 in [-1,1], sample_rate) from uploaded audio bytes. WAV is
    decoded at its native rate (best quality); any other container falls back to
    faster-whisper's decoder at 16 kHz."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            fr, ch, sw, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
            frames = wf.readframes(n)
        if sw == 2:
            a = np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            a = np.frombuffer(frames, np.int32).astype(np.float32) / 2147483648.0
        elif sw == 1:   # 8-bit WAV is unsigned, centered on 128
            a = (np.frombuffer(frames, np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported WAV sample width: {sw * 8}-bit")
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        return np.ascontiguousarray(a, np.float32), fr
    except (wave.Error, EOFError, ValueError):
        from faster_whisper.audio import decode_audio       # mp3/m4a/flac/ogg/…
        a = np.asarray(decode_audio(io.BytesIO(audio_bytes), sampling_rate=16000), np.float32)
        return np.ascontiguousarray(a), 16000


def _encode_wav(mono, fr):
    """mono float32 -> 16-bit mono WAV bytes at sample rate `fr`."""
    pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fr)
        wf.writeframes(pcm)
    return buf.getvalue()


def prepare_reference(audio_bytes, target_secs: float = REF_TARGET_SECS):
    """Make an uploaded clip NeuTTS-ready.

    Returns (wav_bytes, transcript, info) where wav_bytes is 16-bit mono trimmed
    to <= target_secs at a word boundary, transcript is generated from that final
    audio, and info describes what happened (original/final length, whether it was
    trimmed, and any warning). Raises ValueError if the audio can't be decoded.
    """
    from modules import stt

    mono, fr = _decode_mono(audio_bytes)
    if mono.size == 0:
        raise ValueError("decoded audio is empty")
    orig_secs = mono.size / fr

    # 16 kHz mono for Whisper; only the head matters since we cut at/under target.
    if fr != 16000:
        from scipy.signal import resample_poly
        w16 = resample_poly(mono, 16000, fr).astype(np.float32)
    else:
        w16 = mono
    head = w16[: int((target_secs + 4.0) * 16000)]

    segs, _ = stt._get_model().transcribe(
        head, language=config.WHISPER_LANGUAGE, beam_size=1, word_timestamps=True)
    words = [w for s in segs for w in (s.words or [])]

    trimmed = orig_secs > target_secs
    if trimmed and words:
        kept = [w for w in words if w.end <= target_secs] or words[:1]
        mono = mono[: int(kept[-1].end * fr)]
        used = kept
    elif trimmed:                       # no word timing (e.g. near-silent) — hard cut
        mono = mono[: int(target_secs * fr)]
        used = words
    else:
        used = words

    text = " ".join(w.word.strip() for w in used).strip()
    final_secs = mono.size / fr
    info = {
        "orig_secs": round(orig_secs, 1),
        "final_secs": round(final_secs, 1),
        "trimmed": trimmed,
        "sample_rate": fr,
        "warning": (f"clip is only {final_secs:.1f}s — NeuTTS needs ~{REF_MIN_SECS:.0f}s+; "
                    "it may fail to synthesize. Use a longer sample.")
                   if final_secs < REF_MIN_SECS else None,
    }
    return _encode_wav(mono, fr), (text or "This is a reference voice sample."), info
