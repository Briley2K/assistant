"""
Audio capture and playback using arecord/aplay (no portaudio/webrtcvad needed).
Uses RMS energy-based VAD — good enough for a quality mic like the AT2020USB+.
"""
import io
import wave
import subprocess
import collections
import numpy as np

import config

_FRAME_MS   = 30
_FRAME_SAMPLES = int(config.SAMPLE_RATE * _FRAME_MS / 1000)  # 480 samples @ 16 kHz
_FRAME_BYTES   = _FRAME_SAMPLES * 2                           # int16 = 2 bytes/sample

_SILENCE_FRAMES_NEEDED = config.SILENCE_MS // _FRAME_MS
_CALIBRATION_FRAMES    = 12                    # ~360ms to measure the noise floor
_MIN_THRESHOLD         = 40                    # absolute floor so dead-silent rooms still trigger
_MAX_WAIT_FRAMES       = int(8000 / _FRAME_MS)  # give up if no speech starts within 8s
_MAX_UTTER_FRAMES      = int(20000 / _FRAME_MS) # hard cap on a single utterance (20s)
_TRIGGER_WINDOW        = 8                       # recent frames examined to detect speech onset

# Diagnostics from the most recent record_until_silence() call (used by mic_test.py).
last_stats: dict = {}


def _rms(frame_bytes: bytes) -> float:
    samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


def _record_cmd() -> list[str]:
    """pw-record from the PipeWire default source (= the mic chosen in Ubuntu
    Sound settings) unless MIC_DEVICE names a specific target node."""
    cmd = ["pw-record", "--rate", str(config.SAMPLE_RATE),
           "--channels", "1", "--format", "s16", "--raw"]
    if config.MIC_DEVICE is not None:
        cmd += ["--target", str(config.MIC_DEVICE)]
    cmd += ["-"]
    return cmd


def record_until_silence() -> bytes:
    """
    Stream mic audio and return WAV bytes for one utterance.

    The speech threshold is calibrated each call from the room's noise floor, so
    it adapts to mic gain. Returns an (empty) WAV if no speech starts within
    _MAX_WAIT_FRAMES, so the caller never blocks forever.
    """
    proc = subprocess.Popen(_record_cmd(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _read():
        raw = proc.stdout.read(_FRAME_BYTES)
        return raw if raw and len(raw) == _FRAME_BYTES else None

    try:
        # 1. Calibrate noise floor → adaptive threshold.
        floor_samples = []
        for _ in range(_CALIBRATION_FRAMES):
            raw = _read()
            if raw is None:
                break
            floor_samples.append(_rms(raw))
        noise = float(np.median(floor_samples)) if floor_samples else 0.0
        threshold = max(noise * 3.0 + 60, _MIN_THRESHOLD)

        # 2. Wait for speech onset, then collect until trailing silence.
        recent: collections.deque = collections.deque(maxlen=_TRIGGER_WINDOW)
        voiced: list[bytes] = []
        triggered = False
        silent_run = 0
        peak = 0.0
        n = 0

        while n < _MAX_UTTER_FRAMES:
            raw = _read()
            if raw is None:
                break
            n += 1
            level = _rms(raw)
            peak = max(peak, level)
            is_speech = level > threshold

            if not triggered:
                recent.append(raw)
                if is_speech and sum(_rms(f) > threshold for f in recent) >= 3:
                    triggered = True
                    voiced.extend(recent)      # keep the lead-in frames
                elif n >= _MAX_WAIT_FRAMES:
                    last_stats.update(noise_floor=noise, threshold=threshold,
                                      peak_rms=peak, seconds=0.0, triggered=False)
                    return _to_wav_bytes(b"")  # no speech → empty
            else:
                voiced.append(raw)
                silent_run = 0 if is_speech else silent_run + 1
                if silent_run >= _SILENCE_FRAMES_NEEDED:
                    break

        last_stats.update(noise_floor=noise, threshold=threshold, peak_rms=peak,
                          seconds=len(voiced) * _FRAME_MS / 1000, triggered=triggered)
        return _to_wav_bytes(b"".join(voiced))
    finally:
        proc.terminate()
        proc.wait()


def _to_wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def play_audio(wav_bytes: bytes) -> None:
    """Play WAV bytes through the PipeWire default sink (the output chosen in
    Ubuntu Sound settings) via pw-play."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        rate, ch = wf.getframerate(), wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    proc = subprocess.Popen(
        ["pw-play", "--rate", str(rate), "--channels", str(ch),
         "--format", "s16", "--raw", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.communicate(input=pcm)
