"""
Wake word detection — two engines, picked automatically from the phrase:

  - Bundled phrases ("hey jarvis", "alexa", ...) use pretrained openWakeWord
    models: instant detection, near-zero CPU.
  - Any other phrase (e.g. the default "hey cleo") uses a trained Whisper
    listener: the mic is gated by RMS energy, and each speech burst is
    transcribed by the small CPU Whisper (stt.transcribe_spotter) and matched
    against the phrase plus transcription variants learned during a one-time
    training pass (ensure_ready → _train), stored in models/wake/<phrase>.json.
"""
import os
import json
import collections
import subprocess

import numpy as np

import config
from modules import phrases

_model = None                       # openWakeWord model (bundled phrases)
_variants: list[str] | None = None  # learned transcripts (custom phrases)

_TRAIN_SPEEDS = (0.85, 1.0, 1.2)

_FRAME_MS = 30
_FRAME_BYTES = int(config.SAMPLE_RATE * _FRAME_MS / 1000) * 2
_LEAD_FRAMES = 10                       # ~300 ms kept from before speech onset
_END_SILENCE_FRAMES = 500 // _FRAME_MS  # burst ends after ~0.5 s of quiet
_MAX_BURST_FRAMES = 3500 // _FRAME_MS   # ... or at 3.5 s, whichever comes first


# --------------------------------------------------------------------------
# Setup / training
# --------------------------------------------------------------------------
def ensure_ready() -> bool:
    """Prepare detection for the configured phrase. Returns True if a
    training pass ran (i.e. the phrase is newly trained)."""
    if config.WAKE_WORD_MODEL:          # bundled openWakeWord phrase
        _get_model()
        return False
    from modules import stt
    stt.warmup_spotter()
    if _load_variants() is None:
        _train()
        return True
    return False


def _load_variants() -> list[str] | None:
    global _variants
    if _variants is None:
        try:
            with open(config.WAKE_VARIANTS_PATH) as f:
                _variants = json.load(f)["variants"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None
    return _variants


def _train() -> None:
    """One-time training for a custom phrase: synthesize it in many voices and
    speeds, transcribe each sample, and keep the transcription variants the
    recognizer should accept (catches consistent mishearings, e.g. 'cleo' →
    'clio'). Saved to models/wake/<phrase>.json."""
    global _variants
    from modules import stt, kokoro_tts

    phrase = config.WAKE_PHRASE
    print(f"[Wake word] Training recognizer for '{phrase}' (one-time, ~2 min)...")

    # Every English voice (af/am = US, bf/bm = UK). Both text forms — Kokoro
    # pronounces "hey cleo" and "Hey Cleo!" noticeably differently.
    train_voices = [v for v in kokoro_tts.list_voices() if v[0] in "ab"]
    texts = [phrase, phrase.title() + "!"]
    variants = {phrases.normalize(phrase)}
    for voice in train_voices:
        for text in texts:
            for speed in _TRAIN_SPEEDS:
                try:
                    wav = kokoro_tts.synth_wav(text, voice=voice, speed=speed)
                    heard = phrases.normalize(stt.transcribe_spotter(wav))
                except Exception as e:
                    print(f"[Wake word] Sample {voice}@{speed} failed: {e}")
                    continue
                # Keep only transcripts that plausibly are the phrase — a bad
                # TTS sample can make Whisper hallucinate something unrelated,
                # which must not become an accepted wake word.
                if heard and (phrases.contains(heard, phrase)
                              or phrases.similarity(heard, phrase) >= 0.6):
                    variants.add(heard)

    _save_variants(variants)
    print(f"[Wake word] Trained — accepting {len(_variants)} variants: {_variants}")


def _save_variants(variants) -> list[str]:
    """Persist the accepted-transcript set for the current phrase and refresh the
    in-memory cache. Accepts any iterable; stores a sorted, de-duplicated list."""
    global _variants
    _variants = sorted(set(variants))
    os.makedirs(os.path.dirname(config.WAKE_VARIANTS_PATH), exist_ok=True)
    with open(config.WAKE_VARIANTS_PATH, "w") as f:
        json.dump({"phrase": config.WAKE_PHRASE, "variants": _variants}, f, indent=2)
    return _variants


def _accepts_sample(heard: str) -> bool:
    """True if a transcript plausibly IS the wake phrase — used to reject a
    sample captured from background noise or a misfire, so garbage never becomes
    an accepted wake variant. More lenient than detection (the user is genuinely
    saying the phrase here), but still guards against unrelated transcripts."""
    norm = phrases.normalize(heard)
    if not norm:
        return False
    return (phrases.contains(heard, config.WAKE_PHRASE)
            or _windowed_similarity(heard, config.WAKE_PHRASE) >= 0.5)


def enroll_voice(num_samples: int = 5, recorder=None, on_event=None) -> dict:
    """Interactively learn the user's own pronunciation of the wake phrase.

    Records the user saying the phrase `num_samples` times, transcribes each with
    the spotter Whisper, and merges the accepted transcripts into the phrase's
    variant set (on top of any existing synthetic/learned variants) so detection
    adapts to this speaker. Bundled openWakeWord phrases can't be voice-trained
    this way (they use a fixed neural model) — returns an error in that case.

    `recorder()` -> wav bytes is injectable (defaults to one silence-bounded mic
    capture); `on_event(name, data)` receives progress events for the UI/voice
    front-end. Returns a summary dict: {ok, accepted, attempts, variants, samples}."""
    from modules import audio, stt

    def emit(name, **data):
        if on_event:
            on_event(name, data)

    if config.WAKE_WORD_MODEL:        # bundled phrase → neural model, not transcript-based
        emit("unsupported", phrase=config.WAKE_PHRASE)
        return {"ok": False, "error": "voice enrollment only applies to custom wake "
                "phrases; this phrase uses a pretrained model. Adjust the detection "
                "threshold instead."}

    stt.warmup_spotter()
    record = recorder or (lambda: audio.record_until_silence())

    existing = set(_load_variants() or [])
    existing.add(phrases.normalize(config.WAKE_PHRASE))
    learned: list[str] = []
    emit("intro", phrase=config.WAKE_PHRASE, num_samples=num_samples)

    accepted = attempts = 0
    max_attempts = num_samples * 2        # allow re-tries for misfires/silence
    while accepted < num_samples and attempts < max_attempts:
        attempts += 1
        emit("prompt", index=accepted + 1, total=num_samples, attempt=attempts)
        wav = record()
        if not audio.last_stats.get("triggered"):
            emit("no_speech", index=accepted + 1)
            continue
        heard = stt.transcribe_spotter(wav)
        if _accepts_sample(heard):
            norm = phrases.normalize(heard)
            accepted += 1
            if norm not in existing:
                learned.append(norm)
            emit("accepted", index=accepted, total=num_samples, heard=norm)
        else:
            emit("rejected", heard=phrases.normalize(heard))

    if accepted == 0:
        emit("failed", attempts=attempts)
        return {"ok": False, "error": "no usable samples captured", "attempts": attempts}

    variants = _save_variants(existing | set(learned))
    summary = {"ok": True, "accepted": accepted, "attempts": attempts,
               "variants": variants, "samples": learned}
    emit("done", **summary)
    return summary


def _windowed_similarity(heard: str, target: str) -> float:
    """Best fuzzy ratio between the target and any same-length word window of
    the transcript (so 'hey cleo' is found inside 'um hey cleo thanks')."""
    hw, tw = phrases.normalize(heard).split(), phrases.normalize(target).split()
    if not hw or not tw:
        return 0.0
    t = " ".join(tw)
    best = phrases.similarity(" ".join(hw), t)
    for i in range(max(1, len(hw) - len(tw) + 1)):
        win = " ".join(hw[i:i + len(tw)])
        best = max(best, phrases.similarity(win, t))
    return best


def matches_wake(heard: str) -> bool:
    """True if a transcript contains (or closely resembles) the wake phrase."""
    if phrases.contains(heard, config.WAKE_PHRASE):
        return True
    for v in [config.WAKE_PHRASE] + (_load_variants() or []):
        if phrases.contains(heard, v) or _windowed_similarity(heard, v) >= 0.8:
            return True
    return False


# --------------------------------------------------------------------------
# Listening
# --------------------------------------------------------------------------
def listen_for_wake_word() -> None:
    """Block until the configured wake phrase is detected."""
    print(f"[Waiting for wake word — say '{config.WAKE_PHRASE}']", flush=True)
    if config.WAKE_WORD_MODEL:
        _listen_openwakeword()
    else:
        _listen_custom()


def _mic_cmd() -> list[str]:
    cmd = ["pw-record", "--rate", str(config.SAMPLE_RATE),
           "--channels", "1", "--format", "s16", "--raw"]
    if config.MIC_DEVICE is not None:
        cmd += ["--target", str(config.MIC_DEVICE)]
    return cmd + ["-"]


def _get_model():
    global _model
    if _model is None:
        from openwakeword.model import Model
        print(f"[Wake word] Loading '{config.WAKE_WORD_LABEL}' model...")
        _model = Model(wakeword_model_paths=[config.WAKE_WORD_MODEL])
    return _model


def _listen_openwakeword() -> None:
    model = _get_model()
    # openWakeWord expects 16 kHz mono int16, 80 ms chunks = 1280 samples
    chunk_bytes = 1280 * 2
    label = config.WAKE_WORD_LABEL

    proc = subprocess.Popen(_mic_cmd(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            raw = proc.stdout.read(chunk_bytes)
            if not raw or len(raw) < chunk_bytes:
                break
            prediction = model.predict(np.frombuffer(raw, dtype=np.int16))
            if prediction.get(label, 0.0) >= config.WAKE_WORD_THRESHOLD:
                model.reset()
                return
    finally:
        proc.terminate()
        proc.wait()


def _listen_custom() -> None:
    """Energy-gated Whisper listener: transcribe each speech burst with the
    small CPU Whisper and return when it matches the trained phrase."""
    from modules import audio, stt
    stt._get_spotter()

    proc = subprocess.Popen(_mic_cmd(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _read():
        raw = proc.stdout.read(_FRAME_BYTES)
        return raw if raw and len(raw) == _FRAME_BYTES else None

    try:
        # Calibrate the speech threshold from the room's noise floor,
        # mirroring audio.record_until_silence().
        floor = []
        for _ in range(12):
            raw = _read()
            if raw is None:
                return
            floor.append(audio._rms(raw))
        threshold = max(float(np.median(floor)) * 3.0 + 60, 40)

        lead: collections.deque = collections.deque(maxlen=_LEAD_FRAMES)
        burst: list[bytes] = []
        silent_run = 0

        while True:
            raw = _read()
            if raw is None:
                return
            is_speech = audio._rms(raw) > threshold

            if not burst:
                lead.append(raw)
                if is_speech and sum(audio._rms(f) > threshold for f in lead) >= 3:
                    burst = list(lead)
                continue

            burst.append(raw)
            silent_run = 0 if is_speech else silent_run + 1
            if silent_run >= _END_SILENCE_FRAMES or len(burst) >= _MAX_BURST_FRAMES:
                wav = audio._to_wav_bytes(b"".join(burst))
                burst, silent_run = [], 0
                lead.clear()
                heard = stt.transcribe_spotter(wav)
                if heard and matches_wake(heard):
                    return
    finally:
        proc.terminate()
        proc.wait()
