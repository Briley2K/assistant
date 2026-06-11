"""
Wake word detection using openWakeWord, reading mic via arecord subprocess.
"""
import subprocess
import numpy as np
from openwakeword.model import Model

import config

_model: Model | None = None


def _get_model() -> Model:
    global _model
    if _model is None:
        label = config.WAKE_WORD_LABEL
        print(f"[Wake word] Loading '{label}' model...")
        _model = Model(wakeword_model_paths=[config.WAKE_WORD_MODEL])
    return _model


def listen_for_wake_word() -> None:
    """Block until the configured wake word is detected."""
    model = _get_model()

    # openWakeWord expects 16 kHz mono int16, 80 ms chunks = 1280 samples = 2560 bytes
    chunk_bytes = 1280 * 2

    cmd = ["pw-record", "--rate", str(config.SAMPLE_RATE),
           "--channels", "1", "--format", "s16", "--raw"]
    if config.MIC_DEVICE is not None:
        cmd += ["--target", str(config.MIC_DEVICE)]
    cmd += ["-"]

    label = config.WAKE_WORD_LABEL
    print(f"[Waiting for wake word — say '{label.replace('_v0.1', '').replace('_', ' ')}']", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        while True:
            raw = proc.stdout.read(chunk_bytes)
            if not raw or len(raw) < chunk_bytes:
                break

            samples = np.frombuffer(raw, dtype=np.int16)
            prediction = model.predict(samples)

            score = prediction.get(label, 0.0)
            if score >= config.WAKE_WORD_THRESHOLD:
                model.reset()
                return
    finally:
        proc.terminate()
        proc.wait()
