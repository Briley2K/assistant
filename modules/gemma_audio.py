"""
Native audio support for Gemma 4 via llama.cpp's mtmd: a chat handler that
accepts {'type': 'input_audio', ...} content parts (in addition to images) by
building an mtmd audio bitmap from WAV bytes. Lets the model "hear" speech
directly — no Whisper needed.
"""
import io
import wave
import base64
import ctypes
import numpy as np
from scipy.signal import resample_poly
from llama_cpp import llama_chat_format as lcf


class AudioGemma4Handler(lcf.Gemma4ChatHandler):
    """Gemma4 multimodal handler extended to accept input_audio content parts."""

    @staticmethod
    def get_image_urls(messages):
        urls = lcf.Gemma4ChatHandler.get_image_urls(messages)
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for c in m["content"]:
                    if isinstance(c, dict) and c.get("type") == "input_audio":
                        ia = c["input_audio"]
                        urls.append(f"data:audio/{ia.get('format', 'wav')};base64,{ia['data']}")
        return urls

    @staticmethod
    def _convert_content_part_for_template(part, media_marker):
        if isinstance(part, dict) and part.get("type") == "input_audio":
            return {"type": "text", "text": media_marker}
        return lcf.Gemma4ChatHandler._convert_content_part_for_template(part, media_marker)

    def _create_bitmap_from_bytes(self, data: bytes):
        if data[:4] == b"RIFF":   # WAV → audio bitmap
            with wave.open(io.BytesIO(data), "rb") as wf:
                sr = wf.getframerate()
                pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            target = self._mtmd_cpp.mtmd_get_audio_sample_rate(self.mtmd_ctx)
            if target > 0 and sr != target:
                pcm = resample_poly(pcm, target, sr)
            pcm = np.ascontiguousarray(pcm, dtype=np.float32)
            self._audio_ref = pcm   # keep buffer alive across the C call
            ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            bmp = self._mtmd_cpp.mtmd_bitmap_init_from_audio(len(pcm), ptr)
            if bmp is None:
                raise ValueError("mtmd_bitmap_init_from_audio failed")
            return bmp
        return super()._create_bitmap_from_bytes(data)


def audio_part(wav_bytes: bytes) -> dict:
    """Wrap WAV bytes as an OpenAI-style input_audio content part."""
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"},
    }


def image_part(image_bytes: bytes, fmt: str = "png") -> dict:
    """Wrap image bytes as an image_url content part (handled by the base Gemma4
    vision handler — the mmproj is audio+vision)."""
    b64 = base64.b64encode(image_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
