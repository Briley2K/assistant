#!/usr/bin/env python3
"""
Feasibility test: feed audio DIRECTLY to Gemma 4 12B (no Whisper) via its native
audio encoder, using llama.cpp's mtmd. We synthesize a spoken question with Piper
and send ONLY the audio (the answer is never in the text), so a correct reply
proves the model genuinely heard and understood the audio.
"""
import sys, os, io, wave, base64, ctypes
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.signal import resample_poly

import config
from llama_cpp import Llama, llama_chat_format as lcf

MMPROJ = (
    "/run/media/briley/AE24D19024D15C41/Users/Briley/.lmstudio/models/"
    "lmstudio-community/gemma-4-12B-it-GGUF/mmproj-gemma-4-12B-it-BF16.gguf"
)


class AudioGemma4Handler(lcf.Gemma4ChatHandler):
    """Gemma4 handler that also accepts {'type':'input_audio',...} content parts."""

    @staticmethod
    def get_image_urls(messages):
        urls = lcf.Gemma4ChatHandler.get_image_urls(messages)
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for c in m["content"]:
                    if isinstance(c, dict) and c.get("type") == "input_audio":
                        ia = c["input_audio"]
                        urls.append(f"data:audio/{ia.get('format','wav')};base64,{ia['data']}")
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
            self._audio_ref = pcm   # keep alive during init
            ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            bmp = self._mtmd_cpp.mtmd_bitmap_init_from_audio(len(pcm), ptr)
            if bmp is None:
                raise ValueError("mtmd_bitmap_init_from_audio failed")
            return bmp
        return super()._create_bitmap_from_bytes(data)


def synth_wav(text: str) -> bytes:
    from modules import tts
    voice = tts._get_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    return buf.getvalue()


def main():
    question = "What is the capital of France? Answer in one word."
    print(f"[1] Synthesizing spoken question: {question!r}")
    wav = synth_wav(question)

    print("[2] Loading Gemma 4 12B + audio mmproj on GPU...")
    handler = AudioGemma4Handler(clip_model_path=MMPROJ, verbose=False)
    llm = Llama(
        model_path=config.LLM_MODEL_PATH,
        chat_handler=handler,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
    )
    # Force mtmd context init (lazy) so we can query audio support before feeding audio.
    handler._init_mtmd_context(llm)
    supported = handler._mtmd_cpp.mtmd_support_audio(handler.mtmd_ctx)
    print("    audio supported by mtmd:", supported)
    if not supported:
        print("\n=== RESULT: this llama.cpp build does NOT enable Gemma 4's audio "
              "encoder (mtmd_support_audio=False). Native audio not usable here yet; "
              "keep Whisper. Re-run this after a llama.cpp build that adds gemma4 audio. ===")
        return
    print("    mtmd audio sample rate:", handler._mtmd_cpp.mtmd_get_audio_sample_rate(handler.mtmd_ctx))

    b64 = base64.b64encode(wav).decode()
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Listen to the audio and answer it."},
        {"role": "user", "content": [
            {"type": "text", "text": "Answer the question contained in this audio:"},
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        ]},
    ]

    print("[3] Sending AUDIO ONLY (answer not in text) ...")
    out = llm.create_chat_completion(messages=messages, max_tokens=32, temperature=0.0)
    reply = out["choices"][0]["message"]["content"].strip()
    print("\n=== MODEL REPLY:", repr(reply), "===")
    print("PASS — model answered from audio ✓" if "paris" in reply.lower()
          else "Model responded but check whether it heard the audio correctly.")


if __name__ == "__main__":
    main()
