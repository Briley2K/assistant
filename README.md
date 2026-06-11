# Local Voice Assistant

Fully local, voice-activated assistant. Two speech-input modes (set `stt_mode` in
the panel or settings.json):

- **native** (default): **wake word → audio → Gemma 4 (hears directly) → speech.**
  Gemma 4 12B's built-in audio encoder transcribes+understands in one pass — no Whisper.
- **whisper**: **wake word → Whisper STT → Gemma 4 (text) → speech.** Classic pipeline,
  supports remote LLM backends (Ollama/API).

Everything runs on this machine (Gemma 4 12B on GPU, Kokoro/Piper TTS, openWakeWord).

**Voice (TTS):** defaults to **Kokoro** (natural, 55 voices, runs on CPU). Pick the
engine/voice and hit **Test voice** in the control panel. Piper is the lighter fallback.

## Quick start

```bash
cd ~/assistant
bash setup.sh            # one-time: Python deps (already done)
bash download_models.sh  # one-time: Piper voice (already done)
python3 assistant.py     # run it — say "Hey Jarvis"
```

## Control panel (app interface)

A local web app to manage everything — no editing code:

```bash
bash install_service.sh           # installs systemd user services + app launcher
# then open http://localhost:5005  (also appears in your app menu as "Voice Assistant")
```

From the panel you can:
- **Start / Stop** the assistant and toggle **Run on startup** (systemd user service)
- Edit the **pre-prompt** (system prompt)
- Choose the **wake word** (hey jarvis, alexa, hey mycroft, …) and Whisper model
- Manage the **LLM connection**: local Ollama/CPU, or a **remote OpenAI-compatible API**

Settings are saved to `settings.json`, which the assistant reads on start.
Use **Save & restart** in the panel to apply changes to a running assistant.

## Run on Ubuntu startup

`install_service.sh` enables the control panel at boot. To autostart the assistant
itself, click **Enable autostart** in the panel, or:

```bash
systemctl --user enable --now voice-assistant.service   # at login
sudo loginctl enable-linger $USER                        # also before login (headless)
```

## GPU inference (RTX 5080)

- **Speech-to-text: already on GPU** (faster-whisper / CUDA), with automatic CPU
  fallback if the GPU is unavailable.
- **LLM: one command to enable GPU.** A CUDA build of llama-cpp-python is blocked
  on this machine by a glibc 2.41 / CUDA 13.1 header conflict. The fix is verified
  and scripted — run in your terminal (needs `sudo` for one header edit, no downloads):
  ```bash
  bash enable_gpu_llm.sh    # patches the CUDA header, rebuilds w/ CUDA, flips config to GPU
  ```
  Alternatively install Ollama (bundles its own CUDA): see [OLLAMA_SETUP.md](OLLAMA_SETUP.md).
  Either way the backend uses the GPU automatically afterward.

## Layout

| File | Purpose |
|------|---------|
| `assistant.py` | main voice loop |
| `control_panel.py` | Flask web UI (port 5005) |
| `config.py` | defaults + loads `settings.json` |
| `settings.json` | user-editable settings (managed by the panel) |
| `modules/` | `wake_word`, `stt`, `llm` (3 backends), `tts`, `audio` |
| `Modelfile` | imports the Gemma 4 GGUF into Ollama |
| `systemd/`, `install_service.sh` | autostart services + app launcher |
