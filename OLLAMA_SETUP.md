# Enabling GPU inference

Two ways to put the LLM on your **RTX 5080**. STT (Whisper) is already on GPU.

## Option A — patch + rebuild llama-cpp (recommended, no downloads)

A CUDA build of llama-cpp-python fails here because CUDA 13.1's headers clash
with glibc 2.41's `rsqrt` declaration. The fix (adding `__THROW` to two CUDA
declarations so exception specs match) is verified and scripted:

```bash
cd ~/assistant
bash enable_gpu_llm.sh
systemctl --user restart voice-assistant 2>/dev/null || true
```

This uses the CUDA toolkit you already have, needs `sudo` only for one header
edit, and sets `LLM_N_GPU_LAYERS = -1`. Done.

## Option B — Ollama (bundles its own CUDA)

`modules/llm.py` auto-detects Ollama and switches to it automatically — no code
changes needed once it's running.

> Why not done automatically: this Claude session is sandboxed — it can't run
> the `curl | sh` installer (blocked) and can't download the Ollama release
> binary (GitHub release assets are blocked at the network layer here). These
> commands work fine in your own terminal.

## One-time setup (run in your terminal)

```bash
# 1. Install Ollama (ships its own bundled CUDA — sidesteps the gcc15/CUDA13
#    build problem that blocks a GPU llama-cpp build on this machine).
curl -fsSL https://ollama.com/install.sh | sh

# 2. Import your existing Gemma 4 12B GGUF (uses the included Modelfile).
cd ~/assistant
ollama create gemma4-12b -f Modelfile

# 3. (Ollama runs as a background service after install — verify it sees the GPU)
ollama run gemma4-12b "Say hello in five words."
nvidia-smi          # should show an ollama process using VRAM
```

That's it. Next time you run `python3 assistant.py`, the startup log prints:

```
[LLM] Backend: Ollama (GPU)
```

## Switching backends manually

In `config.py`:

```python
LLM_BACKEND = "auto"      # auto (default): Ollama if running, else CPU
LLM_BACKEND = "ollama"    # force GPU (errors if Ollama isn't running)
LLM_BACKEND = "llamacpp"  # force CPU
```

## Alternative paths (if you'd rather not use Ollama)

- **Rebuild llama-cpp-python for CUDA**: needs an older host compiler, since
  CUDA 13.1's nvcc rejects gcc-15. In your terminal:
  ```bash
  sudo apt-get install -y gcc-13 g++-13
  CUDACXX=/usr/local/cuda/bin/nvcc \
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-13" \
  pip install --break-system-packages --force-reinstall --no-cache-dir llama-cpp-python
  ```
  Then set `LLM_N_GPU_LAYERS = -1` and `LLM_BACKEND = "llamacpp"` in config.py.

- **LM Studio server**: you already have LM Studio (`~/Downloads/LM-Studio-*.AppImage`).
  Launch it, load `gemma-4-12B-it`, start its local server (OpenAI-compatible,
  port 1234). Pointing llm.py at it would need a small OpenAI-API backend added.
