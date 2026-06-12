#!/usr/bin/env bash
# One-shot installer: system packages, Python env, all Python deps, and every
# model (Gemma 4 12B LLM, Kokoro + Piper TTS voices, Whisper STT).
#
#     bash setup.sh
#
# When it finishes the assistant is ready to run immediately:
#     .venv/bin/python3 assistant.py
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------
# 1/5 — System packages (only installs what's missing)
# ---------------------------------------------------------------
echo "=== [1/5] System packages ==="
APT_PKGS=()
python3 -c "import ensurepip" 2>/dev/null || APT_PKGS+=("python3-venv")
command -v pw-record >/dev/null || APT_PKGS+=("pipewire-bin")     # mic capture
command -v aplay     >/dev/null || APT_PKGS+=("alsa-utils")       # playback
command -v cc        >/dev/null || APT_PKGS+=("build-essential")  # llama-cpp build
command -v cmake     >/dev/null || APT_PKGS+=("cmake")
# GTK bindings for the status orb overlay (overlay.py, runs on system python).
# Must check the cairo *foreign-struct bridge* (python3-gi-cairo), not just
# "import gi, cairo" — those pass with python3-gi + python3-cairo alone, yet the
# orb's draw handler still can't receive a cairo.Context, so it paints nothing.
/usr/bin/python3 -c "import gi; gi.require_version('Gtk','3.0'); gi.require_foreign('cairo')" 2>/dev/null || \
    APT_PKGS+=("python3-gi" "python3-gi-cairo" "gir1.2-gtk-3.0")

if [ ${#APT_PKGS[@]} -gt 0 ]; then
    echo "Missing: ${APT_PKGS[*]} — installing (needs sudo)..."
    sudo apt-get update -qq
    sudo apt-get install -y "${APT_PKGS[@]}"
else
    echo "All system packages already present."
fi

# ---------------------------------------------------------------
# 2/5 — Python virtual environment
# ---------------------------------------------------------------
echo ""
echo "=== [2/5] Python virtual environment ==="
# --clear rebuilds from scratch: a venv created before python3-venv was present
# comes out empty (no pip), and a plain `venv` over it won't repair it.
if [ -d "$SCRIPT_DIR/.venv" ] && ! "$SCRIPT_DIR/.venv/bin/python3" -m pip --version >/dev/null 2>&1; then
    echo "Existing .venv is incomplete (no pip) — rebuilding it."
fi
python3 -m venv --clear "$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/.venv/bin/activate"
python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
pip install --quiet --upgrade pip

# ---------------------------------------------------------------
# 3/5 — Python packages
# ---------------------------------------------------------------
echo ""
echo "=== [3/5] Python packages ==="
pip install \
    numpy \
    scipy \
    faster-whisper \
    openwakeword \
    piper-tts \
    kokoro-onnx \
    flask \
    huggingface_hub

# ---------------------------------------------------------------
# 4/5 — llama-cpp-python (CUDA if possible, CPU fallback)
# ---------------------------------------------------------------
echo ""
echo "=== [4/5] llama-cpp-python ==="
if python3 -c "import llama_cpp" 2>/dev/null; then
    echo "llama-cpp-python already installed — skipping."
elif [ -x /usr/local/cuda/bin/nvcc ] && \
     CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on" \
     pip install llama-cpp-python --no-cache-dir; then
    echo "Built with CUDA — LLM will run on the GPU."
else
    echo "CUDA build unavailable (no toolkit, or the known gcc15/CUDA13 header"
    echo "conflict — see enable_gpu_llm.sh). Installing CPU build instead..."
    pip install llama-cpp-python --no-cache-dir
    echo "CPU build installed. For GPU later: bash enable_gpu_llm.sh"
fi

# ---------------------------------------------------------------
# 5/5 — Models (Piper + Kokoro TTS, Gemma 4 LLM, Whisper STT)
# ---------------------------------------------------------------
echo ""
echo "=== [5/5] Models ==="
bash "$SCRIPT_DIR/download_models.sh"

echo ""
echo "=== Setup complete — everything is installed. Run the assistant now: ==="
echo ""
echo "    $SCRIPT_DIR/.venv/bin/python3 assistant.py"
echo ""
echo "Optional: bash install_service.sh   # control panel + run-on-startup"
