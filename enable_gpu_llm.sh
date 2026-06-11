#!/usr/bin/env bash
# Enable GPU inference for the local LLM (llama-cpp-python) on this machine.
#
# Why this is needed: CUDA 13.1's headers collide with glibc 2.41's new rsqrt
# declaration, so a CUDA build of llama-cpp-python fails with
#   "exception specification is incompatible with that of previous function rsqrt".
# The fix (verified at the compiler level) is to add __THROW to CUDA's two rsqrt
# host declarations so their exception spec matches glibc's. One sudo edit.
#
# Run this in your own terminal (needs sudo for the one header edit):
#     bash enable_gpu_llm.sh
set -e

HDR="/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h"
CFG="$(cd "$(dirname "$0")" && pwd)/config.py"

echo "=== 1/3  Patching CUDA header for the glibc 2.41 rsqrt conflict ==="
if grep -qE 'rsqrt\(double x\)[[:space:]]+__THROW' "$HDR"; then
    echo "    already patched — skipping."
else
    sudo cp -n "$HDR" "$HDR.orig"          # one-time backup
    sudo sed -i -E \
        -e 's/(double[[:space:]]+rsqrt\(double x\));/\1 __THROW;/' \
        -e 's/(float[[:space:]]+rsqrtf\(float x\));/\1 __THROW;/' \
        "$HDR"
    echo "    patched (backup at $HDR.orig)."
fi

echo "=== 2/3  Rebuilding llama-cpp-python with CUDA (a few minutes) ==="
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DGGML_CUDA=on" \
    pip install --break-system-packages --force-reinstall --no-cache-dir llama-cpp-python

echo "=== 3/3  Switching config.py to full GPU offload ==="
sed -i -E 's/^LLM_N_GPU_LAYERS = 0\b/LLM_N_GPU_LAYERS = -1/' "$CFG"
grep -E '^LLM_N_GPU_LAYERS' "$CFG"

cat <<'EOF'

=== Done — LLM now runs on the RTX 5080 ===
Backend "auto"/"llamacpp" will offload all layers to the GPU.
Restart the assistant (or: systemctl --user restart voice-assistant), then watch
`nvidia-smi` during a reply to confirm VRAM usage.

To undo the header patch:  sudo cp <hdr>.orig <hdr>   (path printed above)
EOF
