"""
GPU compute throttling via NVIDIA MPS.

Caps the share of the GPU's compute units (SMs) Cleo's process may use, so she
doesn't peg the card while other apps/games are running. This limits *compute*,
not VRAM. It works by:
  1. starting the per-user MPS control daemon (if not already up), and
  2. setting CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for THIS process only — so only
     Cleo is throttled; anything else you run on the GPU is untouched.

The env var is read by the CUDA driver when the process first touches the GPU,
so apply_compute_limit() MUST run before any CUDA work (i.e. before the model
loads). That also means a changed percentage only takes effect on restart.
"""
import os
import shutil
import subprocess

import config

_ENV_VAR = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"


def _mps_daemon_running() -> bool:
    try:
        return subprocess.run(["pgrep", "-x", "nvidia-cuda-mps-control"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


def _start_mps_daemon() -> bool:
    """Start the MPS control daemon (idempotent). Returns True if it's up."""
    if _mps_daemon_running():
        return True
    if shutil.which("nvidia-cuda-mps-control") is None:
        print("[GPU] MPS not installed — cannot cap GPU compute. Running unthrottled.")
        return False
    try:
        # -d daemonizes and returns immediately; uses the default pipe dir
        # (/tmp/nvidia-mps), which clients also default to, so they connect.
        subprocess.run(["nvidia-cuda-mps-control", "-d"],
                       capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[GPU] Could not start MPS daemon: {e}. Running unthrottled.")
        return False
    return _mps_daemon_running()


def apply_compute_limit() -> None:
    """Constrain this process to config.GPU_COMPUTE_PERCENT of GPU compute.
    No-op at 100 (no limit) or when there's no NVIDIA GPU. Call once, before
    the model is loaded."""
    pct = config.GPU_COMPUTE_PERCENT
    if pct >= 100:
        return   # no limit requested
    if shutil.which("nvidia-smi") is None:
        return   # no NVIDIA GPU on this machine — nothing to throttle
    if not _start_mps_daemon():
        return   # couldn't enable MPS — leave the process unthrottled
    os.environ[_ENV_VAR] = str(pct)
    print(f"[GPU] Compute capped to {pct}% of the GPU via MPS (this process only).")


def status() -> str:
    """One-line human-readable summary of the active cap (for logs/UI)."""
    pct = config.GPU_COMPUTE_PERCENT
    if pct >= 100:
        return "GPU compute: unlimited"
    if os.environ.get(_ENV_VAR) == str(pct):
        return f"GPU compute: capped at {pct}% (MPS active)"
    return f"GPU compute: {pct}% requested (not active — restart to apply)"
