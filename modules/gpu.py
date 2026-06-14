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
import time
import ctypes
import shutil
import subprocess

import config

_ENV_VAR = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

# MPS thread % currently in effect for THIS process (so wake-time re-capping only
# resets the CUDA context when the value actually changes). Tracks what we last
# applied; the driver binds it at primary-context creation (i.e. next model load).
_active_percent: int = 100


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
    """Constrain this process to config.GPU_COMPUTE_PERCENT of GPU compute, and —
    when the adaptive throttle is on — bring MPS up even at 100% so the process is
    an MPS client from its first CUDA touch, which is what lets set_compute_percent
    re-cap it live at wake. No-op when there's no NVIDIA GPU. Call once, before the
    model is loaded."""
    global _active_percent
    pct = config.GPU_COMPUTE_PERCENT
    if pct >= 100 and not config.GPU_ADAPTIVE_THROTTLE:
        return   # no static limit and no adaptive throttle — nothing to set up
    if shutil.which("nvidia-smi") is None:
        return   # no NVIDIA GPU on this machine — nothing to throttle
    if not _start_mps_daemon():
        return   # couldn't enable MPS — leave the process unthrottled
    os.environ[_ENV_VAR] = str(pct)
    _active_percent = pct
    if pct < 100:
        print(f"[GPU] Compute capped to {pct}% of the GPU via MPS (this process only).")
    else:
        print("[GPU] MPS enabled (adaptive throttle ready); no static cap.")


def current_percent() -> int:
    """The MPS thread % last applied to this process."""
    return _active_percent


def _smi(query: str) -> list[int]:
    """Run one nvidia-smi --query-gpu and return its integer values (per device)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if r.returncode != 0:
        return []
    return [int(x) for x in r.stdout.split() if x.strip().lstrip("-").isdigit()]


def sample_load(seconds: float):
    """Average GPU utilization (%) over ~`seconds`, plus the current free/total
    VRAM (MB) of the largest device. Returns (avg_util, free_mb, total_mb), or
    None if there's no usable NVIDIA GPU. Cleo is unloaded while asleep, so the
    utilization measured here reflects OTHER apps using the card."""
    if shutil.which("nvidia-smi") is None:
        return None
    interval = 0.5
    n = max(1, round(seconds / interval))
    utils = []
    for i in range(n):
        u = _smi("utilization.gpu")
        if u:
            utils.append(max(u))
        if i < n - 1:
            time.sleep(interval)
    free, total = _smi("memory.free"), _smi("memory.total")
    if not utils or not free or not total:
        return None
    return round(sum(utils) / len(utils)), max(free), max(total)


def free_mb():
    """Free VRAM (MB) on the largest device right now, or None if no usable GPU."""
    vals = _smi("memory.free")
    return max(vals) if vals else None


def free_mb():
    """Free VRAM (MB) on the largest CUDA device right now, or None if there's no
    usable NVIDIA GPU."""
    if shutil.which("nvidia-smi") is None:
        return None
    vals = _smi("memory.free")
    return max(vals) if vals else None


def adaptive_compute_percent(avg_util: int):
    """The compute cap to apply at wake given the recent average GPU utilization
    of OTHER apps. None when no extra throttle is warranted — the GPU is mostly
    idle, or the free headroom already meets the user's baseline cap."""
    if avg_util < config.GPU_INTERFERENCE_PERCENT:
        return None                       # GPU mostly free — run at baseline
    headroom = max(config.GPU_MIN_COMPUTE_PERCENT, 100 - avg_util)
    cap = min(config.GPU_COMPUTE_PERCENT, headroom)
    return cap if cap < config.GPU_COMPUTE_PERCENT else None


def _reset_primary_context() -> bool:
    """Destroy this process's CUDA primary context so the next model load creates
    a fresh one that picks up the current MPS thread-% env var. Uses the CUDA
    driver API (libcuda) — version-independent, unlike the runtime — and is safe
    only when no GPU work is live in this process (true at wake: the LLM and
    Whisper are unloaded). Returns True on success."""
    cuda = None
    for lib in ("libcuda.so", "libcuda.so.1"):
        try:
            cuda = ctypes.CDLL(lib)
            break
        except OSError:
            continue
    if cuda is None:
        print("[GPU] libcuda not found — can't rebind the MPS cap without a restart.")
        return False
    try:
        if cuda.cuInit(0) != 0:
            return False
        dev = ctypes.c_int(0)
        if cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return False
        rc = cuda.cuDevicePrimaryCtxReset(dev)
        if rc != 0:
            print(f"[GPU] cuDevicePrimaryCtxReset failed (rc={rc}).")
            return False
        return True
    except (OSError, AttributeError) as e:
        print(f"[GPU] CUDA context reset error: {e}.")
        return False


def set_compute_percent(pct: int) -> bool:
    """Re-cap this process's GPU compute to `pct`% at runtime (used at wake).
    Ensures MPS is up, sets the per-process thread %, and resets the CUDA primary
    context so the value binds on the next model load. Returns True only if it was
    actually applied — callers should announce a throttle only on True. Call while
    the LLM/Whisper are unloaded (i.e. before warmup), or the reset is unsafe."""
    global _active_percent
    if shutil.which("nvidia-smi") is None:
        return False
    if not _start_mps_daemon():
        return False
    os.environ[_ENV_VAR] = str(pct)
    if not _reset_primary_context():
        return False                      # cap won't bind without a fresh context
    _active_percent = pct
    return True


def status() -> str:
    """One-line human-readable summary of the active cap (for logs/UI)."""
    pct = config.GPU_COMPUTE_PERCENT
    if pct >= 100:
        return "GPU compute: unlimited"
    if os.environ.get(_ENV_VAR) == str(pct):
        return f"GPU compute: capped at {pct}% (MPS active)"
    return f"GPU compute: {pct}% requested (not active — restart to apply)"
