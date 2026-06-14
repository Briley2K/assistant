#!/usr/bin/env python3
"""
Voice Assistant control panel — a small local web app (http://localhost:5005).

Lets you start/stop the assistant, toggle run-on-boot (systemd user service),
edit the pre-prompt, choose the wake word, and manage the LLM / remote-API
connection. All settings are persisted to settings.json, which config.py reads.
"""
import os
import sys
import time
import json
import threading
import subprocess

from flask import Flask, request, redirect, url_for, render_template_string

BASE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE, "settings.json")
CHATLOG_PATH = os.path.join(BASE, "logs", "chat.jsonl")   # written by modules/chatlog.py
LIVE_PATH = os.path.join(BASE, "logs", "live.json")       # written by modules/live.py
SERVICE = "voice-assistant.service"
PANEL_SERVICE = "voice-assistant-panel.service"   # this control panel itself

BUNDLED_WAKE_WORDS = ["hey jarvis", "alexa", "hey mycroft", "hey marvin", "timer"]
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
BACKENDS = ["auto", "ollama", "llamacpp", "api"]
LLM_DEVICES = ["auto", "gpu", "cpu"]
STT_MODES = ["native", "whisper"]
TTS_ENGINES = ["kokoro", "piper", "neutts", "chatterbox"]
NEUTTS_BACKBONES = ["neuphonic/neutts-air-q4-gguf", "neuphonic/neutts-air-q8-gguf"]


def neutts_voice_names():
    """Bundled NeuTTS reference-voice names, from config (or a default)."""
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        return config.neutts_voices()
    except Exception:
        return ["jo"]


def skill_list():
    """(name, description) for every registered skill, for the panel."""
    try:
        import sys
        sys.path.insert(0, BASE)
        from modules import skills
        return [(s.name, s.description) for s in skills.all_skills()]
    except Exception:
        return []


def _custom_files(dir_path: str, exts: tuple) -> list:
    """Sorted filenames in a custom-model dir matching the given extensions."""
    try:
        return sorted(f for f in os.listdir(dir_path)
                      if f.lower().endswith(exts) and os.path.isfile(os.path.join(dir_path, f)))
    except OSError:
        return []


def llm_models():
    """(key, label) for every selectable LLM: registry models, any uploaded custom
    GGUFs (key 'custom:<file>'), plus a 'None' option (image-only mode)."""
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        items = [(k, v["label"]) for k, v in config.LLM_MODELS.items()]
        items += [("custom:" + f, "Custom: " + f)
                  for f in _custom_files(config.CUSTOM_GGUF_DIR, (".gguf",))]
    except Exception:
        items = [("gemma4-12b", "Gemma 4 12B")]
    return [("none", "None — no language model (image-only)")] + items


def image_models():
    """(key, label) for image models: catalog, any uploaded custom checkpoints
    (key 'custom:<file>'), plus a 'None' option."""
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        items = [(k, v["label"]) for k, v in config.IMAGEGEN_MODELS.items()]
        items += [("custom:" + f, "Custom: " + f)
                  for f in _custom_files(config.CUSTOM_IMAGE_DIR, (".safetensors", ".ckpt"))]
    except Exception:
        items = []
    return [("none", "None — image generation off")] + items


def image_meta() -> dict:
    """{key: {vram, size}} for the image models, for the panel's VRAM hint JS."""
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        return {k: {"vram": v.get("vram_gb"), "size": v.get("size")}
                for k, v in config.IMAGEGEN_MODELS.items()}
    except Exception:
        return {}


def video_models():
    """(key, label) for video models: catalog, uploaded custom checkpoints
    (key 'custom:<file>'), plus a 'None' option."""
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        items = [(k, v["label"]) for k, v in config.VIDEOGEN_MODELS.items()]
        items += [("custom:" + f, "Custom: " + f)
                  for f in _custom_files(config.CUSTOM_VIDEO_DIR, (".safetensors", ".ckpt"))]
    except Exception:
        items = []
    return [("none", "None — video generation off")] + items


def video_meta() -> dict:
    """{key: {vram, size}} for video models, for the panel's VRAM hint JS."""
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        return {k: {"vram": v.get("vram_gb"), "size": v.get("size")}
                for k, v in config.VIDEOGEN_MODELS.items()}
    except Exception:
        return {}


_IMAGE_DL_MARKER = os.path.join(BASE, "logs", ".image_downloading")


def _hf_repo_dir(repo: str) -> str:
    """The HuggingFace hub cache folder for a repo (shared by the main env and the
    image helper venv)."""
    base = (os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"))
    return os.path.join(base, "models--" + repo.replace("/", "--"))


def _image_downloaded(repo: str) -> bool:
    """True if the model's weights are cached: a completed diffusers snapshot
    (model_index.json present) with no in-flight .incomplete blobs."""
    if not repo:
        return False
    d = _hf_repo_dir(repo)
    blobs, snaps = os.path.join(d, "blobs"), os.path.join(d, "snapshots")
    if not os.path.isdir(snaps):
        return False
    try:
        if os.path.isdir(blobs) and any(f.endswith(".incomplete") for f in os.listdir(blobs)):
            return False
        for rev in os.listdir(snaps):
            if os.path.exists(os.path.join(snaps, rev, "model_index.json")):
                return True
    except OSError:
        pass
    return False


def _image_downloaded_bytes(repo: str) -> int:
    """Bytes present in the repo's blob cache (counts .incomplete partials)."""
    blobs = os.path.join(_hf_repo_dir(repo), "blobs")
    total = 0
    try:
        for fn in os.listdir(blobs):
            try:
                total += os.path.getsize(os.path.join(blobs, fn))
            except OSError:
                pass
    except OSError:
        pass
    return total


def image_status(s: dict) -> dict:
    """Status of the image-gen backend for the panel: selected model, whether its
    weights are downloaded / downloading, and whether the helper venv is installed
    and the model is loaded / loading / errored."""
    info = {"model": s.get("image_model", "none"), "enabled": False,
            "installed": False, "ready": False, "loading": False,
            "device": s.get("image_device", "cuda"), "error": None, "repo": None,
            "downloaded": False, "downloading": False, "custom": False,
            "dl_gb": None, "downloaded_gb": None, "pct": None}
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        model = info["model"]
        info["installed"] = (os.path.exists(config.IMAGEGEN_PYTHON)
                             and os.path.exists(config.IMAGEGEN_SERVER))

        # Uploaded custom checkpoint: local file, already "downloaded" if present.
        if model.startswith("custom:"):
            fn = model.split(":", 1)[1]
            path = os.path.join(config.CUSTOM_IMAGE_DIR, fn)
            info["custom"] = True
            info["enabled"] = True
            info["repo"] = path
            info["downloaded"] = os.path.exists(path)
            if info["downloaded"] and info["installed"]:
                from modules import imagegen
                h = imagegen._health()
                if h is not None and h.get("model") == path:
                    info["ready"] = bool(h.get("ready"))
                    info["loading"] = not h.get("ready") and not h.get("error")
                    info["error"] = h.get("error")
            return info

        entry = config.IMAGEGEN_MODELS.get(model) if model != "none" else None
        info["enabled"] = entry is not None
        info["repo"] = entry["repo"] if entry else None
        repo = info["repo"]
        if not info["enabled"] or not repo:
            return info
        info["dl_gb"] = entry.get("dl_gb")

        # An in-flight download is marked by logs/.image_downloading (its repo id).
        marker_repo = None
        try:
            with open(_IMAGE_DL_MARKER) as f:
                marker_repo = f.read().strip()
        except OSError:
            pass
        if marker_repo == repo:
            info["downloading"] = True
            got = _image_downloaded_bytes(repo)
            info["downloaded_gb"] = round(got / 1e9, 2)
            if info["dl_gb"]:
                info["pct"] = max(0, min(99, round(got / (info["dl_gb"] * 1e9) * 100)))
            return info

        info["downloaded"] = _image_downloaded(repo)
        if info["downloaded"] and info["installed"]:
            from modules import imagegen
            h = imagegen._health()           # None if not running yet
            if h is not None and h.get("model") == repo:   # ignore a helper on a different model
                info["ready"] = bool(h.get("ready"))
                info["loading"] = not h.get("ready") and not h.get("error")
                info["error"] = h.get("error")
    except Exception:
        pass
    return info


_VIDEO_DL_MARKER = os.path.join(BASE, "logs", ".video_downloading")


def video_status(s: dict) -> dict:
    """Status of the video-gen backend (mirrors image_status): selected model,
    weights downloaded / downloading, helper installed / loaded / loading / error."""
    info = {"model": s.get("video_model", "none"), "enabled": False,
            "installed": False, "ready": False, "loading": False,
            "device": s.get("video_device", "cuda"), "error": None, "repo": None,
            "downloaded": False, "downloading": False, "custom": False,
            "dl_gb": None, "downloaded_gb": None, "pct": None}
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        model = info["model"]
        info["installed"] = (os.path.exists(config.VIDEOGEN_PYTHON)
                             and os.path.exists(config.VIDEOGEN_SERVER))
        if model == "none":
            return info

        if model.startswith("custom:"):
            fn = model.split(":", 1)[1]
            path = os.path.join(config.CUSTOM_VIDEO_DIR, fn)
            info.update(custom=True, enabled=True, repo=path,
                        downloaded=os.path.exists(path))
            if info["downloaded"] and info["installed"]:
                from modules import videogen
                h = videogen._health()
                if h is not None and h.get("model") == path:
                    info["ready"] = bool(h.get("ready"))
                    info["loading"] = not h.get("ready") and not h.get("error")
                    info["error"] = h.get("error")
            return info

        entry = config.VIDEOGEN_MODELS.get(model)
        info["enabled"] = entry is not None
        if not entry:
            return info
        repo = entry["repo"]
        info["repo"] = repo
        info["dl_gb"] = entry.get("dl_gb")

        marker_repo = None
        try:
            with open(_VIDEO_DL_MARKER) as f:
                marker_repo = f.read().strip()
        except OSError:
            pass
        if marker_repo == repo:
            info["downloading"] = True
            got = _image_downloaded_bytes(repo)        # same HF-cache blob scan
            info["downloaded_gb"] = round(got / 1e9, 2)
            if info["dl_gb"]:
                info["pct"] = max(0, min(99, round(got / (info["dl_gb"] * 1e9) * 100)))
            return info

        info["downloaded"] = _image_downloaded(repo)   # same diffusers-snapshot check
        if info["downloaded"] and info["installed"]:
            from modules import videogen
            h = videogen._health()
            if h is not None and h.get("model") == repo:
                info["ready"] = bool(h.get("ready"))
                info["loading"] = not h.get("ready") and not h.get("error")
                info["error"] = h.get("error")
    except Exception:
        pass
    return info


def _selected_quant_name(s: dict, key: str):
    """The quant chosen for `key` in live settings, else its default, else None.
    Resolved from the passed-in settings (not config._S, which is import-time)."""
    import config
    quants = config.model_quants(key)
    if not quants:
        return None
    want = (s.get("model_quants") or {}).get(key)
    match = next((q for q in quants if q["name"] == want), None)
    if not match:
        match = next((q for q in quants if q.get("default")), quants[0])
    return match["name"]


def model_status(s: dict) -> dict:
    """Whether the selected model's local GGUF is present / downloading, read
    from live settings + the static registry (not cached config state). The GGUF
    is resolved for the selected quant when the model has variants."""
    info = {"key": s.get("llm_model", "gemma4-12b"), "present": False,
            "size_gb": None, "downloading": False, "downloadable": False,
            "label": "", "has_quants": False, "quant": None, "none": False, "custom": False}
    if info["key"] == "none":
        info["none"] = True
        info["label"] = "No language model"
        return info
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        key = info["key"]
        if key.startswith("custom:"):
            fn = key.split(":", 1)[1]
            p = os.path.join(config.CUSTOM_GGUF_DIR, fn)
            info["label"] = "Custom: " + fn
            info["present"] = os.path.exists(p)
            info["custom"] = True
            if info["present"]:
                info["size_gb"] = round(os.path.getsize(p) / 1e9, 1)
            return info
        m = config.LLM_MODELS.get(key) or config.LLM_MODELS["gemma4-12b"]
        info["label"] = m.get("label", key)
        info["downloadable"] = bool(m.get("hf_repo"))
        quants = config.model_quants(key)
        info["has_quants"] = bool(quants)
        if quants:
            qname = _selected_quant_name(s, key)
            info["quant"] = qname
            gguf = next((q["gguf"] for q in quants if q["name"] == qname), m["gguf"])
        else:
            gguf = m["gguf"]
        found = next((os.path.join(d, gguf) for d in m["dirs"]
                      if os.path.exists(os.path.join(d, gguf))), None)
        info["present"] = found is not None
        if found:
            info["size_gb"] = round(os.path.getsize(found) / 1e9, 1)
        info["downloading"] = os.path.exists(os.path.join(m["dirs"][0], ".downloading"))
    except Exception:
        pass
    return info


def gpu_total_gb():
    """Total VRAM (GB) of the largest CUDA device, or None if no usable GPU."""
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        mb = config.gpu_total_mb()
        return round(mb / 1024, 1) if mb else None
    except Exception:
        return None


def quant_data() -> dict:
    """Per-model data the VRAM-fit bar needs in JS: each model's quant sizes
    (and whether each is already downloaded), the co-loaded mmproj size for
    native-audio models, the fixed VRAM overhead, and — for single-file models —
    the on-disk size if present. Keyed by model key."""
    out = {}
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        for key, m in config.LLM_MODELS.items():
            native = bool(m.get("mmproj"))
            mmproj_gb = 0.0
            if native:
                for d in m["dirs"]:
                    p = os.path.join(d, m["mmproj"])
                    if os.path.exists(p):
                        mmproj_gb = round(os.path.getsize(p) / 1e9, 2)
                        break
            quants = []
            for q in config.model_quants(key):
                present = any(os.path.exists(os.path.join(d, q["gguf"])) for d in m["dirs"])
                quants.append({"name": q["name"], "gb": q.get("size_gb"),
                               "present": present, "def": bool(q.get("default"))})
            file_gb = None
            if not quants:
                for d in m["dirs"]:
                    p = os.path.join(d, m["gguf"])
                    if os.path.exists(p):
                        file_gb = round(os.path.getsize(p) / 1e9, 2)
                        break
            out[key] = {"native": native, "mmproj_gb": mmproj_gb,
                        "overhead_gb": round(config.VRAM_OVERHEAD_GB, 2),
                        "quants": quants, "file_gb": file_gb}
        # Uploaded custom GGUFs: single-file, text-only; size from the file.
        for f in _custom_files(config.CUSTOM_GGUF_DIR, (".gguf",)):
            p = os.path.join(config.CUSTOM_GGUF_DIR, f)
            out["custom:" + f] = {
                "native": False, "mmproj_gb": 0.0,
                "overhead_gb": round(config.VRAM_OVERHEAD_GB, 2),
                "quants": [], "file_gb": round(os.path.getsize(p) / 1e9, 2)}
    except Exception:
        pass
    return out


def file_blacklist_default() -> list:
    """The default read-access blacklist from config, shown pre-filled in the
    panel so the user keeps the secret-shielding patterns unless they edit them."""
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        return config.DEFAULT_FILE_BLACKLIST
    except Exception:
        return []


def kokoro_voice_names():
    import numpy as np
    try:
        return sorted(np.load(os.path.join(BASE, "models", "kokoro", "voices-v1.0.npz")).keys())
    except Exception:
        return ["af_heart"]

app = Flask(__name__)
# Custom model uploads are multi-GB (GGUFs / SD checkpoints) — lift the request
# size cap. Werkzeug spools the upload to a temp file on disk, so memory is fine.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024 * 1024   # 64 GB


# --------------------------------------------------------------------------
# Settings + systemd helpers
# --------------------------------------------------------------------------
def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(s: dict) -> None:
    with open(SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=2)


def systemctl(*args) -> tuple[int, str]:
    try:
        r = subprocess.run(
            ["systemctl", "--user", *args],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def service_state() -> dict:
    _, active = systemctl("is-active", SERVICE)
    _, enabled = systemctl("is-enabled", SERVICE)
    installed = os.path.exists(
        os.path.expanduser(f"~/.config/systemd/user/{SERVICE}")
    )
    return {"active": active, "enabled": enabled, "installed": installed}


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    import config
    return render_template_string(
        PAGE,
        s=load_settings(),
        state=service_state(),
        bundled_wake_words=BUNDLED_WAKE_WORDS,
        whisper_models=WHISPER_MODELS,
        backends=BACKENDS,
        llm_models=llm_models(),
        image_models=image_models(),
        image_status=image_status(load_settings()),
        image_meta=image_meta(),
        video_models=video_models(),
        video_status=video_status(load_settings()),
        video_meta=video_meta(),
        model_status=model_status(load_settings()),
        gpu_total_gb=gpu_total_gb(),
        quant_data=quant_data(),
        chatterbox_vram_gb=config.CHATTERBOX_VRAM_GB,
        llm_devices=LLM_DEVICES,
        stt_modes=STT_MODES,
        tts_engines=TTS_ENGINES,
        kokoro_voices=kokoro_voice_names(),
        neutts_voices=neutts_voice_names(),
        neutts_backbones=NEUTTS_BACKBONES,
        skills=skill_list(),
        disabled=set(load_settings().get("skills_disabled", [])),
        file_blacklist_default=file_blacklist_default(),
        msg=request.args.get("msg", ""),
    )


@app.route("/save", methods=["POST"])
def save():
    f = request.form
    s = load_settings()
    s["system_prompt"]       = f.get("system_prompt", "").strip()
    s["stt_mode"]            = f.get("stt_mode", "native")
    s["tts_engine"]          = f.get("tts_engine", "kokoro")
    s["kokoro_voice"]        = f.get("kokoro_voice", "af_heart")
    s["neutts_voice"]        = f.get("neutts_voice", "jo")
    s["neutts_backbone"]     = f.get("neutts_backbone", "neuphonic/neutts-air-q4-gguf")
    s["kokoro_speed"]        = float(f.get("kokoro_speed", 1.0))
    for _k, _d in (("chatterbox_exaggeration", 0.5), ("chatterbox_cfg", 0.5),
                   ("chatterbox_temperature", 0.8)):
        try:
            s[_k] = float(f.get(_k, _d))
        except (TypeError, ValueError):
            s[_k] = _d
    try:
        s["tts_volume"]      = max(0.0, min(1.0, int(f.get("tts_volume", 100)) / 100.0))
    except (TypeError, ValueError):
        s["tts_volume"]      = 1.0
    s["wake_word"]           = " ".join(f.get("wake_word", "").lower().split()) or "hey cleo"
    s["wake_word_threshold"] = float(f.get("wake_word_threshold", 0.5))
    s["overlay"]             = bool(f.get("overlay"))
    s["sleep_command"]       = f.get("sleep_command", "").strip() or "go to sleep"
    s["wake_command"]        = f.get("wake_command", "").strip() or "wake up"
    s["sleep_reply"]         = f.get("sleep_reply", "").strip() or "Going to sleep."
    s["wake_reply"]          = f.get("wake_reply", "").strip() or "Awake and ready."
    s["llm_model"]           = f.get("llm_model", "gemma4-12b")
    s["image_model"]         = f.get("image_model", "none")
    s["image_device"]        = f.get("image_device", "cuda")
    s["video_model"]         = f.get("video_model", "none")
    s["video_device"]        = f.get("video_device", "cuda")
    # Per-model quant choice (only for models that ship variants). Keep prior
    # choices for other models; only update the one currently selected.
    _quant = f.get("llm_quant", "").strip()
    if _quant:
        _mq = dict(s.get("model_quants") or {})
        _mq[s["llm_model"]] = _quant
        s["model_quants"] = _mq
    s["followup_mode"]       = bool(f.get("followup_mode"))
    s["followup_prompt"]     = f.get("followup_prompt", "").strip() or "Is there anything else I can help with?"
    s["followup_signoff"]    = f.get("followup_signoff", "").strip() or "Okay, I'll be here if you need me."
    s["screen_view"]         = bool(f.get("screen_view"))
    s["side_panel"]          = bool(f.get("side_panel"))
    s["llm_backend"]         = f.get("llm_backend", "auto")
    s["llm_device"]          = f.get("llm_device", "auto")
    s["llm_gpu_layers"]      = f.get("llm_gpu_layers", "").strip()
    try:
        s["gpu_compute_percent"] = max(10, min(100, int(f.get("gpu_compute_percent", 100))))
    except (TypeError, ValueError):
        s["gpu_compute_percent"] = 100
    try:
        s["cpu_compute_percent"] = max(10, min(100, int(f.get("cpu_compute_percent", 100))))
    except (TypeError, ValueError):
        s["cpu_compute_percent"] = 100

    s["skills"]        = bool(f.get("skills"))
    enabled_skills     = set(f.getlist("skill_on"))
    all_skill_names    = [n for n, _ in skill_list()]
    s["skills_disabled"] = [n for n in all_skill_names if n not in enabled_skills]
    s["weather_place"] = f.get("weather_place", "").strip()
    s["weather_units"] = f.get("weather_units", "fahrenheit")

    # File access (read-only): master switch + allowed roots / blacklist, one
    # path-or-glob per line in each textarea.
    def _lines(name):
        return [ln.strip() for ln in f.get(name, "").splitlines() if ln.strip()]
    s["file_access"]           = bool(f.get("file_access"))
    s["file_access_roots"]     = _lines("file_access_roots")
    s["file_access_blacklist"] = _lines("file_access_blacklist")
    try:
        s["file_access_max_kb"] = max(1, min(4096, int(f.get("file_access_max_kb", 256))))
    except (TypeError, ValueError):
        s["file_access_max_kb"] = 256
    s["ollama_host"]         = f.get("ollama_host", "").strip()
    s["ollama_model"]        = f.get("ollama_model", "").strip()
    s["whisper_model"]       = f.get("whisper_model", "small")

    api = s.get("api", {})
    api["base_url"] = f.get("api_base_url", "").strip()
    api["model"]    = f.get("api_model", "").strip()
    new_key = f.get("api_key", "").strip()
    if new_key:                      # blank = keep existing key
        api["api_key"] = new_key
    s["api"] = api

    save_settings(s)

    msg = "Settings saved."
    if f.get("restart") and service_state()["active"] == "active":
        code, _ = systemctl("restart", SERVICE)
        msg += " Assistant restarted." if code == 0 else " (restart failed)"
    return redirect(url_for("index", msg=msg))


@app.route("/test_voice", methods=["POST"])
def test_voice():
    """Synthesize a short sample with the selected voice and play it locally."""
    import sys
    sys.path.insert(0, BASE)
    f = request.form
    engine = f.get("tts_engine", "kokoro")
    sample = "Hello, this is a preview of the selected voice."
    try:
        from modules import audio
        label = engine
        if engine == "kokoro":
            from modules import kokoro_tts
            voice = f.get("kokoro_voice", "af_heart")
            wav = kokoro_tts.synth_wav(sample, voice=voice)
            label = f"kokoro: {voice}"
        elif engine == "neutts":
            from modules import neutts_tts
            voice = f.get("neutts_voice", "jo")
            wav = neutts_tts.synth_wav(sample, voice=voice)   # first call may load the model (~15-30s)
            label = f"neutts: {voice}"
        elif engine == "chatterbox":
            from modules import chatterbox_tts
            voice = f.get("neutts_voice", "sophon_2")         # shares the cloned-voice clips
            wav = chatterbox_tts.synth_wav(sample, voice=voice)   # first call loads the GPU model
            label = f"chatterbox: {voice}"
        else:
            import io, wave
            from modules import tts
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                tts._get_voice().synthesize_wav(sample, wf)
            wav = buf.getvalue()
        audio.play_audio(wav)
        msg = f"Played sample ({label})."
    except Exception as e:
        msg = f"Voice test failed: {e}"
    return redirect(url_for("index", msg=msg))


@app.route("/neutts_upload", methods=["POST"])
def neutts_upload():
    """Add a custom NeuTTS reference voice. The uploaded clip is downmixed to
    mono and trimmed to NeuTTS' usable length (a too-long reference overflows the
    model and fails), then transcribed (Whisper) to a MATCHING .txt, and both are
    saved under neutts_test/samples/<name>.{wav,txt} so the voice just works."""
    from flask import jsonify
    import sys
    sys.path.insert(0, BASE)

    f = request.files.get("ref_audio")
    raw = (request.form.get("ref_name") or "").strip().lower()
    name = "".join(c for c in raw.replace(" ", "_") if c.isalnum() or c in "_-")
    if not f or not name:
        return jsonify({"error": "Need an audio file and a name."}), 400

    data = f.read()
    samples = os.path.join(BASE, "neutts_test", "samples")
    os.makedirs(samples, exist_ok=True)
    try:
        from modules import voice_prep
        wav_out, transcript, info = voice_prep.prepare_reference(data)
    except Exception as e:
        return jsonify({"error": f"Couldn't read/transcribe — use a clean audio file. ({e})"}), 400

    with open(os.path.join(samples, f"{name}.wav"), "wb") as out:
        out.write(wav_out)
    with open(os.path.join(samples, f"{name}.txt"), "w") as out:
        out.write(transcript)
    return jsonify({"ok": True, "name": name, "transcript": transcript, **info})


@app.route("/control", methods=["POST"])
def control():
    action = request.form.get("action")
    actions = {
        "start":   ("start", SERVICE),
        "stop":    ("stop", SERVICE),
        "restart": ("restart", SERVICE),
        "enable":  ("enable", SERVICE),
        "disable": ("disable", SERVICE),
    }
    if action not in actions:
        return redirect(url_for("index", msg="Unknown action"))
    code, out = systemctl(*actions[action])
    msg = f"'{action}' ok." if code == 0 else f"'{action}' failed: {out or 'no systemd user session'}"
    return redirect(url_for("index", msg=msg))


def _download_progress(dest_dir: str, gguf: str, total_gb) -> tuple:
    """(downloading, downloaded_bytes, total_bytes) for an in-flight download.
    Progress is read from the partial .incomplete file huggingface_hub writes
    under the dest's cache; total is the registry's expected size."""
    downloading = os.path.exists(os.path.join(dest_dir, ".downloading"))
    downloaded = 0
    cache = os.path.join(dest_dir, ".cache", "huggingface", "download")
    try:
        incs = [os.path.join(cache, f) for f in os.listdir(cache) if f.endswith(".incomplete")]
        if incs:
            downloaded = max(os.path.getsize(p) for p in incs)
    except OSError:
        pass
    final = os.path.join(dest_dir, gguf)
    if os.path.exists(final):
        downloaded = os.path.getsize(final)
    total = int(total_gb * 1e9) if total_gb else None
    return downloading, downloaded, total


@app.route("/model_status.json")
def model_status_json():
    """Live status for a model+quant (present / downloading / progress), so the
    panel can update the Model card without a page reload or a Save."""
    from flask import jsonify
    import sys
    sys.path.insert(0, BASE)
    import config
    key = request.args.get("key") or load_settings().get("llm_model", "gemma4-12b")
    quant = request.args.get("quant", "").strip()
    if key == "none":
        return jsonify({"key": "none", "none": True, "present": False,
                        "downloadable": False, "downloading": False,
                        "has_quants": False, "quant": None})
    if key.startswith("custom:"):
        fn = key.split(":", 1)[1]
        p = os.path.join(config.CUSTOM_GGUF_DIR, fn)
        present = os.path.exists(p)
        return jsonify({"key": key, "custom": True, "present": present,
                        "downloadable": False, "downloading": False,
                        "has_quants": False, "quant": None,
                        "size_gb": round(os.path.getsize(p) / 1e9, 1) if present else None})
    m = config.LLM_MODELS.get(key)
    if not m:
        return jsonify({"error": "unknown model"}), 404
    quants = config.model_quants(key)
    size_gb, gguf = None, m["gguf"]
    if quants:
        q = next((x for x in quants if x["name"] == quant), None) \
            or next((x for x in quants if x.get("default")), quants[0])
        gguf, size_gb, quant = q["gguf"], q.get("size_gb"), q["name"]
    dest = m["dirs"][0]
    found = next((os.path.join(d, gguf) for d in m["dirs"]
                  if os.path.exists(os.path.join(d, gguf))), None)
    downloading, dl_bytes, total_bytes = _download_progress(dest, gguf, size_gb)
    present = found is not None and not downloading
    pct = (max(0, min(100, round(dl_bytes / total_bytes * 100)))
           if downloading and total_bytes else None)
    return jsonify({
        "key": key, "quant": quant or None, "gguf": gguf,
        "present": present,
        "size_gb": round(os.path.getsize(found) / 1e9, 1) if found else size_gb,
        "downloadable": bool(m.get("hf_repo")),
        "has_quants": bool(quants),
        "downloading": downloading,
        "downloaded_gb": round(dl_bytes / 1e9, 2) if downloading else None,
        "total_gb": size_gb,
        "pct": pct,
    })


@app.route("/download_model", methods=["POST"])
def download_model():
    """Start downloading the selected model's GGUF in the background (detached),
    streaming progress to logs/model_download.log. Returns JSON for an AJAX call
    (json=1) so the panel can show a live progress bar; otherwise redirects."""
    import sys
    from flask import jsonify
    # Use the model currently chosen in the dropdown (the form submits with this
    # button), so Download works even before the selection has been saved.
    key = request.form.get("llm_model") or load_settings().get("llm_model", "gemma4-12b")
    quant = request.form.get("llm_quant", "").strip()
    want_json = request.form.get("json") == "1"
    script = os.path.join(BASE, "download_model.py")
    log = os.path.join(BASE, "logs", "model_download.log")
    argv = [sys.executable, script, key] + ([quant] if quant else [])
    try:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        lf = open(log, "w")
        subprocess.Popen(
            argv,
            stdout=lf, stderr=lf, start_new_session=True,
        )
    except OSError as e:
        if want_json:
            return jsonify({"error": str(e)}), 500
        return redirect(url_for("index", msg=f"Couldn't start download: {e}"))
    if want_json:
        return jsonify({"started": True, "key": key, "quant": quant or None})
    return redirect(url_for("index", msg=(
        f"Downloading '{key}'{(' · ' + quant) if quant else ''} in the background — "
        "this can take a while (several GB). "
        "Watch logs/model_download.log; the Model section shows when it's ready.")))


@app.route("/delete_model", methods=["POST"])
def delete_model():
    """Delete a model's local GGUF to free disk space. For a quant model it
    removes only the named quant's file; the registry entry (and re-download)
    stays. Only ever removes a known GGUF filename inside the model's own dirs."""
    from flask import jsonify
    import sys
    sys.path.insert(0, BASE)
    import config
    data = request.get_json(silent=True) or request.form
    key = data.get("key", "")
    quant = (data.get("quant") or "").strip()
    m = config.LLM_MODELS.get(key)
    if not m:
        return jsonify({"error": "unknown model"}), 404
    quants = config.model_quants(key)
    if quants:
        q = next((x for x in quants if x["name"] == quant), None)
        if not q:
            return jsonify({"error": f"unknown quant '{quant}'"}), 400
        gguf = q["gguf"]
    else:
        gguf = m["gguf"]
    removed, freed = [], 0
    for d in m["dirs"]:
        p = os.path.join(d, gguf)
        if os.path.exists(p):
            try:
                freed += os.path.getsize(p)
                os.remove(p)
                removed.append(p)
            except OSError as e:
                return jsonify({"error": str(e)}), 500
    if not removed:
        return jsonify({"ok": True, "removed": [], "note": "nothing to delete"})
    return jsonify({"ok": True, "removed": removed, "freed_gb": round(freed / 1e9, 1)})


@app.route("/restart_panel", methods=["POST"])
def restart_panel():
    """Restart this control panel itself. Restarting synchronously would kill the
    process mid-response, so we hand the restart to a detached child that waits a
    moment first — letting this page (which auto-reloads) return cleanly."""
    try:
        subprocess.Popen(
            ["bash", "-c",
             f"sleep 1; systemctl --user restart {PANEL_SERVICE}"],
            start_new_session=True,   # survive our own death during the restart
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        return redirect(url_for("index", msg=f"Panel restart failed: {e}"))
    return render_template_string(RESTART_PANEL_PAGE)


SCREEN_HELPER = os.path.join(BASE, "screen_capture_helper.py")
SCREEN_TOKEN = os.path.expanduser("~/.cache/cleo_screencast_token.txt")


@app.route("/screen_grant", methods=["POST"])
def screen_grant():
    """Re-trigger the GNOME 'share your screen' dialog so the user can (re)grant
    screen access — picking 'Entire screen' lets Cleo view any monitor by voice.
    Forgets the old grant first, then launches the portal helper detached (it
    blocks waiting for the dialog, so we don't wait on it here)."""
    try:
        os.remove(SCREEN_TOKEN)
    except OSError:
        pass
    try:
        subprocess.Popen(
            ["/usr/bin/python3", SCREEN_HELPER, "--list", "--watchdog", "150"],
            start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        return redirect(url_for("index", msg=f"Couldn't start screen grant: {e}"))
    return redirect(url_for("index", msg=(
        "A 'Share your screen' dialog should appear — pick 'Entire screen' "
        "(or ctrl-click all monitors) and click Share. Cleo will remember it.")))


RESTART_PANEL_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Restarting…</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; max-width: 480px; margin: 6rem auto;
         padding: 0 1rem; background:#0f1117; color:#e8eaf0; text-align:center; }
  body::before { content:""; position:fixed; inset:0; z-index:-1;
    background:radial-gradient(900px 500px at 50% -180px, rgba(77,124,255,.16), transparent 70%); }
  .spin { width:2.2rem; height:2.2rem; margin:1.5rem auto; border:3px solid #262c3a;
          border-top-color:#4d7cff; border-radius:50%; animation:spin 1s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  a { color:#6b9bff; }
</style>
<script>
  // The panel is bouncing; poll until it answers, then go home.
  function ping(){ fetch('/', {cache:'no-store'})
    .then(r => { if (r.ok) location.href = '/'; else setTimeout(ping, 700); })
    .catch(() => setTimeout(ping, 700)); }
  setTimeout(ping, 1500);
</script></head><body>
<h2>♻️ Restarting control panel…</h2>
<div class="spin"></div>
<p>This page will return automatically. If it doesn't, <a href="/">click here</a>.</p>
</body></html>"""


# --------------------------------------------------------------------------
# Conversation log (live chat view)
# --------------------------------------------------------------------------
def read_chat(limit: int = 300) -> list:
    try:
        with open(CHATLOG_PATH) as f:
            lines = f.readlines()[-limit:]
    except FileNotFoundError:
        return []
    turns = []
    for line in lines:
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return turns


@app.route("/chat")
def chat():
    return render_template_string(CHAT_PAGE)


def read_live():
    """The in-progress turn, or None if the assistant is idle/stale."""
    try:
        with open(LIVE_PATH) as f:
            d = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not d or d.get("phase", "idle") == "idle":
        return None
    if time.time() - d.get("ts", 0) > 60:   # assistant likely died mid-turn
        return None
    return d


@app.route("/chat.json")
def chat_json():
    from flask import jsonify
    return jsonify({"turns": read_chat(), "live": read_live()})


@app.route("/chat/image/<path:name>")
def chat_image(name):
    """Serve a generated image (written by modules/imagegen.py into logs/images).
    The chat view references these via [[IMAGE:name]] markers in assistant turns."""
    from flask import send_from_directory, abort
    safe = os.path.basename(name)               # no path traversal
    img_dir = os.path.join(BASE, "logs", "images")
    if not safe or not os.path.exists(os.path.join(img_dir, safe)):
        abort(404)
    return send_from_directory(img_dir, safe)


@app.route("/chat/video/<path:name>")
def chat_video(name):
    """Serve a generated video (written by modules/videogen.py into logs/videos).
    The chat view references these via [[VIDEO:name]] markers in assistant turns."""
    from flask import send_from_directory, abort
    safe = os.path.basename(name)
    vid_dir = os.path.join(BASE, "logs", "videos")
    if not safe or not os.path.exists(os.path.join(vid_dir, safe)):
        abort(404)
    return send_from_directory(vid_dir, safe)


@app.route("/image")
def image_page():
    return render_template_string(IMAGE_PAGE)


@app.route("/image/status.json")
def image_status_json():
    from flask import jsonify
    s = load_settings()
    model = request.args.get("model")        # live dropdown value, before Save
    if model is not None:
        s = dict(s); s["image_model"] = model
    return jsonify(image_status(s))


@app.route("/image/download", methods=["POST"])
def image_download():
    """Pre-download the selected image model's weights into the HF cache in the
    background (detached), streaming progress to logs/image_download.log. Uses the
    model from the dropdown so it works before the selection is saved."""
    from flask import jsonify
    import sys
    sys.path.insert(0, BASE)
    import config
    key = request.form.get("image_model") or load_settings().get("image_model", "none")
    if key not in config.IMAGEGEN_MODELS:
        return jsonify({"error": "Select an image model first."}), 400
    script = os.path.join(BASE, "download_image_model.py")
    log = os.path.join(BASE, "logs", "image_download.log")
    try:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        lf = open(log, "w")
        subprocess.Popen([sys.executable, script, key],
                         stdout=lf, stderr=lf, start_new_session=True)
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"started": True, "key": key})


def _save_upload(field: str, dest_dir: str, exts: tuple):
    """Stream an uploaded model file into dest_dir. Returns (filename, error)."""
    from werkzeug.utils import secure_filename
    f = request.files.get(field)
    if f is None or not f.filename:
        return None, "No file provided."
    name = secure_filename(f.filename)
    if not name.lower().endswith(exts):
        return None, f"File must be one of: {', '.join(exts)}"
    os.makedirs(dest_dir, exist_ok=True)
    try:
        f.save(os.path.join(dest_dir, name))        # streams to disk
    except OSError as e:
        return None, str(e)
    return name, None


@app.route("/llm/upload", methods=["POST"])
def llm_upload():
    """Upload a custom .gguf into models/custom-gguf; it then appears in the LLM
    dropdown as 'Custom: <file>' (key 'custom:<file>')."""
    from flask import jsonify
    import sys
    sys.path.insert(0, BASE)
    import config
    name, err = _save_upload("file", config.CUSTOM_GGUF_DIR, (".gguf",))
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True, "name": name, "key": "custom:" + name})


@app.route("/image/upload", methods=["POST"])
def image_upload():
    """Upload a custom Stable Diffusion checkpoint (.safetensors/.ckpt) into
    models/custom-image; it appears in the image dropdown as 'Custom: <file>'."""
    from flask import jsonify
    import sys
    sys.path.insert(0, BASE)
    import config
    name, err = _save_upload("file", config.CUSTOM_IMAGE_DIR, (".safetensors", ".ckpt"))
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True, "name": name, "key": "custom:" + name})


@app.route("/image/generate", methods=["POST"])
def image_generate():
    """Generate one image from a prompt and return its URL. Used by the standalone
    /image page; works without the assistant or any LLM running."""
    from flask import jsonify
    import sys
    sys.path.insert(0, BASE)
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Enter a prompt."}), 400
    try:
        import config
        if not config.IMAGEGEN_ENABLED:
            return jsonify({"error": "No image model selected — pick one in the panel."}), 400
        from modules import imagegen
        if not (os.path.exists(config.IMAGEGEN_PYTHON) and os.path.exists(config.IMAGEGEN_SERVER)):
            return jsonify({"error": "Image venv not installed — run imagegen/setup_imagegen.sh."}), 400
        kw = {}
        try:
            if int(data.get("steps", 0)) > 0:
                kw["steps"] = int(data["steps"])
        except (TypeError, ValueError):
            pass
        try:
            if data.get("cfg") not in (None, "", 0, "0"):
                kw["guidance"] = float(data["cfg"])
        except (TypeError, ValueError):
            pass
        sampler = (data.get("sampler") or "").strip()
        if sampler and sampler != "default":
            kw["sampler"] = sampler
        for dim in ("width", "height"):
            try:
                v = int(data.get(dim, 0))
                if v > 0:
                    kw[dim] = v
            except (TypeError, ValueError):
                pass
        name = imagegen.save_image(prompt, **kw)
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500
    return jsonify({"ok": True, "url": "/chat/image/" + name, "name": name, "prompt": prompt})


# --------------------------------------------------------------------------
# Video generation (mirrors the image routes)
# --------------------------------------------------------------------------
@app.route("/video")
def video_page():
    return render_template_string(VIDEO_PAGE)


@app.route("/video/status.json")
def video_status_json():
    from flask import jsonify
    s = load_settings()
    model = request.args.get("model")        # live dropdown value, before Save
    if model is not None:
        s = dict(s); s["video_model"] = model
    return jsonify(video_status(s))


@app.route("/video/download", methods=["POST"])
def video_download():
    """Pre-download the selected video model's weights into the HF cache in the
    background (detached), streaming progress to logs/video_download.log."""
    from flask import jsonify
    import sys
    sys.path.insert(0, BASE)
    import config
    key = request.form.get("video_model") or load_settings().get("video_model", "none")
    if key not in config.VIDEOGEN_MODELS:
        return jsonify({"error": "Select a video model first."}), 400
    script = os.path.join(BASE, "download_video_model.py")
    log = os.path.join(BASE, "logs", "video_download.log")
    try:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        lf = open(log, "w")
        subprocess.Popen([sys.executable, script, key],
                         stdout=lf, stderr=lf, start_new_session=True)
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"started": True, "key": key})


@app.route("/video/upload", methods=["POST"])
def video_upload():
    """Upload a custom video checkpoint (.safetensors/.ckpt) into models/custom-video."""
    from flask import jsonify
    import sys
    sys.path.insert(0, BASE)
    import config
    name, err = _save_upload("file", config.CUSTOM_VIDEO_DIR, (".safetensors", ".ckpt"))
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True, "name": name, "key": "custom:" + name})


@app.route("/video/generate", methods=["POST"])
def video_generate():
    """Generate one video from a prompt and return its URL. Used by the /video page;
    works without the assistant or any LLM running."""
    from flask import jsonify
    import sys
    sys.path.insert(0, BASE)
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Enter a prompt."}), 400
    try:
        import config
        if not config.VIDEOGEN_ENABLED:
            return jsonify({"error": "No video model selected — pick one in the panel."}), 400
        from modules import videogen
        if not (os.path.exists(config.VIDEOGEN_PYTHON) and os.path.exists(config.VIDEOGEN_SERVER)):
            return jsonify({"error": "Video venv not installed — run video/setup_video.sh."}), 400
        kw = {}
        try:
            if int(data.get("frames", 0)) > 0:
                kw["frames"] = int(data["frames"])
            if int(data.get("steps", 0)) > 0:
                kw["steps"] = int(data["steps"])
        except (TypeError, ValueError):
            pass
        name = videogen.save_video(prompt, **kw)
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500
    return jsonify({"ok": True, "url": "/chat/video/" + name, "name": name, "prompt": prompt})


@app.route("/chat/clear", methods=["POST"])
def chat_clear():
    try:
        open(CHATLOG_PATH, "w").close()
    except OSError:
        pass
    try:                                  # also drop any queued typed messages
        import sys
        sys.path.insert(0, BASE)
        from modules import textq
        textq.clear()
    except Exception:
        pass
    return ("", 204)


@app.route("/chat/send", methods=["POST"])
def chat_send():
    """Hand a typed message to the running assistant, which answers it through
    its normal pipeline (reusing the already-loaded model — no second copy). We
    log the user turn here so it shows instantly, then queue the text; the
    assistant streams the reply into the live view and logs it when done."""
    from flask import jsonify
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty message."}), 400

    import sys
    sys.path.insert(0, BASE)
    try:
        from modules import chatlog, textq
    except Exception as e:
        return jsonify({"error": f"Could not load the assistant modules: {e}"}), 500

    chatlog.log("user", text)
    textq.send(text)

    # Best-effort heads-up if the assistant clearly isn't running to answer it.
    note = None
    st = service_state()
    if st["installed"] and st["active"] != "active":
        note = ("The assistant service isn't running, so this won't be answered "
                "until you Start it from the panel.")
    return jsonify({"queued": True, "note": note})


# --------------------------------------------------------------------------
# Wake-word voice enrollment (interactive — runs in this process so it can use
# the mic/speakers; the assistant service is paused during it to free the mic).
# --------------------------------------------------------------------------
_enroll = {"running": False, "phase": "idle", "prompt": "",
           "log": [], "summary": None, "error": None}
_enroll_lock = threading.Lock()


def _enroll_set(**fields) -> None:
    with _enroll_lock:
        _enroll.update(fields)


def _enroll_log(kind: str, text: str) -> None:
    with _enroll_lock:
        _enroll["log"].append({"kind": kind, "text": text})


def _enroll_event(name: str, data: dict) -> None:
    """Map enroll_voice() progress events into the shared state the browser polls,
    and speak the user-facing prompts aloud so the user knows when to talk."""
    from modules import tts
    phrase = data.get("phrase", "")
    speak = None
    if name == "intro":
        speak = (f"Let's teach me your voice. When I prompt you, say your wake "
                 f"phrase. We'll do this {data['num_samples']} times.")
        _enroll_set(phase="recording", prompt=speak)
    elif name == "prompt":
        speak = f"Say it now. Number {data['index']} of {data['total']}."
        _enroll_set(phase="recording",
                    prompt=f"🎤 Say your wake phrase now — {data['index']} of {data['total']}")
    elif name == "accepted":
        _enroll_log("ok", f"✓ heard “{data['heard']}”  ({data['index']}/{data['total']})")
    elif name == "rejected":
        speak = "Hmm, that didn't sound right. Let's try again."
        _enroll_log("bad", f"✗ didn't catch that (“{data['heard']}”) — retrying")
    elif name == "no_speech":
        speak = "I didn't hear anything. Let's try again."
        _enroll_log("bad", "✗ no speech detected — retrying")
    elif name == "unsupported":
        _enroll_set(prompt="This phrase uses a built-in model — nothing to train.")
    elif name == "failed":
        _enroll_log("bad", "No usable samples captured.")
    elif name == "done":
        n = len(data.get("samples") or [])
        msg = (f"Learned {n} new pronunciation(s) from your voice."
               if n else "Your voice already matched — no new variants needed.")
        _enroll_log("ok", msg)
    if speak:
        try:
            tts.speak(speak)
        except Exception:
            pass


def _enroll_worker(num_samples: int) -> None:
    restart = False
    try:
        sys.path.insert(0, BASE)
        st = service_state()
        if st["installed"] and st["active"] == "active":
            _enroll_set(phase="preparing",
                        prompt="Pausing the assistant so it doesn't grab the mic…")
            systemctl("stop", SERVICE)
            restart = True
            time.sleep(1.5)            # let it release the mic/audio device

        _enroll_set(phase="preparing", prompt="Loading speech models…")
        from modules import tts, wake_word
        tts.warmup()
        summary = wake_word.enroll_voice(num_samples=num_samples, on_event=_enroll_event)
        if summary.get("ok"):
            _enroll_set(phase="done", summary=summary,
                        prompt="All set — I've learned how you say it.")
        else:
            _enroll_set(phase="error", error=summary.get("error", "enrollment failed"),
                        prompt=summary.get("error", "Enrollment failed."))
    except Exception as e:
        _enroll_set(phase="error", error=str(e), prompt=f"Enrollment failed: {e}")
        _enroll_log("bad", str(e))
    finally:
        if restart:
            systemctl("start", SERVICE)
            _enroll_log("ok", "Restarted the assistant.")
        with _enroll_lock:
            _enroll["running"] = False


@app.route("/wake/enroll/start", methods=["POST"])
def wake_enroll_start():
    from flask import jsonify
    with _enroll_lock:
        if _enroll["running"]:
            return jsonify({"error": "Enrollment already in progress."}), 409
        try:
            num = max(1, min(20, int((request.get_json(silent=True) or {}).get("samples", 5))))
        except (TypeError, ValueError):
            num = 5
        _enroll.update(running=True, phase="preparing", prompt="Starting…",
                       log=[], summary=None, error=None)
    threading.Thread(target=_enroll_worker, args=(num,), daemon=True,
                     name="wake-enroll").start()
    return jsonify({"started": True, "samples": num})


@app.route("/wake/enroll/status")
def wake_enroll_status():
    from flask import jsonify
    with _enroll_lock:
        return jsonify(dict(_enroll))


PAGE = """<!doctype html>
<html lang="en" data-theme="midnight"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Assistant</title>
<script>try{document.documentElement.dataset.theme=localStorage.getItem('cleo_theme')||'midnight'}catch(e){}</script>
<style>
  *{ box-sizing:border-box; }
  :root{ color-scheme:dark; --radius:14px;
    --bg:#0f1117; --card:#1a1e27; --card-border:#262c3a; --card-hover:#39435c;
    --input-bg:#11141c; --input-border:#2e3545; --text:#e8eaf0; --dim:#9aa3b5; --faint:#6b7385;
    --accent:#4d7cff; --accent2:#6b5cff; --accent-text:#fff;
    --ok-bg:#14361f; --ok-text:#5fdd8f; --bad-bg:#3a1a1a; --bad-text:#f08a8a;
    --warn-bg:#3a3320; --warn-text:#e3c878; --msg-bg:#1b2740; --msg-border:#2f4063; --shadow:rgba(0,0,0,.35); }
  [data-theme="nord"]{ color-scheme:dark; --bg:#262b35; --card:#3b4252; --card-border:#434c5e; --card-hover:#4c566a;
    --input-bg:#2e3440; --input-border:#434c5e; --text:#eceff4; --dim:#d8dee9; --faint:#8b95a7;
    --accent:#88c0d0; --accent2:#81a1c1; --accent-text:#2e3440;
    --ok-bg:#3b4a3f; --ok-text:#a3be8c; --bad-bg:#4a3a3d; --bad-text:#bf616a;
    --warn-bg:#4a4334; --warn-text:#ebcb8b; --msg-bg:#3b4252; --msg-border:#4c566a; --shadow:rgba(0,0,0,.3); }
  [data-theme="dracula"]{ color-scheme:dark; --bg:#1b1c25; --card:#282a36; --card-border:#383a4a; --card-hover:#6272a4;
    --input-bg:#1e1f29; --input-border:#44475a; --text:#f8f8f2; --dim:#bcc0d6; --faint:#6272a4;
    --accent:#bd93f9; --accent2:#ff79c6; --accent-text:#21222c;
    --ok-bg:#2d4a37; --ok-text:#50fa7b; --bad-bg:#4a2d33; --bad-text:#ff5555;
    --warn-bg:#4a4530; --warn-text:#f1fa8c; --msg-bg:#343746; --msg-border:#44475a; --shadow:rgba(0,0,0,.4); }
  [data-theme="emerald"]{ color-scheme:dark; --bg:#0c1310; --card:#14211b; --card-border:#21342a; --card-hover:#2f5742;
    --input-bg:#0e1713; --input-border:#243a2e; --text:#e6f0ea; --dim:#9bb3a6; --faint:#6a8579;
    --accent:#2dd4a7; --accent2:#14b8a6; --accent-text:#04241c;
    --ok-bg:#14361f; --ok-text:#5fdd8f; --bad-bg:#3a1a1a; --bad-text:#f08a8a;
    --warn-bg:#3a3320; --warn-text:#e3c878; --msg-bg:#14271f; --msg-border:#25402f; --shadow:rgba(0,0,0,.35); }
  [data-theme="synthwave"]{ color-scheme:dark; --bg:#181029; --card:#241a3a; --card-border:#3d2c5e; --card-hover:#6d4bb0;
    --input-bg:#1c1230; --input-border:#3d2c5e; --text:#f5e6ff; --dim:#c4a8e0; --faint:#8a72b0;
    --accent:#ff2e97; --accent2:#7b5cff; --accent-text:#fff;
    --ok-bg:#243a3a; --ok-text:#3affc8; --bad-bg:#4a2030; --bad-text:#ff6f9d;
    --warn-bg:#3a3320; --warn-text:#ffd86b; --msg-bg:#2a1f4a; --msg-border:#4a3470; --shadow:rgba(255,46,151,.18); }
  [data-theme="light"]{ color-scheme:light; --bg:#eef1f6; --card:#ffffff; --card-border:#dde3ec; --card-hover:#b9c4d6;
    --input-bg:#f6f8fc; --input-border:#d2dae6; --text:#1c2430; --dim:#5a6678; --faint:#8a95a6;
    --accent:#3b6fe0; --accent2:#6b5cff; --accent-text:#fff;
    --ok-bg:#d8f3e2; --ok-text:#1d7a45; --bad-bg:#fbe0e0; --bad-text:#c0392b;
    --warn-bg:#fcf0d0; --warn-text:#8a6d1a; --msg-bg:#e2ecff; --msg-border:#c3d6f7; --shadow:rgba(20,30,50,.12); }

  body{ font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; max-width:820px; margin:0 auto;
        padding:0 1rem 2rem; background:var(--bg); color:var(--text); line-height:1.5; -webkit-font-smoothing:antialiased; }
  body::before{ content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
    background:radial-gradient(1100px 600px at 50% -240px, color-mix(in srgb,var(--accent) 16%,transparent), transparent 70%); }
  ::selection{ background:color-mix(in srgb,var(--accent) 45%,transparent); }
  h1{ font-size:1.35rem; margin:0; letter-spacing:.01em; }
  h2{ font-size:1.02rem; margin:0 0 .6rem; color:var(--text); font-weight:600; display:flex; align-items:center; gap:.5rem; }
  a{ color:var(--accent); text-decoration:none; } a:hover{ text-decoration:underline; }
  code{ background:var(--input-bg); border:1px solid var(--input-border); border-radius:5px; padding:.05rem .35rem; font-size:.82em; }

  .topbar{ position:sticky; top:0; z-index:30; display:flex; align-items:center; gap:1rem; margin:0 -1rem 1.2rem;
    padding:.85rem 1rem; background:color-mix(in srgb,var(--bg) 82%,transparent); backdrop-filter:blur(12px);
    border-bottom:1px solid var(--card-border); }
  .topbar h1{ flex:1; }
  .swatchbar{ display:flex; gap:.45rem; align-items:center; }
  .swatch{ width:22px; height:22px; min-width:0; padding:0; border-radius:50%; cursor:pointer; background:var(--sw,#888);
    border:2px solid transparent; box-shadow:0 1px 4px var(--shadow); transition:transform .12s ease, border-color .12s; }
  .swatch:hover{ transform:scale(1.18); filter:none; }
  .swatch.active{ border-color:var(--text); transform:scale(1.1); }

  .card{ background:var(--card); border:1px solid var(--card-border); border-radius:var(--radius); padding:1.15rem 1.3rem;
    margin:1.1rem 0; box-shadow:0 2px 12px var(--shadow); transition:border-color .2s; }
  .card:hover{ border-color:var(--card-hover); }

  /* Collapsible sections (set up by JS at load — collapsed by default). */
  .card>h2.card-h2{ cursor:pointer; user-select:none; display:flex; align-items:center; gap:.5rem; }
  .card-caret{ display:inline-block; font-size:.7em; opacity:.6; transition:transform .15s; }
  .card:not(.collapsed)>h2.card-h2 .card-caret{ transform:rotate(90deg); }
  .card.collapsed>.card-body{ display:none; }
  /* A lone toggle stays in the header so it works while the section is collapsed. */
  .hdr-switch{ margin-left:auto; }
  .hdr-switch label{ margin:0; color:var(--text); }

  label{ display:block; margin:.75rem 0 .25rem; font-size:.8rem; color:var(--dim); font-weight:500; }
  input,select,textarea{ width:100%; box-sizing:border-box; background:var(--input-bg); color:var(--text);
    border:1px solid var(--input-border); border-radius:9px; padding:.55rem .65rem; font-size:.9rem; font-family:inherit;
    transition:border-color .15s, box-shadow .15s; }
  input:focus,select:focus,textarea:focus{ outline:none; border-color:var(--accent);
    box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 25%,transparent); }
  textarea{ min-height:96px; resize:vertical; line-height:1.5; }
  select{ appearance:none; -webkit-appearance:none; cursor:pointer; padding-right:2rem; background-repeat:no-repeat;
    background-position:right .75rem center;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%23888' stroke-width='1.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E"); }

  input[type="checkbox"]{ appearance:none; -webkit-appearance:none; position:relative; flex:0 0 auto; width:44px !important;
    height:24px; padding:0; border:none; border-radius:999px; cursor:pointer; background:var(--input-border);
    transition:background .2s; vertical-align:middle; box-shadow:none; }
  input[type="checkbox"]::after{ content:""; position:absolute; top:3px; left:3px; width:18px; height:18px; border-radius:50%;
    background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.4); transition:transform .2s; }
  input[type="checkbox"]:checked{ background:var(--accent); }
  input[type="checkbox"]:checked::after{ transform:translateX(20px); }
  input[type="checkbox"]:focus-visible{ outline:none; box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 30%,transparent); }

  input[type="range"]{ -webkit-appearance:none; appearance:none; height:8px; padding:0; border:none; border-radius:999px;
    background:var(--input-border); cursor:pointer; }
  input[type="range"]::-webkit-slider-thumb{ -webkit-appearance:none; width:20px; height:20px; border-radius:50%;
    background:var(--accent); border:3px solid var(--card); box-shadow:0 0 0 1px var(--accent); cursor:pointer; }
  input[type="range"]::-moz-range-thumb{ width:16px; height:16px; border-radius:50%; background:var(--accent);
    border:3px solid var(--card); cursor:pointer; }

  button{ background:linear-gradient(135deg,var(--accent),var(--accent2)); color:var(--accent-text); border:0;
    border-radius:9px; padding:.55rem 1.05rem; font-size:.9rem; font-weight:600; cursor:pointer; font-family:inherit;
    box-shadow:0 2px 8px var(--shadow); transition:transform .08s, filter .2s, box-shadow .2s; }
  button:hover{ transform:translateY(-1px); filter:brightness(1.08); box-shadow:0 5px 16px var(--shadow); }
  button:active{ transform:translateY(0); filter:brightness(.98); }
  button.secondary{ background:var(--input-bg); color:var(--text); border:1px solid var(--input-border); box-shadow:none; }
  button.secondary:hover{ background:var(--card); border-color:var(--accent); filter:none; }
  button.danger{ background:linear-gradient(135deg,#e25a4d,#b3402f); color:#fff; }

  .pill{ display:inline-flex; align-items:center; gap:.3rem; padding:.2rem .65rem; border-radius:999px; font-size:.76rem; font-weight:600; }
  .on{ background:var(--ok-bg); color:var(--ok-text); } .off{ background:var(--bad-bg); color:var(--bad-text); }
  .msg{ background:var(--msg-bg); border:1px solid var(--msg-border); padding:.7rem 1rem; border-radius:10px; margin:.4rem 0 1rem; font-size:.9rem; }
  .btns{ display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.8rem; }
  .hint{ font-size:.78rem; color:var(--faint); margin-top:.35rem; line-height:1.5; }
  .row{ display:flex; gap:1rem; flex-wrap:wrap; } .row>div{ flex:1 1 180px; }

  .vbar{ position:relative; height:18px; border-radius:999px; background:var(--input-bg);
    border:1px solid var(--input-border); overflow:hidden; margin-top:.2rem; }
  .vfill{ height:100%; width:0; border-radius:999px; transition:width .25s ease, background .25s; }
  .vbar.stack{ display:flex; }
  .vbar.stack .vseg{ height:100%; width:0; flex:0 0 auto; transition:width .25s ease, background .25s; }
  .vsw{ display:inline-block; width:.7rem; height:.7rem; border-radius:3px; vertical-align:middle; margin-right:.15rem; }
  .vbar-scale{ display:flex; justify-content:space-between; font-size:.7rem; color:var(--faint); margin-top:.2rem; }

  .savebar{ position:sticky; bottom:0; z-index:20; display:flex; gap:.6rem; flex-wrap:wrap; margin:1.2rem -1rem 0;
    padding:.85rem 1rem; background:color-mix(in srgb,var(--bg) 85%,transparent); backdrop-filter:blur(10px);
    border-top:1px solid var(--card-border); }
  .savebar button{ flex:0 0 auto; }

  ::-webkit-scrollbar{ width:11px; height:11px; }
  ::-webkit-scrollbar-thumb{ background:var(--card-border); border-radius:999px; border:3px solid var(--bg); }
  ::-webkit-scrollbar-thumb:hover{ background:var(--card-hover); }
</style></head><body>
<div class="topbar">
  <h1>🎙️ Voice Assistant</h1>
  <div class="swatchbar" id="themeBar"></div>
</div>
{% if msg %}<div class="msg">{{ msg }}</div>{% endif %}

<div class="card">
  <h2 style="margin-top:0">⚡ Status</h2>
  {% set active = state.active == 'active' %}
  Service: <span class="pill {{ 'on' if active else 'off' }}">{{ state.active }}</span>
  &nbsp; Run on startup:
  <span class="pill {{ 'on' if state.enabled == 'enabled' else 'off' }}">{{ state.enabled }}</span>
  {% if not state.installed %}
    <p class="hint">systemd service not installed yet — run <code>bash install_service.sh</code> once.</p>
  {% endif %}
  <form method="post" action="/control" class="btns">
    <button name="action" value="start">Start</button>
    <button name="action" value="stop" class="danger">Stop</button>
    <button name="action" value="restart" class="secondary">Restart</button>
    <button name="action" value="enable" class="secondary">Enable autostart</button>
    <button name="action" value="disable" class="secondary">Disable autostart</button>
  </form>
  <p class="hint">Autostart runs at login. For start-before-login, also run
     <code>sudo loginctl enable-linger $USER</code>.</p>
  <div class="btns">
    <a href="/chat" target="_blank"><button type="button">💬 View conversation</button></a>
    <form method="post" action="/restart_panel" style="display:inline">
      <button type="submit" class="secondary"
              title="Restart this web control panel (not the assistant)">♻️ Restart control panel</button>
    </form>
  </div>
  <p class="hint">The Start/Stop/Restart buttons above control the <b>assistant</b>.
     "Restart control panel" restarts this web UI itself — use it after changing panel code.</p>
</div>

<form method="post" action="/save">
<div class="card">
  <h2 style="margin-top:0">📝 Pre-prompt (system prompt)</h2>
  <textarea name="system_prompt">{{ s.system_prompt or '' }}</textarea>
</div>

<div class="card">
  <h2 style="margin-top:0">👂 Wake word &amp; speech</h2>
  <label>Speech input</label>
  <select name="stt_mode">
    {% for m in stt_modes %}
    <option value="{{ m }}" {{ 'selected' if s.stt_mode==m else '' }}>{{ m }}</option>
    {% endfor %}
  </select>
  <p class="hint">native = Gemma 4 hears audio directly (no Whisper). whisper = transcribe first.</p>
  <div class="row">
    <div>
      <label>Wake word / phrase</label>
      <input name="wake_word" placeholder="hey cleo"
             value="{{ (s.wake_word or 'hey cleo').replace('_', ' ') }}">
    </div>
    <div>
      <label>Detection threshold (0–1)</label>
      <input type="number" name="wake_word_threshold" step="0.05" min="0" max="1"
             value="{{ s.wake_word_threshold or 0.5 }}">
    </div>
    <div>
      <label>Whisper (STT) model</label>
      <select name="whisper_model">
        {% for m in whisper_models %}
        <option value="{{ m }}" {{ 'selected' if s.whisper_model==m else '' }}>{{ m }}</option>
        {% endfor %}
      </select>
    </div>
  </div>
  <p class="hint">Any phrase works — a new phrase is trained automatically when the assistant
     (re)starts (~30 s), and it announces when the wake word is ready.
     These have pretrained instant models: {{ bundled_wake_words|join(', ') }}
     (threshold applies to those only).</p>

  <div style="border-top:1px solid #2c313c; margin-top:1rem; padding-top:.8rem">
    <label style="margin-top:0">Train the wake word to your voice</label>
    <div class="row" style="align-items:flex-end">
      <div style="max-width:9rem">
        <label style="margin-top:0">Samples</label>
        <input type="number" id="enrollSamples" value="5" min="1" max="20" step="1">
      </div>
      <div style="flex:0 0 auto">
        <button type="button" id="enrollBtn" onclick="startEnroll()">🎤 Train to my voice</button>
      </div>
    </div>
    <div id="enrollStatus" class="hint" style="display:none; margin-top:.6rem;
         background:#11131a; border:1px solid #2c313c; border-radius:8px; padding:.7rem .9rem">
      <div id="enrollPrompt" style="font-size:1rem; color:#e6e6e6"></div>
      <div id="enrollLog" style="margin-top:.4rem; font-family:ui-monospace,monospace; font-size:.8rem"></div>
    </div>
    <p class="hint">Cleo will ask you to say your wake phrase a few times and learn how
       your voice is heard. The assistant is paused during training (to free the mic)
       and restarted after. Only for custom phrases — say each prompt in a quiet spot.</p>
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0">😴 Sleep mode</h2>
  <div class="row">
    <div><label>Sleep command</label><input name="sleep_command" value="{{ s.sleep_command or 'go to sleep' }}"></div>
    <div><label>Wake command</label><input name="wake_command" value="{{ s.wake_command or 'wake up' }}"></div>
  </div>
  <div class="row">
    <div><label>Sleep reply</label><input name="sleep_reply" value="{{ s.sleep_reply or 'Going to sleep.' }}"></div>
    <div><label>Wake reply</label><input name="wake_reply" value="{{ s.wake_reply or 'Awake and ready.' }}"></div>
  </div>
  <p class="hint">Saying "&lt;wake word&gt;, &lt;sleep command&gt;" unloads the LLM to free GPU memory but keeps
     listening for the wake word. "&lt;wake word&gt;, &lt;wake command&gt;" reloads the model, then it speaks the wake reply.</p>
</div>

<div class="card">
  <h2 style="margin-top:0">💬 Conversation flow</h2>
  <label style="display:flex; align-items:center; gap:.5rem; margin:.2rem 0;">
    <input type="checkbox" name="followup_mode" value="1" style="width:auto"
           {{ 'checked' if s.get('followup_mode', True) else '' }}>
    Keep listening after each reply (follow-up mode)
  </label>
  <p class="hint">After Cleo answers she keeps listening — no wake word needed for the next thing.
     When you pause she asks if there's anything else; she only stops (until the next wake word)
     when you decline ("no thanks", "that's all") or stay quiet. Off = one reply per wake word.</p>
  <div class="row">
    <div><label>Follow-up prompt</label>
      <input name="followup_prompt" value="{{ s.followup_prompt or 'Is there anything else I can help with?' }}"></div>
    <div><label>Sign-off</label>
      <input name="followup_signoff" value="{{ s.followup_signoff or \"Okay, I'll be here if you need me.\" }}"></div>
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0">🖥️ Screen viewing</h2>
  <label style="display:flex; align-items:center; gap:.5rem; margin:.2rem 0;">
    <input type="checkbox" name="screen_view" value="1" style="width:auto"
           {{ 'checked' if s.get('screen_view', True) else '' }}>
    Let Cleo look at the screen on request ("look at my screen")
  </label>
  <p class="hint">Native-audio mode only (the model needs the audio+vision model). Cleo captures a
     monitor via the GNOME screen-share portal and looks at it. You grant access once below;
     after that, captures are silent.</p>
  <div class="btns">
    <button type="submit" formaction="/screen_grant" class="secondary"
            title="Pop the GNOME share dialog to (re)grant screen access">🖥️ Set up / re-grant screen access</button>
  </div>
  <p class="hint">Click this, then in the GNOME dialog pick <b>Entire screen</b> (or ctrl-click all
     monitors) so Cleo can view any of them by voice. Picking a single monitor limits it to that one.</p>
</div>

<div class="card">
  <h2 style="margin-top:0">📑 Side panel</h2>
  <label style="display:flex; align-items:center; gap:.5rem; margin:.2rem 0;">
    <input type="checkbox" name="side_panel" value="1" style="width:auto"
           {{ 'checked' if s.get('side_panel', True) else '' }}>
    Show code / stories in a side overlay instead of reading them aloud
  </label>
  <p class="hint">When a reply contains something to read or copy — code, a story, a poem, a list —
     Cleo pops a side panel (conversation on top, the text below with a Copy button) and just gives a
     short spoken intro. Plain conversational replies are unaffected.</p>
</div>

<div class="card">
  <h2 style="margin-top:0">🔮 Status orb</h2>
  <label style="display:flex; align-items:center; gap:.5rem; margin:.2rem 0;">
    <input type="checkbox" name="overlay" value="1" style="width:auto"
           {{ 'checked' if s.get('overlay', True) else '' }}>
    Show the Siri-style orb in the top-right of the screen while listening / replying
  </label>
</div>

<div class="card">
  <h2 style="margin-top:0">🔊 Voice (text-to-speech)</h2>
  <label>Engine</label>
  <select name="tts_engine" id="ttsEngine" onchange="updateVoiceOpts()">
    {% for e in tts_engines %}
    <option value="{{ e }}" {{ 'selected' if s.tts_engine==e else '' }}>{{ e }}</option>
    {% endfor %}
  </select>

  <div class="row" id="kokoroOpts">
    <div>
      <label>Kokoro voice</label>
      <select name="kokoro_voice">
        {% for v in kokoro_voices %}
        <option value="{{ v }}" {{ 'selected' if s.kokoro_voice==v else '' }}>{{ v }}</option>
        {% endfor %}
      </select>
    </div>
    <div>
      <label>Speed</label>
      <input type="number" name="kokoro_speed" step="0.05" min="0.5" max="2.0"
             value="{{ s.kokoro_speed or 1.0 }}">
    </div>
  </div>

  <div class="row" id="cloneOpts">
    <div>
      <label>Cloned voice (NeuTTS / Chatterbox)</label>
      <select name="neutts_voice">
        {% for v in neutts_voices %}
        <option value="{{ v }}" {{ 'selected' if (s.neutts_voice or 'jo')==v else '' }}>{{ v }}</option>
        {% endfor %}
      </select>
    </div>
    <div id="neuttsModelOpt">
      <label>NeuTTS model</label>
      <select name="neutts_backbone">
        {% for b in neutts_backbones %}
        <option value="{{ b }}" {{ 'selected' if (s.neutts_backbone or neutts_backbones[0])==b else '' }}>
          {{ 'q4 (fast)' if 'q4' in b else 'q8 (higher quality)' }}</option>
        {% endfor %}
      </select>
    </div>
  </div>
  <div id="chatterboxOpts">
    <p class="hint">Chatterbox runs on the <b>GPU</b> and clones the voice selected above. Tune its delivery:</p>
    <div class="row">
      <div>
        <label>Expressiveness: <b id="cbExagLabel">{{ '%.2f'|format(s.chatterbox_exaggeration or 0.5) }}</b></label>
        <input type="range" name="chatterbox_exaggeration" min="0.25" max="1.0" step="0.05"
               value="{{ s.chatterbox_exaggeration or 0.5 }}"
               oninput="document.getElementById('cbExagLabel').textContent=(+this.value).toFixed(2)">
      </div>
      <div>
        <label>Pace / CFG: <b id="cbCfgLabel">{{ '%.2f'|format(s.chatterbox_cfg or 0.5) }}</b></label>
        <input type="range" name="chatterbox_cfg" min="0.2" max="1.0" step="0.05"
               value="{{ s.chatterbox_cfg or 0.5 }}"
               oninput="document.getElementById('cbCfgLabel').textContent=(+this.value).toFixed(2)">
      </div>
      <div>
        <label>Temperature: <b id="cbTempLabel">{{ '%.2f'|format(s.chatterbox_temperature or 0.8) }}</b></label>
        <input type="range" name="chatterbox_temperature" min="0.1" max="1.5" step="0.05"
               value="{{ s.chatterbox_temperature or 0.8 }}"
               oninput="document.getElementById('cbTempLabel').textContent=(+this.value).toFixed(2)">
      </div>
    </div>
    <p class="hint">Higher expressiveness = more emotion; lower CFG = slower, calmer. Applies on the next restart.</p>
    <div id="cbVramBox" style="margin-top:.8rem">
      <label style="display:flex; justify-content:space-between; align-items:center; gap:.5rem;">
        <span>GPU VRAM — model <b id="cbModelGb">—</b> + Chatterbox <b id="cbCbGb">—</b> = <b id="cbTotalGb">—</b></span>
        <span id="cbVerdict" class="pill" style="display:none"></span></label>
      <div class="vbar stack">
        <div class="vseg" id="cbModelSeg" style="background:var(--accent)"></div>
        <div class="vseg" id="cbCbSeg" style="background:#a371f7"></div>
      </div>
      <div class="vbar-scale"><span>0</span><span id="cbTotalScale"></span></div>
      <p class="hint" style="margin-top:.35rem">
        <i class="vsw" style="background:var(--accent)"></i> Language model
        &nbsp;&nbsp;<i class="vsw" style="background:#a371f7"></i> Chatterbox TTS
        <span id="cbVramNote"></span></p>
    </div>
  </div>
  <div id="cloneUpload">
    <p class="hint"><b>NeuTTS</b> and <b>Chatterbox</b> both <b>clone the reference voice</b> you upload — quality
       depends on the clip. For a great voice, upload a clean <b>mono clip, ~10–15s</b> (calm, clear, no
       music/noise). It's auto-trimmed to the usable length, transcribed, and added to the voice list above.
       Applies on the next restart.</p>
    <div class="row">
      <div><label>New voice name</label><input type="text" id="refName" placeholder="e.g. sophon"></div>
      <div><label>Reference audio</label><input type="file" id="refFile" accept="audio/*,.wav,.mp3,.m4a,.flac"></div>
      <div style="display:flex; align-items:flex-end;">
        <button type="button" class="secondary" onclick="uploadRef()">⬆ Upload &amp; add voice</button></div>
    </div>
    <p class="hint" id="refMsg"></p>
  </div>
  <script>
    function updateVoiceOpts(){
      var e = document.getElementById('ttsEngine').value;
      var cloned = (e === 'neutts' || e === 'chatterbox');   // both clone a reference voice
      function show(id, on){ var x=document.getElementById(id); if(x) x.style.display = on ? '' : 'none'; }
      show('kokoroOpts', e === 'kokoro');
      show('cloneOpts', cloned);            // shared cloned-voice selector
      show('neuttsModelOpt', e === 'neutts'); // q4/q8 is NeuTTS-only
      show('chatterboxOpts', e === 'chatterbox');
      show('cloneUpload', cloned);
      if(typeof updateCbVram === 'function') updateCbVram();   // refresh the VRAM bar
    }
    async function uploadRef(){
      var name = document.getElementById('refName').value.trim();
      var file = document.getElementById('refFile').files[0];
      var msg = document.getElementById('refMsg');
      if (!name || !file){ msg.textContent = 'Enter a name and choose a WAV file.'; return; }
      msg.textContent = 'Uploading & transcribing (Whisper)… this can take a moment.';
      var fd = new FormData(); fd.append('ref_name', name); fd.append('ref_audio', file);
      try {
        var r = await fetch('/neutts_upload', {method:'POST', body: fd});
        var d = await r.json();
        if (!r.ok){ msg.textContent = d.error || 'Upload failed.'; return; }
        msg.textContent = 'Added "' + d.name + '". Transcript: “' + d.transcript + '” — reloading…';
        setTimeout(function(){ location.reload(); }, 1500);
      } catch(e){ msg.textContent = 'Upload failed: ' + e; }
    }
    updateVoiceOpts();
  </script>
  {% set vol_pct = (s.get('tts_volume', 1.0) * 100) | round | int %}
  <label>Volume: <b id="volLabel">{{ vol_pct }}%</b></label>
  <input type="range" name="tts_volume" min="0" max="100" step="5" value="{{ vol_pct }}"
         oninput="document.getElementById('volLabel').textContent=this.value+'%'">
  <p class="hint">How loud Cleo speaks (100% = full scale), separate from your system volume.
     Applies on the next thing she says — no restart needed.</p>
  <p class="hint">Voice prefixes: af/am = US female/male, bf/bm = British, plus ef, ff, hf, if, jf, zf … (other languages).
     <button type="submit" formaction="/test_voice" class="secondary">🔊 Test voice</button>
     (uses the selected engine/voice above)</p>
</div>

<div class="card">
  <h2 style="margin-top:0">🧰 Skills (tools)</h2>
  <label style="display:flex; align-items:center; gap:.5rem; margin:.2rem 0;">
    <input type="checkbox" name="skills" value="1" style="width:auto"
           {{ 'checked' if s.get('skills', True) else '' }}>
    Enable skills — let Cleo call tools (time, weather, timers…)
  </label>
  {% for name, desc in skills %}
  <label style="display:flex; align-items:center; gap:.5rem; margin:.15rem 0; font-size:.82rem;">
    <input type="checkbox" name="skill_on" value="{{ name }}" style="width:auto"
           {{ 'checked' if name not in disabled else '' }}>
    <b>{{ name }}</b> — {{ desc }}
  </label>
  {% endfor %}
  <div class="row" style="margin-top:.6rem">
    <div><label>Weather home location</label>
      <input name="weather_place" value="{{ s.weather_place or '' }}" placeholder="Dallas, Texas"></div>
    <div><label>Temperature units</label>
      <select name="weather_units">
        <option value="fahrenheit" {{ 'selected' if (s.weather_units or 'fahrenheit')=='fahrenheit' else '' }}>Fahrenheit</option>
        <option value="celsius" {{ 'selected' if s.weather_units=='celsius' else '' }}>Celsius</option>
      </select></div>
  </div>
  <p class="hint">Weather uses open-meteo (free, no key). The home location is used when you don't name a place.</p>
</div>

<div class="card">
  <h2 style="margin-top:0">📁 File access (read-only)</h2>
  <label style="display:flex; align-items:center; gap:.5rem; margin:.2rem 0;">
    <input type="checkbox" name="file_access" value="1" style="width:auto"
           {{ 'checked' if s.get('file_access', False) else '' }}>
    Let Cleo read files on this machine
  </label>
  <p class="hint">When on, Cleo can read files and list folders — but <b>only</b> inside the folders
     you list below, and never anything on the blocked list. It's strictly read-only (she can't
     write, move, or delete). With no allowed folders set, nothing is readable.</p>

  <label>Allowed folders / files (one per line)</label>
  <textarea name="file_access_roots" placeholder="/home/briley/Documents&#10;/home/briley/notes.txt"
            style="min-height:78px; font-family:ui-monospace,monospace; font-size:.82rem">{{ (s.file_access_roots or []) | join('\\n') }}</textarea>
  <p class="hint">Absolute paths. A folder grants access to everything beneath it. <code>~</code> and
     <code>$VARS</code> are expanded. Symlinks are resolved before checking, so a link can't reach
     outside these.</p>

  <label>Blocked paths / patterns (one per line)</label>
  <textarea name="file_access_blacklist"
            style="min-height:120px; font-family:ui-monospace,monospace; font-size:.82rem">{{ (s.file_access_blacklist if s.file_access_blacklist is defined else file_blacklist_default) | join('\\n') }}</textarea>
  <p class="hint">Hidden from Cleo even inside an allowed folder — for listings and reads alike.
     A line with <code>*</code>, <code>?</code> or <code>[</code> is a glob (matched against the full
     path and the bare filename, e.g. <code>*.env</code>, <code>*/.ssh/*</code>); anything else is a
     literal file or folder to block. The defaults shield common secrets and Cleo's own settings —
     edit as you like.</p>

  <label>Max file size to read (KB)</label>
  <input type="number" name="file_access_max_kb" min="1" max="4096" step="1"
         value="{{ s.file_access_max_kb or 256 }}" style="max-width:10rem">
  <p class="hint">Reads stop at this size so a huge file can't flood the model. Binary files are
     refused outright. Changes apply on the next assistant restart.</p>
</div>

<div class="card">
  <h2 style="margin-top:0">🧠 LLM connection</h2>
  {% set selkey = s.llm_model or 'gemma4-12b' %}
  <label>Model</label>
  <select name="llm_model" id="llmModel" onchange="onModelChange()">
    {% for key, label in llm_models %}
    <option value="{{ key }}" {{ 'selected' if selkey==key else '' }}>{{ label }}</option>
    {% endfor %}
  </select>
  <p class="hint">Picks which model to run. Text-only models (e.g. Nemotron) automatically use
     Whisper for speech — native audio needs an audio-capable model like Gemma 4.</p>

  <div id="quantRow" style="{{ '' if quant_data.get(selkey) and quant_data[selkey].quants else 'display:none' }}">
    <label>Quantization</label>
    <select name="llm_quant" id="llmQuant" onchange="onQuantChange()">
      {% if quant_data.get(selkey) %}
        {% for q in quant_data[selkey].quants %}
        <option value="{{ q.name }}" {{ 'selected' if model_status.quant==q.name else '' }}>
          {{ q.name }} — {{ q.gb }} GB{{ ' · downloaded' if q.present else '' }}</option>
        {% endfor %}
      {% endif %}
    </select>
    <p class="hint">Higher quant = better quality but more VRAM/RAM. The bar below estimates whether
       the whole model fits on your GPU. After changing this, Download the quant and Save &amp; restart.</p>
  </div>

  <div id="vramBox" style="margin-top:.7rem; display:none">
    <label style="margin-top:0; display:flex; align-items:center; gap:.5rem; flex-wrap:wrap">
      Estimated VRAM <b id="vramNeed"></b>
      <span id="vramVerdict" class="pill" style="display:none"></span></label>
    <div class="vbar"><div class="vfill" id="vramFill"></div></div>
    <div class="vbar-scale"><span>0</span><span id="vramTotal"></span></div>
    <p class="hint" id="vramNote"></p>
  </div>

  <div class="hint" style="display:flex; align-items:center; gap:.6rem; flex-wrap:wrap;">
    <span id="msPill" class="pill off">checking…</span>
    <button type="button" id="dlBtn" class="secondary" style="display:none"
            onclick="startDownload()" title="Download this GGUF from HuggingFace">⬇ Download</button>
    <button type="button" id="delBtn" class="danger" style="display:none"
            onclick="deleteModel()" title="Delete this local GGUF to free disk space">🗑 Delete</button>
  </div>
  <div id="dlProgress" style="display:none; margin-top:.5rem">
    <div class="vbar"><div class="vfill" id="dlFill" style="background:var(--accent)"></div></div>
    <div class="vbar-scale"><span id="dlText"></span><span id="dlPct"></span></div>
  </div>
  <div class="row" style="margin-top:.5rem; align-items:center;">
    <input type="file" id="llmUploadFile" accept=".gguf" style="font-size:.8rem;">
    <button type="button" class="secondary" onclick="uploadModel('llmUploadFile','/llm/upload','llmModel',onModelChange)">⬆ Upload GGUF</button>
  </div>
  <div id="llmUpProgress" style="display:none; margin-top:.4rem">
    <div class="vbar"><div class="vfill" id="llmUpFill" style="background:var(--accent2)"></div></div>
    <div class="vbar-scale"><span id="llmUpText"></span><span id="llmUpPct"></span></div>
  </div>
  <p class="hint">A local file is only needed for the <b>llama-cpp</b> backend. For Ollama use
     <code>ollama pull</code>; for a remote model use the API backend below — neither needs a download.
     Or <b>upload your own .gguf</b> above — it appears as "Custom: …" in the model list (text-only, llama-cpp).</p>
  <label>Backend</label>
  <select name="llm_backend">
    {% for b in backends %}
    <option value="{{ b }}" {{ 'selected' if s.llm_backend==b else '' }}>{{ b }}</option>
    {% endfor %}
  </select>
  <p class="hint">auto = Ollama if running, else local llama-cpp. "api" = remote endpoint below.</p>
  <label>Device (local llama-cpp)</label>
  <select name="llm_device">
    {% for d in llm_devices %}
    <option value="{{ d }}" {{ 'selected' if (s.llm_device or 'auto')==d else '' }}>{{ d }}</option>
    {% endfor %}
  </select>
  <p class="hint">auto = GPU if it has enough free VRAM for the model, otherwise CPU + RAM.
     gpu / cpu force one. (Needs a CUDA-enabled llama-cpp build for GPU — see enable_gpu_llm.sh.)</p>
  <label>GPU layers (partial offload)</label>
  <input name="llm_gpu_layers" placeholder="auto"
         value="{{ s.llm_gpu_layers or '' }}">
  <p class="hint">How many of the model's layers to put on the GPU; the rest run on the CPU.
     For big models that don't fit in VRAM (e.g. Nemotron 30B on a 16&nbsp;GB card), set a number
     like <code>20</code> — raise it until VRAM is nearly full, lower it if you hit out-of-memory.
     <code>auto</code> (blank) follows the Device setting; <code>all</code> = whole model on GPU,
     <code>none</code> = CPU only. Overrides Device when set. Applies on the next assistant restart.</p>
  <label>GPU compute limit: <b id="gpuPctLabel">{{ (s.gpu_compute_percent or 100) }}%</b></label>
  <input type="range" name="gpu_compute_percent" min="10" max="100" step="5"
         value="{{ s.gpu_compute_percent or 100 }}"
         oninput="document.getElementById('gpuPctLabel').textContent=this.value+'%'">
  <p class="hint">Caps the share of GPU <b>compute</b> (not VRAM) Cleo may use, via NVIDIA MPS —
     only Cleo is throttled. 100% = no limit. Takes effect on the next assistant restart.</p>
  <label>CPU compute limit: <b id="cpuPctLabel">{{ (s.cpu_compute_percent or 100) }}%</b></label>
  <input type="range" name="cpu_compute_percent" min="10" max="100" step="5"
         value="{{ s.cpu_compute_percent or 100 }}"
         oninput="document.getElementById('cpuPctLabel').textContent=this.value+'%'">
  <p class="hint">Caps the share of CPU <b>cores</b> Cleo may use when running a model on the CPU
     (llama-cpp threads + OpenMP). Running a big model on all cores at once can hard-lock the
     machine; this leaves cores free for the desktop. 100% = no limit. Takes effect on restart.</p>
  <div class="row">
    <div><label>Ollama host</label><input name="ollama_host" value="{{ s.ollama_host or '' }}"></div>
    <div><label>Ollama model (override — blank = model default)</label>
         <input name="ollama_model" placeholder="uses selected model's tag" value="{{ s.ollama_model or '' }}"></div>
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0">🎨 Image connection</h2>
  <label>Image model</label>
  <select name="image_model" id="imageModel" onchange="onImageModelChange()">
    {% for key, label in image_models %}
    <option value="{{ key }}" {{ 'selected' if (s.image_model or 'none')==key else '' }}>{{ label }}</option>
    {% endfor %}
  </select>
  <p class="hint">Local Stable Diffusion (diffusers), running in its own venv just like the voice
     models. Weights download from HuggingFace on first generation. "None" turns image generation
     off. With a model selected, Cleo can draw images on request — and you can prompt directly on the
     <a href="/image" target="_blank">image page</a> (works even with the language model set to None).</p>

  <div class="row">
    <div><label>Device</label>
      <select name="image_device" id="imageDevice" onchange="refreshImageStatus()">
        <option value="cuda" {{ 'selected' if (s.image_device or 'cuda')=='cuda' else '' }}>cuda (GPU)</option>
        <option value="cpu" {{ 'selected' if s.image_device=='cpu' else '' }}>cpu (slow, no VRAM)</option>
      </select>
    </div>
    <div><label>Working-set estimate</label>
      <div id="imgVram" class="hint" style="margin-top:.55rem">—</div></div>
  </div>

  <div class="hint" style="display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; margin-top:.4rem;">
    <span id="imgPill" class="pill off">checking…</span>
    <button type="button" id="imgDlBtn" class="secondary" style="display:none"
            onclick="downloadImageModel()" title="Download this model's weights from HuggingFace">⬇ Download</button>
    <a href="/image" target="_blank"><button type="button" class="secondary">🎨 Open image page</button></a>
  </div>
  <div id="imgDlProgress" style="display:none; margin-top:.5rem">
    <div class="vbar"><div class="vfill" id="imgDlFill" style="background:var(--accent)"></div></div>
    <div class="vbar-scale"><span id="imgDlText"></span><span id="imgDlPct"></span></div>
  </div>
  <p class="hint" id="imgSetupNote" style="display:none">
     The image venv isn't installed yet — run <code>bash imagegen/setup_imagegen.sh</code> to create it
     (needed to <i>generate</i>; downloading weights works without it).</p>
  <div class="row" style="margin-top:.5rem; align-items:center;">
    <input type="file" id="imgUploadFile" accept=".safetensors,.ckpt" style="font-size:.8rem;">
    <button type="button" class="secondary" onclick="uploadModel('imgUploadFile','/image/upload','imageModel',onImageModelChange)">⬆ Upload checkpoint</button>
  </div>
  <div id="imgUpProgress" style="display:none; margin-top:.4rem">
    <div class="vbar"><div class="vfill" id="imgUpFill" style="background:var(--accent2)"></div></div>
    <div class="vbar-scale"><span id="imgUpText"></span><span id="imgUpPct"></span></div>
  </div>
  <p class="hint">Upload your own Stable Diffusion checkpoint (<code>.safetensors</code>/<code>.ckpt</code>) —
     it appears as "Custom: …". Put "xl" in the filename for SDXL checkpoints so it renders at 1024px.</p>
</div>

<div class="card">
  <h2 style="margin-top:0">🎬 Video connection</h2>
  <label>Video model</label>
  <select name="video_model" id="videoModel" onchange="onVideoModelChange()">
    {% for key, label in video_models %}
    <option value="{{ key }}" {{ 'selected' if (s.video_model or 'none')==key else '' }}>{{ label }}</option>
    {% endfor %}
  </select>
  <p class="hint">Local text-to-video (diffusers: Wan, LTX-2, …), in its own venv. Weights download from
     HuggingFace on first generation. "None" turns video off. With a model selected, Cleo can make
     short clips on request, and you can prompt directly on the
     <a href="/video" target="_blank">video page</a>. Video is slow and VRAM-heavy — Wan 1.3B / LTX
     fit ~16 GB; the 14B models need 24 GB+.</p>

  <div class="row">
    <div><label>Device</label>
      <select name="video_device" id="videoDevice" onchange="refreshVideoStatus()">
        <option value="cuda" {{ 'selected' if (s.video_device or 'cuda')=='cuda' else '' }}>cuda (GPU)</option>
        <option value="cpu" {{ 'selected' if s.video_device=='cpu' else '' }}>cpu (very slow, no VRAM)</option>
      </select>
    </div>
    <div><label>Working-set estimate</label>
      <div id="vidVram" class="hint" style="margin-top:.55rem">—</div></div>
  </div>

  <div class="hint" style="display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; margin-top:.4rem;">
    <span id="vidPill" class="pill off">checking…</span>
    <button type="button" id="vidDlBtn" class="secondary" style="display:none"
            onclick="downloadVideoModel()" title="Download this model's weights from HuggingFace">⬇ Download</button>
    <a href="/video" target="_blank"><button type="button" class="secondary">🎬 Open video page</button></a>
  </div>
  <div id="vidDlProgress" style="display:none; margin-top:.5rem">
    <div class="vbar"><div class="vfill" id="vidDlFill" style="background:var(--accent)"></div></div>
    <div class="vbar-scale"><span id="vidDlText"></span><span id="vidDlPct"></span></div>
  </div>
  <p class="hint" id="vidSetupNote" style="display:none">
     The video venv isn't installed yet — run <code>bash video/setup_video.sh</code> to create it
     (needed to <i>generate</i>; downloading weights works without it).</p>
  <div class="row" style="margin-top:.5rem; align-items:center;">
    <input type="file" id="vidUploadFile" accept=".safetensors,.ckpt" style="font-size:.8rem;">
    <button type="button" class="secondary" onclick="uploadModel('vidUploadFile','/video/upload','videoModel',onVideoModelChange)">⬆ Upload checkpoint</button>
  </div>
  <div id="vidUpProgress" style="display:none; margin-top:.4rem">
    <div class="vbar"><div class="vfill" id="vidUpFill" style="background:var(--accent2)"></div></div>
    <div class="vbar-scale"><span id="vidUpText"></span><span id="vidUpPct"></span></div>
  </div>
  <p class="hint">Upload your own video checkpoint (<code>.safetensors</code>/<code>.ckpt</code>) —
     it appears as "Custom: …" (single-file loading is best-effort for video models).</p>
</div>

<div class="card">
  <h2 style="margin-top:0">🌐 Remote API (OpenAI-compatible)</h2>
  <label>Base URL</label>
  <input name="api_base_url" placeholder="https://api.openai.com/v1" value="{{ s.api.base_url if s.api else '' }}">
  <div class="row">
    <div><label>Model</label><input name="api_model" placeholder="gpt-4o-mini" value="{{ s.api.model if s.api else '' }}"></div>
    <div><label>API key {% if s.api and s.api.api_key %}(set — blank keeps it){% endif %}</label>
         <input name="api_key" type="password" placeholder="sk-..."></div>
  </div>
  <p class="hint">Key is stored locally in settings.json on this machine.</p>
</div>

<div class="savebar">
  <button type="submit">💾 Save settings</button>
  <button type="submit" name="restart" value="1" class="secondary">💾 Save &amp; restart assistant</button>
</div>
</form>

<script>
// --- Wake-word voice enrollment ---
let enrolling = false;
async function startEnroll(){
  if (enrolling) return;
  const btn = document.getElementById('enrollBtn');
  const samples = parseInt(document.getElementById('enrollSamples').value) || 5;
  const box = document.getElementById('enrollStatus');
  const promptEl = document.getElementById('enrollPrompt');
  const logEl = document.getElementById('enrollLog');
  box.style.display = 'block'; logEl.innerHTML = '';
  promptEl.textContent = 'Starting…';
  try {
    const r = await fetch('/wake/enroll/start', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({samples})});
    const d = await r.json().catch(()=>({}));
    if (!r.ok){ promptEl.textContent = d.error || 'Could not start.'; return; }
  } catch(e){ promptEl.textContent = 'Could not start: ' + e; return; }
  enrolling = true; btn.disabled = true; btn.textContent = 'Training…';
  pollEnroll();
}
async function pollEnroll(){
  const btn = document.getElementById('enrollBtn');
  const promptEl = document.getElementById('enrollPrompt');
  const logEl = document.getElementById('enrollLog');
  try {
    const r = await fetch('/wake/enroll/status');
    const s = await r.json();
    promptEl.textContent = s.prompt || '';
    logEl.innerHTML = (s.log||[]).map(l =>
      '<div style="color:'+(l.kind==='ok'?'#7fd17f':l.kind==='bad'?'#e08f8f':'#aab')+'">'
      + l.text.replace(/</g,'&lt;') + '</div>').join('');
    if (s.running){ setTimeout(pollEnroll, 600); return; }
  } catch(e){ promptEl.textContent = 'Lost contact with the panel: ' + e; }
  enrolling = false; btn.disabled = false; btn.textContent = '🎤 Train to my voice';
}

// --- Themes ---
const THEMES = [['midnight','Midnight','#4d7cff'],['nord','Nord','#88c0d0'],
  ['dracula','Dracula','#bd93f9'],['emerald','Emerald','#2dd4a7'],
  ['synthwave','Synthwave','#ff2e97'],['light','Light','#3b6fe0']];
function applyTheme(t){
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('cleo_theme', t); } catch(e){}
  document.querySelectorAll('.swatch').forEach(s => s.classList.toggle('active', s.dataset.t === t));
}
function initThemes(){
  const bar = document.getElementById('themeBar');
  if (bar) bar.innerHTML = THEMES.map(([id,n,c]) =>
    '<button type="button" class="swatch" data-t="'+id+'" title="'+n+'" style="--sw:'+c
    +'" onclick="applyTheme(\\''+id+'\\')"></button>').join('');
  let s = 'midnight';
  try { s = localStorage.getItem('cleo_theme') || 'midnight'; } catch(e){}
  applyTheme(s);
}
initThemes();

// --- Model quant picker + VRAM-fit bar ---
const QUANT_DATA = {{ quant_data|tojson }};
const GPU_TOTAL_GB = {{ gpu_total_gb if gpu_total_gb is not none else 'null' }};
const CHATTERBOX_VRAM_GB = {{ chatterbox_vram_gb }};
// Chatterbox runs on the GPU on top of the language model; show the stacked VRAM
// when it's the selected TTS engine so the user can see if both fit.
function updateCbVram(){
  const box = document.getElementById('cbVramBox');
  if(!box) return;
  const eng = (document.getElementById('ttsEngine')||{}).value;
  if(eng !== 'chatterbox'){ box.style.display='none'; return; }
  box.style.display='';
  const model = modelVramGb() || 0;
  const cb = CHATTERBOX_VRAM_GB;
  const total = Math.round((model + cb)*10)/10;
  document.getElementById('cbModelGb').textContent = '~'+model+' GB';
  document.getElementById('cbCbGb').textContent = '~'+cb+' GB';
  document.getElementById('cbTotalGb').textContent = '~'+total+' GB';
  const mSeg = document.getElementById('cbModelSeg');
  const cSeg = document.getElementById('cbCbSeg');
  const verdict = document.getElementById('cbVerdict');
  const note = document.getElementById('cbVramNote');
  const scale = document.getElementById('cbTotalScale');
  if(GPU_TOTAL_GB == null){
    mSeg.style.width='0%'; cSeg.style.width='0%';
    verdict.style.display='none'; scale.textContent='no GPU';
    note.textContent = ' — no NVIDIA GPU detected; Chatterbox needs a GPU (or set its device to CPU).';
    return;
  }
  scale.textContent = GPU_TOTAL_GB+' GB';
  mSeg.style.width = Math.round(model/GPU_TOTAL_GB*100)+'%';   // overflow is clipped by .vbar
  cSeg.style.width = Math.round(cb/GPU_TOTAL_GB*100)+'%';
  const fits = total <= GPU_TOTAL_GB;
  const tight = fits && total > GPU_TOTAL_GB*0.9;
  verdict.style.display='';
  verdict.textContent = !fits ? 'too big for GPU' : (tight ? 'tight fit' : 'fits on GPU');
  verdict.style.background = !fits ? 'var(--bad-bg)' : (tight ? 'var(--warn-bg)' : 'var(--ok-bg)');
  verdict.style.color = !fits ? 'var(--bad-text)' : (tight ? 'var(--warn-text)' : 'var(--ok-text)');
  note.textContent = fits
    ? (' — model + Chatterbox together need ~'+total+' GB of '+GPU_TOTAL_GB+' GB.')
    : (' — together they exceed VRAM (~'+total+' GB vs '+GPU_TOTAL_GB+' GB); lower the LLM quant or run one on CPU.');
}
function onModelChange(){
  // Rebuild the quant options for the newly-selected model (defaulting to the
  // model's default quant), then recompute the bar.
  const key = document.getElementById('llmModel').value;
  const sel = document.getElementById('llmQuant');
  const row = document.getElementById('quantRow');
  const d = QUANT_DATA[key];
  if(!d || !d.quants || !d.quants.length){ row.style.display='none'; sel.innerHTML=''; computeVram(); refreshModelStatus(); return; }
  row.style.display='';
  sel.innerHTML = d.quants.map(q =>
    '<option value="'+q.name+'">'+q.name+' — '+q.gb+' GB'+(q.present?' · downloaded':'')+'</option>').join('');
  const def = d.quants.find(q=>q.def) || d.quants[0];
  sel.value = def.name;
  computeVram();
  refreshModelStatus();
}
function onQuantChange(){ computeVram(); refreshModelStatus(); }
function modelVramGb(){
  // Estimated VRAM (GB) the selected model+quant needs fully on the GPU, or null.
  const d = QUANT_DATA[document.getElementById('llmModel').value];
  if(!d) return null;
  let sizeGb = null;
  if(d.quants && d.quants.length){
    const q = d.quants.find(x => x.name === document.getElementById('llmQuant').value);
    sizeGb = q ? q.gb : null;
  } else { sizeGb = d.file_gb; }
  if(sizeGb == null) return null;
  return Math.round((sizeGb + (d.overhead_gb||0) + (d.native ? (d.mmproj_gb||0) : 0))*10)/10;
}
function computeVram(){ computeVramCore(); updateCbVram(); }
function computeVramCore(){
  const box = document.getElementById('vramBox');
  const need = modelVramGb();
  if(need == null){ box.style.display='none'; return; }   // size unknown (single-file, not downloaded)
  box.style.display='';
  document.getElementById('vramNeed').textContent = '~'+need+' GB';
  const fill = document.getElementById('vramFill');
  const verdict = document.getElementById('vramVerdict');
  const note = document.getElementById('vramNote');
  const totalEl = document.getElementById('vramTotal');
  if(GPU_TOTAL_GB == null){
    fill.style.width='0%'; verdict.style.display='none'; totalEl.textContent='no GPU';
    note.textContent = 'No NVIDIA GPU detected — the model runs on CPU + RAM, needing about '+need+' GB of RAM.';
    return;
  }
  totalEl.textContent = GPU_TOTAL_GB+' GB';
  verdict.style.display='';
  fill.style.width = Math.min(100, Math.round(need/GPU_TOTAL_GB*100))+'%';
  const fits = need <= GPU_TOTAL_GB;
  const tight = fits && need > GPU_TOTAL_GB*0.9;
  fill.style.background = !fits ? '#e25a4d' : (tight ? '#e3a008' : '#3fb950');
  verdict.textContent = !fits ? 'too big for GPU' : (tight ? 'tight fit' : 'fits on GPU');
  verdict.style.background = !fits ? 'var(--bad-bg)' : (tight ? 'var(--warn-bg)' : 'var(--ok-bg)');
  verdict.style.color = !fits ? 'var(--bad-text)' : (tight ? 'var(--warn-text)' : 'var(--ok-text)');
  note.textContent = fits
    ? ('Should fit fully on the GPU (~'+need+' GB of '+GPU_TOTAL_GB+' GB). Other apps using VRAM '
       + 'cut into free space; the "auto" device falls back to CPU if free VRAM is short at launch.')
    : ('Likely too big to fully offload (~'+need+' GB needed vs '+GPU_TOTAL_GB+' GB total). It will '
       + 'run partly or fully on CPU + RAM — or set GPU layers below for a partial offload.');
}

// --- Live model status (present / download progress / delete), no page reload ---
let dlPollTimer = null;
function curModel(){ return document.getElementById('llmModel').value; }
function curQuant(){
  const row = document.getElementById('quantRow');
  const sel = document.getElementById('llmQuant');
  return (sel && row.style.display !== 'none') ? sel.value : '';
}
async function refreshModelStatus(){
  const key = curModel(), quant = curQuant();
  let d;
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 8000);   // don't hang forever
    const resp = await fetch('/model_status.json?key='+encodeURIComponent(key)
                           +'&quant='+encodeURIComponent(quant),
                           {cache:'no-store', signal: ctl.signal});
    clearTimeout(t);
    d = await resp.json();
  } catch(e){
    // A failed/slow check must NOT leave the pill stuck on "checking…" forever:
    // show that the check didn't complete and retry, so it self-heals.
    if (curModel() === key){
      const pill = document.getElementById('msPill');
      pill.className = 'pill off'; pill.style.background = ''; pill.style.color = '';
      pill.textContent = 'status unavailable — retrying…';
    }
    clearTimeout(dlPollTimer);
    dlPollTimer = setTimeout(refreshModelStatus, 3000);
    return;
  }
  if (curModel() !== key) return;                 // selection changed mid-fetch
  const pill = document.getElementById('msPill');
  const dlBtn = document.getElementById('dlBtn');
  const delBtn = document.getElementById('delBtn');
  const prog = document.getElementById('dlProgress');
  clearTimeout(dlPollTimer);
  if (d.none){
    prog.style.display = 'none';
    pill.className = 'pill off'; pill.style.background = ''; pill.style.color = '';
    pill.textContent = 'no language model — image-only mode';
    dlBtn.style.display = 'none'; delBtn.style.display = 'none';
    return;
  }
  if (d.downloading){
    pill.className = 'pill'; pill.style.background = 'var(--warn-bg)'; pill.style.color = 'var(--warn-text)';
    pill.textContent = '⬇ downloading…' + (d.quant ? (' ' + d.quant) : '');
    dlBtn.style.display = 'none'; delBtn.style.display = 'none';
    prog.style.display = '';
    const fill = document.getElementById('dlFill');
    if (d.pct != null){ fill.style.opacity = '1'; fill.style.width = d.pct + '%';
                        document.getElementById('dlPct').textContent = d.pct + '%'; }
    else { fill.style.opacity = '.4'; fill.style.width = '100%';
           document.getElementById('dlPct').textContent = ''; }
    document.getElementById('dlText').textContent =
      (d.downloaded_gb != null ? d.downloaded_gb + ' GB' : '') + (d.total_gb ? (' / ' + d.total_gb + ' GB') : '');
    dlPollTimer = setTimeout(refreshModelStatus, 1200);
    return;
  }
  prog.style.display = 'none';
  pill.style.background = ''; pill.style.color = '';
  if (d.present){
    pill.className = 'pill on';
    pill.textContent = (d.custom ? 'uploaded' : 'local file ready')
      + (d.quant ? (' · ' + d.quant) : '') + (d.size_gb ? (' · ' + d.size_gb + ' GB') : '');
    dlBtn.style.display = 'none';
    delBtn.style.display = d.custom ? 'none' : '';   // /delete_model only knows registry models
  } else {
    pill.className = 'pill off';
    pill.textContent = 'no local file' + (d.quant ? (' · ' + d.quant) : '');
    dlBtn.style.display = d.downloadable ? '' : 'none';
    delBtn.style.display = 'none';
  }
}
async function startDownload(){
  const fd = new FormData();
  fd.append('llm_model', curModel());
  if (curQuant()) fd.append('llm_quant', curQuant());
  fd.append('json', '1');
  const pill = document.getElementById('msPill');
  pill.className = 'pill'; pill.textContent = 'starting…';
  try { const r = await fetch('/download_model', {method:'POST', body: fd});
        const j = await r.json().catch(()=>({}));
        if (j.error){ alert('Download failed: ' + j.error); }
  } catch(e){ alert('Download failed: ' + e); }
  setTimeout(refreshModelStatus, 700);
}
async function deleteModel(){
  const key = curModel(), quant = curQuant();
  if (!confirm('Delete the local file for ' + key + (quant ? (' (' + quant + ')') : '')
               + '? You can re-download it later.')) return;
  const fd = new FormData(); fd.append('key', key); if (quant) fd.append('quant', quant);
  try { const r = await fetch('/delete_model', {method:'POST', body: fd});
        const j = await r.json().catch(()=>({}));
        if (j.error){ alert('Delete failed: ' + j.error); }
  } catch(e){ alert('Delete failed: ' + e); }
  refreshModelStatus();
}
computeVram();
refreshModelStatus();

// --- Image connection (model status + VRAM hint) ---
const IMAGE_META = {{ image_meta|tojson }};
let imgPollTimer = null;
function curImageModel(){ const el = document.getElementById('imageModel'); return el ? el.value : 'none'; }
function updateImgVram(){
  const box = document.getElementById('imgVram');
  if (!box) return;
  const meta = IMAGE_META[curImageModel()];
  if (!meta || meta.vram == null){ box.textContent = '—'; return; }
  let txt = '~'+meta.vram+' GB'+(meta.size ? (' · '+meta.size+'px') : '');
  if (GPU_TOTAL_GB != null) txt += ' of '+GPU_TOTAL_GB+' GB'
    + (meta.vram > GPU_TOTAL_GB ? ' — too big for GPU' : '');
  box.textContent = txt;
}
function onImageModelChange(){ updateImgVram(); refreshImageStatus(); }
async function refreshImageStatus(){
  const pill = document.getElementById('imgPill');
  const note = document.getElementById('imgSetupNote');
  const dlBtn = document.getElementById('imgDlBtn');
  const prog = document.getElementById('imgDlProgress');
  if (!pill) return;
  let d;
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 8000);
    const resp = await fetch('/image/status.json?model='+encodeURIComponent(curImageModel()),
                             {cache:'no-store', signal: ctl.signal});
    clearTimeout(t);
    d = await resp.json();
  } catch(e){
    pill.className = 'pill off'; pill.textContent = 'status unavailable — retrying…';
    clearTimeout(imgPollTimer); imgPollTimer = setTimeout(refreshImageStatus, 3000); return;
  }
  clearTimeout(imgPollTimer);
  pill.style.background = ''; pill.style.color = '';
  dlBtn.style.display = 'none'; prog.style.display = 'none';
  if (note) note.style.display = 'none';

  if (!d.enabled){ pill.className = 'pill off'; pill.textContent = 'image generation off'; return; }

  if (d.downloading){
    pill.className = 'pill'; pill.style.background = 'var(--warn-bg)'; pill.style.color = 'var(--warn-text)';
    pill.textContent = '⬇ downloading weights…';
    prog.style.display = '';
    const fill = document.getElementById('imgDlFill');
    if (d.pct != null){ fill.style.opacity = '1'; fill.style.width = d.pct + '%';
                        document.getElementById('imgDlPct').textContent = d.pct + '%'; }
    else { fill.style.opacity = '.4'; fill.style.width = '100%'; document.getElementById('imgDlPct').textContent = ''; }
    document.getElementById('imgDlText').textContent =
      (d.downloaded_gb != null ? d.downloaded_gb + ' GB' : '') + (d.dl_gb ? (' / ~' + d.dl_gb + ' GB') : '');
    imgPollTimer = setTimeout(refreshImageStatus, 1500);
    return;
  }
  if (!d.downloaded){
    pill.className = 'pill off';
    pill.textContent = 'not downloaded' + (d.dl_gb ? (' · ~' + d.dl_gb + ' GB') : '');
    dlBtn.style.display = '';
    if (note && !d.installed) note.style.display = '';
    return;
  }
  // Downloaded:
  if (!d.installed){ pill.className = 'pill on'; pill.textContent = 'downloaded · venv not installed';
                     if (note) note.style.display = ''; return; }
  if (d.error){ pill.className = 'pill off'; pill.style.background='var(--bad-bg)'; pill.style.color='var(--bad-text)';
                pill.textContent = 'load failed — see logs'; return; }
  if (d.ready){ pill.className = 'pill on'; pill.textContent = 'model loaded ('+(d.device||'')+')'; return; }
  if (d.loading){ pill.className = 'pill'; pill.style.background='var(--warn-bg)'; pill.style.color='var(--warn-text)';
                  pill.textContent = 'loading model…'; imgPollTimer = setTimeout(refreshImageStatus, 2000); return; }
  pill.className = 'pill on'; pill.textContent = 'downloaded · ready (loads on first image)';
}
async function downloadImageModel(){
  const key = curImageModel();
  if (key === 'none'){ alert('Select an image model first.'); return; }
  const fd = new FormData(); fd.append('image_model', key);
  const pill = document.getElementById('imgPill');
  pill.className = 'pill'; pill.textContent = 'starting…';
  try { const r = await fetch('/image/download', {method:'POST', body: fd});
        const j = await r.json().catch(()=>({}));
        if (j.error){ alert('Download failed: ' + j.error); } }
  catch(e){ alert('Download failed: ' + e); }
  setTimeout(refreshImageStatus, 700);
}
// Upload a custom model file (GGUF or SD checkpoint) with a live progress bar.
function uploadModel(fileInputId, endpoint, selectId, onDone){
  const inp = document.getElementById(fileInputId);
  const file = inp && inp.files && inp.files[0];
  if (!file){ alert('Choose a file first.'); return; }
  const P = (selectId === 'llmModel') ? 'llmUp' : (selectId === 'videoModel') ? 'vidUp' : 'imgUp';
  const prog = document.getElementById(P+'Progress');
  const fill = document.getElementById(P+'Fill');
  const text = document.getElementById(P+'Text');
  const pct  = document.getElementById(P+'Pct');
  prog.style.display = ''; fill.style.width = '0%'; pct.textContent = '0%';
  text.textContent = 'uploading '+file.name+'…';
  const fd = new FormData(); fd.append('file', file);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', endpoint);
  xhr.upload.onprogress = e => {
    if (e.lengthComputable){ const p = Math.round(e.loaded/e.total*100);
      fill.style.width = p+'%'; pct.textContent = p+'%';
      text.textContent = (e.loaded/1e9).toFixed(2)+' / '+(e.total/1e9).toFixed(2)+' GB'; }
  };
  xhr.onload = () => {
    let d = {}; try { d = JSON.parse(xhr.responseText); } catch(e){}
    if (xhr.status >= 200 && xhr.status < 300 && d.key){
      fill.style.width = '100%'; pct.textContent = '100%';
      text.textContent = 'uploaded ✓ — Save & restart to use it';
      const sel = document.getElementById(selectId);
      if (!Array.prototype.some.call(sel.options, o => o.value === d.key)){
        const o = document.createElement('option');
        o.value = d.key; o.textContent = 'Custom: ' + d.name; sel.appendChild(o);
      }
      sel.value = d.key; inp.value = '';
      if (onDone) onDone();
    } else { text.textContent = 'upload failed: ' + (d.error || ('HTTP '+xhr.status)); }
  };
  xhr.onerror = () => { text.textContent = 'upload failed (network error).'; };
  xhr.send(fd);
}
updateImgVram();
refreshImageStatus();

// --- Video connection (model status + VRAM hint) ---
const VIDEO_META = {{ video_meta|tojson }};
let vidPollTimer = null;
function curVideoModel(){ const el = document.getElementById('videoModel'); return el ? el.value : 'none'; }
function updateVidVram(){
  const box = document.getElementById('vidVram');
  if (!box) return;
  const meta = VIDEO_META[curVideoModel()];
  if (!meta || meta.vram == null){ box.textContent = '—'; return; }
  let txt = '~'+meta.vram+' GB'+(meta.size ? (' · '+meta.size+'px') : '');
  if (GPU_TOTAL_GB != null) txt += ' of '+GPU_TOTAL_GB+' GB'
    + (meta.vram > GPU_TOTAL_GB ? ' — too big for GPU' : '');
  box.textContent = txt;
}
function onVideoModelChange(){ updateVidVram(); refreshVideoStatus(); }
async function refreshVideoStatus(){
  const pill = document.getElementById('vidPill');
  const note = document.getElementById('vidSetupNote');
  const dlBtn = document.getElementById('vidDlBtn');
  const prog = document.getElementById('vidDlProgress');
  if (!pill) return;
  let d;
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 8000);
    const resp = await fetch('/video/status.json?model='+encodeURIComponent(curVideoModel()),
                             {cache:'no-store', signal: ctl.signal});
    clearTimeout(t);
    d = await resp.json();
  } catch(e){
    pill.className = 'pill off'; pill.textContent = 'status unavailable — retrying…';
    clearTimeout(vidPollTimer); vidPollTimer = setTimeout(refreshVideoStatus, 3000); return;
  }
  clearTimeout(vidPollTimer);
  pill.style.background = ''; pill.style.color = '';
  dlBtn.style.display = 'none'; prog.style.display = 'none';
  if (note) note.style.display = 'none';

  if (!d.enabled){ pill.className = 'pill off'; pill.textContent = 'video generation off'; return; }
  if (d.downloading){
    pill.className = 'pill'; pill.style.background = 'var(--warn-bg)'; pill.style.color = 'var(--warn-text)';
    pill.textContent = '⬇ downloading weights…';
    prog.style.display = '';
    const fill = document.getElementById('vidDlFill');
    if (d.pct != null){ fill.style.opacity = '1'; fill.style.width = d.pct + '%';
                        document.getElementById('vidDlPct').textContent = d.pct + '%'; }
    else { fill.style.opacity = '.4'; fill.style.width = '100%'; document.getElementById('vidDlPct').textContent = ''; }
    document.getElementById('vidDlText').textContent =
      (d.downloaded_gb != null ? d.downloaded_gb + ' GB' : '') + (d.dl_gb ? (' / ~' + d.dl_gb + ' GB') : '');
    vidPollTimer = setTimeout(refreshVideoStatus, 1500);
    return;
  }
  if (!d.downloaded && !d.custom){
    pill.className = 'pill off';
    pill.textContent = 'not downloaded' + (d.dl_gb ? (' · ~' + d.dl_gb + ' GB') : '');
    dlBtn.style.display = '';
    if (note && !d.installed) note.style.display = '';
    return;
  }
  if (!d.installed){ pill.className = 'pill on'; pill.textContent = 'downloaded · venv not installed';
                     if (note) note.style.display = ''; return; }
  if (d.error){ pill.className = 'pill off'; pill.style.background='var(--bad-bg)'; pill.style.color='var(--bad-text)';
                pill.textContent = 'load failed — see logs'; return; }
  if (d.ready){ pill.className = 'pill on'; pill.textContent = 'model loaded ('+(d.device||'')+')'; return; }
  if (d.loading){ pill.className = 'pill'; pill.style.background='var(--warn-bg)'; pill.style.color='var(--warn-text)';
                  pill.textContent = 'loading model…'; vidPollTimer = setTimeout(refreshVideoStatus, 2000); return; }
  pill.className = 'pill on'; pill.textContent = (d.custom ? 'uploaded' : 'downloaded') + ' · ready (loads on first video)';
}
async function downloadVideoModel(){
  const key = curVideoModel();
  if (key === 'none'){ alert('Select a video model first.'); return; }
  const fd = new FormData(); fd.append('video_model', key);
  const pill = document.getElementById('vidPill');
  pill.className = 'pill'; pill.textContent = 'starting…';
  try { const r = await fetch('/video/download', {method:'POST', body: fd});
        const j = await r.json().catch(()=>({}));
        if (j.error){ alert('Download failed: ' + j.error); } }
  catch(e){ alert('Download failed: ' + e); }
  setTimeout(refreshVideoStatus, 700);
}
updateVidVram();
refreshVideoStatus();

// Make each settings section collapsible (collapsed by default). A section whose
// only control is a single switch keeps that switch in the header, so it stays
// usable while collapsed.
(function(){
  document.querySelectorAll('.card').forEach(function(card){
    var h2 = card.querySelector(':scope > h2');
    if(!h2) return;
    var body = document.createElement('div');
    body.className = 'card-body';
    while(h2.nextSibling){ body.appendChild(h2.nextSibling); }
    card.appendChild(body);

    var controls = body.querySelectorAll('input, select, textarea, button');
    var lone = (controls.length === 1 && controls[0].type === 'checkbox') ? controls[0] : null;

    h2.classList.add('card-h2');
    var caret = document.createElement('span');
    caret.className = 'card-caret';
    caret.textContent = '▸';            // ▸
    h2.insertBefore(caret, h2.firstChild);

    if(lone){                                 // surface the single switch in the header
      var holder = document.createElement('span');
      holder.className = 'hdr-switch';
      holder.appendChild(lone.closest('label') || lone);
      h2.appendChild(holder);
      holder.addEventListener('click', function(e){ e.stopPropagation(); });
    }

    card.classList.add('collapsed');
    h2.addEventListener('click', function(){ card.classList.toggle('collapsed'); });
  });
})();
</script>
</body></html>"""


CHAT_PAGE = """<!doctype html>
<html lang="en" data-theme="midnight"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conversation — Voice Assistant</title>
<script>try{document.documentElement.dataset.theme=localStorage.getItem('cleo_theme')||'midnight'}catch(e){}</script>
<style>
  *{ box-sizing:border-box; }
  :root{ color-scheme:dark;
    --bg:#0f1117; --card:#1a1e27; --card-border:#262c3a; --card-hover:#39435c;
    --input-bg:#11141c; --input-border:#2e3545; --text:#e8eaf0; --dim:#9aa3b5; --faint:#6b7385;
    --accent:#4d7cff; --accent2:#6b5cff; --accent-text:#fff; --ok-text:#5fdd8f; --shadow:rgba(0,0,0,.35); }
  [data-theme="nord"]{ color-scheme:dark; --bg:#262b35; --card:#3b4252; --card-border:#434c5e; --card-hover:#4c566a;
    --input-bg:#2e3440; --input-border:#434c5e; --text:#eceff4; --dim:#d8dee9; --faint:#8b95a7;
    --accent:#88c0d0; --accent2:#81a1c1; --accent-text:#2e3440; --ok-text:#a3be8c; --shadow:rgba(0,0,0,.3); }
  [data-theme="dracula"]{ color-scheme:dark; --bg:#1b1c25; --card:#282a36; --card-border:#383a4a; --card-hover:#6272a4;
    --input-bg:#1e1f29; --input-border:#44475a; --text:#f8f8f2; --dim:#bcc0d6; --faint:#6272a4;
    --accent:#bd93f9; --accent2:#ff79c6; --accent-text:#21222c; --ok-text:#50fa7b; --shadow:rgba(0,0,0,.4); }
  [data-theme="emerald"]{ color-scheme:dark; --bg:#0c1310; --card:#14211b; --card-border:#21342a; --card-hover:#2f5742;
    --input-bg:#0e1713; --input-border:#243a2e; --text:#e6f0ea; --dim:#9bb3a6; --faint:#6a8579;
    --accent:#2dd4a7; --accent2:#14b8a6; --accent-text:#04241c; --ok-text:#5fdd8f; --shadow:rgba(0,0,0,.35); }
  [data-theme="synthwave"]{ color-scheme:dark; --bg:#181029; --card:#241a3a; --card-border:#3d2c5e; --card-hover:#6d4bb0;
    --input-bg:#1c1230; --input-border:#3d2c5e; --text:#f5e6ff; --dim:#c4a8e0; --faint:#8a72b0;
    --accent:#ff2e97; --accent2:#7b5cff; --accent-text:#fff; --ok-text:#3affc8; --shadow:rgba(255,46,151,.18); }
  [data-theme="light"]{ color-scheme:light; --bg:#eef1f6; --card:#ffffff; --card-border:#dde3ec; --card-hover:#b9c4d6;
    --input-bg:#f6f8fc; --input-border:#d2dae6; --text:#1c2430; --dim:#5a6678; --faint:#8a95a6;
    --accent:#3b6fe0; --accent2:#6b5cff; --accent-text:#fff; --ok-text:#1d7a45; --shadow:rgba(20,30,50,.12); }

  body{ font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; max-width:820px; margin:0 auto;
        padding:0 1rem 6rem; background:var(--bg); color:var(--text); line-height:1.45; -webkit-font-smoothing:antialiased; }
  body::before{ content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
    background:radial-gradient(1100px 600px at 50% -240px, color-mix(in srgb,var(--accent) 16%,transparent), transparent 70%); }
  header{ position:sticky; top:0; z-index:20; background:color-mix(in srgb,var(--bg) 82%,transparent);
    backdrop-filter:blur(12px); padding:.9rem 0 .7rem; border-bottom:1px solid var(--card-border);
    display:flex; align-items:center; gap:1rem; }
  h1{ font-size:1.2rem; margin:0; }
  .swatchbar{ display:flex; gap:.45rem; align-items:center; flex:1; }
  .swatch{ width:20px; height:20px; min-width:0; padding:0; border-radius:50%; cursor:pointer; background:var(--sw,#888);
    border:2px solid transparent; box-shadow:0 1px 4px var(--shadow); transition:transform .12s ease, border-color .12s; }
  .swatch:hover{ transform:scale(1.18); filter:none; }
  .swatch.active{ border-color:var(--text); transform:scale(1.1); }
  .dot{ width:.6rem; height:.6rem; border-radius:50%; background:var(--faint); display:inline-block; transition:background .3s, box-shadow .3s; }
  .dot.live{ background:var(--ok-text); box-shadow:0 0 8px var(--ok-text); }
  button{ background:var(--input-bg); color:var(--text); border:1px solid var(--input-border); border-radius:8px;
    padding:.45rem .85rem; font-size:.85rem; cursor:pointer; font-weight:600; font-family:inherit;
    transition:border-color .15s, background .15s, filter .15s; }
  button:hover{ border-color:var(--accent); background:var(--card); }
  .turn{ display:flex; margin:.6rem 0; }
  .turn.user{ justify-content:flex-end; }
  .bubble{ max-width:78%; padding:.6rem .9rem; border-radius:16px; line-height:1.4;
           white-space:pre-wrap; word-wrap:break-word; box-shadow:0 1px 6px var(--shadow); }
  .user .bubble{ background:linear-gradient(135deg,var(--accent),var(--accent2)); color:var(--accent-text); border-bottom-right-radius:5px; }
  .assistant .bubble{ background:var(--card); border:1px solid var(--card-border); border-bottom-left-radius:5px; }
  .genimg{ display:block; max-width:min(78%,420px); margin:.35rem 0; border-radius:14px;
           border:1px solid var(--card-border); box-shadow:0 1px 6px var(--shadow); cursor:pointer; }
  .time{ font-size:.66rem; color:var(--faint); margin:.15rem .35rem 0; }
  .meta{ font-size:.66rem; color:var(--dim); margin:.2rem .35rem 0; display:flex; gap:.7rem; flex-wrap:wrap; }
  .meta b{ color:var(--accent); font-weight:600; }
  .empty{ color:var(--faint); text-align:center; margin-top:3.5rem; }
  .cursor{ display:inline-block; width:.5ch; animation:blink 1s steps(1) infinite; color:var(--accent); }
  @keyframes blink{ 50%{ opacity:0; } }
  .pulse .bubble{ border-color:var(--accent); box-shadow:0 0 0 1px color-mix(in srgb,var(--accent) 40%,transparent); }
  .phase{ font-style:italic; color:var(--dim); }
  .summary{ margin:1.5rem 0 .4rem; padding:.8rem 1rem; background:var(--card); border:1px solid var(--card-border);
    border-radius:12px; font-size:.78rem; color:var(--dim); }
  .summary h2{ font-size:.8rem; margin:0 0 .5rem; color:var(--accent); }
  .summary .grid{ display:flex; gap:1.6rem; flex-wrap:wrap; }
  .summary b{ color:var(--text); font-size:1.05rem; display:block; }
  #composer{ position:fixed; left:0; right:0; bottom:0; z-index:20; background:color-mix(in srgb,var(--bg) 88%,transparent);
    backdrop-filter:blur(12px); border-top:1px solid var(--card-border); padding:.7rem 1rem; }
  #composer .inner{ max-width:820px; margin:0 auto; display:flex; gap:.5rem; }
  #msg{ flex:1; background:var(--input-bg); color:var(--text); border:1px solid var(--input-border); border-radius:10px;
    padding:.65rem .8rem; font-size:.92rem; font-family:inherit; }
  #msg:focus{ outline:none; border-color:var(--accent); box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 25%,transparent); }
  #sendbtn{ background:linear-gradient(135deg,var(--accent),var(--accent2)); color:var(--accent-text); border:0;
    padding:.6rem 1.2rem; font-weight:600; }
  #sendbtn:hover{ filter:brightness(1.08); border:0; }
  #sendbtn:disabled{ opacity:.5; cursor:default; }
  ::-webkit-scrollbar{ width:11px; }
  ::-webkit-scrollbar-thumb{ background:var(--card-border); border-radius:999px; border:3px solid var(--bg); }
</style></head><body>
<header>
  <h1>💬 Conversation</h1>
  <div class="swatchbar" id="themeBar"></div>
  <span class="dot" id="live" title="live status"></span>
  <button onclick="clearChat()">Clear</button>
</header>
<div id="log"><p class="empty">No conversation yet. Say the wake word or type a message below to start.</p></div>

<form id="composer" onsubmit="send(); return false;">
  <div class="inner">
    <input id="msg" autocomplete="off" placeholder="Type a message to Cleo…">
    <button id="sendbtn" type="submit">Send</button>
  </div>
</form>
<script>
let lastSig = "";
const esc = s => (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');
function fmt(ts){ const d = new Date(ts*1000);
  return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}); }
function metaHtml(m){
  if (!m) return '';
  const parts = [];
  if (m.ttft_ms != null) parts.push('⚡ <b>'+m.ttft_ms+' ms</b> to first token');
  if (m.tok_per_sec != null) parts.push('<b>'+m.tok_per_sec+'</b> tok/s'
    + (m.tokens != null ? ' ('+m.tokens+' tok)' : ''));
  if (m.voice_ms != null) parts.push('🔊 <b>'+m.voice_ms+' ms</b> to speech');
  if (!parts.length) return '';
  return '<div class="meta">'+parts.join('')+'</div>';
}
// Pull [[IMAGE:name]] markers out of a turn's text, returning the remaining
// prose and the <img> tags for any generated images.
function splitImages(text){
  const imgs = [];
  let body = (text||'').replace(/\\[\\[IMAGE:([^\\]]+)\\]\\]/g, (m, n) => {
    imgs.push('<img class="genimg" src="/chat/image/'+encodeURIComponent(n.trim())
            + '" loading="lazy" onclick="window.open(this.src)">');
    return '';
  });
  body = body.replace(/\\[\\[VIDEO:([^\\]]+)\\]\\]/g, (m, n) => {
    imgs.push('<video class="genimg" src="/chat/video/'+encodeURIComponent(n.trim())
            + '" controls preload="metadata"></video>');
    return '';
  });
  return { body: body.trim(), imgs: imgs.join('') };
}
function turnHtml(role, text, ts, meta){
  const who = role === 'user' ? 'user' : 'assistant';
  const { body, imgs } = splitImages(text);
  const bubble = body ? '<div class="bubble">'+esc(body)+'</div>' : '';
  return '<div class="turn '+who+'"><div>'+bubble+imgs
       + (ts ? '<div class="time" style="text-align:'+(who==='user'?'right':'left')+'">'+fmt(ts)+'</div>' : '')
       + (who==='assistant' ? metaHtml(meta) : '') + '</div></div>';
}
// The in-progress turn: your words (or a status placeholder) + Cleo's reply
// streaming in token by token with a blinking cursor.
function liveHtml(l){
  const PH = {listening:'🎤 listening…', thinking:'💭 thinking…', speaking:'🔊 speaking…'};
  let html = '';
  if (l.user) html += turnHtml('user', l.user, null, null);
  else html += '<div class="turn user"><div><div class="bubble pulse"><span class="phase">'
             + (PH[l.phase]||'…') + '</span></div></div></div>';
  if (l.assistant) html += '<div class="turn assistant pulse"><div><div class="bubble">'
             + esc(l.assistant) + '<span class="cursor">▌</span></div></div></div>';
  else if (l.phase === 'speaking')
    html += '<div class="turn assistant pulse"><div><div class="bubble"><span class="cursor">▌</span></div></div></div>';
  return html;
}
// Conversation-wide averages over every measured assistant turn.
function summaryHtml(turns){
  const m = turns.filter(t => t.role==='assistant' && t.meta).map(t => t.meta);
  if (!m.length) return '';
  const avg = (k) => { const v = m.filter(x=>x[k]!=null).map(x=>x[k]);
    return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null; };
  const ttft = avg('ttft_ms'), tps = avg('tok_per_sec'), voice = avg('voice_ms');
  const tot = m.reduce((a,x)=>a+(x.tokens||0),0);
  const cell = (label,val) => val==null ? '' : '<div><b>'+val+'</b>'+label+'</div>';
  return '<div class="summary"><h2>📊 Conversation stats — '+m.length+' replies</h2><div class="grid">'
    + cell(' ms avg to first token', ttft!=null?Math.round(ttft):null)
    + cell(' tok/s avg', tps!=null?Math.round(tps*10)/10:null)
    + cell(' ms avg to speech', voice!=null?Math.round(voice):null)
    + cell(' tokens total', tot||null)
    + '</div></div>';
}
let pollTimer = null, sending = false;
async function poll(){
  let dot = document.getElementById('live');
  let nextDelay = 1500;
  try {
    const data = await (await fetch('/chat.json')).json();
    const turns = data.turns || [];
    const l = data.live;
    dot.classList.add('live');
    if (l) nextDelay = 450;   // a turn is happening — refresh quickly
    // Signature so we only re-render (and disturb scrolling) on real changes.
    const sig = turns.length + '|' + (turns.length?turns[turns.length-1].ts:'')
              + '|' + (l ? l.phase+':'+(l.user||'')+':'+l.assistant.length : 'idle');
    if (sig === lastSig) return;
    lastSig = sig;
    const atBottom = window.innerHeight + window.scrollY >= document.body.offsetHeight - 80;
    const log = document.getElementById('log');
    if (!turns.length && !l){
      log.innerHTML = '<p class="empty">No conversation yet. Say the wake word or type a message below to start.</p>';
      return;
    }
    log.innerHTML = turns.map(t => turnHtml(t.role, t.text, t.ts, t.meta)).join('')
                  + (l ? liveHtml(l) : '')
                  + summaryHtml(turns);
    if (atBottom) window.scrollTo(0, document.body.scrollHeight);
  } catch(e){ dot.classList.remove('live'); }
  finally { clearTimeout(pollTimer); pollTimer = setTimeout(poll, nextDelay); }
}
// Force an immediate re-render without spawning a second poll loop.
function kick(){ clearTimeout(pollTimer); lastSig = ''; poll(); }
async function clearChat(){
  await fetch('/chat/clear', {method:'POST'}); kick();
}
async function send(){
  const inp = document.getElementById('msg'), btn = document.getElementById('sendbtn');
  const text = inp.value.trim();
  if (!text || sending) return;
  sending = true; inp.value = ''; inp.disabled = btn.disabled = true; btn.textContent = '…';
  kick();                                   // show the user bubble the server just logged
  try {
    const r = await fetch('/chat/send', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
    const d = await r.json().catch(()=>({}));
    if (!r.ok){ alert(d.error || 'Send failed.'); }
    else if (d.note){ alert(d.note); }
  } catch(e){ alert('Send failed: ' + e); }
  finally {
    sending = false; inp.disabled = btn.disabled = false; btn.textContent = 'Send';
    inp.focus(); kick();
  }
}
// --- Themes ---
const THEMES = [['midnight','Midnight','#4d7cff'],['nord','Nord','#88c0d0'],
  ['dracula','Dracula','#bd93f9'],['emerald','Emerald','#2dd4a7'],
  ['synthwave','Synthwave','#ff2e97'],['light','Light','#3b6fe0']];
function applyTheme(t){
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('cleo_theme', t); } catch(e){}
  document.querySelectorAll('.swatch').forEach(s => s.classList.toggle('active', s.dataset.t === t));
}
(function initThemes(){
  const bar = document.getElementById('themeBar');
  if (bar) bar.innerHTML = THEMES.map(([id,n,c]) =>
    '<button type="button" class="swatch" data-t="'+id+'" title="'+n+'" style="--sw:'+c
    +'" onclick="applyTheme(\\''+id+'\\')"></button>').join('');
  let s = 'midnight';
  try { s = localStorage.getItem('cleo_theme') || 'midnight'; } catch(e){}
  applyTheme(s);
})();
poll();
</script>
</body></html>"""


IMAGE_PAGE = """<!doctype html>
<html lang="en" data-theme="midnight"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image generation — Voice Assistant</title>
<script>try{document.documentElement.dataset.theme=localStorage.getItem('cleo_theme')||'midnight'}catch(e){}</script>
<style>
  *{ box-sizing:border-box; }
  :root{ color-scheme:dark;
    --bg:#0f1117; --card:#1a1e27; --card-border:#262c3a; --input-bg:#11141c; --input-border:#2e3545;
    --text:#e8eaf0; --dim:#9aa3b5; --faint:#6b7385; --accent:#4d7cff; --accent2:#6b5cff;
    --accent-text:#fff; --ok-text:#5fdd8f; --bad-text:#ff7a7a; --shadow:rgba(0,0,0,.35); }
  body{ font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; max-width:820px; margin:0 auto;
        padding:0 1rem 5rem; background:var(--bg); color:var(--text); line-height:1.45; }
  body::before{ content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
    background:radial-gradient(1100px 600px at 50% -240px, color-mix(in srgb,var(--accent) 16%,transparent), transparent 70%); }
  header{ position:sticky; top:0; z-index:20; background:color-mix(in srgb,var(--bg) 82%,transparent);
    backdrop-filter:blur(12px); padding:.9rem 0 .7rem; border-bottom:1px solid var(--card-border);
    display:flex; align-items:center; gap:1rem; }
  h1{ font-size:1.2rem; margin:0; flex:1; }
  a{ color:var(--accent); }
  textarea{ width:100%; background:var(--input-bg); color:var(--text); border:1px solid var(--input-border);
    border-radius:10px; padding:.7rem .9rem; font-family:inherit; font-size:.95rem; resize:vertical; min-height:4.2rem; }
  .row{ display:flex; gap:.7rem; align-items:center; margin-top:.6rem; flex-wrap:wrap; }
  button{ background:linear-gradient(135deg,var(--accent),var(--accent2)); color:var(--accent-text); border:0;
    border-radius:9px; padding:.6rem 1.1rem; font-size:.92rem; cursor:pointer; font-weight:700; font-family:inherit; }
  button:disabled{ filter:grayscale(.5) brightness(.7); cursor:default; }
  input[type=number]{ width:5.5rem; background:var(--input-bg); color:var(--text);
    border:1px solid var(--input-border); border-radius:8px; padding:.45rem .5rem; font-family:inherit; }
  select{ background:var(--input-bg); color:var(--text); border:1px solid var(--input-border);
    border-radius:8px; padding:.45rem .5rem; font-family:inherit; }
  .pill{ font-size:.72rem; padding:.18rem .55rem; border-radius:999px; border:1px solid var(--card-border);
    background:var(--input-bg); color:var(--dim); }
  .pill.on{ color:var(--ok-text); border-color:color-mix(in srgb,var(--ok-text) 45%,transparent); }
  .pill.bad{ color:var(--bad-text); border-color:color-mix(in srgb,var(--bad-text) 45%,transparent); }
  .hint{ color:var(--faint); font-size:.8rem; }
  .gallery{ margin-top:1.3rem; display:flex; flex-direction:column; gap:1.1rem; }
  .shot{ background:var(--card); border:1px solid var(--card-border); border-radius:14px; padding:.7rem;
    box-shadow:0 1px 8px var(--shadow); }
  .shot img{ width:100%; border-radius:10px; display:block; }
  .shot .cap{ color:var(--dim); font-size:.82rem; margin-top:.5rem; }
  .empty{ color:var(--faint); text-align:center; margin-top:2.5rem; }
</style></head>
<body>
<header>
  <h1>🎨 Image generation</h1>
  <span id="statePill" class="pill">checking…</span>
  <a href="/">← panel</a>
</header>

<textarea id="prompt" placeholder="Describe an image to generate — e.g. 'a watercolor fox in a snowy forest at dusk'"></textarea>
<div class="row">
  <button id="gen" onclick="generate()">Generate</button>
  <label class="hint">W <input type="number" id="width" min="0" max="2048" step="64" value="0" title="0 = model default. Use multiples of 64."></label>
  <label class="hint">H <input type="number" id="height" min="0" max="2048" step="64" value="0" title="0 = model default. Use multiples of 64."></label>
  <label class="hint">steps <input type="number" id="steps" min="0" max="50" value="0" title="0 = model default"></label>
  <label class="hint">CFG <input type="number" id="cfg" min="0" max="30" step="0.5" value="0" title="0 = model default. Turbo models ignore CFG."></label>
  <label class="hint">sampler
    <select id="sampler" title="Scheduler / sampling algorithm">
      <option value="default">default</option>
      <option value="euler">Euler</option>
      <option value="euler_a">Euler a</option>
      <option value="dpmpp_2m">DPM++ 2M</option>
      <option value="dpmpp_2m_karras">DPM++ 2M Karras</option>
      <option value="dpmpp_sde">DPM++ SDE</option>
      <option value="unipc">UniPC</option>
      <option value="ddim">DDIM</option>
      <option value="lms">LMS</option>
      <option value="heun">Heun</option>
    </select>
  </label>
  <span id="msg" class="hint"></span>
</div>

<div id="gallery" class="gallery"><p class="empty" id="empty">Generated images will appear here.</p></div>

<script>
const esc = s => (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
let busy = false;
async function refreshState(){
  const pill = document.getElementById('statePill');
  let d;
  try { d = await (await fetch('/image/status.json', {cache:'no-store'})).json(); }
  catch(e){ pill.className='pill bad'; pill.textContent='panel unreachable'; return; }
  if (!d.enabled){ pill.className='pill bad'; pill.textContent='no image model selected'; }
  else if (!d.installed){ pill.className='pill bad'; pill.textContent='image venv not installed'; }
  else if (d.error){ pill.className='pill bad'; pill.textContent='load failed'; }
  else if (d.ready){ pill.className='pill on'; pill.textContent='model loaded'; }
  else if (d.loading){ pill.className='pill'; pill.textContent='loading model…'; setTimeout(refreshState, 2000); }
  else { pill.className='pill'; pill.textContent='ready'; }
}
async function generate(){
  if (busy) return;
  const prompt = document.getElementById('prompt').value.trim();
  const msg = document.getElementById('msg'), btn = document.getElementById('gen');
  if (!prompt){ msg.textContent = 'Enter a prompt first.'; return; }
  const steps = parseInt(document.getElementById('steps').value) || 0;
  const cfg = parseFloat(document.getElementById('cfg').value) || 0;
  const sampler = document.getElementById('sampler').value;
  const width = parseInt(document.getElementById('width').value) || 0;
  const height = parseInt(document.getElementById('height').value) || 0;
  busy = true; btn.disabled = true; btn.textContent = 'Generating…';
  msg.textContent = 'Working… (first run loads/downloads the model — can take a while)';
  try {
    const r = await fetch('/image/generate', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({prompt, steps, cfg, sampler, width, height})});
    const d = await r.json().catch(()=>({}));
    if (!r.ok || d.error){ msg.textContent = d.error || 'Generation failed.'; }
    else {
      msg.textContent = '';
      const e = document.getElementById('empty'); if (e) e.remove();
      const g = document.getElementById('gallery');
      const div = document.createElement('div');
      div.className = 'shot';
      div.innerHTML = '<img src="'+d.url+'" loading="lazy" onclick="window.open(this.src)">'
                    + '<div class="cap">'+esc(d.prompt)+'</div>';
      g.insertBefore(div, g.firstChild);
    }
  } catch(e){ msg.textContent = 'Generation failed: ' + e; }
  finally { busy = false; btn.disabled = false; btn.textContent = 'Generate'; refreshState(); }
}
document.getElementById('prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) generate();
});
refreshState();
</script>
</body></html>"""


VIDEO_PAGE = """<!doctype html>
<html lang="en" data-theme="midnight"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video generation — Voice Assistant</title>
<script>try{document.documentElement.dataset.theme=localStorage.getItem('cleo_theme')||'midnight'}catch(e){}</script>
<style>
  *{ box-sizing:border-box; }
  :root{ color-scheme:dark;
    --bg:#0f1117; --card:#1a1e27; --card-border:#262c3a; --input-bg:#11141c; --input-border:#2e3545;
    --text:#e8eaf0; --dim:#9aa3b5; --faint:#6b7385; --accent:#4d7cff; --accent2:#6b5cff;
    --accent-text:#fff; --ok-text:#5fdd8f; --bad-text:#ff7a7a; --shadow:rgba(0,0,0,.35); }
  body{ font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; max-width:820px; margin:0 auto;
        padding:0 1rem 5rem; background:var(--bg); color:var(--text); line-height:1.45; }
  body::before{ content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
    background:radial-gradient(1100px 600px at 50% -240px, color-mix(in srgb,var(--accent) 16%,transparent), transparent 70%); }
  header{ position:sticky; top:0; z-index:20; background:color-mix(in srgb,var(--bg) 82%,transparent);
    backdrop-filter:blur(12px); padding:.9rem 0 .7rem; border-bottom:1px solid var(--card-border);
    display:flex; align-items:center; gap:1rem; }
  h1{ font-size:1.2rem; margin:0; flex:1; }
  a{ color:var(--accent); }
  textarea{ width:100%; background:var(--input-bg); color:var(--text); border:1px solid var(--input-border);
    border-radius:10px; padding:.7rem .9rem; font-family:inherit; font-size:.95rem; resize:vertical; min-height:4.2rem; }
  .row{ display:flex; gap:.7rem; align-items:center; margin-top:.6rem; flex-wrap:wrap; }
  button{ background:linear-gradient(135deg,var(--accent),var(--accent2)); color:var(--accent-text); border:0;
    border-radius:9px; padding:.6rem 1.1rem; font-size:.92rem; cursor:pointer; font-weight:700; font-family:inherit; }
  button:disabled{ filter:grayscale(.5) brightness(.7); cursor:default; }
  input[type=number]{ width:5.5rem; background:var(--input-bg); color:var(--text);
    border:1px solid var(--input-border); border-radius:8px; padding:.45rem .5rem; font-family:inherit; }
  .pill{ font-size:.72rem; padding:.18rem .55rem; border-radius:999px; border:1px solid var(--card-border);
    background:var(--input-bg); color:var(--dim); }
  .pill.on{ color:var(--ok-text); border-color:color-mix(in srgb,var(--ok-text) 45%,transparent); }
  .pill.bad{ color:var(--bad-text); border-color:color-mix(in srgb,var(--bad-text) 45%,transparent); }
  .hint{ color:var(--faint); font-size:.8rem; }
  .gallery{ margin-top:1.3rem; display:flex; flex-direction:column; gap:1.1rem; }
  .shot{ background:var(--card); border:1px solid var(--card-border); border-radius:14px; padding:.7rem;
    box-shadow:0 1px 8px var(--shadow); }
  .shot video{ width:100%; border-radius:10px; display:block; background:#000; }
  .shot .cap{ color:var(--dim); font-size:.82rem; margin-top:.5rem; }
  .empty{ color:var(--faint); text-align:center; margin-top:2.5rem; }
</style></head>
<body>
<header>
  <h1>🎬 Video generation</h1>
  <span id="statePill" class="pill">checking…</span>
  <a href="/">← panel</a>
</header>

<textarea id="prompt" placeholder="Describe a video to generate — e.g. 'a fox running through a snowy forest, cinematic, slow motion'"></textarea>
<div class="row">
  <button id="gen" onclick="generate()">Generate</button>
  <label class="hint">frames <input type="number" id="frames" min="0" max="257" value="0" title="0 = model default"></label>
  <label class="hint">steps <input type="number" id="steps" min="0" max="100" value="0" title="0 = model default"></label>
  <span id="msg" class="hint"></span>
</div>
<p class="hint">Heads-up: video generation is slow — often a minute or more per clip, and the first run
   loads/downloads the model.</p>

<div id="gallery" class="gallery"><p class="empty" id="empty">Generated videos will appear here.</p></div>

<script>
const esc = s => (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
let busy = false;
async function refreshState(){
  const pill = document.getElementById('statePill');
  let d;
  try { d = await (await fetch('/video/status.json', {cache:'no-store'})).json(); }
  catch(e){ pill.className='pill bad'; pill.textContent='panel unreachable'; return; }
  if (!d.enabled){ pill.className='pill bad'; pill.textContent='no video model selected'; }
  else if (!d.installed){ pill.className='pill bad'; pill.textContent='video venv not installed'; }
  else if (d.error){ pill.className='pill bad'; pill.textContent='load failed'; }
  else if (d.ready){ pill.className='pill on'; pill.textContent='model loaded'; }
  else if (d.loading){ pill.className='pill'; pill.textContent='loading model…'; setTimeout(refreshState, 2000); }
  else if (d.downloading){ pill.className='pill'; pill.textContent='downloading weights…'; setTimeout(refreshState, 2500); }
  else { pill.className='pill'; pill.textContent='ready'; }
}
async function generate(){
  if (busy) return;
  const prompt = document.getElementById('prompt').value.trim();
  const msg = document.getElementById('msg'), btn = document.getElementById('gen');
  if (!prompt){ msg.textContent = 'Enter a prompt first.'; return; }
  const frames = parseInt(document.getElementById('frames').value) || 0;
  const steps = parseInt(document.getElementById('steps').value) || 0;
  busy = true; btn.disabled = true; btn.textContent = 'Generating…';
  msg.textContent = 'Working… video can take a while (first run loads/downloads the model).';
  try {
    const r = await fetch('/video/generate', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({prompt, frames, steps})});
    const d = await r.json().catch(()=>({}));
    if (!r.ok || d.error){ msg.textContent = d.error || 'Generation failed.'; }
    else {
      msg.textContent = '';
      const e = document.getElementById('empty'); if (e) e.remove();
      const g = document.getElementById('gallery');
      const div = document.createElement('div');
      div.className = 'shot';
      div.innerHTML = '<video src="'+d.url+'" controls autoplay loop muted playsinline></video>'
                    + '<div class="cap">'+esc(d.prompt)+'</div>';
      g.insertBefore(div, g.firstChild);
    }
  } catch(e){ msg.textContent = 'Generation failed: ' + e; }
  finally { busy = false; btn.disabled = false; btn.textContent = 'Generate'; refreshState(); }
}
document.getElementById('prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) generate();
});
refreshState();
</script>
</body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=False)
