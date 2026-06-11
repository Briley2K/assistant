#!/usr/bin/env python3
"""
Voice Assistant control panel — a small local web app (http://localhost:5005).

Lets you start/stop the assistant, toggle run-on-boot (systemd user service),
edit the pre-prompt, choose the wake word, and manage the LLM / remote-API
connection. All settings are persisted to settings.json, which config.py reads.
"""
import os
import json
import subprocess

from flask import Flask, request, redirect, url_for, render_template_string

BASE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE, "settings.json")
SERVICE = "voice-assistant.service"

WAKE_WORDS = ["hey_jarvis", "alexa", "hey_mycroft", "hey_marvin", "timer"]
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
BACKENDS = ["auto", "ollama", "llamacpp", "api"]
STT_MODES = ["native", "whisper"]
TTS_ENGINES = ["kokoro", "piper"]


def kokoro_voice_names():
    import numpy as np
    try:
        return sorted(np.load(os.path.join(BASE, "models", "kokoro", "voices-v1.0.npz")).keys())
    except Exception:
        return ["af_heart"]

app = Flask(__name__)


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
    return render_template_string(
        PAGE,
        s=load_settings(),
        state=service_state(),
        wake_words=WAKE_WORDS,
        whisper_models=WHISPER_MODELS,
        backends=BACKENDS,
        stt_modes=STT_MODES,
        tts_engines=TTS_ENGINES,
        kokoro_voices=kokoro_voice_names(),
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
    s["kokoro_speed"]        = float(f.get("kokoro_speed", 1.0))
    s["wake_word"]           = f.get("wake_word", "hey_jarvis")
    s["wake_word_threshold"] = float(f.get("wake_word_threshold", 0.5))
    s["llm_backend"]         = f.get("llm_backend", "auto")
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
    engine, voice = f.get("tts_engine", "kokoro"), f.get("kokoro_voice", "af_heart")
    sample = "Hello, this is a preview of the selected voice."
    try:
        from modules import audio
        if engine == "kokoro":
            from modules import kokoro_tts
            wav = kokoro_tts.synth_wav(sample, voice=voice)
        else:
            import io, wave
            from modules import tts
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                tts._get_voice().synthesize_wav(sample, wf)
            wav = buf.getvalue()
        audio.play_audio(wav)
        msg = f"Played sample ({engine}{': ' + voice if engine == 'kokoro' else ''})."
    except Exception as e:
        msg = f"Voice test failed: {e}"
    return redirect(url_for("index", msg=msg))


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


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Voice Assistant</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
         padding: 0 1rem; background:#15171c; color:#e6e6e6; }
  h1 { font-size: 1.5rem; } h2 { font-size: 1.05rem; margin-top: 1.8rem; color:#9ecbff; }
  .card { background:#1d2027; border:1px solid #2c313c; border-radius:10px; padding:1rem 1.2rem; margin:1rem 0; }
  label { display:block; margin:.7rem 0 .2rem; font-size:.85rem; color:#aab; }
  input, select, textarea { width:100%; box-sizing:border-box; background:#11131a; color:#e6e6e6;
         border:1px solid #333a47; border-radius:6px; padding:.5rem; font-size:.9rem; }
  textarea { min-height:90px; resize:vertical; }
  .row { display:flex; gap:1rem; } .row>div { flex:1; }
  button { background:#2d6cdf; color:#fff; border:0; border-radius:6px; padding:.55rem 1rem;
           font-size:.9rem; cursor:pointer; } button:hover { background:#3b7cf0; }
  button.secondary { background:#333a47; } button.danger { background:#b3402f; }
  .pill { display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.78rem; }
  .on { background:#1f5135; color:#7ee2a8; } .off { background:#4a2020; color:#f0a3a3; }
  .msg { background:#23314a; border:1px solid #345; padding:.6rem .9rem; border-radius:8px; }
  .btns { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.6rem; }
  .hint { font-size:.78rem; color:#778; margin-top:.3rem; }
</style></head><body>
<h1>🎙️ Voice Assistant</h1>
{% if msg %}<div class="msg">{{ msg }}</div>{% endif %}

<div class="card">
  <h2 style="margin-top:0">Status</h2>
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
</div>

<form method="post" action="/save">
<div class="card">
  <h2 style="margin-top:0">Pre-prompt (system prompt)</h2>
  <textarea name="system_prompt">{{ s.system_prompt or '' }}</textarea>
</div>

<div class="card">
  <h2 style="margin-top:0">Wake word &amp; speech</h2>
  <label>Speech input</label>
  <select name="stt_mode">
    {% for m in stt_modes %}
    <option value="{{ m }}" {{ 'selected' if s.stt_mode==m else '' }}>{{ m }}</option>
    {% endfor %}
  </select>
  <p class="hint">native = Gemma 4 hears audio directly (no Whisper). whisper = transcribe first.</p>
  <div class="row">
    <div>
      <label>Wake word</label>
      <select name="wake_word">
        {% for w in wake_words %}
        <option value="{{ w }}" {{ 'selected' if s.wake_word==w else '' }}>{{ w.replace('_',' ') }}</option>
        {% endfor %}
      </select>
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
</div>

<div class="card">
  <h2 style="margin-top:0">Voice (text-to-speech)</h2>
  <div class="row">
    <div>
      <label>Engine</label>
      <select name="tts_engine">
        {% for e in tts_engines %}
        <option value="{{ e }}" {{ 'selected' if s.tts_engine==e else '' }}>{{ e }}</option>
        {% endfor %}
      </select>
    </div>
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
  <p class="hint">Voice prefixes: af/am = US female/male, bf/bm = British, plus ef, ff, hf, if, jf, zf … (other languages).
     <button type="submit" formaction="/test_voice" class="secondary">🔊 Test voice</button>
     (uses the selected engine/voice above)</p>
</div>

<div class="card">
  <h2 style="margin-top:0">LLM connection</h2>
  <label>Backend</label>
  <select name="llm_backend">
    {% for b in backends %}
    <option value="{{ b }}" {{ 'selected' if s.llm_backend==b else '' }}>{{ b }}</option>
    {% endfor %}
  </select>
  <p class="hint">auto = Ollama if running, else local CPU. "api" = remote endpoint below.</p>
  <div class="row">
    <div><label>Ollama host</label><input name="ollama_host" value="{{ s.ollama_host or '' }}"></div>
    <div><label>Ollama model</label><input name="ollama_model" value="{{ s.ollama_model or '' }}"></div>
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0">Remote API (OpenAI-compatible)</h2>
  <label>Base URL</label>
  <input name="api_base_url" placeholder="https://api.openai.com/v1" value="{{ s.api.base_url if s.api else '' }}">
  <div class="row">
    <div><label>Model</label><input name="api_model" placeholder="gpt-4o-mini" value="{{ s.api.model if s.api else '' }}"></div>
    <div><label>API key {% if s.api and s.api.api_key %}(set — blank keeps it){% endif %}</label>
         <input name="api_key" type="password" placeholder="sk-..."></div>
  </div>
  <p class="hint">Key is stored locally in settings.json on this machine.</p>
</div>

<div class="btns">
  <button type="submit">Save</button>
  <button type="submit" name="restart" value="1" class="secondary">Save &amp; restart assistant</button>
</div>
</form>
</body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=False)
