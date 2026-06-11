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
CHATLOG_PATH = os.path.join(BASE, "logs", "chat.jsonl")   # written by modules/chatlog.py
SERVICE = "voice-assistant.service"

BUNDLED_WAKE_WORDS = ["hey jarvis", "alexa", "hey mycroft", "hey marvin", "timer"]
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
BACKENDS = ["auto", "ollama", "llamacpp", "api"]
LLM_DEVICES = ["auto", "gpu", "cpu"]
STT_MODES = ["native", "whisper"]
TTS_ENGINES = ["kokoro", "piper"]


def skill_list():
    """(name, description) for every registered skill, for the panel."""
    try:
        import sys
        sys.path.insert(0, BASE)
        from modules import skills
        return [(s.name, s.description) for s in skills.all_skills()]
    except Exception:
        return []


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
        bundled_wake_words=BUNDLED_WAKE_WORDS,
        whisper_models=WHISPER_MODELS,
        backends=BACKENDS,
        llm_devices=LLM_DEVICES,
        stt_modes=STT_MODES,
        tts_engines=TTS_ENGINES,
        kokoro_voices=kokoro_voice_names(),
        skills=skill_list(),
        disabled=set(load_settings().get("skills_disabled", [])),
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
    s["wake_word"]           = " ".join(f.get("wake_word", "").lower().split()) or "hey cleo"
    s["wake_word_threshold"] = float(f.get("wake_word_threshold", 0.5))
    s["overlay"]             = bool(f.get("overlay"))
    s["sleep_command"]       = f.get("sleep_command", "").strip() or "go to sleep"
    s["wake_command"]        = f.get("wake_command", "").strip() or "wake up"
    s["sleep_reply"]         = f.get("sleep_reply", "").strip() or "Going to sleep."
    s["wake_reply"]          = f.get("wake_reply", "").strip() or "Awake and ready."
    s["llm_backend"]         = f.get("llm_backend", "auto")
    s["llm_device"]          = f.get("llm_device", "auto")

    s["skills"]        = bool(f.get("skills"))
    enabled_skills     = set(f.getlist("skill_on"))
    all_skill_names    = [n for n, _ in skill_list()]
    s["skills_disabled"] = [n for n in all_skill_names if n not in enabled_skills]
    s["weather_place"] = f.get("weather_place", "").strip()
    s["weather_units"] = f.get("weather_units", "fahrenheit")
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


@app.route("/chat.json")
def chat_json():
    from flask import jsonify
    return jsonify(read_chat())


@app.route("/chat/clear", methods=["POST"])
def chat_clear():
    try:
        open(CHATLOG_PATH, "w").close()
    except OSError:
        pass
    return ("", 204)


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
  <div class="btns"><a href="/chat" target="_blank"><button type="button">💬 View conversation</button></a></div>
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
</div>

<div class="card">
  <h2 style="margin-top:0">Sleep mode</h2>
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
  <h2 style="margin-top:0">Status orb</h2>
  <label style="display:flex; align-items:center; gap:.5rem; margin:.2rem 0;">
    <input type="checkbox" name="overlay" value="1" style="width:auto"
           {{ 'checked' if s.get('overlay', True) else '' }}>
    Show the Siri-style orb in the top-right of the screen while listening / replying
  </label>
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
  <h2 style="margin-top:0">Skills (tools)</h2>
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
  <h2 style="margin-top:0">LLM connection</h2>
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


CHAT_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Conversation — Voice Assistant</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 0 auto;
         padding: 0 1rem 6rem; background:#15171c; color:#e6e6e6; }
  header { position:sticky; top:0; background:#15171c; padding:1rem 0 .6rem;
           border-bottom:1px solid #2c313c; display:flex; align-items:center; gap:1rem; }
  h1 { font-size:1.2rem; margin:0; flex:1; }
  .dot { width:.6rem; height:.6rem; border-radius:50%; background:#555; display:inline-block; }
  .dot.live { background:#7ee2a8; }
  button { background:#333a47; color:#fff; border:0; border-radius:6px; padding:.45rem .8rem;
           font-size:.85rem; cursor:pointer; } button:hover { filter:brightness(1.2); }
  .turn { display:flex; margin:.6rem 0; }
  .turn.user { justify-content:flex-end; }
  .bubble { max-width:78%; padding:.55rem .85rem; border-radius:14px; line-height:1.35;
            white-space:pre-wrap; word-wrap:break-word; }
  .user .bubble { background:#2d6cdf; color:#fff; border-bottom-right-radius:4px; }
  .assistant .bubble { background:#1d2027; border:1px solid #2c313c; border-bottom-left-radius:4px; }
  .time { font-size:.66rem; color:#778; margin:.15rem .3rem 0; }
  .empty { color:#778; text-align:center; margin-top:3rem; }
</style></head><body>
<header>
  <h1>💬 Conversation</h1>
  <span class="dot" id="live"></span>
  <button onclick="clearChat()">Clear</button>
</header>
<div id="log"><p class="empty">No conversation yet. Say the wake word to start talking.</p></div>
<script>
let lastCount = -1;
function fmt(ts){ const d = new Date(ts*1000);
  return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}); }
async function poll(){
  let live = document.getElementById('live');
  try {
    const turns = await (await fetch('/chat.json')).json();
    live.classList.add('live');
    if (turns.length === lastCount) return;       // nothing new
    const atBottom = window.innerHeight + window.scrollY >= document.body.offsetHeight - 80;
    lastCount = turns.length;
    const log = document.getElementById('log');
    if (!turns.length){ log.innerHTML = '<p class="empty">No conversation yet. Say the wake word to start talking.</p>'; return; }
    log.innerHTML = turns.map(t => {
      const who = t.role === 'user' ? 'user' : 'assistant';
      const text = t.text.replace(/&/g,'&amp;').replace(/</g,'&lt;');
      return '<div class="turn '+who+'"><div><div class="bubble">'+text+'</div>'
           + '<div class="time" style="text-align:'+(who==='user'?'right':'left')+'">'+fmt(t.ts)+'</div></div></div>';
    }).join('');
    if (atBottom) window.scrollTo(0, document.body.scrollHeight);
  } catch(e){ live.classList.remove('live'); }
}
async function clearChat(){
  await fetch('/chat/clear', {method:'POST'}); lastCount = -1; poll();
}
poll(); setInterval(poll, 1500);
</script>
</body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=False)
