#!/usr/bin/env python3
"""
Voice Assistant control panel — a small local web app (http://localhost:5005).

Lets you start/stop the assistant, toggle run-on-boot (systemd user service),
edit the pre-prompt, choose the wake word, and manage the LLM / remote-API
connection. All settings are persisted to settings.json, which config.py reads.
"""
import os
import time
import json
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


def llm_models():
    """(key, label) for every selectable LLM, from the config registry."""
    try:
        import sys
        sys.path.insert(0, BASE)
        import config
        return [(k, v["label"]) for k, v in config.LLM_MODELS.items()]
    except Exception:
        return [("gemma4-12b", "Gemma 4 12B")]


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
        llm_models=llm_models(),
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
    s["followup_mode"]       = bool(f.get("followup_mode"))
    s["followup_prompt"]     = f.get("followup_prompt", "").strip() or "Is there anything else I can help with?"
    s["followup_signoff"]    = f.get("followup_signoff", "").strip() or "Okay, I'll be here if you need me."
    s["screen_view"]         = bool(f.get("screen_view"))
    s["side_panel"]          = bool(f.get("side_panel"))
    s["llm_backend"]         = f.get("llm_backend", "auto")
    s["llm_device"]          = f.get("llm_device", "auto")
    try:
        s["gpu_compute_percent"] = max(10, min(100, int(f.get("gpu_compute_percent", 100))))
    except (TypeError, ValueError):
        s["gpu_compute_percent"] = 100

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
<html><head><meta charset="utf-8"><title>Restarting…</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; max-width: 480px; margin: 6rem auto;
         padding: 0 1rem; background:#15171c; color:#e6e6e6; text-align:center; }
  .spin { width:2rem; height:2rem; margin:1.5rem auto; border:3px solid #2c313c;
          border-top-color:#2d6cdf; border-radius:50%; animation:spin 1s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  a { color:#9ecbff; }
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
  <h2 style="margin-top:0">Conversation flow</h2>
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
  <h2 style="margin-top:0">Screen viewing</h2>
  <label style="display:flex; align-items:center; gap:.5rem; margin:.2rem 0;">
    <input type="checkbox" name="screen_view" value="1" style="width:auto"
           {{ 'checked' if s.get('screen_view', True) else '' }}>
    Let Cleo look at the screen on request ("look at my screen")
  </label>
  <p class="hint">Native-audio mode only (the model needs the audio+vision model). Cleo captures a
     monitor via the GNOME screen-share portal and looks at it. You grant access once below;
     after that, captures are silent.</p>
  <div class="btns">
    <form method="post" action="/screen_grant" style="display:inline">
      <button type="submit" class="secondary"
              title="Pop the GNOME share dialog to (re)grant screen access">🖥️ Set up / re-grant screen access</button>
    </form>
  </div>
  <p class="hint">Click this, then in the GNOME dialog pick <b>Entire screen</b> (or ctrl-click all
     monitors) so Cleo can view any of them by voice. Picking a single monitor limits it to that one.</p>
</div>

<div class="card">
  <h2 style="margin-top:0">Side panel</h2>
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
  <label>Model</label>
  <select name="llm_model">
    {% for key, label in llm_models %}
    <option value="{{ key }}" {{ 'selected' if (s.llm_model or 'gemma4-12b')==key else '' }}>{{ label }}</option>
    {% endfor %}
  </select>
  <p class="hint">Picks which model to run. Text-only models (e.g. Nemotron) automatically use
     Whisper for speech — native audio needs an audio-capable model like Gemma 4.
     Nemotron runs via Ollama: <code>ollama pull nemotron-3-nano-30b</code> first, and make sure
     the tag matches the Ollama field below (blank = the model's default tag).</p>
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
  <label>GPU compute limit: <b id="gpuPctLabel">{{ (s.gpu_compute_percent or 100) }}%</b></label>
  <input type="range" name="gpu_compute_percent" min="10" max="100" step="5"
         value="{{ s.gpu_compute_percent or 100 }}"
         oninput="document.getElementById('gpuPctLabel').textContent=this.value+'%'">
  <p class="hint">Caps the share of GPU <b>compute</b> (not VRAM) Cleo may use, via NVIDIA MPS —
     only Cleo is throttled. 100% = no limit. Takes effect on the next assistant restart.</p>
  <div class="row">
    <div><label>Ollama host</label><input name="ollama_host" value="{{ s.ollama_host or '' }}"></div>
    <div><label>Ollama model (override — blank = model default)</label>
         <input name="ollama_model" placeholder="uses selected model's tag" value="{{ s.ollama_model or '' }}"></div>
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
  .meta { font-size:.66rem; color:#8a93a6; margin:.15rem .3rem 0;
          display:flex; gap:.7rem; flex-wrap:wrap; }
  .meta b { color:#9ecbff; font-weight:600; }
  .empty { color:#778; text-align:center; margin-top:3rem; }
  .cursor { display:inline-block; width:.5ch; animation:blink 1s steps(1) infinite; color:#9ecbff; }
  @keyframes blink { 50% { opacity:0; } }
  .pulse .bubble { border-color:#3b7cf0; }
  .phase { font-style:italic; color:#8a93a6; }
  .summary { margin:1.4rem 0 .4rem; padding:.7rem .9rem; background:#1a1d24;
             border:1px solid #2c313c; border-radius:10px; font-size:.78rem; color:#aeb6c4; }
  .summary h2 { font-size:.8rem; margin:0 0 .4rem; color:#9ecbff; }
  .summary .grid { display:flex; gap:1.4rem; flex-wrap:wrap; }
  .summary b { color:#e6e6e6; font-size:1rem; display:block; }
</style></head><body>
<header>
  <h1>💬 Conversation</h1>
  <span class="dot" id="live"></span>
  <button onclick="clearChat()">Clear</button>
</header>
<div id="log"><p class="empty">No conversation yet. Say the wake word to start talking.</p></div>
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
function turnHtml(role, text, ts, meta){
  const who = role === 'user' ? 'user' : 'assistant';
  return '<div class="turn '+who+'"><div><div class="bubble">'+esc(text)+'</div>'
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
      log.innerHTML = '<p class="empty">No conversation yet. Say the wake word to start talking.</p>';
      return;
    }
    log.innerHTML = turns.map(t => turnHtml(t.role, t.text, t.ts, t.meta)).join('')
                  + (l ? liveHtml(l) : '')
                  + summaryHtml(turns);
    if (atBottom) window.scrollTo(0, document.body.scrollHeight);
  } catch(e){ dot.classList.remove('live'); }
  finally { setTimeout(poll, nextDelay); }
}
async function clearChat(){
  await fetch('/chat/clear', {method:'POST'}); lastSig = ''; poll();
}
poll();
</script>
</body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=False)
