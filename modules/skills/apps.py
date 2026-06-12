"""
Desktop-control skills: open applications, open websites, and play a song on
YouTube. These let the user say things like "open Firefox", "open YouTube", or
"play Bohemian Rhapsody on YouTube" and have Cleo actually launch it.

Everything runs on the user's graphical session (DISPLAY / WAYLAND_DISPLAY).
Apps are launched detached (Popen, no wait) so a turn never blocks on the app;
nothing is read back from the launched process.
"""
import os
import re
import glob
import shlex
import urllib.parse
import urllib.request
import urllib.error
import subprocess

import config
from modules import phrases
from modules.skills import skill


# --------------------------------------------------------------------------
# Launching helpers
# --------------------------------------------------------------------------
def _launch_env() -> dict:
    """Environment for a launched GUI app, minus snap-injected library paths
    (present when the assistant is started from e.g. a VSCode snap terminal)
    that break system GTK/Qt apps. Mirrors assistant._clean_env."""
    env = {}
    for k, v in os.environ.items():
        if k.startswith("SNAP") or k.endswith("_VSCODE_SNAP_ORIG"):
            continue
        if k == "PATH":
            env[k] = ":".join(p for p in v.split(":") if "/snap/" not in p)
        elif "/snap/" in v:
            orig = os.environ.get(f"{k}_VSCODE_SNAP_ORIG", "")
            if orig:
                env[k] = orig
        else:
            env[k] = v
    return env


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _spawn(argv: list[str]) -> None:
    """Start a process fully detached from the assistant (its own session, no
    inherited stdio), so it outlives the turn and never ties up the model."""
    subprocess.Popen(
        argv, env=_launch_env(), start_new_session=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _open_url(url: str) -> None:
    _spawn(["xdg-open", url])


# --------------------------------------------------------------------------
# Application index (parsed from .desktop entries, fuzzy-matched by name)
# --------------------------------------------------------------------------
_APP_DIRS = [
    "/usr/share/applications",
    "/usr/local/share/applications",
    os.path.expanduser("~/.local/share/applications"),
    "/var/lib/flatpak/exports/share/applications",
    os.path.expanduser("~/.local/share/flatpak/exports/share/applications"),
    "/var/lib/snapd/desktop/applications",
]

_app_index: list[dict] | None = None


def _parse_desktop(path: str):
    """Pull the launchable bits out of a .desktop file: display name, the id
    gtk-launch uses (basename without .desktop), and whether it's hidden."""
    name = None
    no_display = hidden = is_app = False
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            in_entry = False
            for line in f:
                line = line.strip()
                if line.startswith("["):
                    in_entry = line == "[Desktop Entry]"
                    continue
                if not in_entry or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if key == "Name" and name is None:
                    name = val
                elif key == "Type":
                    is_app = val == "Application"
                elif key == "NoDisplay":
                    no_display = val.lower() == "true"
                elif key == "Hidden":
                    hidden = val.lower() == "true"
    except OSError:
        return None
    if not name or not is_app or no_display or hidden:
        return None
    return {"name": name, "id": os.path.basename(path)[:-len(".desktop")]}


def _build_index() -> list[dict]:
    seen, apps = set(), []
    for d in _APP_DIRS:
        for path in glob.glob(os.path.join(d, "*.desktop")):
            entry = _parse_desktop(path)
            if entry and entry["id"] not in seen:
                seen.add(entry["id"])
                apps.append(entry)
    return apps


def _apps() -> list[dict]:
    global _app_index
    if _app_index is None:
        _app_index = _build_index()
    return _app_index


def _find_app(query: str):
    """Best fuzzy match for a spoken app name, or None if nothing's close.
    Prefers a substring hit, then falls back to token similarity."""
    q = phrases.normalize(query)
    if not q:
        return None
    best, best_score = None, 0.0
    for app in _apps():
        n = phrases.normalize(app["name"])
        if not n:
            continue
        if q == n or q in n.split() or n in q:
            return app                      # exact / whole-word / contained
        score = phrases.similarity(q, n)
        if q in n or n in q:                # partial substring — strong signal
            score = max(score, 0.85)
        if score > best_score:
            best, best_score = app, score
    return best if best_score >= 0.6 else None


# --------------------------------------------------------------------------
# Website resolution — no per-site table. A name like "youtube" becomes a domain
# guess ("youtube.com") that we verify actually resolves; anything unguessable
# falls back to a web search. So new sites work without ever editing this file.
# --------------------------------------------------------------------------
def _reachable(url: str) -> bool:
    """True if `url` responds without a client/server error. Follows redirects
    (urllib does so for GET), so e.g. twitch.com -> twitch.tv counts as reachable."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status < 400
    except urllib.error.HTTPError as e:
        return e.code < 400          # a 3xx/401/403 still means the site exists
    except (urllib.error.URLError, OSError):
        return False


def _resolve_site(target: str) -> str | None:
    """Turn a website name or URL into a full URL.
      - a full URL or bare domain ('example.com') -> opened directly
      - a single-word name ('youtube', 'github')  -> '<name>.com' if it resolves
      - anything else (multi-word, no domain)      -> a web search for it
    Returns None only for empty input."""
    t = (target or "").strip()
    if not t:
        return None
    if re.match(r"^https?://", t, re.I):
        return t
    # A bare domain like "example.com" or "maps.google.com" — open it directly.
    if re.match(r"^[\w-]+(\.[\w-]+)+(/.*)?$", t):
        return "https://" + t
    # A single bare word ("youtube", "reddit"): guess the .com and verify it's real.
    slug = re.sub(r"[^a-z0-9]", "", t.lower())
    if slug and " " not in t.strip():
        guess = f"https://{slug}.com"
        if _reachable(guess):
            return guess
    # Multi-word or unresolvable — search for it instead of failing.
    return "https://www.google.com/search?q=" + urllib.parse.quote(t)


# --------------------------------------------------------------------------
# YouTube: resolve a search query to its first video and play it
# --------------------------------------------------------------------------
def _first_youtube_id(query: str) -> str | None:
    """Scrape the first video id off YouTube's search results page. Returns None
    on any network/parse failure (caller falls back to the search page)."""
    url = ("https://www.youtube.com/results?search_query="
           + urllib.parse.quote(query) + "&sp=EgIQAQ%253D%253D")  # sp = filter: videos only
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", "ignore")
    except (urllib.error.URLError, OSError):
        return None
    m = re.search(r'"videoId":"([\w-]{11})"', html)
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# Skills
# --------------------------------------------------------------------------
@skill("open_app",
       "Open or launch a desktop application by name (e.g. Firefox, files, calculator).",
       {"name": "the application's name as the user said it"})
def _open_app(args):
    if not _has_display():
        return {"error": "no graphical session available to open apps"}
    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "no app name given"}
    app = _find_app(name)
    if not app:
        return {"error": f"couldn't find an app called '{name}'"}
    try:
        _spawn(["gtk-launch", app["id"]])
    except OSError as e:
        return {"error": f"failed to launch: {e}"}
    return {"ok": True, "opened": app["name"]}


@skill("open_website",
       "Open a website in the browser, by name (youtube, gmail, reddit, ...), a "
       "domain, or a full URL. Unknown names fall back to a web search.",
       {"target": "a site name, a domain, or a full https URL"})
def _open_website(args):
    if not _has_display():
        return {"error": "no graphical session available to open a browser"}
    target = (args.get("target") or args.get("name") or args.get("url") or "").strip()
    url = _resolve_site(target)
    if not url:
        return {"error": "no website given"}
    _open_url(url)
    return {"ok": True, "opened": target or url}


@skill("play_youtube",
       "Search YouTube for a song or video and start playing the first result.",
       {"query": "what to play, e.g. 'Bohemian Rhapsody' or 'lofi study mix'"})
def _play_youtube(args):
    if not _has_display():
        return {"error": "no graphical session available to play video"}
    query = (args.get("query") or args.get("song") or args.get("name") or "").strip()
    if not query:
        return {"error": "nothing to play — no search query given"}
    video_id = _first_youtube_id(query)
    if video_id:
        # autoplay=1 makes the watch page start playing without a click.
        _open_url(f"https://www.youtube.com/watch?v={video_id}&autoplay=1")
        return {"ok": True, "playing": query}
    # Couldn't resolve a specific video — fall back to the search results page.
    _open_url("https://www.youtube.com/results?search_query=" + urllib.parse.quote(query))
    return {"ok": True, "searched": query,
            "note": "opened YouTube search results (couldn't auto-pick a video)"}
