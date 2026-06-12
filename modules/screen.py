"""
Screen viewing for Cleo. Captures a monitor through the ScreenCast portal helper
(screen_capture_helper.py, run under the system python3 since the venv has no gi)
and returns PNG bytes the model can see. After the one-time GNOME share grant,
captures are silent.

shared_monitors() lists what the persisted grant actually shares (a subset of
the physical monitors); capture() grabs one of them. Friendly labels (left /
center / right) come from the xrandr layout, ordered left-to-right.
"""
import os
import re
import json
import tempfile
import subprocess

_HELPER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screen_capture_helper.py")
_SYS_PY = "/usr/bin/python3"   # has gi + GStreamer + pipewiresrc; the venv doesn't
_MAP_PATH = os.path.expanduser("~/.cache/cleo_screencast_map.json")  # id→[x,y], written by the helper


def _pos_map():
    """Persisted id→position map (positions are unreliable on portal restore)."""
    try:
        with open(_MAP_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _run_helper(args, timeout):
    try:
        r = subprocess.run([_SYS_PY, _HELPER, *args],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.strip().splitlines()
    if not out:
        return None
    try:
        return json.loads(out[-1])   # helper prints one JSON line on stdout
    except json.JSONDecodeError:
        return None


def _xrandr_monitors():
    """Physical monitors: name, geometry, primary flag — parsed from xrandr."""
    try:
        out = subprocess.run(["xrandr", "--listmonitors"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    mons = []
    for line in out.splitlines():
        # e.g. " 0: +*DP-4 2560/600x1440/340+1920+0  DP-4"
        m = re.match(r"\s*\d+:\s+\+(\*?)\S+\s+(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)\s+(\S+)", line)
        if not m:
            continue
        mons.append({"primary": m.group(1) == "*",
                     "w": int(m.group(2)), "h": int(m.group(3)),
                     "x": int(m.group(4)), "y": int(m.group(5)),
                     "name": m.group(6)})
    return mons


def _labelled():
    """xrandr monitors with a spoken position label and a 1-based index."""
    mons = sorted(_xrandr_monitors(), key=lambda m: m["x"])
    n = len(mons)
    for i, m in enumerate(mons):
        if n <= 1:
            m["label"] = "screen"
        elif n == 2:
            m["label"] = "left" if i == 0 else "right"
        elif n == 3:
            m["label"] = ("left", "center", "right")[i]
        else:
            m["label"] = f"screen {i + 1}"
        m["index"] = i + 1
    return mons


def shared_monitors(timeout=20):
    """Monitors the persisted ScreenCast grant currently shares, each tagged with
    its friendly label. Empty if nothing is shared / the helper fails."""
    res = _run_helper(["--list"], timeout)
    if not res or not res.get("ok"):
        return []
    physical = _labelled()
    pmap = _pos_map()
    out = []
    for s in res.get("streams", []):
        sid = s.get("id", "")
        # Trust the persisted position for this stable id over the live one,
        # which GNOME often reports wrong on a restored session.
        px, py = pmap.get(sid, [s["x"], s["y"]])
        match = (min(physical, key=lambda m: abs(m["x"] - px) + abs(m["y"] - py))
                 if physical else None)
        out.append({"id": sid, "x": px, "y": py, "w": s["w"], "h": s["h"],
                    "label": match["label"] if match else "screen",
                    "name": match["name"] if match else "",
                    "index": match["index"] if match else 1,
                    "primary": match["primary"] if match else False})
    return out


def capture(monitor=None, timeout=30):
    """Grab a PNG of `monitor` (a dict from shared_monitors) or the largest shared
    screen. Returns (png_bytes, None) or (None, error_message)."""
    path = tempfile.mktemp(suffix=".png")
    args = ["--capture", path]
    if monitor and monitor.get("id"):
        args += ["--id", str(monitor["id"])]   # stable; positions drift on restore
    elif monitor:
        args += ["--x", str(monitor["x"]), "--y", str(monitor["y"])]
    res = _run_helper(args, timeout)
    if not res or not res.get("ok"):
        return None, (res or {}).get("error", "screen capture failed")
    try:
        with open(path, "rb") as f:
            return f.read(), None
    except OSError as e:
        return None, str(e)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def resolve(spoken, monitors):
    """Pick the shared monitor the user named (left/center/right, main/primary,
    a number, or the connector name). None if no confident match."""
    t = " ".join(re.sub(r"[^a-z0-9 ]", " ", (spoken or "").lower()).split())
    if not t or not monitors:
        return None
    words = set(t.split())

    pos = {"left": "left", "right": "right", "center": "center", "centre": "center",
           "middle": "center", "central": "center"}
    for w in words:
        if w in pos:
            for m in monitors:
                if m["label"] == pos[w]:
                    return m
    if words & {"main", "primary", "default"}:
        for m in monitors:
            if m.get("primary"):
                return m
        for m in monitors:
            if m["label"] == "center":
                return m

    nums = {"one": 1, "first": 1, "1": 1, "two": 2, "second": 2, "2": 2,
            "three": 3, "third": 3, "3": 3, "four": 4, "fourth": 4, "4": 4}
    for w in words:
        if w in nums:
            for m in monitors:
                if m.get("index") == nums[w]:
                    return m

    flat = t.replace(" ", "").replace("-", "")
    for m in monitors:
        if m.get("name") and m["name"].lower().replace("-", "") in flat:
            return m
    return None
