#!/usr/bin/env python3
"""
Screen capture via the XDG ScreenCast portal + PipeWire, for GNOME/Wayland where
nothing else can grab the screen non-interactively.

Runs under the SYSTEM python3 (needs python3-gi + GStreamer + pipewiresrc, which
the venv lacks) — the assistant shells out to it. The portal grant is remembered
via a restore token (persist_mode=2): the user approves a "share your screen"
dialog once, after which captures are silent.

Usage:
  screen_capture_helper.py --list                 # JSON of shared streams (geometry)
  screen_capture_helper.py --capture OUT.png [--x X --y Y]   # grab one stream

--capture with --x/--y grabs the stream whose monitor sits at that logical
position (matching xrandr); without it, the first/largest stream. On success it
prints JSON {"ok": true, "stream": {...}} to stdout; on failure {"ok": false,
"error": "..."}. The first run (no token yet) raises the GNOME permission dialog.
"""
import os
import sys
import json
import argparse

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gio, GLib, Gst

TOKEN_PATH = os.path.expanduser("~/.cache/cleo_screencast_token.txt")
# GNOME reports correct stream positions only on a fresh/clean list; on later
# restores two monitors can report the same position. The stream "id" stays
# stable, so we remember id→position from any all-distinct (trustworthy) list and
# rely on that thereafter.
MAP_PATH = os.path.expanduser("~/.cache/cleo_screencast_map.json")

# Portal source-type / mode flags
MONITOR, WINDOW, VIRTUAL = 1, 2, 4
CURSOR_EMBEDDED = 2
PERSIST_PERSISTENT = 2


def _read_token():
    try:
        with open(TOKEN_PATH) as f:
            return f.read().strip()
    except OSError:
        return ""


def _write_token(tok):
    if not tok:
        return
    try:
        os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
        with open(TOKEN_PATH, "w") as f:
            f.write(tok)
    except OSError:
        pass


class ScreenCast:
    """Drives one CreateSession → SelectSources → Start → OpenPipeWireRemote
    portal exchange, restoring the remembered grant when possible."""

    PORTAL = "org.freedesktop.portal.Desktop"
    OBJ = "/org/freedesktop/portal/desktop"

    def __init__(self):
        self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self.unique = self.bus.get_unique_name()[1:].replace(".", "_")
        self.loop = GLib.MainLoop()
        self._counter = 0
        self.session = None
        self.streams = []
        self.restore_token = ""

    def _new_token(self, kind):
        self._counter += 1
        return f"cleo_{kind}_{self._counter}"

    def _call(self, method, body, on_response):
        """Call a portal method that replies via a Request Response signal."""
        token = self._new_token("req")
        handle = f"{self.OBJ}/request/{self.unique}/{token}"
        # options is always the last entry of body — inject our handle_token
        body[-1]["handle_token"] = GLib.Variant("s", token)

        sub = self.bus.signal_subscribe(
            self.PORTAL, "org.freedesktop.portal.Request", "Response", handle,
            None, Gio.DBusSignalFlags.NONE,
            lambda *a: on_response(a[-1].unpack()))
        self.bus.call_sync(self.PORTAL, self.OBJ, "org.freedesktop.portal.ScreenCast",
                           method, GLib.Variant(self._sig(method), tuple(body)),
                           None, Gio.DBusCallFlags.NONE, -1, None)
        return sub

    @staticmethod
    def _sig(method):
        return {
            "CreateSession": "(a{sv})",
            "SelectSources": "(oa{sv})",
            "Start": "(osa{sv})",
        }[method]

    def run(self, watchdog=40):
        GLib.timeout_add_seconds(watchdog, self._watchdog)
        self._create_session()
        self.loop.run()
        return self.streams

    def _watchdog(self):
        self._fail("timed out waiting for the screen-share dialog/response")
        return False

    def _create_session(self):
        print("[helper] CreateSession…", file=sys.stderr, flush=True)
        opts = {"session_handle_token": GLib.Variant("s", self._new_token("sess"))}
        self._call("CreateSession", [opts], self._on_session)

    def _on_session(self, resp):
        code, results = resp
        print(f"[helper] CreateSession response code={code}", file=sys.stderr, flush=True)
        if code != 0:
            self._fail("CreateSession denied")
            return
        self.session = results["session_handle"]
        self._select_sources()

    def _select_sources(self):
        opts = {
            "types": GLib.Variant("u", MONITOR | VIRTUAL),
            "multiple": GLib.Variant("b", True),
            "cursor_mode": GLib.Variant("u", CURSOR_EMBEDDED),
            "persist_mode": GLib.Variant("u", PERSIST_PERSISTENT),
        }
        tok = _read_token()
        if tok:
            opts["restore_token"] = GLib.Variant("s", tok)
        self._call("SelectSources", [self.session, opts], self._on_sources)

    def _on_sources(self, resp):
        code, _ = resp
        print(f"[helper] SelectSources response code={code}", file=sys.stderr, flush=True)
        if code != 0:
            self._fail("SelectSources denied")
            return
        self._start()

    def _start(self):
        print("[helper] Start (dialog should appear now)…", file=sys.stderr, flush=True)
        self._call("Start", [self.session, "", {}], self._on_start)

    def _on_start(self, resp):
        code, results = resp
        print(f"[helper] Start response code={code} streams={len(results.get('streams', []))}",
              file=sys.stderr, flush=True)
        if code != 0:
            self._fail("screen-share permission denied")
            return
        _write_token(results.get("restore_token", ""))
        for node_id, props in results.get("streams", []):
            pos = props.get("position", (0, 0))
            size = props.get("size", (0, 0))
            self.streams.append({"node": int(node_id),
                                 "x": int(pos[0]), "y": int(pos[1]),
                                 "w": int(size[0]), "h": int(size[1]),
                                 "id": props.get("id", ""),
                                 "mapping_id": props.get("mapping_id", ""),
                                 "source_type": int(props.get("source_type", 0))})
        self._maybe_remember_positions()
        self.loop.quit()

    def _maybe_remember_positions(self):
        """Persist id→position when this list is trustworthy (all positions
        distinct) — later restores can report duplicates we shouldn't trust."""
        positions = [(s["x"], s["y"]) for s in self.streams]
        if not positions or len(set(positions)) != len(positions):
            return
        try:
            os.makedirs(os.path.dirname(MAP_PATH), exist_ok=True)
            with open(MAP_PATH, "w") as f:
                json.dump({s["id"]: [s["x"], s["y"]] for s in self.streams}, f)
        except OSError:
            pass

    def open_pipewire_fd(self):
        ret, fdlist = self.bus.call_with_unix_fd_list_sync(
            self.PORTAL, self.OBJ, "org.freedesktop.portal.ScreenCast",
            "OpenPipeWireRemote", GLib.Variant("(oa{sv})", (self.session, {})),
            GLib.VariantType("(h)"), Gio.DBusCallFlags.NONE, -1, None, None)
        return fdlist.get(ret.unpack()[0])

    def _fail(self, msg):
        print(json.dumps({"ok": False, "error": msg}))
        self.loop.quit()
        sys.exit(2)


def _grab_frame(fd, node, out_path):
    """Pull a frame from the PipeWire node and write it as PNG via GStreamer.
    Warms up a few frames first so we don't snapshot a blank initial buffer."""
    Gst.init(None)
    desc = (f"pipewiresrc fd={fd} path={node} always-copy=true do-timestamp=true "
            f"! video/x-raw ! videoconvert ! videorate ! image/png "
            f"! pngenc snapshot=false ! multifilesink location={out_path}")
    # Simpler & robust: grab N buffers, keep the last written file.
    desc = (f"pipewiresrc fd={fd} path={node} num-buffers=6 always-copy=true "
            f"! videoconvert ! pngenc snapshot=false "
            f"! multifilesink location={out_path} max-files=1")
    pipeline = Gst.parse_launch(desc)
    pipeline.set_state(Gst.State.PLAYING)
    msg = pipeline.get_bus().timed_pop_filtered(
        10 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipeline.set_state(Gst.State.NULL)
    if msg and msg.type == Gst.MessageType.ERROR:
        err, _ = msg.parse_error()
        raise RuntimeError(str(err))
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("no frame captured")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--capture", metavar="OUT")
    ap.add_argument("--x", type=int)
    ap.add_argument("--y", type=int)
    ap.add_argument("--id", dest="stream_id",
                    help="capture the stream with this stable portal id (preferred)")
    ap.add_argument("--watchdog", type=int, default=40,
                    help="seconds to wait for the portal/dialog before giving up")
    args = ap.parse_args()

    sc = ScreenCast()
    try:
        streams = sc.run(watchdog=args.watchdog)
    except GLib.Error as e:
        print(json.dumps({"ok": False, "error": f"portal error: {e}"}))
        return 2
    if not streams:
        print(json.dumps({"ok": False, "error": "no streams shared"}))
        return 2

    if args.list:
        print(json.dumps({"ok": True, "streams": streams}))
        return 0

    # Prefer the stable id (positions are unreliable on restore); fall back to
    # position, then to the widest stream (most likely the main display).
    target = None
    if args.stream_id is not None:
        target = next((s for s in streams if s["id"] == args.stream_id), None)
    if target is None and args.x is not None and args.y is not None:
        target = min(streams, key=lambda s: abs(s["x"] - args.x) + abs(s["y"] - args.y))
    if target is None:
        target = max(streams, key=lambda s: s["w"] * s["h"])

    fd = sc.open_pipewire_fd()
    try:
        _grab_frame(fd, target["node"], args.capture)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 2
    print(json.dumps({"ok": True, "stream": target}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
