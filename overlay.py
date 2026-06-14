#!/usr/bin/env python3
"""
Siri-style status orb — a small always-on-top, click-through overlay in the
top-right corner of the screen. A colorful blob morphs while the assistant is
listening / thinking / speaking, and fades out when idle.

Runs as its own process (system python3 — needs GTK, which the venv doesn't
have) and is driven by one-word UDP datagrams from modules/status.py:
    idle | listening | thinking | speaking

Runs on XWayland (GDK_BACKEND=x11) so the popup can position itself; native
Wayland GNOME offers no way for a window to choose its own spot.

Test a single frame without a display:  python3 overlay.py --render-test out.png
"""
import os
import sys
import json
import math
import socket

SIZE = 100        # px, requested orb size
MARGIN = 14       # px from screen edges
PORT = int(os.environ.get("OVERLAY_PORT", "5006"))
FPS = 30

ACTIVE_STATES = {"listening", "thinking", "speaking"}

# Per-state color palettes (r, g, b) — three additive blobs each.
PALETTES = {
    "listening": [(0.15, 0.75, 1.00), (0.55, 0.35, 1.00), (0.05, 1.00, 0.80)],
    "thinking":  [(0.60, 0.40, 1.00), (0.25, 0.55, 1.00), (0.95, 0.40, 0.90)],
    "speaking":  [(1.00, 0.30, 0.55), (0.20, 0.85, 1.00), (0.70, 0.45, 1.00)],
}
SPEED = {"listening": 1.0, "thinking": 0.55, "speaking": 1.6}


def draw_orb(cr, state: str, phase: float, level: float) -> None:
    """Draw one frame of the orb onto a cairo context (SIZE x SIZE)."""
    import cairo

    cr.save()
    cr.set_operator(cairo.OPERATOR_CLEAR)
    cr.paint()
    cr.restore()
    if level <= 0.01:
        return

    pal = PALETTES.get(state, PALETTES["listening"])
    cx = cy = SIZE / 2
    base = SIZE * 0.27 * (0.85 + 0.15 * level)

    # Three additive blobs, each a circle warped by sine harmonics and slowly
    # orbiting the center — the overlap of the colors gives the Siri look.
    for i, (r, g, b) in enumerate(pal):
        ph = phase * (0.9 + 0.37 * i) + i * 2.094
        amp = level * (0.10 + 0.08 * math.sin(phase * 1.9 + i * 1.3))
        ox = cx + level * 5.0 * math.sin(phase * 0.8 + i * 2.1)
        oy = cy + level * 5.0 * math.cos(phase * 0.66 + i * 1.7)

        cr.new_path()
        steps = 90
        for s in range(steps + 1):
            th = 2 * math.pi * s / steps
            wobble = (0.55 * math.sin(3 * th + ph)
                      + 0.30 * math.sin(5 * th - 1.7 * ph)
                      + 0.15 * math.sin(8 * th + 0.6 * ph))
            rad = base * (1.0 + amp * 2.2 * wobble)
            x, y = ox + rad * math.cos(th), oy + rad * math.sin(th)
            (cr.move_to if s == 0 else cr.line_to)(x, y)
        cr.close_path()

        grad = cairo.RadialGradient(ox, oy, base * 0.1, ox, oy, base * 1.35)
        grad.add_color_stop_rgba(0.0, r, g, b, 0.85 * level)
        grad.add_color_stop_rgba(0.75, r, g, b, 0.45 * level)
        grad.add_color_stop_rgba(1.0, r, g, b, 0.0)
        cr.set_source(grad)
        cr.set_operator(cairo.OPERATOR_ADD)
        cr.fill()

    # Soft white core, gently breathing.
    core = base * (0.42 + 0.05 * math.sin(phase * 2.3))
    grad = cairo.RadialGradient(cx, cy, 0, cx, cy, core)
    grad.add_color_stop_rgba(0.0, 1, 1, 1, 0.9 * level)
    grad.add_color_stop_rgba(1.0, 1, 1, 1, 0.0)
    cr.set_source(grad)
    cr.set_operator(cairo.OPERATOR_ADD)
    cr.arc(cx, cy, core, 0, 2 * math.pi)
    cr.fill()


def render_test(path: str) -> None:
    """Render one frame of each state to a PNG strip (no display needed)."""
    import cairo
    states = ["listening", "thinking", "speaking"]
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, SIZE * len(states), SIZE)
    for i, st in enumerate(states):
        sub = cairo.ImageSurface(cairo.FORMAT_ARGB32, SIZE, SIZE)
        draw_orb(cairo.Context(sub), st, phase=2.7, level=1.0)
        cr = cairo.Context(surf)
        # dark backdrop so the additive colors are visible in the PNG
        cr.rectangle(i * SIZE, 0, SIZE, SIZE)
        cr.set_source_rgb(0.08, 0.09, 0.11)
        cr.fill()
        cr.set_source_surface(sub, i * SIZE, 0)
        cr.paint()
    surf.write_to_png(path)
    print(f"Wrote {path}")


def main() -> None:
    # Must be set before GTK initializes (see module docstring).
    os.environ["GDK_BACKEND"] = "x11"
    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk, Gdk, GLib, Pango
    import cairo

    class Orb(Gtk.Window):
        def __init__(self):
            # A normal TOPLEVEL (not POPUP/override-redirect): GNOME's Mutter
            # refuses to composite standalone override-redirect XWayland windows
            # on Wayland, so the orb stayed invisible. A decoration-less TOPLEVEL
            # with a NOTIFICATION type hint *is* composited, stays on top, and
            # doesn't steal focus or show in the taskbar.
            super().__init__(type=Gtk.WindowType.TOPLEVEL)
            self.state = "idle"
            self.phase = 0.0
            self.level = 0.0
            self.ticking = False

            self.set_decorated(False)
            self.set_resizable(False)
            self.set_skip_taskbar_hint(True)
            self.set_skip_pager_hint(True)
            self.set_keep_above(True)
            self.set_accept_focus(False)
            self.set_type_hint(Gdk.WindowTypeHint.NOTIFICATION)

            self.set_app_paintable(True)
            visual = self.get_screen().get_rgba_visual()
            if visual:
                self.set_visual(visual)
            self.set_default_size(SIZE, SIZE)
            self.resize(SIZE, SIZE)
            self.connect("draw", self.on_draw)
            self.connect("realize", self.on_realize)
            self.move_to_corner()

        def move_to_corner(self):
            display = Gdk.Display.get_default()
            monitor = display.get_primary_monitor() or display.get_monitor(0)
            geo = monitor.get_geometry()
            self.move(geo.x + geo.width - SIZE - MARGIN, geo.y + MARGIN)

        def on_realize(self, _w):
            try:  # let clicks pass through the orb
                self.input_shape_combine_region(cairo.Region())
            except (TypeError, AttributeError):
                pass

        def on_draw(self, _w, cr):
            draw_orb(cr, self.state, self.phase, self.level)
            return False

        def set_state(self, state: str):
            self.state = state if state in ACTIVE_STATES else "idle"
            if self.state in ACTIVE_STATES and not self.ticking:
                self.show_all()
                self.move_to_corner()   # re-assert after map; Mutter may re-place
                self.ticking = True
                GLib.timeout_add(1000 // FPS, self.tick)

        def tick(self):
            self.phase += (1.0 / FPS) * 2.0 * SPEED.get(self.state, 1.0)
            target = 1.0 if self.state in ACTIVE_STATES else 0.0
            self.level += (target - self.level) * 0.14
            if target == 0.0 and self.level < 0.02:
                self.level = 0.0
                self.ticking = False
                self.hide()
                return False
            self.queue_draw()
            return True

    BASE = os.path.dirname(os.path.abspath(__file__))
    SETTINGS_PATH = os.path.join(BASE, "settings.json")

    def side_panel_enabled():
        """The live 'side_panel' toggle from settings.json (default on). Read on
        each poll rather than once, so turning the switch off hides the panel
        without a restart — and a stale logs/panel.json can never resurrect it
        while the feature is off. (The overlay runs under system python without
        the config module, so it reads settings.json directly.)"""
        try:
            with open(SETTINGS_PATH) as f:
                return bool(json.load(f).get("side_panel", True))
        except (OSError, ValueError):
            return True

    def tail_jsonl(path, n):
        try:
            with open(path) as f:
                lines = f.readlines()[-n:]
        except OSError:
            return []
        rows = []
        for ln in lines:
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
        return rows

    class SidePanel(Gtk.Window):
        """A docked panel on the right of the primary monitor. Pops up when the
        assistant publishes an artifact (code, a story…) to logs/panel.json: the
        conversation (as chat bubbles) sits on top, the artifact below with
        syntax highlighting and a Copy button. A slim handle on the left edge
        collapses the panel to just the handle and back. Polls the file so it
        appears on its own; a ✕ dismisses it until the next artifact."""

        POLL_MS = 700
        HANDLE_W = 26
        FLARE = 18      # px radius of the concave fillets that sweep into the screen edge
        RADIUS = 14     # px radius of the rounded left corners

        def __init__(self):
            super().__init__(type=Gtk.WindowType.TOPLEVEL)
            self.panel_path = os.path.join(BASE, "logs", "panel.json")
            self.chat_path = os.path.join(BASE, "logs", "chat.jsonl")
            self.last_ts = 0.0          # newest artifact we've shown
            self.dismissed_ts = 0.0     # artifact the user closed (don't re-pop)
            self.content = ""
            self._title = "Text"
            self._last_conv = None      # last conversation drawn (skip redraws)
            self.collapsed = False
            self._build()
            GLib.timeout_add(self.POLL_MS, self._poll)

        def _pick_monitor(self):
            """Target the main display: the primary monitor, falling back to the
            largest. (Logged so positioning can be checked from the service.)"""
            display = Gdk.Display.get_default()
            mons = [display.get_monitor(i) for i in range(display.get_n_monitors())]
            target = display.get_primary_monitor()
            if target is None and mons:
                target = max(mons, key=lambda m: m.get_geometry().width * m.get_geometry().height)
            try:
                with open(os.path.join(BASE, "logs", "overlay_debug.log"), "a") as f:
                    for i, m in enumerate(mons):
                        g = m.get_geometry()
                        star = " <-- target" if m is target else ""
                        f.write(f"mon[{i}] {m.get_model()} ({g.x},{g.y} {g.width}x{g.height}){star}\n")
            except OSError:
                pass
            return (target or mons[0]).get_geometry()

        def _build(self):
            self.set_decorated(False)
            self.set_skip_taskbar_hint(True)
            self.set_skip_pager_hint(True)
            self.set_keep_above(True)
            # DOCK: the one type Mutter does not re-place. NOTIFICATION worked for
            # the tiny orb, but a large window gets snapped to whatever monitor
            # Mutter fancies (measured: want x=3906 → forced to 4480, every time).
            # Docks position themselves (like taskbars) and still take clicks.
            self.set_type_hint(Gdk.WindowTypeHint.DOCK)
            self.set_position(Gtk.WindowPosition.NONE)   # don't let GTK auto-place us
            # Mutter doesn't give DOCK windows focus on click, which breaks
            # text-drag selection in the code view — request it explicitly.
            self.set_accept_focus(True)
            self.connect("button-press-event", self._on_button_press)

            geo = self._pick_monitor()
            self._w = min(560, int(geo.width * 0.34))
            self._h = int(geo.height * 0.80)              # 80% of the screen, centered
            self._x = geo.x + geo.width - self._w         # flush with the screen edge
            self._y = geo.y + (geo.height - self._h) // 2
            # collapsed: only the handle remains, hugging the screen edge
            self._x_col = geo.x + geo.width - self.HANDLE_W
            self.set_default_size(self._w, self._h)
            self.resize(self._w, self._h)
            self.move(self._x, self._y)
            self._apply_css()

            # Transparent window; we paint the concave edge fillets ourselves so
            # the panel appears to sweep out of the screen edge (CSS can't do
            # inverse corners). The body is inset by FLARE top and bottom.
            self.set_app_paintable(True)
            visual = self.get_screen().get_rgba_visual()
            if visual:
                self.set_visual(visual)
            self.connect("draw", self._on_flare_draw)

            # [ handle | content ] — the handle stays visible when collapsed.
            outer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            outer.get_style_context().add_class("cleo-root")
            outer.set_margin_top(self.FLARE)
            outer.set_margin_bottom(self.FLARE)
            self.add(outer)

            self.handle = Gtk.Button(label="❯")
            self.handle.set_tooltip_text("Collapse / expand")
            self.handle.get_style_context().add_class("cleo-handle")
            self.handle.connect("clicked", lambda _b: self._set_collapsed(not self.collapsed))
            outer.pack_start(self.handle, False, False, 0)

            self.body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            outer.pack_start(self.body, True, True, 0)

            header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            header.get_style_context().add_class("cleo-header")
            self.title_lbl = Gtk.Label(label="Cleo")
            self.title_lbl.set_xalign(0.0)
            self.title_lbl.get_style_context().add_class("cleo-title")
            header.pack_start(self.title_lbl, True, True, 0)
            copy_btn = Gtk.Button(label="Copy")
            copy_btn.connect("clicked", self._copy)
            header.pack_start(copy_btn, False, False, 0)
            close_btn = Gtk.Button(label="✕")
            close_btn.connect("clicked", self._on_close)
            header.pack_start(close_btn, False, False, 0)
            self.body.pack_start(header, False, False, 0)

            conv_hdr = Gtk.Label(label="CONVERSATION")
            conv_hdr.set_xalign(0.0)
            conv_hdr.get_style_context().add_class("cleo-sec")
            self.body.pack_start(conv_hdr, False, False, 0)
            self.conv_scroll = Gtk.ScrolledWindow()
            self.conv_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            self.conv_scroll.set_size_request(-1, int(self._h * 0.30))
            self.conv_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            self.conv_box.set_margin_start(8)
            self.conv_box.set_margin_end(8)
            self.conv_box.set_margin_top(4)
            self.conv_box.set_margin_bottom(8)
            self.conv_scroll.add(self.conv_box)
            self.body.pack_start(self.conv_scroll, False, False, 0)

            self.art_hdr = Gtk.Label(label="")
            self.art_hdr.set_xalign(0.0)
            self.art_hdr.get_style_context().add_class("cleo-sec")
            self.body.pack_start(self.art_hdr, False, False, 0)
            art_scroll = Gtk.ScrolledWindow()
            art_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
            self.art_view = Gtk.TextView()
            self.art_view.set_editable(False)
            self.art_view.set_monospace(True)
            self.art_view.set_wrap_mode(Gtk.WrapMode.NONE)
            self.art_view.set_left_margin(10)
            self.art_view.set_top_margin(8)
            self.art_view.get_style_context().add_class("cleo-art")
            art_scroll.add(self.art_view)
            self.body.pack_start(art_scroll, True, True, 0)
            self._make_tags()

        def _on_button_press(self, _w, _e):
            """Mutter won't focus a DOCK on click by itself — ask explicitly so
            text-drag selection works in the views."""
            gdkwin = self.get_window()
            if gdkwin is not None:
                gdkwin.focus(Gdk.CURRENT_TIME)
            return False

        def _set_collapsed(self, collapsed):
            """Slide the panel away leaving only the handle (and back)."""
            self.collapsed = collapsed
            self.handle.set_label("❮" if collapsed else "❯")
            self.body.set_visible(not collapsed)
            if collapsed:
                self.resize(self.HANDLE_W, self._h)
                self.move(self._x_col, self._y)
            else:
                self.resize(self._w, self._h)
                self.move(self._x, self._y)
            self.queue_draw()                  # flares differ between states
            GLib.timeout_add(80, self._reposition)
            GLib.timeout_add(120, self._update_input_shape)

        def _update_input_shape(self):
            """Only the visible panel takes clicks — the transparent strips above
            and below the body (left of the fillets) pass them through."""
            if self.get_window() is None:
                return False
            W = self.get_allocated_width()
            H = self.get_allocated_height()
            F = self.FLARE
            rects = [cairo.RectangleInt(0, F, W, max(1, H - 2 * F))]
            if not self.collapsed:             # the two fillet squares
                rects.append(cairo.RectangleInt(W - F, 0, F, F))
                rects.append(cairo.RectangleInt(W - F, H - F, F, F))
            try:
                self.input_shape_combine_region(cairo.Region(rects))
            except (TypeError, AttributeError):
                pass                           # old pycairo — window stays clickable
            return False

        def _on_flare_draw(self, _w, cr):
            """Window background: transparent, plus two concave fillets at the
            screen-edge side (top-right and bottom-right) so the panel looks like
            it slopes out of the edge. The body's left corners are rounded via
            CSS; the fillets are pure cairo — CSS has no inverse corners."""
            cr.save()
            cr.set_operator(cairo.OPERATOR_CLEAR)
            cr.paint()                          # fully transparent canvas
            cr.restore()
            if self.collapsed:
                return False                    # just the handle — no fillets
            W = self.get_allocated_width()
            H = self.get_allocated_height()
            F = self.FLARE
            # top fillet — header color, sweeping the header into the edge
            cr.move_to(W, 0)
            cr.arc(W - F, 0, F, 0, math.pi / 2)     # concave arc to (W-F, F)
            cr.line_to(W, F)
            cr.close_path()
            cr.set_source_rgb(0x1D / 255, 0x20 / 255, 0x27 / 255)
            cr.fill()
            # bottom fillet — terminal color, sweeping the code pane into the edge
            cr.move_to(W, H)
            cr.arc_negative(W - F, H, F, 0, -math.pi / 2)   # to (W-F, H-F)
            cr.line_to(W, H - F)
            cr.close_path()
            cr.set_source_rgb(0x0A / 255, 0x0C / 255, 0x10 / 255)
            cr.fill()
            return False

        def _apply_css(self):
            css = b"""
            .cleo-root { background:#15171c; border-radius:14px 0 0 14px; }
            .cleo-header { background:#1d2027; padding:8px; border-bottom:1px solid #2c313c; }
            .cleo-title { color:#9ecbff; font-weight:bold; }
            .cleo-sec { color:#8a93a6; font-size:10px; padding:8px 8px 4px; letter-spacing:1px; }
            /* collapse handle: slim strip, always visible; it owns the panel's
               rounded left corners (it spans the full height on the left) */
            .cleo-handle { background:#1d2027; color:#8a93a6; border:0;
                           border-radius:14px 0 0 14px;
                           padding:0 4px; border-right:1px solid #2c313c; }
            .cleo-handle:hover { background:#262b35; color:#e6e6e6; }
            /* chat bubbles */
            .bub-user { background:#2d6cdf; color:#ffffff; border-radius:13px;
                        padding:6px 11px; }
            .bub-cleo { background:#262b35; color:#e6e6e6; border:1px solid #2c313c;
                        border-radius:13px; padding:6px 11px; }
            .bub-who { color:#667; font-size:9px; padding:0 6px; }
            /* bottom section: code terminal look */
            .cleo-art { border-top:1px solid #2c313c; }
            .cleo-art, .cleo-art text { background:#0a0c10; color:#d8dee9;
                font-family:'JetBrains Mono','DejaVu Sans Mono',monospace; }
            .cleo-art text selection { background:#264f78; color:#ffffff; }
            .cleo-art text selection:backdrop { background:#264f78; color:#ffffff; }
            button { background:#2d6cdf; color:#ffffff; border:0; border-radius:5px; padding:3px 10px; }
            button:hover { background:#3b7cf0; }
            """
            prov = Gtk.CssProvider()
            prov.load_from_data(css)
            Gtk.StyleContext.add_provider_for_screen(
                self.get_screen(), prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        def _copy(self, _b):
            cb = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
            cb.set_text(self.content, -1)
            cb.store()
            self.title_lbl.set_text("Copied ✓")
            GLib.timeout_add(1200, lambda: (self.title_lbl.set_text(self._title), False)[1])

        def _on_close(self, _b):
            self.dismissed_ts = self.last_ts
            self.hide()

        def _refresh_conversation(self):
            import re as _re
            turns = tail_jsonl(self.chat_path, 14)
            # Code blocks live in the artifact section — keep bubbles readable.
            for t in turns:
                t["text"] = _re.sub(r"```.*?```", "〔code below〕",
                                    t.get("text") or "", flags=_re.S).strip()
            sig = json.dumps(turns)
            if sig == self._last_conv:
                return                       # unchanged — don't rebuild (avoids scroll jump)
            self._last_conv = sig

            for child in self.conv_box.get_children():
                self.conv_box.remove(child)
            for t in turns:
                if not t["text"]:
                    continue
                is_user = t.get("role") == "user"
                row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
                who = Gtk.Label(label="You" if is_user else "Cleo")
                who.set_halign(Gtk.Align.END if is_user else Gtk.Align.START)
                who.get_style_context().add_class("bub-who")
                lbl = Gtk.Label(label=t["text"])
                lbl.set_line_wrap(True)
                lbl.set_line_wrap_mode(Pango.WrapMode.WORD_CHAR)
                lbl.set_max_width_chars(38)
                lbl.set_xalign(0.0)
                lbl.set_selectable(True)
                lbl.get_style_context().add_class("bub-user" if is_user else "bub-cleo")
                lbl.set_halign(Gtk.Align.END if is_user else Gtk.Align.START)
                row.pack_start(who, False, False, 0)
                row.pack_start(lbl, False, False, 0)
                self.conv_box.pack_start(row, False, False, 0)
            self.conv_box.show_all()
            GLib.timeout_add(60, self._scroll_conv_to_end)

        def _scroll_conv_to_end(self):
            adj = self.conv_scroll.get_vadjustment()
            adj.set_value(adj.get_upper() - adj.get_page_size())
            return False

        # ------ artifact syntax highlighting (no GtkSourceView dependency) ------
        _KEYWORDS = {
            "python": "def class return if elif else for while in not and or is None True False "
                      "import from as with try except finally raise lambda yield pass break "
                      "continue global nonlocal assert del async await match case",
            "javascript": "function return if else for while do switch case break continue new "
                          "var let const class extends super this typeof instanceof in of null "
                          "undefined true false async await yield import export from default "
                          "try catch finally throw delete void",
            "bash": "if then else elif fi for while do done case esac function in echo exit "
                    "return local export readonly shift source true false",
            "c": "int char float double void long short unsigned signed struct union enum "
                 "typedef static const return if else for while do switch case break continue "
                 "goto sizeof extern volatile inline NULL true false bool include define",
        }
        _KEYWORDS["js"] = _KEYWORDS["typescript"] = _KEYWORDS["javascript"]
        _KEYWORDS["sh"] = _KEYWORDS["shell"] = _KEYWORDS["bash"]
        _KEYWORDS["cpp"] = _KEYWORDS["c++"] = _KEYWORDS["java"] = _KEYWORDS["c"]

        def _make_tags(self):
            b = self.art_view.get_buffer()
            self._tags = {
                "kw":  b.create_tag("kw",  foreground="#c678dd"),                  # keywords: purple
                "str": b.create_tag("str", foreground="#98c379"),                  # strings: green
                "com": b.create_tag("com", foreground="#5c6370",
                                    style=Pango.Style.ITALIC),                     # comments: gray italic
                "num": b.create_tag("num", foreground="#d19a66"),                  # numbers: orange
                "fn":  b.create_tag("fn",  foreground="#61afef"),                  # calls/defs: blue
                "dec": b.create_tag("dec", foreground="#e5c07b"),                  # decorators: yellow
            }

        def _set_artifact(self, content, title, kind):
            import re as _re
            b = self.art_view.get_buffer()
            b.set_text(content)
            if kind != "code":
                return
            lang = (title.split() or ["?"])[0].lower()
            kws = self._KEYWORDS.get(lang, self._KEYWORDS["python"])

            def apply(tag, start, end):
                b.apply_tag(self._tags[tag], b.get_iter_at_offset(start),
                            b.get_iter_at_offset(end))

            protected = []   # string/comment spans — keywords don't apply inside
            com_re = r"#[^\n]*" if lang in ("python", "bash", "sh", "shell") \
                     else r"//[^\n]*|/\*.*?\*/"
            for pat, tag in ((r'""".*?"""|\'\'\'.*?\'\'\'|"(?:\\.|[^"\\\n])*"'
                              r"|'(?:\\.|[^'\\\n])*'", "str"), (com_re, "com")):
                for m in _re.finditer(pat, content, _re.S):
                    if any(m.start() < e and m.end() > s for s, e in protected):
                        continue   # e.g. a '#' inside a string is not a comment
                    apply(tag, m.start(), m.end())
                    protected.append((m.start(), m.end()))

            def free(m):
                return not any(m.start() < e and m.end() > s for s, e in protected)

            for m in _re.finditer(r"\b(?:%s)\b" % "|".join(kws.split()), content):
                if free(m):
                    apply("kw", m.start(), m.end())
            for m in _re.finditer(r"\b\d+(?:\.\d+)?\b", content):
                if free(m):
                    apply("num", m.start(), m.end())
            for m in _re.finditer(r"\b([A-Za-z_]\w*)\s*\(", content):
                if free(m) and m.group(1) not in kws.split():
                    apply("fn", m.start(1), m.end(1))
            for m in _re.finditer(r"^\s*@[\w.]+", content, _re.M):
                if free(m):
                    apply("dec", m.start(), m.end())

        def _debug(self, msg):
            try:
                with open(os.path.join(BASE, "logs", "overlay_debug.log"), "a") as f:
                    f.write(msg + "\n")
            except OSError:
                pass

        def _reposition(self, attempt=0):
            """Move to the desired spot (which depends on collapsed state), then
            measure where the WM actually put us and re-move compensating for the
            offset. Repeats a few times because Mutter can re-place after map."""
            tx = self._x_col if self.collapsed else self._x
            ax, ay = self.get_position()
            dx, dy = ax - tx, ay - self._y
            if abs(dx) > 2 or abs(dy) > 2:
                self._debug(f"reposition[{attempt}]: want=({tx},{self._y}) "
                            f"actual=({ax},{ay}) delta=({dx},{dy})")
                self.move(tx - dx, self._y - dy)   # counteract the WM's offset
            if attempt < 4:
                GLib.timeout_add(120, self._reposition, attempt + 1)
            return False

        def command(self, cmd):
            """Voice-driven visibility: 'show' pops the panel (expanded, with
            whatever conversation/artifact exists); 'hide' removes it entirely.
            The collapse handle stays a manual, on-panel control."""
            if cmd == "show":
                if not side_panel_enabled():
                    return                   # feature off — ignore show requests
                self._refresh_conversation()
                self.move(self._x, self._y)
                self.show_all()
                self._set_collapsed(False)
                GLib.timeout_add(150, self._update_input_shape)
            elif cmd == "hide":
                self.hide()

        def _poll(self):
            if not side_panel_enabled():      # switch off → keep the panel hidden
                if self.get_visible():
                    self.hide()
                return True
            if self.get_visible() and not self.collapsed:
                self._refresh_conversation()
            try:
                with open(self.panel_path) as f:
                    d = json.load(f)
            except (OSError, ValueError):
                return True
            ts = d.get("ts", 0.0)
            if ts and ts > self.last_ts and ts != self.dismissed_ts:
                self.last_ts = ts
                self.content = d.get("content", "")
                self._title = d.get("title", "Text")
                self.title_lbl.set_text(self._title)
                self.art_hdr.set_text(self._title.upper())
                self._set_artifact(self.content, self._title, d.get("kind", "text"))
                self._refresh_conversation()
                self.move(self._x, self._y)
                self.show_all()
                self._set_collapsed(False)   # new content always expands the panel
                GLib.timeout_add(150, self._update_input_shape)
            return True

    orb = Orb()
    side = SidePanel()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", PORT))
    except OSError:
        sys.exit(0)   # another overlay instance already owns the port
    sock.setblocking(False)

    # Exit when the parent (assistant.py) goes away: it holds our stdin pipe,
    # so a hangup on stdin means the assistant died — even by SIGKILL.
    if not sys.stdin.isatty():
        GLib.io_add_watch(sys.stdin.fileno(), GLib.IO_HUP | GLib.IO_ERR,
                          lambda *_a: (Gtk.main_quit(), False)[1])

    def on_udp(_fd, _cond):
        try:
            while True:
                data, _ = sock.recvfrom(64)
                msg = data.decode(errors="ignore").strip()
                if msg.startswith("panel:"):          # side-panel show/hide
                    side.command(msg.split(":", 1)[1])
                else:
                    orb.set_state(msg)
        except BlockingIOError:
            pass
        return True

    GLib.io_add_watch(sock.fileno(), GLib.IO_IN, on_udp)
    Gtk.main()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--render-test":
        render_test(sys.argv[2])
    else:
        main()
