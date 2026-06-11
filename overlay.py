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
    from gi.repository import Gtk, Gdk, GLib
    import cairo

    class Orb(Gtk.Window):
        def __init__(self):
            # POPUP = unmanaged window: no decorations, stays on top, and we
            # control its position ourselves (works through XWayland).
            super().__init__(type=Gtk.WindowType.POPUP)
            self.state = "idle"
            self.phase = 0.0
            self.level = 0.0
            self.ticking = False

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
                self.move_to_corner()
                self.show_all()
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

    orb = Orb()

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
                orb.set_state(data.decode(errors="ignore").strip())
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
