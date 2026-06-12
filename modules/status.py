"""
Fire-and-forget status broadcasts to the orb overlay (overlay.py).
States: idle | listening | thinking | speaking
Side-panel commands are sent on the same socket as "panel:show" / "panel:hide".
"""
import socket

import config

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def set_state(state: str) -> None:
    try:
        _sock.sendto(state.encode(), ("127.0.0.1", config.OVERLAY_PORT))
    except OSError:
        pass


def panel(cmd: str) -> None:
    """Side-panel visibility command for the overlay: 'show' or 'hide'."""
    try:
        _sock.sendto(f"panel:{cmd}".encode(), ("127.0.0.1", config.OVERLAY_PORT))
    except OSError:
        pass
