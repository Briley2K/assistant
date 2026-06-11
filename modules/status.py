"""
Fire-and-forget status broadcasts to the orb overlay (overlay.py).
States: idle | listening | thinking | speaking
"""
import socket

import config

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def set_state(state: str) -> None:
    try:
        _sock.sendto(state.encode(), ("127.0.0.1", config.OVERLAY_PORT))
    except OSError:
        pass
