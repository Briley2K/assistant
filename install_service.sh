#!/usr/bin/env bash
# Installs the systemd *user* services (no sudo needed) and a desktop launcher.
set -e

BASE="$(cd "$(dirname "$0")" && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
APP_DIR="$HOME/.local/share/applications"

# Both services run "$BASE/.venv/bin/python3 ...", so the environment must exist
# first — otherwise the units install but silently fail to start.
if ! "$BASE/.venv/bin/python3" -m pip --version >/dev/null 2>&1; then
    echo "ERROR: Python environment not ready at $BASE/.venv"
    echo "Run the installer first:   bash setup.sh"
    exit 1
fi

mkdir -p "$UNIT_DIR" "$APP_DIR"

echo "=== Installing systemd user units ==="
cp "$BASE/systemd/voice-assistant.service"       "$UNIT_DIR/"
cp "$BASE/systemd/voice-assistant-panel.service" "$UNIT_DIR/"
systemctl --user daemon-reload

echo "=== Enabling + starting the control panel ==="
systemctl --user enable --now voice-assistant-panel.service

echo "=== Installing app launcher ==="
cp "$BASE/voice-assistant.desktop" "$APP_DIR/"
update-desktop-database "$APP_DIR" 2>/dev/null || true

cat <<EOF

=== Done ===
Control panel:   http://localhost:5005   (also in your app menu: "Voice Assistant")

From the panel you can Start the assistant and toggle "Enable autostart".
Or from the terminal:
    systemctl --user enable --now voice-assistant.service   # run now + on login

To start before you log in (e.g. headless boot):
    sudo loginctl enable-linger $USER
EOF
