#!/usr/bin/env bash
# PyFlayer installer for Raspberry Pi OS (64-bit Bookworm/Trixie) and other Debian/Ubuntu systems.
#
#   ./install.sh                  install app + desktop shortcut + download server/Java/game data
#   ./install.sh --nightly 23:00  also train automatically every night at 23:00 for 7 hours
#   ./install.sh --no-setup       skip the (one-time ~150 MB) Minecraft/Java download
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$APP_DIR/.venv"
NIGHTLY=""
DO_SETUP=1
while [ $# -gt 0 ]; do
  case "$1" in
    --nightly) NIGHTLY="${2:-23:00}"; shift 2 ;;
    --no-setup) DO_SETUP=0; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING:\033[0m %s\n' "$*"; }

ARCH="$(uname -m)"
RAM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
say "PyFlayer installer - $ARCH, ${RAM_MB} MB RAM"
if [ "$ARCH" = "armv7l" ] || [ "$ARCH" = "armv6l" ]; then
  warn "32-bit OS detected. Minecraft 26.x needs Java 25, which is only built for 64-bit ARM."
  warn "Install the 64-bit Raspberry Pi OS to run the server locally, or set server.host to another machine."
fi
if [ "$RAM_MB" -lt 3500 ]; then
  warn "Less than 4 GB RAM: the local server will get ~1-1.8 GB. A Pi 4/5 with 4-8 GB is recommended."
  warn "Consider a bigger swap file: sudo dphys-swapfile swapoff; sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile; sudo dphys-swapfile setup; sudo dphys-swapfile swapon"
fi

# ---------------------------------------------------------------- system packages
PKGS="python3 python3-venv python3-pip python3-tk python3-numpy ca-certificates"
MISSING=""
for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"; done
if [ -n "$MISSING" ]; then
  say "Installing system packages:$MISSING"
  if command -v sudo >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then SUDO=sudo; else SUDO=""; fi
  $SUDO apt-get update -q
  $SUDO apt-get install -y -q $MISSING
fi

# ---------------------------------------------------------------- python environment
if [ ! -x "$VENV/bin/python" ]; then
  say "Creating virtual environment (.venv, using system numpy/tk)"
  python3 -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$APP_DIR/requirements.txt"
"$VENV/bin/python" -c "import tkinter, numpy, openai" || { echo "python dependencies missing"; exit 1; }

# ---------------------------------------------------------------- launcher
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/pyflayer" <<EOF
#!/usr/bin/env bash
cd "$APP_DIR" && exec "$VENV/bin/python" -m pyflayer "\$@"
EOF
chmod +x "$HOME/.local/bin/pyflayer"
chmod +x "$APP_DIR/run.sh" 2>/dev/null || true

# ---------------------------------------------------------------- desktop entry
mkdir -p "$HOME/.local/share/applications"
DESKTOP_FILE="$HOME/.local/share/applications/pyflayer.desktop"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=PyFlayer AI
Comment=Minecraft 26.x AI bot with local server and overnight training
Exec=$HOME/.local/bin/pyflayer gui
Icon=$APP_DIR/assets/pyflayer-256.png
Terminal=false
Categories=Game;Education;
StartupNotify=true
EOF
chmod +x "$DESKTOP_FILE"
if [ -d "$HOME/Desktop" ]; then
  cp "$DESKTOP_FILE" "$HOME/Desktop/pyflayer.desktop"
  chmod +x "$HOME/Desktop/pyflayer.desktop"
  command -v gio >/dev/null 2>&1 && gio set "$HOME/Desktop/pyflayer.desktop" metadata::trusted true 2>/dev/null || true
fi

# ---------------------------------------------------------------- systemd user units
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/pyflayer-train.service" <<EOF
[Unit]
Description=PyFlayer overnight Minecraft AI training
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$VENV/bin/python -m pyflayer train
Nice=5
TimeoutStopSec=120
KillSignal=SIGINT
EOF
if [ -n "$NIGHTLY" ]; then
  cat > "$UNIT_DIR/pyflayer-train.timer" <<EOF
[Unit]
Description=Start PyFlayer training every night

[Timer]
OnCalendar=*-*-* $NIGHTLY:00
Persistent=false

[Install]
WantedBy=timers.target
EOF
  systemctl --user daemon-reload 2>/dev/null || true
  systemctl --user enable --now pyflayer-train.timer 2>/dev/null && say "Nightly training enabled at $NIGHTLY" \
    || warn "could not enable the systemd user timer (run: systemctl --user enable --now pyflayer-train.timer)"
  command -v loginctl >/dev/null 2>&1 && loginctl enable-linger "$USER" 2>/dev/null || true
else
  systemctl --user daemon-reload 2>/dev/null || true
fi

# ---------------------------------------------------------------- one-time downloads
if [ "$DO_SETUP" = "1" ]; then
  say "Downloading Minecraft server, Java runtime and generating game data (one time)..."
  "$VENV/bin/python" -m pyflayer setup
fi

say "Done!"
echo "  Start the app:        PyFlayer AI (menu: Games) or  ~/.local/bin/pyflayer gui"
echo "  Set your AI endpoint: ~/.local/bin/pyflayer config llm.base_url=http://HOST:PORT/v1 llm.api_key=KEY llm.model=MODEL"
echo "  Train overnight now:  ~/.local/bin/pyflayer train --hours 7     (or: systemctl --user start pyflayer-train)"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "  (add ~/.local/bin to your PATH to type just 'pyflayer')";; esac
