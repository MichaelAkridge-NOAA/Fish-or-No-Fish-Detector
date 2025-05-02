#!/usr/bin/env bash
set -e

export DISPLAY=:99
GAME_DIR=".devcontainer/yes-no-fish"

# ── 1. Launch virtual X server & lightweight window manager once ──────────
if ! pgrep -x Xvfb >/dev/null ; then
  echo "🎮 Starting virtual X server on $DISPLAY …"
  Xvfb :99 -screen 0 1280x800x24 &   # 1280×800 24-bit
  fluxbox 2>/tmp/fluxbox.log &       # tiny WM so pygame gets focus
  websockify --web=/usr/share/novnc 6080 localhost:5900 &  # noVNC → :6080
  echo "🔗 When the container is up, expose **Port 6080** and open the URL."
fi

# ── 2. Run the game (kills & restarts on every postStart) ─────────────────
echo "🐠 Launching Fish Arcade …"
cd "$GAME_DIR"
export SDL_VIDEODRIVER=x11          # use the X server, not dummy
python app.py
