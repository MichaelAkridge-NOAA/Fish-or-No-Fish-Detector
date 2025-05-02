#!/usr/bin/env bash
set -e

export DISPLAY=:99
export SDL_VIDEODRIVER=x11
export SDL_AUDIODRIVER=dummy
GAME_DIR=".devcontainer/yes-no-fish"

# ── 1.  Start Xvfb + WM +  x11vnc  + websockify  (once) ───────────────
if ! pgrep -x Xvfb >/dev/null ; then
  echo "🎮 Starting virtual X server on $DISPLAY …"
  Xvfb :99 -screen 0 1280x800x24 &                 # virtual display
  fluxbox 2>/tmp/fluxbox.log &                     # tiny window manager

  echo "🔑 Starting x11vnc on :5900 …"
  x11vnc -display :99 -nopw -forever -shared -rfbport 5900 2>/tmp/x11vnc.log &

  echo "🌐 Starting websockify (noVNC) on :6080 …"
  websockify --web=/usr/share/novnc 6080 localhost:5900 &

  echo "🔗 Port 6080 will open automatically in the browser."
fi

# ── 2.  Run the game (restarts on every postStart) ─────────────────────
cd "$GAME_DIR"
python app.py
