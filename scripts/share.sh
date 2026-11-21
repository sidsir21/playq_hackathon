#!/bin/bash
# Share the PlayQ demo on a public HTTPS link in ~1 minute (Cloudflare quick tunnel, no account needed).
#   ./scripts/share.sh            # password-protected public mode (uploads off, live runs capped)
# Stop with Ctrl+C. The app keeps running on THIS laptop, so keep it plugged in and open (caffeinate stops sleep).
set -e
cd "$(dirname "$0")/.."
PORT=8501
CF=./bin/cloudflared

if [ ! -x "$CF" ]; then
  case "$(uname -m)" in arm64) ARCH=arm64;; *) ARCH=amd64;; esac
  echo "Downloading cloudflared ($ARCH) from github.com/cloudflare/cloudflared ..."
  curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-$ARCH.tgz" | tar -xz -C bin
  chmod +x "$CF"
fi

# Access code: from .env (PLAYQ_DEMO_PASSWORD) or a random one printed below
CODE=$(grep -E '^PLAYQ_DEMO_PASSWORD=.+' .env 2>/dev/null | head -1 | cut -d= -f2-)
[ -z "$CODE" ] && CODE=$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 6)
export PLAYQ_DEMO_PASSWORD="$CODE" PLAYQ_PUBLIC_MODE=1

pkill -f "streamlit run app/streamlit_app.py" 2>/dev/null || true
.venv/bin/streamlit run app/streamlit_app.py --server.port $PORT --server.headless true > /tmp/playq_streamlit.log 2>&1 &
for i in $(seq 1 30); do curl -s localhost:$PORT/_stcore/health >/dev/null && break; sleep 1; done

LOG=/tmp/playq_tunnel.log; : > "$LOG"
$CF tunnel --url http://localhost:$PORT --no-autoupdate > "$LOG" 2>&1 &
TUN=$!
trap 'kill $TUN 2>/dev/null; pkill -f "streamlit run app/streamlit_app.py" 2>/dev/null; rm -f data/public_url.txt; exit' INT TERM EXIT
for i in $(seq 1 40); do URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1); [ -n "$URL" ] && break; sleep 1; done
[ -z "$URL" ] && { echo "Tunnel did not start; see $LOG"; exit 1; }
echo "$URL" > data/public_url.txt     # the app's QR code picks this up on the next page load

echo; echo "  PlayQ is live:  $URL"; echo "  Access code:    $CODE"; echo "  (laptop must stay on; press Ctrl+C to take it offline)"; echo
caffeinate -dimsu -w $TUN
