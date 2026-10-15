#!/bin/sh
# Start Ollama on boot (cron @reboot) if it isn't already running.
# Improvements: flock PID guard, error handling, log rotation, portable paths.

set -eu
set -o pipefail

OLLAMA_BIN="${OLLAMA_BIN:-/home/jimmy/ollama_root/bin/ollama}"
OLLAMA_LOG="${OLLAMA_LOG:-/home/jimmy/ollama.log}"
OLLAMA_PIDFILE="${OLLAMA_PIDFILE:-/tmp/ollama-pi.pid}"
OLLAMA_LINES=2000

# --- log rotation (keep last N lines) ---
if [ -f "$OLLAMA_LOG" ] && [ "$(wc -l < "$OLLAMA_LOG")" -gt "$OLLAMA_LINES" ]; then
    tail -n "$OLLAMA_LINES" "$OLLAMA_LOG" > "${OLLAMA_LOG}.tmp" && mv "${OLLAMA_LOG}.tmp" "$OLLAMA_LOG"
fi

# --- PID file guard (prevent double-start race) ---
# Atomic lock guard (replaces TOCTOU-prone PID file check)
exec 200>"${OLLAMA_PIDFILE}.lock"
if ! flock -n 200; then
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Another instance is starting Ollama, exiting" >> "$OLLAMA_LOG"
    exit 0
fi

if [ -f "$OLLAMA_PIDFILE" ]; then
    OLD_PID=$(cat "$OLLAMA_PIDFILE" 2>/dev/null || true)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Ollama already running (PID $OLD_PID)" >> "$OLLAMA_LOG"
        exit 0
    fi
    rm -f "$OLLAMA_PIDFILE"
fi

# --- also check pgrep as fallback ---
if command -v pgrep >/dev/null 2>&1; then
    if pgrep -f "ollama serve" > /dev/null 2>&1; then
        echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Ollama already running (pgrep match)" >> "$OLLAMA_LOG"
        exit 0
    fi
fi

# --- start ---
if [ ! -x "$OLLAMA_BIN" ]; then
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') ERROR: Ollama binary not found at $OLLAMA_BIN" >> "$OLLAMA_LOG"
    exit 1
fi

nohup "$OLLAMA_BIN" serve >> "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
# Detach so the shell can exit without the child being signalled on teardown.
disown 2>/dev/null || true
echo "$OLLAMA_PID" > "$OLLAMA_PIDFILE"
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') Started Ollama (PID $OLLAMA_PID)" >> "$OLLAMA_LOG"
