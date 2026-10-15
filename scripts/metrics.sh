#!/bin/sh
# Log current system resource percentages (cron every minute).
# Output: UTC timestamp + CPU idle %, memory used %, root disk usage %.
# Improvements: single-pass awk, division-by-zero guard, flock, log rotation.

METRICS_LOG="${METRICS_LOG:-/home/jimmy/ollama_root/metrics.log}"
METRICS_LINES=10000

# --- log rotation ---
if [ -f "$METRICS_LOG" ] && [ "$(wc -l < "$METRICS_LOG")" -gt "$METRICS_LINES" ]; then
    tail -n 5000 "$METRICS_LOG" > "${METRICS_LOG}.tmp" && mv "${METRICS_LOG}.tmp" "$METRICS_LOG"
fi

# --- flock to prevent overlapping runs ---
(
    flock -n 9 || exit 0

    # Single awk pass for CPU calculation (POSIX-compatible, no process substitution)
    _cpu_tmp=$(mktemp)
    awk '/^cpu /' /proc/stat > "$_cpu_tmp"
    sleep 1
    awk '/^cpu /' /proc/stat >> "$_cpu_tmp"
    cpu_idle=$(awk '
        /^cpu / {
            idle = $5
            total = $2 + $3 + $4 + $5 + $6 + $7 + $8 + $9
            sample_count++
            if (sample_count == 1) { idle1 = idle; total1 = total }
            if (sample_count == 2) { idle2 = idle; total2 = total }
        }
        END {
            if (total2 > total1) {
                printf "%d", 100 * (idle2 - idle1) / (total2 - total1)
            } else {
                printf "0"
            }
        }
    ' "$_cpu_tmp")
    rm -f "$_cpu_tmp"

    mem_used=$(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END {
        if (t > 0) printf "%d", int((t - a) * 100 / t)
        else printf "0"
    }' /proc/meminfo)

    disk_used=$(df / | awk 'NR==2 {gsub(/%/, "", $5); print $5}')

    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') cpu_idle=${cpu_idle}% ram_used=${mem_used}% disk_used=${disk_used}%" >> "$METRICS_LOG"
) 9>/tmp/metrics-pi.lock
