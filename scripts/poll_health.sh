#!/usr/bin/env bash
# Poll /health until 200 or a timeout; show service-log progress between checks.
set -u
URL="${1:-http://127.0.0.1:8090/health}"
STEPS="${2:-50}"    # number of 10s tries
for i in $(seq 1 "$STEPS"); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 "$URL" 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo
    echo "READY after ~$((i*10))s"
    curl -s "$URL"; echo
    exit 0
  fi
  prog=$(journalctl --user -u decisionmaker-serve -n 4 --no-pager 2>/dev/null \
      | grep -oE "(loading|ready|serving|Downloading|Error|Traceback|CUDA|[0-9]+%|[0-9.]+[GM]?B/s)" | tail -1)
  printf '\r[%03d/%ds] http=%-6s %-24s' "$((i*10))" "$((STEPS*10))" "${code:-none}" "${prog:-download/load}"
  sleep 10
done
echo
echo "TIMEOUT waiting for $URL"
exit 1