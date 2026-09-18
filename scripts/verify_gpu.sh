#!/usr/bin/env bash
# Verify live GPU truth on this box and confirm the quadlet's pinned UUID is
# still a real device. /dev/nvidiaN MINORS FLIP ACROSS REBOOTS — never trust a
# cached/CDI mapping; this reads the authoritative source.
#
#   scripts/verify_gpu.sh            # list live GPUs + UUIDs
#   scripts/verify_gpu.sh --serve     # list AND validate the serve quadlet pin
set -euo pipefail

SERVE_QUADLET="${DECISIONMAKER_REPO:-$HOME/workspace/decisionmaker}/deploy/decisionmaker-serve.container"

echo "== live NVIDIA devices (/proc/driver/nvidia/gpus/*/information) =="
declare -A uuid2model
for f in /proc/driver/nvidia/gpus/*/information; do
    model=$(sed -n 's/^Model:[[:space:]]*//p' "$f")
    uuid=$(awk -F'\t' '/^GPU UUID:/{print $2}' "$f" | tr -d ' \t')
    minor=$(awk -F'\t' '/^Device Minor:/{print $2}' "$f" | tr -d ' \t')
    printf '  minor=%-4s uuid=%-40s %s\n' "$minor" "$uuid" "$model"
    uuid2model["$uuid"]="$model"
done

if [[ "${1:-}" == "--serve" ]]; then
    # Keep the FULL value (incl. the GPU- prefix) so it matches the live uuid.
    pinned=$(sed -n 's/^Environment=CUDA_VISIBLE_DEVICES=//p' "$SERVE_QUADLET" | tr -d ' \t')
    if [[ -z "$pinned" ]]; then
        echo "!! could not read a pinned GPU UUID from $SERVE_QUADLET"
        exit 1
    fi
    printf '\n== serve pin validation ==\n  pinned UUID: %s\n' "$pinned"
    if [[ -n "${uuid2model[$pinned]:-}" ]]; then
        echo "  OK: pinned UUID matches a live device: ${uuid2model[$pinned]}"
    else
        echo "  !! PINNED UUID NOT FOUND on this boot — /dev/nvidiaN minors flipped."
        echo "     pick the current UUID of the desired GPU and update the quadlet,"
        echo "     then: systemctl --user daemon-reload && systemctl --user restart decisionmaker-serve"
        exit 1
    fi
fi