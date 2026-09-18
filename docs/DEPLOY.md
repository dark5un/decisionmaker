# Deploying Decision-Maker serve (rootless Podman, GPU-pinned)

Phase 4 is CODE-COMPLETE and verified live at the HTTP layer (tests + the running
service on this box). These steps build the container and install the quadlet;
nothing below mutates your system unless you run it.

Verify first: the GPU has to be real and live, never a cached map.
```bash
cd ~/workspace/decisionmaker
scripts/verify_gpu.sh --serve     # lists live GPUs + UUIDs and checks the quadlet pin
```

## Step 1 — build the serve image (one-time; ~2–3 GB, several minutes)
```bash
cd ~/workspace/decisionmaker
podman build -f deploy/Containerfile.serve -t decisionmaker-serve:local .
```

## Step 2 — install + start (one-time, then every reboot autostarts)
```bash
mkdir -p ~/.config/containers/systemd
cp deploy/decisionmaker.network deploy/decisionmaker-serve.container \
     deploy/healthcheck.sh ~/.config/containers/systemd/
chmod +x ~/.config/containers/systemd/healthcheck.sh
systemctl --user daemon-reload
systemctl --user start decisionmaker-serve
```

The first start downloads the pinned model once (`Qwen/Qwen3-0.6B`, rev
`c1899de289a04d12100db370d81485cdf75e47ca`, ~1.2 GB) into
`~/.cache/huggingface`, then loads it once and serves. Later starts are load-once,
warm. Give it up to `TimeoutStartSec=1800` on first boot (image + model).

If the image/model pull fails or /health does not go green, see Troubleshooting.

## Step 3 — verify (the Phase 4 gate)
```bash
curl -s http://127.0.0.1:8090/health                  # {"status":"ready",...}
curl -s -X POST http://127.0.0.1:8090/v1/decisionmaker \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/domain_shape.request.json | python3 -m json.tool
```
Gate: `/health` = ready, and every answer's `probabilities` sums to 1.

## Operations
| Action | Command |
|---|---|
| Stop / start | `systemctl --user stop|start decisionmaker-serve` |
| Logs | `journalctl --user -u decisionmaker-serve -f` |
| Watch quadlets | `systemctl --user list-units 'decisionmaker*'` |
| Confirm GPU in use | `journalctl --user -u decisionmaker-serve -b | grep -i cuda` |

## GPU pinning (the fragile part)
`/dev/nvidiaN` minors flip across reboots. The quadlet pins by UUID
(`CUDA_VISIBLE_DEVICES=GPU-58e5f98d-… = RTX 4070 Ti`). Live check 2026-09-18: the
5090 is NOT idle (llama-server holds ~26.6 GiB), so decisionmaker uses the 4070 Ti
(~8 GiB free; 0.6B fp32 ≈ 2.5 GiB). If it ever stops matching / runs out of GPU
memory:
```bash
scripts/verify_gpu.sh                # read the CURRENT UUIDs from /proc
# edit ~/.config/containers/systemd/decisionmaker-serve.container to the correct UUID
systemctl --user daemon-reload && systemctl --user restart decisionmaker-serve
```

## Troubleshooting
- Image build fails on torch/CUDA → the pytorch base tag may be stale on your
  arch; pick the matching `pytorch/pytorch:<torch>-cuda<ver>-cudnn<ver>-runtime`.
- `/health` 503 forever → the model didn't load; check logs. The engine refuses
  to claim GPU success without a device, so re-verify GPU pinning if it says no
  CUDA device.
- Port 8090 busy → change `PublishPort` and `DECISIONMAKER_PORT` together.
- Mismatched repo volume shows stale code → the repo volume is read-only `:ro`;
  rebuild time you `podman build` only, code changes are live via the mount.

## Train (Phase 5 — live)
Fine-tunes the decision head against `data/seed.jsonl` and runs the ECE gate.
`deploy/decisionmaker-train.container` is the real entrypoint (`python -m train.train`),
writing a frozen run dir under `runs/` (checkpoint + config + predictions + ECE report).
The gate exit code is 0 on pass, 2 on fail; results are in `docs/CALIBRATION.md`.

```bash
systemctl --user start decisionmaker-train       # one-shot; check with --user status/logs
```