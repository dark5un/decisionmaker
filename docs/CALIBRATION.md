# Calibration results (Phase 5, live runs 2026-09-18)

Head-only fine-tune on the RTX 5090 (backbone frozen, bf16; head fp32, lr 3e-4,
3 epochs, 36 steps) against `data/seed.jsonl` (data_sha256
9a10060b...7fe47c0). Gate = boolean ECE over 10 fixed bins (test and OOD), threshold 0.20.

| loss   | test boolean ECE | ood boolean ECE | gate |
|--------|---------------|--------------|------|
| ce     | 0.0261        | 0.0424       | PASS |
| brier  | 0.0002        | 0.0147       | PASS |
| paired | 0.0436        | 0.0247       | PASS |

Takeaways (honest):
- All three proper-scoring objectives pass the gate. On this single seed, vector-Brier is
  strongest; the RLCD-style paired proper-reward does NOT beat plain CE on test (it is better
  on OOD). It is not a guaranteed win — matches the plan's risk register.
- Absolute ECE values are under-stated: predictions concentrate ~0.4, so most fixed bins are
  empty and a weighted ECE adds zero for them. Rank the arms, not the raw numbers.
- One draw per arm; a fresh seed would sharpen ordering. Rerun instructions: edit the Exec=
  line in `deploy/decisionmaker-train.container` (--loss, --run-dir), reinstall to
  `~/.config/containers/systemd/`, daemon-reload, start.

GPU/training facts learned live (hard-won):
- Head-only training MUST freeze the backbone (`requires_grad_(False)` + bf16); fp32 with the
  full autograd graph blew ~8 GiB and OOM'd even on a 12 GB card.
- The `Image=` quadlet key cannot carry an inline `#` comment — podman parses the whole line
  as the reference.
- GPU choice is a live decision, not a cached one: the 5090 was 98% busy (llama-server) at 14:10
  but fully free by 16:05 after stopping llama; training is pinned to the 5090 now.