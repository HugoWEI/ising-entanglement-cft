# Phase-resolved modular-capacity revival research

**Research status (29 October 2026):** preliminary, reproducible computational research. Neither a peer-reviewed publication nor a fully certified numerical theorem.

This directory preserves the research code, mathematical notes, and data for two related experiments:

- `certificate/`: an Airy first-return prediction checked against an independent fourth-order two-site Hermite–Volterra witness. Includes the witness datasets, conditional first-peak enclosures, tests and numerical figures. Start with [certificate/README.md](certificate/README.md) and [certificate/FINAL_RESULTS.json](certificate/FINAL_RESULTS.json).
- `airy_audit/`: finite-order Bessel-vs-Airy pointwise interval-assisted series calculations, projector-error transfer, and trace-response diagnostics. Start with [airy_audit/RESEARCH_NOTE.md](airy_audit/RESEARCH_NOTE.md).

From each subdirectory, install its `requirements.txt` and run `python -m pytest -q` (or the documented unittest commands). Plot generation and more expensive validation scripts have separate commands in the research notes.

## Scientific limitations

1. The first-peak enclosures are conditional on ordinary floating-point evaluations; they are **not** outward-rounded interval certifications.
2. The finite-order Airy enclosures are **pointwise**, not uniform in time, and depend on the correctness of interval primitives.
3. The a posteriori peak certificate uses a full two-site Volterra witness; a standalone reduced Airy predictor theorem remains unfinished.
4. No claim of novelty over existing literature is made. Compare carefully with Capizzi & Eisler, *SciPost Physics* **14**, 070 (2023), arXiv:2209.03297 and standard Olver–Airy asymptotics (NIST DLMF §10.20).

Legacy Ising and other scattering studies remain in the top-level `src/`, `scripts/`, and `docs/` directories.

**Source provenance:** exported from the reproducibility packages prepared on 29 October 2026; latest suites checked locally before upload (5 and 6 passing tests, respectively).
