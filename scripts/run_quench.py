"""Exploratory global field quench with a fixed defect.

Usage:
    python scripts/run_quench.py

This is a finite-size PILOT, not a universal CFT fit or new result.
Writes results/quench_trajectory.csv and figures/quench_dynamics.png.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dynamics import quench_trajectory

LENGTH = 24
FIELD_INITIAL = 2.0
FIELD_FINAL = 1.0
DEFECTS = (0., 0.25, 0.5, 0.75, 1.)
TIMES = np.linspace(0., 8., 33)


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    rows = []
    for defect in DEFECTS:
        s, c = quench_trajectory(
            length=LENGTH,
            defect=defect,
            field_initial=FIELD_INITIAL,
            field_final=FIELD_FINAL,
            times=TIMES,
        )
        axes[0].plot(TIMES, s, label=f"defect={defect:g}")
        axes[1].plot(TIMES, c, label=f"defect={defect:g}")
        for time, entropy, capacity in zip(TIMES, s, c):
            rows.append({
                "L": LENGTH,
                "field_initial": FIELD_INITIAL,
                "field_final": FIELD_FINAL,
                "defect": defect,
                "time": f"{time:.10g}",
                "entropy": f"{entropy:.12g}",
                "capacity": f"{capacity:.12g}",
            })
    for ax, ylabel in zip(axes, ["S_A(time) [nats]", "C_A(time) [nats²]"]):
        ax.set(xlabel="Time [hbar/J]", ylabel=ylabel)
        ax.legend(fontsize=8)
    fig.suptitle("Ising global-field quench with fixed defect: pilot L=24")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "quench_dynamics.png", dpi=180)
    plt.close(fig)
    with (ROOT / "results" / "quench_trajectory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} data points. Exploratory, not a universal scaling claim.")


if __name__ == "__main__":
    main()
