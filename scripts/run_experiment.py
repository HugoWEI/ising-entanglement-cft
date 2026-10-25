"""Reproduce a finite-size half-chain entropy experiment and save a CSV + PNG."""
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
from ising import half_chain_entropy, fit_effective_c

LENGTHS = (16, 24, 32, 48, 64)
DEFECTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for defect in DEFECTS:
        entropies = [half_chain_entropy(L, defect) for L in LENGTHS]
        c_fit, _ = fit_effective_c(LENGTHS, entropies)
        for L, S in zip(LENGTHS, entropies):
            rows.append({"L": L, "defect": defect,
                         "entropy_nats": f"{S:.12g}",
                         "fitted_c_eff": f"{c_fit:.8g}"})
        ax.plot(np.log(LENGTHS), entropies, marker="o",
                label=f"t={defect:g}; fitted c_eff={c_fit:.3f}")
        print(f"t={defect:.2f}: finite-size fitted c_eff={c_fit:.6f}")

    with (ROOT / "results" / "half_chain_entropy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("L", "defect", "entropy_nats", "fitted_c_eff"))
        writer.writeheader()
        writer.writerows(rows)
    ax.set(xlabel="ln L", ylabel="S(L/2) [nats]",
           title="Critical open Ising chain: half-chain entanglement")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "entropy_vs_logL.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
