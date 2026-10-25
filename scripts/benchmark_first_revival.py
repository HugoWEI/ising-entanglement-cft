"""Preliminary capacity-revival benchmark; NOT a coherent many-body phase theory.

Contrasts matched-transmission null baseline (predicted peak shift = 0)
with a single-capacity-weighted Wigner-delay estimate, using the *first*
broad capacity peak defined in src/revival_benchmark.py. The second method
is physically motivated but heuristic. Any failure is scientifically useful.

The central onsite impurities are (u,v) and its spatial reverse (v,u).
This is a number-conserving tight-binding model, NOT the Ising model.

Usage: python scripts/benchmark_first_revival.py
Outputs: CSV and PNG under results/ and figures/.
"""
from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from scattering_phase import initial_modes, trajectory
from revival_benchmark import (capacity_weighted_delay_difference,
                               first_broad_capacity_peak)

# Original exploratory pair followed by two distinct independent controls.
PAIRS=((1.2,.35),(.7,1.4),(.5,.2))
LENGTHS=(48,64)
DT=0.2
INITIAL_STAGGER=2.5


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    records=[]
    for pair_id,pair in enumerate(PAIRS):
        kstar,predicted=capacity_weighted_delay_difference(pair,INITIAL_STAGGER)
        for length in LENGTHS:
            times=np.arange(0,1.8*length+DT*.1,DT)
            occupied=initial_modes(length,"staggered",stagger=INITIAL_STAGGER)
            a=trajectory(length,pair,times,occupied)
            b=trajectory(length,pair[::-1],times,occupied)
            first_a,height_a=first_broad_capacity_peak(times,a[:,1],length)
            first_b,height_b=first_broad_capacity_peak(times,b[:,1],length)
            observed=first_a-first_b
            records.append({
                "defect_pair":f"{pair[0]:g},{pair[1]:g}",
                "L":length,
                "kstar":kstar,
                "first_peak_uv":first_a,
                "first_peak_vu":first_b,
                "peak_capacity_uv":height_a,
                "peak_capacity_vu":height_b,
                "observed_peak_shift":observed,
                "transmission_only_null_shift":0.0,
                "heuristic_wigner_shift":predicted,
                "abs_error_null":abs(observed),
                "abs_error_wigner":abs(observed-predicted),
            })
            print("pair=",pair,"L=",length,"peak shift=",round(observed,3),
                  "delay prediction=",round(predicted,3))
    with (ROOT/"results"/"first_revival_benchmark.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig,ax=plt.subplots(figsize=(8,4.5))
    labels=[]
    observed=[]
    predicted=[]
    for row in records:
        labels.append(f"({row['defect_pair']}) L={row['L']}")
        observed.append(row["observed_peak_shift"])
        predicted.append(row["heuristic_wigner_shift"])
    x=np.arange(len(records))
    ax.plot(x,observed,"o-",label="Exact capacity-envelope peak shift")
    ax.plot(x,predicted,"s--",label="Capacity-weighted Wigner-delay heuristic")
    ax.axhline(0,color="gray",linestyle=":",label="Matched-T null baseline")
    ax.set_xticks(x,labels,rotation=25,ha="right")
    ax.set_ylabel("First broad capacity-peak time difference")
    ax.set_title("Preliminary test: one Wigner delay is not a universal revival predictor")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"first_revival_benchmark.png",dpi=180)
    plt.close(fig)
    print("Mean absolute shift prediction error (null):",
          np.nanmean([r["abs_error_null"] for r in records]))
    print("Mean absolute shift prediction error (Wigner heuristic):",
          np.nanmean([r["abs_error_wigner"] for r in records]))
    print("A difference is NOT proof of a coherent many-body theory.")


if __name__=="__main__":
    main()
