"""Strong-defect capacity from an exact parity scattering contour integral.

This is an ALL-DELTA *implicit spectral formula*, NOT a new compact
analytical law for first capacity-revival peak times or heights.
It uses the eigensystem of the symmetric reference only.

An independent dense eigendecomposition of each asymmetric Hamiltonian
is used as a held-out reference, never to construct predictions.

Outputs:
  results/parity_contour_capacity_benchmark.csv
  figures/parity_contour_capacity_strong_defect.png

Run: python scripts/benchmark_parity_contour_capacity.py
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
from parity_contour_capacity import ContourCapacity
from scattering_phase import initial_modes,trajectory
from revival_benchmark import first_broad_capacity_peak

CASES=((24,.775,.85),(48,.775,.85),(48,.25,.85),
       (64,.775,.85),(24,.775,1.6))
DT=.4
NODES=768


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    rows=[]
    example=None
    for length,w,delta in CASES:
        times=np.arange(0,int(round(1.7*length/DT))+1)*DT
        initial=initial_modes(length,"staggered",stagger=2.5)
        formula=ContourCapacity(length,w,initial)
        for sign in (+1,-1):
            d=sign*delta
            numerical=formula.capacity_trajectory(d,times,NODES)
            # Independent validation only: asymmetric eigensolver.
            exact=trajectory(length,(w+d/2,w-d/2),times,initial)
            mask=(times>=.4*length)&(times<=1.65*length)
            err=numerical["values"]-exact
            predicted_peak=first_broad_capacity_peak(
                times,numerical["values"][:,1],length)
            exact_peak=first_broad_capacity_peak(
                times,exact[:,1],length)
            record=dict(
                L=length,mean_potential=w,delta=d,
                quadrature_nodes=NODES,
                max_time=float(np.max(times)),
                revival_capacity_RMSE=float(np.sqrt(np.mean(err[mask,1]**2))),
                revival_capacity_max_abs=float(np.max(abs(err[mask,1]))),
                revival_entropy_RMSE=float(np.sqrt(np.mean(err[mask,0]**2))),
                predicted_broad_peak_time=predicted_peak[0],
                exact_broad_peak_time=exact_peak[0],
                predicted_broad_peak_height=predicted_peak[1],
                exact_broad_peak_height=exact_peak[1],
                max_occupied_isometry_error=numerical["max_isometry_error"],
                max_reduced_occupation=numerical["max_occupation"],
            )
            rows.append(record)
            print(f"L={length},w={w},delta={d:+.2f}: "
                  f"capacity RMSE={record['revival_capacity_RMSE']:.3g}; "
                  f"unitarity error={record['max_occupied_isometry_error']:.3g}")
            if length==48 and w==.775 and delta==.85 and sign==+1:
                example={"times":times,"exact":exact[:,1],
                         "contour":numerical["values"][:,1]}
    with (ROOT/"results"/"parity_contour_capacity_benchmark.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)

    fig,ax=plt.subplots(figsize=(8.6,4.5))
    ax.plot(example["times"],example["exact"],lw=2.2,
            label="Independent full-defect eigensolver")
    ax.plot(example["times"],example["contour"],ls="--",
            label="Symmetric reference + rank-two contour")
    ax.set(xlabel="Time [inverse hopping]",ylabel="Capacity [nats²]",
           title="Strong asymmetry delta=.85: exact scattering contour reproduces capacity")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"parity_contour_capacity_strong_defect.png",dpi=190)
    plt.close(fig)
    print("An exact formal contour expression is not a new simple revival-peak law.")


if __name__=="__main__":
    main()
