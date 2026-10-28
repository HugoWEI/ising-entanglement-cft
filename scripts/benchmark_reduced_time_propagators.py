"""Benchmark two unitary rank-two time solvers versus exact Gaussian evolution.

No method here is a closed-form revival theory: both approximately
integrate the original finite-chain Hamiltonian and are compared with
independent *exact full-chain eigenbasis* evolution.

Compare:
  - Cayley implicit midpoint with a 2x2 Woodbury inverse;
  - Strang splitting: exact clean propagation, exact 2-site defect kick.
For both, capacity is calculated from restricted correlation eigenvalues.

Usage: python scripts/benchmark_reduced_time_propagators.py

Outputs:
  results/reduced_time_propagator_benchmark.csv
  figures/reduced_time_propagator_comparison.png
  figures/reduced_time_step_convergence.png
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
from scattering_phase import initial_modes,trajectory
from rank2_cayley import RankTwoCayley
from clean_defect_strang import CleanDefectStrang
from revival_benchmark import first_broad_capacity_peak

PAIRS=((1.2,.35),(.7,1.4),(.5,.2))
LENGTHS=(48,64)
STEPS=(.2,.1,.05)
OBS_STEP=.2
INITIAL_MASS=2.5


def peak(t,y,L):
    return first_broad_capacity_peak(t,y,L)


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    rows=[]
    example=None
    for n in LENGTHS:
        times=np.arange(0,round(1.8*n/OBS_STEP)+1)*OBS_STEP
        f0=initial_modes(n,"staggered",stagger=INITIAL_MASS)
        mask=(times>=.4*n)&(times<1.65*n)
        for pair in PAIRS:
            exact=trajectory(n,pair,times,f0)
            exact_peak,exact_height=peak(times,exact[:,1],n)
            for step in STEPS:
                for cls in (RankTwoCayley,CleanDefectStrang):
                    solver=cls(n,pair,step)
                    approx,orth_error=solver.trajectory(
                        f0,times,return_unitarity_error=True)
                    capdiff=approx[:,1]-exact[:,1]
                    pred_peak,pred_height=peak(times,approx[:,1],n)
                    row={
                        "L":n,"onsite_u":pair[0],"onsite_v":pair[1],
                        "solver":cls.__name__,"time_step":step,
                        "revival_rmse_capacity":float(np.sqrt(np.mean(capdiff[mask]**2))),
                        "revival_mae_capacity":float(np.mean(np.abs(capdiff[mask]))),
                        "max_abs_capacity_error":float(np.max(abs(capdiff[mask]))),
                        "exact_broad_peak_time":exact_peak,
                        "approx_broad_peak_time":pred_peak,
                        "peak_time_error":abs(exact_peak-pred_peak),
                        "exact_broad_peak_height":exact_height,
                        "approx_broad_peak_height":pred_height,
                        "peak_height_error":abs(exact_height-pred_height),
                        "max_occupied_orthogonality_error":orth_error,
                    }
                    rows.append(row)
                    print(f"L={n}, defect={pair}, dt={step}, solver={cls.__name__}:"
                          f" revival RMSE={row['revival_rmse_capacity']:.6f},"
                          f" peak timing error={row['peak_time_error']:.3f}")
                    if n==48 and pair==(1.2,.35) and step==.2:
                        if example is None:example={"t":times,"exact":exact[:,1]}
                        example[cls.__name__]=approx[:,1]
    with (ROOT/"results"/"reduced_time_propagator_benchmark.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig,ax=plt.subplots(figsize=(8.6,4.7))
    ax.plot(example["t"],example["exact"],label="Exact defective-chain dynamics",lw=2.3)
    ax.plot(example["t"],example["RankTwoCayley"],ls="--",label="Cayley, dt=0.2")
    ax.plot(example["t"],example["CleanDefectStrang"],ls=":",lw=2.3,label="Clean+defect Strang, dt=0.2")
    ax.set(xlabel="Time [inverse hopping]",ylabel="Modular capacity [nats²]",
           title="48 sites: finite-step numerical methods versus exact dynamics")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"reduced_time_propagator_comparison.png",dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.6))
    for solver in ("RankTwoCayley","CleanDefectStrang"):
        subset=[r for r in rows if r["L"]==48 and
                (r["onsite_u"],r["onsite_v"])==(1.2,.35) and
                r["solver"]==solver]
        ax.loglog([r["time_step"] for r in subset],
                  [r["revival_rmse_capacity"] for r in subset],
                  marker="o",label=solver)
    ax.set(xlabel="Numerical time step",ylabel="RMSE of capacity in revival window",
           title="Convergence of two unitary low-rank numerical schemes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"reduced_time_step_convergence.png",dpi=180)
    plt.close(fig)

    for step in STEPS:
        relevant=[r for r in rows if r["time_step"]==step]
        for method in ("RankTwoCayley","CleanDefectStrang"):
            vals=[r["revival_rmse_capacity"] for r in relevant if r["solver"]==method]
            print(f"step={step:g} {method}: mean RMSE across six cases = {np.mean(vals):.6f}")
    print("These are finite-step solvers, NOT a predictive scattering theory.")


if __name__=="__main__":
    main()
