"""Coherent boundary returns + two-site Volterra dynamics benchmark.

This deliberately compares direct propagation (0 image orders), one
boundary-return image order, and two image orders. The Volterra equation
resums coherent visits to the defect; approximate quadrature may violate
unitarity, therefore a *reported nearest-isometry projection* is used.

This is a scattering-path numerical approximation, not yet a new
closed-form quasiparticle prediction of capacity-revival peaks.

Run: python scripts/benchmark_boundary_image_volterra.py

Outputs:
  results/boundary_image_volterra_benchmark.csv
  figures/boundary_image_volterra_capacity.png
  figures/boundary_image_propagator_error.png
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
from boundary_images import finite_clean_image_propagator,exact_clean_sine_propagator
from boundary_image_volterra import volterra_trajectory
from revival_benchmark import first_broad_capacity_peak

LENGTHS=(24,48)
ONSITE=(1.2,.35)
STEP=.2
IMAGE_ORDERS=(0,1,2)


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    summaries=[]
    example=None
    for n in LENGTHS:
        occupied=initial_modes(n,"staggered",stagger=2.5)
        times=np.arange(0,int(round(1.6*n/STEP))+1)*STEP
        exact=trajectory(n,ONSITE,times,occupied)
        mask=(times>=.4*n)&(times<=1.6*n)
        exact_peak,exact_height=first_broad_capacity_peak(
            times,exact[:,1],n)
        for order in IMAGE_ORDERS:
            result=volterra_trajectory(
                n,ONSITE,occupied,step=STEP,final_time=1.6*n,
                image_order=order,physicality="polar")
            if not np.allclose(result["times"],times,atol=1e-11):
                raise RuntimeError("time grid mismatch")
            observed=result["entropy_capacity"]
            errors=observed[:,1]-exact[:,1]
            predicted_peak,predicted_height=first_broad_capacity_peak(
                times,observed[:,1],n)
            row={
                "L":n,
                "image_order":order,
                "time_step":STEP,
                "revival_capacity_rmse":float(np.sqrt(np.mean(errors[mask]**2))),
                "revival_capacity_mae":float(np.mean(np.abs(errors[mask]))),
                "max_raw_orbital_isometry_error":float(
                    np.max(result["raw_isometry_error"])),
                "max_corrected_occupation":float(
                    np.max(result["occupation_bounds"][:,1])),
                "exact_peak_time":exact_peak,
                "predicted_peak_time":predicted_peak,
                "exact_peak_height":exact_height,
                "predicted_peak_height":predicted_height,
            }
            summaries.append(row)
            print(f"L={n}, image orders={order}: "
                  f"RMSE={row['revival_capacity_rmse']:.6f}, "
                  f"raw isometry error={row['max_raw_orbital_isometry_error']:.5f}")
            if n==48:
                if example is None:
                    example={"times":times,"exact":exact[:,1]}
                example[order]=observed[:,1]
    with (ROOT/"results"/"boundary_image_volterra_benchmark.csv").open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=list(summaries[0].keys()))
        w.writeheader();w.writerows(summaries)

    fig,ax=plt.subplots(figsize=(8.3,4.6))
    ax.plot(example["times"],example["exact"],lw=2.4,label="Exact lattice capacity")
    for order in (1,2):
        ax.plot(example["times"],example[order],ls="--" if order==1 else ":",
                lw=1.8,label=f"Volterra, {order} image order(s)")
    ax.set(xlabel="Time [inverse hopping]",ylabel="Modular capacity [nats²]",
           title="Scattering-path approximation with explicit physicality repair")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"boundary_image_volterra_capacity.png",dpi=180)
    plt.close(fig)

    times=np.linspace(0,1.8*48,40)
    curves={}
    for order in IMAGE_ORDERS:
        errors=[]
        for time in times:
            approx=finite_clean_image_propagator(48,float(time),order)
            exact=exact_clean_sine_propagator(48,float(time))
            errors.append(float(np.linalg.norm(approx-exact)/np.linalg.norm(exact)))
        curves[order]=np.asarray(errors)
    fig,ax=plt.subplots(figsize=(8.0,4.4))
    for order,errors in curves.items():
        ax.semilogy(times,np.maximum(errors,1e-15),label=f"Image order {order}")
    ax.set(xlabel="Time [inverse hopping]",
           ylabel="Relative Frobenius error of clean propagator",
           title="Missing reflected paths increasingly matter at long times")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"boundary_image_propagator_error.png",dpi=180)
    plt.close(fig)

    print("This benchmark does not yet derive a new analytical peak law.")


if __name__=="__main__":
    main()
