"""Held-out strong-defect peak test from symmetric scattering + bound pole.

Prediction inputs: *symmetric* reference h(w,w), initial Slater modes,
two-channel T-matrix contour, bound-state pole/projector, and
PREDECLARED eight-block Krylov compression. NO asymmetric h(delta)
diagonalization during the prediction.

Independent full asymmetric Gaussian trajectories are used ONLY to
score the resulting predicted peak timings/heights.

The contour is near-exact quantum evolution and computationally
expensive; we claim a concrete conditional reduced *entanglement*
formula, NOT a new efficient simple physical first-revival law.

Outputs:
  results/scattering_pole_krylov_peaks.csv
  results/scattering_pole_krylov_timeseries.csv
  figures/scattering_pole_krylov_first_rev.png
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
from scattering_pole_krylov_capacity import ScatteringPoleKrylovCapacity
from scattering_phase import initial_modes,trajectory
from revival_benchmark import first_broad_capacity_peak

CONFIGURATIONS=((48,.775,.85),(64,.775,.85),(48,1.05,-.7),
                # Prospective extra test: L=56, onsite pair (1.1,0.1),
                # selected before scoring its capacity trajectory.
                (56,.6,1.0))
DT=.4
NODES=768
KRYLOV_BLOCKS=8


def write_rows(file,rows):
    if not rows:
        raise ValueError("no records")
    with (ROOT/"results"/file).open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    summary=[]
    series=[]
    figure_data={}
    for length,mean,delta in CONFIGURATIONS:
        times=np.arange(int(round(1.8*length/DT))+1)*DT
        occupied=initial_modes(length,"staggered",stagger=2.5)
        model=ScatteringPoleKrylovCapacity(length,mean,occupied)
        for orientation in (+1,-1):
            d=orientation*delta
            predicted=model.trajectory(d,times,
                                       krylov_blocks=KRYLOV_BLOCKS,
                                       quadrature_nodes=NODES)
            cap=predicted["capacities"]
            actual=trajectory(length,
                              (mean+d/2,mean-d/2),times,occupied)[:,1]
            mask=(times>=.4*length)&(times<1.65*length)
            predicted_peak=first_broad_capacity_peak(
                times,cap[:,0],length)
            exact_peak=first_broad_capacity_peak(times,actual,length)
            deph_peak=first_broad_capacity_peak(times,cap[:,1],length)
            row=dict(
                L=length,mean_potential=mean,asymmetry_delta=d,
                retained_krylov_blocks=KRYLOV_BLOCKS,
                retained_dimension=predicted["maximum_krylov_dimension"],
                subsystem_dimension=length//2,
                bound_energy=predicted["bound_energy"],
                exact_first_broad_peak_time=exact_peak[0],
                predicted_first_broad_peak_time=predicted_peak[0],
                dephased_first_broad_peak_time=deph_peak[0],
                absolute_peak_time_error=abs(exact_peak[0]-predicted_peak[0]),
                exact_first_broad_peak_capacity=exact_peak[1],
                predicted_first_broad_peak_capacity=predicted_peak[1],
                absolute_peak_height_error=abs(exact_peak[1]-predicted_peak[1]),
                revival_capacity_RMSE=float(np.sqrt(
                    np.mean((cap[mask,0]-actual[mask])**2))),
                revival_dephased_capacity_RMSE=float(np.sqrt(
                    np.mean((cap[mask,1]-actual[mask])**2))),
                contour_capacity_RMSE=float(np.sqrt(
                    np.mean((cap[mask,2]-actual[mask])**2))),
                maximum_contour_occupied_isometry_error=
                    predicted["maximum_contour_isometry_error"],
            )
            summary.append(row)
            print(f"L={length}, w={mean}, delta={d:+.2f}:"
                  f" peak predicted {predicted_peak[0]:.2f},"
                  f" measured {exact_peak[0]:.2f},"
                  f" capacity RMSE {row['revival_capacity_RMSE']:.5g};"
                  f" no asymmetric diagonalization for prediction")
            for t,a,p in zip(times,actual,cap):
                series.append(dict(
                    L=length,mean_potential=mean,delta=d,time=t,
                    capacity_exact=a,
                    capacity_scattering_krylov=p[0],
                    capacity_dephased=p[1],
                    capacity_full_contour_diagnostic=p[2],
                ))
            if length==48 and mean==.775:
                figure_data[orientation]=(times,actual,cap)
    write_rows("scattering_pole_krylov_peaks.csv",summary)
    write_rows("scattering_pole_krylov_timeseries.csv",series)

    fig,axes=plt.subplots(2,1,figsize=(8.5,7.2),sharex=True)
    for ax,orientation in zip(axes,(+1,-1)):
        t,exact,cap=figure_data[orientation]
        ax.plot(t,exact,lw=2.0,label="Held-out exact asymmetric simulation")
        ax.plot(t,cap[:,0],ls="--",label="Symmetric scattering + bound pole + Krylov")
        ax.plot(t,cap[:,1],ls=":",label="Dephased Gaussian baseline")
        ax.set_ylabel("Modular capacity [nats²]")
        ax.set_title(f"L=48, mean w=.775, defect asymmetry {orientation*.85:+.2f}")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time [inverse hopping]")
    fig.suptitle("Strong-asymmetry capacity: a scattering-only conditional formula")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"scattering_pole_krylov_first_rev.png",dpi=180)
    plt.close(fig)

    print("This still evaluates a near-exact energy contour for the band.")
    print("It does not establish a simple original strong-defect peak law.")


if __name__=="__main__":
    main()
