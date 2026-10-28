"""Mechanism benchmark: bound-band coherence, parity reversal and Krylov compression.

This is a source-separated ANALYSIS, not a new strong-defect first peak
formula. The exact band evolution is supplied by defective-chain
diagonalization here, so comparing Krylov dimensions only isolates
the accuracy of the rank-two entanglement-correction approximation.

The dephased covariance has NO bound-band coherences and is a valid
Gaussian reference, but it corresponds to a different (mixed) state.
It does not constitute removing an impurity from the same quench.

Outputs:
  results/bound_coherence_reversal.csv
  results/bound_coherence_krylov.csv
  figures/bound_coherence_reversal.png
  figures/bound_coherence_krylov_convergence.png
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
from scattering_phase import initial_modes
from bound_coherence_determinant import SingleBoundCoherence
from bound_coherence_krylov import predict_capacity_from_dephased
from revival_benchmark import first_broad_capacity_peak

LENGTHS=(48,64)
PAIRS=((1.2,.35),(.7,1.4),(.5,.2))
STEPS=(0,1,2,4,6,8,10)
DT=.2


def save_csv(name,rows):
    if not rows:
        raise ValueError("No rows")
    with (ROOT/"results"/name).open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    differences=[]
    krylov=[]
    example={}
    for n in LENGTHS:
        times=np.arange(0,1.8*n+.01,DT)
        initial=initial_modes(n,"staggered",stagger=2.5)
        window=(times>=.4*n)&(times<1.65*n)
        for u,v in PAIRS:
            runs=[]
            for p in ((u,v),(v,u)):
                model=SingleBoundCoherence(n,p,initial)
                trajectory=model.trajectory(times)
                cached_pieces=[model.components(tt) for tt in times]
                full,ref=trajectory[:,0],trajectory[:,1]
                full_peak=first_broad_capacity_peak(times,full,n)
                deph_peak=first_broad_capacity_peak(times,ref,n)
                max_det_error=0.
                for tt in (0.,n*.5,n*1.2):
                    for z in (.2+.4j,1.7+.12j):
                        max_det_error=max(max_det_error,
                            abs(model.entanglement_determinant_ratio(tt,z)-
                                model.direct_determinant_ratio(tt,z)))
                if max_det_error>2e-10:
                    raise RuntimeError("rank-two determinant identity failed")
                run=dict(pair=f"{p[0]:g},{p[1]:g}",full=full,deph=ref,
                         peak_full=full_peak,peak_deph=deph_peak,
                         bound_energy=model.bound_energy,
                         determinant_error=max_det_error)
                runs.append(run)
                for steps in STEPS:
                    if steps==0:
                        approx=ref
                        max_dim=0
                    else:
                        predictions=[
                            predict_capacity_from_dephased(
                                pieces["dephased"],pieces["X"],
                                pieces["J"],num_steps=steps)
                            for pieces in cached_pieces
                        ]
                        approx=np.asarray([a for a,_ in predictions])
                        max_dim=max(dim for _,dim in predictions)
                    error=approx-full
                    approx_peak=first_broad_capacity_peak(times,approx,n)
                    krylov.append(dict(
                        L=n,defect_pair=run["pair"],
                        krylov_blocks=steps,
                        maximum_reduced_dimension=max_dim,
                        subsystem_dimension=n//2,
                        revival_capacity_RMSE=float(
                            np.sqrt(np.mean(error[window]**2))),
                        revival_capacity_MAE=float(
                            np.mean(abs(error[window]))),
                        full_first_peak_time=full_peak[0],
                        approximate_first_peak_time=approx_peak[0],
                        peak_time_absolute_error=abs(full_peak[0]-approx_peak[0]),
                        full_first_peak_height=full_peak[1],
                        approximate_first_peak_height=approx_peak[1],
                    ))
                    if n==48 and p in ((1.2,.35),(.35,1.2)) and steps==8:
                        example[f"krylov_{run['pair']}"]=approx
                if n==48 and p in ((1.2,.35),(.35,1.2)):
                    example[f"full_{run['pair']}"]=full
                    example[f"deph_{run['pair']}"]=ref
                    example["times"]=times
            forward,reverse=runs
            dt_full=forward["peak_full"][0]-reverse["peak_full"][0]
            dt_deph=forward["peak_deph"][0]-reverse["peak_deph"][0]
            d_full=forward["full"]-reverse["full"]
            d_deph=forward["deph"]-reverse["deph"]
            differences.append(dict(
                L=n,onsite_u=u,onsite_v=v,
                bound_energy=forward["bound_energy"],
                forward_full_peak_time=forward["peak_full"][0],
                reverse_full_peak_time=reverse["peak_full"][0],
                forward_deph_peak_time=forward["peak_deph"][0],
                reverse_deph_peak_time=reverse["peak_deph"][0],
                reversal_full_peak_shift=dt_full,
                reversal_deph_peak_shift=dt_deph,
                shift_change_on_dephasing=dt_full-dt_deph,
                revival_RMSE_full_reversal_contrast=float(
                    np.sqrt(np.mean(d_full[window]**2))),
                revival_RMSE_deph_reversal_contrast=float(
                    np.sqrt(np.mean(d_deph[window]**2))),
                revival_RMSE_coherence_part_of_contrast=float(
                    np.sqrt(np.mean((d_full[window]-d_deph[window])**2))),
                forward_mean_abs_full_minus_deph=float(
                    np.mean(abs(forward["full"][window]-forward["deph"][window]))),
                max_determinant_ratio_error=max(
                    forward["determinant_error"],reverse["determinant_error"]),
            ))
            print(f"L={n}, pair={(u,v)}: full reversal Δt={dt_full:+.2f},"
                  f" dephased Δt={dt_deph:+.2f};"
                  f" determinant test={differences[-1]['max_determinant_ratio_error']:.2e}")
    save_csv("bound_coherence_reversal.csv",differences)
    save_csv("bound_coherence_krylov.csv",krylov)
    fig,axes=plt.subplots(2,1,figsize=(8.3,7.1),sharex=True)
    for ax,p in zip(axes,((1.2,.35),(.35,1.2))):
        key=f"{p[0]:g},{p[1]:g}"
        ax.plot(example["times"],example[f"full_{key}"],
                label="Exact pure-state capacity",lw=2.0)
        ax.plot(example["times"],example[f"deph_{key}"],
                label="Bound-band dephased Gaussian reference",ls="--")
        ax.plot(example["times"],example[f"krylov_{key}"],
                label="8-block Krylov coherent correction",ls=":")
        ax.set_ylabel("Capacity [nats²]")
        ax.set_title(f"L=48, reversed defect potentials ({key})")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time [inverse hopping]")
    fig.suptitle("Bound-band coherence, not bound population alone, shifts this revival")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"bound_coherence_reversal.png",dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.3,4.7))
    for n in LENGTHS:
        for p in ((1.2,.35),(.35,1.2)):
            key=f"{p[0]:g},{p[1]:g}"
            subset=[row for row in krylov if row["L"]==n and
                    row["defect_pair"]==key]
            ax.plot([row["maximum_reduced_dimension"] for row in subset],
                    [row["revival_capacity_RMSE"] for row in subset],
                    marker="o",label=f"L={n}, ({key})")
    ax.set(xlabel="Retained entanglement-subspace dimension",
           ylabel="Revival-window capacity RMSE",
           title="Rank-two coherence needs a multi-mode spectral response")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"bound_coherence_krylov_convergence.png",dpi=180)
    plt.close(fig)
    print("These are exploratory numerical diagnostics and a conditional")
    print("reduced correction: exact band propagation is still used here.")


if __name__=="__main__":
    main()
