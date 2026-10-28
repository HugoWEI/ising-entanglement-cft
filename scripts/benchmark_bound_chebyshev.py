"""A-priori bounded continuum-WALK capacity without complex contour.

For a positive two-site impurity with ONE bound pole, the new
finite sum predicts full strong-defect capacity from:
  - a scalar parity secular root and rank-one bound projector;
  - coherent Chebyshev walks T_m(h/2) in the band;
  - Bessel amplitudes J_m(2t);
  - polar correction that preserves a valid pure Gaussian state.

No asymmetric Hamiltonian eigendecomposition or full time-contour
integration is needed for PREDICTIONS. Independent diagonalization is
performed afterwards only for benchmark reference.

The analytic spectral-norm walk error and FFT-alias estimates give a
conservative a-priori bound for the quantum orbitals and an
endpoint-safe modulus bound for capacity, conditional on exact
bound-pole information and exact arithmetic. Floating-point error
is separately checked and is NOT part of that formal bound.

The first-local-maximum certificate checks only the smoothed
discrete index within a predetermined sampling grid. It does NOT
prove uniqueness or identity of the first broad peak among all
local maxima.

Outputs:
  results/bound_chebyshev_revivals.csv
  figures/bound_chebyshev_convergence.png
  figures/bound_chebyshev_capacity.png

Run: python scripts/benchmark_bound_chebyshev.py
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
from bound_projected_chebyshev import BoundProjectedChebyshev
from scattering_phase import initial_modes,trajectory
from revival_benchmark import first_broad_capacity_peak, gaussian_smooth

CASES=((48,.775,.85),(48,.775,-.85),
       (64,.775,.85),(64,.775,-.85),
       (56,.6,1.),(56,.6,-1.),
       (48,1.05,.7),(48,1.05,-.7))
TARGETS=(1e-3,1e-7,1e-12)
DT=.4


def local_index_certificate(t,capacity,peak_time,L,capacity_bound):
    """Check sufficient condition for SAME local-max grid index.

    Since smoothing has nonnegative weights summing to one,
    the smoothed approximate and exact curves differ <= bound
    at every sampled time. If both neighbor differences exceed
    2*bound, the sample remains a local maximum of the exact
    smoothed sequence in exact arithmetic.
    """
    if not np.isfinite(peak_time) or not np.isfinite(capacity_bound):
        return False,float("nan")
    sm=gaussian_smooth(capacity,2./DT)
    idx=int(np.argmin(abs(t-peak_time)))
    if not (1<=idx<len(t)-1):
        return False,float("nan")
    gap=float(min(sm[idx]-sm[idx-1],sm[idx]-sm[idx+1]))
    return bool(gap>2*capacity_bound),float(gap-2*capacity_bound)


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    rows=[]
    example={}
    for n,w,delta in CASES:
        t=np.arange(int(round(1.8*n/DT))+1)*DT
        f0=initial_modes(n,"staggered",stagger=2.5)
        predictor=BoundProjectedChebyshev(n,w,delta,f0)
        # The asymmetric solver is used ONLY for post-hoc scoring.
        exact=trajectory(n,(w+delta/2,w-delta/2),t,f0)
        exact_peak=first_broad_capacity_peak(t,exact[:,1],n)
        window=(t>=.4*n)&(t<1.65*n)
        for target in TARGETS:
            result=predictor.trajectory(t,target=target)
            cap=result["values"][:,1]
            peak=first_broad_capacity_peak(t,cap,n)
            rmse=float(np.sqrt(np.mean((cap[window]-exact[window,1])**2)))
            maxerror=float(np.max(abs(cap-exact[:,1])))
            certificate,margin=local_index_certificate(
                t,cap,peak[0],n,result["formal_capacity_error_bound"])
            row=dict(
                L=n,mean_potential=w,asymmetry_delta=delta,
                target_orbital_error=target,
                polynomial_degree=result["degree"],
                fft_points=result["fft_points"],
                bound_energy=result["bound_energy"],
                theoretical_walk_error_bound=result["analytic_walk_truncation_bound"],
                theoretical_fft_alias_error_bound=result["analytic_FFT_alias_bound"],
                formal_total_orbital_error_bound=result["formal_amplitude_error_bound"],
                formal_capacity_error_bound=result["formal_capacity_error_bound"],
                actual_max_abs_capacity_error=maxerror,
                actual_revival_capacity_RMSE=rmse,
                actual_peak_time=exact_peak[0],
                walk_predicted_peak_time=peak[0],
                peak_time_grid_error=abs(exact_peak[0]-peak[0]),
                actual_peak_height=exact_peak[1],
                walk_predicted_peak_height=peak[1],
                peak_height_error=abs(exact_peak[1]-peak[1]),
                certified_same_local_grid_maximum=certificate,
                local_peak_certificate_margin=margin,
                maximum_raw_orbital_isometry_error=
                    result["max_raw_occupied_isometry_error"],
                maximum_polar_repair_fro=result["max_polar_change_fro"],
            )
            rows.append(row)
            print(f"L={n} onsite=({w+delta/2:g},{w-delta/2:g}),"
                  f" target={target:.0e}, M={result['degree']}: "
                  f"peak {peak[0]:.1f} vs exact {exact_peak[0]:.1f},"
                  f" RMSE={rmse:.2e}, analytical cap bound="
                  f"{result['formal_capacity_error_bound']:.2e},"
                  f" certified_local_index={certificate}")
            if target==1e-12:
                if maxerror > result["formal_capacity_error_bound"]+2e-8:
                    raise RuntimeError("measured error exceeds formal analytical bound")
                if row["maximum_raw_orbital_isometry_error"]>3e-8:
                    raise RuntimeError("Chebyshev walk lost orbital orthonormality")
            if n==48 and w==.775 and delta==.85:
                example[target]=(t,cap,exact[:,1])
    with (ROOT/"results"/"bound_chebyshev_revivals.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig,ax=plt.subplots(figsize=(8.3,4.5))
    for n,w,d in CASES:
        if (w,d) not in ((.775,.85),(.6,1.)):
            continue
        sub=[r for r in rows if r["L"]==n and
             r["mean_potential"]==w and r["asymmetry_delta"]==d]
        ax.semilogy([r["polynomial_degree"] for r in sub],
                    [r["actual_revival_capacity_RMSE"] for r in sub],
                    marker="o",label=f"L={n}, (u,v)=({w+d/2:g},{w-d/2:g})")
    ax.set(xlabel="Maximum coherent-walk polynomial order M",
           ylabel="Capacity RMSE in revival window",
           title="A-priori Bessel-tail-controlled continuum convergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"bound_chebyshev_convergence.png",dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,4.6))
    t,_,exact=example[1e-12]
    ax.plot(t,exact,label="Held-out full-chain exact",lw=2.2)
    for target in TARGETS:
        _,values,_=example[target]
        ax.plot(t,values,label=f"Pole+walk, tolerance {target:.0e}",ls="--")
    ax.set(xlabel="Time [inverse hopping]",ylabel="Capacity [nats²]",
           title="Exact pole + finite coherent walks: strong defect L=48")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"bound_chebyshev_capacity.png",dpi=180)
    plt.close(fig)

    print("Chebyshev propagation is established literature (Tal-Ezer & Kosloff 1984).")
    print("The Bessel tail bounds certify truncation, not floating-point/pole errors.")
    print("A compact few-reflection revival law remains an open research goal.")


if __name__=="__main__":
    main()
