"""Strong-defect phase-curvature peak predictions: NO Hamiltonian diagonalization.

Theory: clean sine 2x2 bound pole + projected Bessel/Chebyshev
continuum walks => C(t,0), chi_theta(t,0), kappa_theta(t,0).
From this SINGLE reference phase, first/second-order capacity
curves at theta=+/-eta are predicted without executing the
phase-shifted time evolution.

Exact asymmetrical eigensolver is reserved for held-out scoring.
The Chebyshev orbital-tail bound does not by itself certify
the error in spectral phase curvature or in peak derivatives.

Outputs:
  results/walk_phase_quadratic_peak.csv
  figures/walk_phase_quadratic_peak.png
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
from walk_phase_response import WalkPhaseResponse
from bound_coherence_determinant import SingleBoundCoherence
from capacity_peak_response import tracked_peak,find_reference_peak

CASES=((48,(1.2,.35)),(64,(1.2,.35)),
       (48,(.7,1.4)),(56,(1.1,.1)))
OFFSETS=(.08,.3,.6)
DT=.2

def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    rows=[]
    for n,pair in CASES:
        times=np.arange(0,1.8*n+.01,DT)
        f0=initial_modes(n,"staggered",stagger=2.5)
        solver=WalkPhaseResponse(n,(pair[0]+pair[1])/2,
                                 pair[0]-pair[1],f0)
        theory=solver.trajectory(
            times,phase=0.,target=1e-12,include_curvature=True)
        C,chi,kappa=(theory["values"][:,j] for j in range(3))
        peak=find_reference_peak(times,C,n)
        if peak is None:
            print(f"L={n},pair={pair}: reference peak undefined")
            continue
        t0=peak[0]
        validation=SingleBoundCoherence(n,pair,f0)
        for offset in OFFSETS:
            for sign in (-1,1):
                eta=sign*offset
                first=tracked_peak(times,C+eta*chi,t0,n)
                second=tracked_peak(times,C+eta*chi+.5*eta**2*kappa,t0,n)
                # Held-out exact reference, never used to fit theory
                exact=tracked_peak(
                    times,validation.trajectory(times,phase_kick=eta)[:,0],
                    t0,n)
                valid=first is not None and second is not None and exact is not None
                row=dict(
                    L=n,u=pair[0],v=pair[1],phase_offset=eta,
                    polynomial_degree=theory["degree"],
                    orbital_tail_bound=theory["a_priori_orbital_tail_bound"],
                    numerical_isometry_error=theory["max_occupied_isometry_error"],
                    reference_peak_time=t0,
                    predicted_linear_time=first[0] if first else np.nan,
                    predicted_quadratic_time=second[0] if second else np.nan,
                    heldout_exact_time=exact[0] if exact else np.nan,
                    linear_time_error=abs(first[0]-exact[0]) if valid else np.nan,
                    quadratic_time_error=abs(second[0]-exact[0]) if valid else np.nan,
                    predicted_linear_height=first[1] if first else np.nan,
                    predicted_quadratic_height=second[1] if second else np.nan,
                    heldout_exact_height=exact[1] if exact else np.nan,
                    quadratic_height_error=abs(second[1]-exact[1]) if valid else np.nan,
                    stable_local_peak_branch=valid,
                )
                rows.append(row)
                print(f"L={n}, pair={pair}, phase={eta:+.2f}:"
                      f" exact={row['heldout_exact_time']:.5f}"
                      f", linear={row['predicted_linear_time']:.5f}"
                      f", quadratic={row['predicted_quadratic_time']:.5f}")
    with (ROOT/"results"/"walk_phase_quadratic_peak.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)

    fig,axes=plt.subplots(1,2,figsize=(10.7,4.5))
    for n,pair in CASES:
        selected=[r for r in rows if r["L"]==n and
                  (r["u"],r["v"])==pair]
        label=f"L={n} ({pair[0]:g},{pair[1]:g})"
        axes[0].plot([r["phase_offset"] for r in selected],
                     [r["linear_time_error"] for r in selected],
                     marker="o",label=label)
        axes[1].plot([r["phase_offset"] for r in selected],
                     [r["quadratic_time_error"] for r in selected],
                     marker="o",label=label)
    for ax in axes:
        ax.set_xlabel("Initial bound–continuum phase shift [rad]")
        ax.set_ylabel("Absolute peak-time prediction error")
    axes[0].set_title("First-order phase model")
    axes[1].set_title("Second-order phase model")
    axes[0].legend(fontsize=7)
    fig.suptitle("Quadratic revival-peak prediction from coherent finite walks")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"walk_phase_quadratic_peak.png",dpi=185)
    plt.close(fig)

    for offset in OFFSETS:
        subset=[r for r in rows if r["stable_local_peak_branch"] and
                abs(r["phase_offset"])==offset]
        if subset:
            e1=np.mean([r["linear_time_error"] for r in subset])
            e2=np.mean([r["quadratic_time_error"] for r in subset])
            print(f"|phase|={offset:g}, mean absolute error:"
                  f" linear={e1:.6g}, quadratic={e2:.6g},"
                  f" reduction={100*(1-e2/e1):.1f}%")
    print("Chebyshev bound certifies orbital propagation truncation only;")
    print("curvature and peak-time errors are independently scored.")


if __name__=="__main__":
    main()
