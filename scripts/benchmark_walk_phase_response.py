"""No-Hamiltonian-diagonalization coherence-phase FIRST-PEAK predictor.

Theory side:
  clean-chain 2x2 bound pole and projector + finite coherent Bessel
  walks + exact rank-two phase susceptibility + implicit peak
  differentiation, using ONLY theta=0 dynamics.

Held-out test side:
  exact single-bound Gaussian evolution for +/-0.02 and +/-0.1 rad,
  used ONLY after the prediction has been constructed.

The band-walk orbital truncation is bounded a priori. The final
susceptibility and peak-time derivatives are NOT independently
certified by that bound because f'(p) has endpoint singularities.
This script reports numerical errors explicitly.

Outputs:
  results/walk_phase_peak_response.csv
  figures/walk_phase_peak_response.png
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
from capacity_peak_response import tracked_peak

CASES=((48,(1.2,.35)),(64,(1.2,.35)),
       (48,(.7,1.4)),(56,(1.1,.1)))
DELTAS=(.02,.1,.3)
DT=.2


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    results=[]
    for n,pair in CASES:
        occupied=initial_modes(n,"staggered",stagger=2.5)
        model=WalkPhaseResponse(n,(pair[0]+pair[1])/2,
                                pair[0]-pair[1],occupied)
        times=np.arange(0,1.8*n+.001,DT)
        # Construct THE reference response once using the finite walk.
        ref=model.trajectory(times,phase=0.,target=1e-12)
        from capacity_peak_response import predict_reversed_peaks
        exact_reference=SingleBoundCoherence(n,pair,occupied)
        for eta in DELTAS:
            pred=predict_reversed_peaks(
                times,ref["values"][:,0],ref["values"][:,1],n,eta)
            if pred is None:
                raise RuntimeError("reference phase peak undefined")
            # Validate using the independent defective-chain
            # eigensolver only AFTER calculating the prediction.
            forward=exact_reference.trajectory(times,phase_kick=eta)[:,0]
            reverse=exact_reference.trajectory(times,phase_kick=-eta)[:,0]
            plus=tracked_peak(times,forward,
                              pred["reference_peak_time"],n)
            minus=tracked_peak(times,reverse,
                               pred["reference_peak_time"],n)
            valid=plus is not None and minus is not None
            exact_diff=plus[0]-minus[0] if valid else np.nan
            row=dict(
                L=n,onsite_u=pair[0],onsite_v=pair[1],
                phase_offset=eta,
                walk_polynomial_order=ref["degree"],
                walk_orbital_tail_bound=ref["a_priori_orbital_tail_bound"],
                max_walk_isometry_error=ref["max_occupied_isometry_error"],
                predicted_phase_peak_time_contrast=pred["predicted_time_contrast"],
                independently_measured_phase_peak_time_contrast=exact_diff,
                absolute_time_contrast_error=abs(
                    pred["predicted_time_contrast"]-exact_diff),
                predicted_phase_peak_height_contrast=pred["predicted_height_contrast"],
                independently_measured_peak_height_contrast=(
                    plus[1]-minus[1] if valid else np.nan),
                stable_tracked_peak=valid,
                no_full_hamiltonian_diag_for_prediction=True,
            )
            results.append(row)
            print(f"L={n}, defect={pair}, +/-theta={eta:.2f}:"
                  f" predicted dt={pred['predicted_time_contrast']:+.6f},"
                  f" exact dt={exact_diff:+.6f},"
                  f" walks={ref['degree']},"
                  f" orbital tail bound={ref['a_priori_orbital_tail_bound']:.1e}")
    out=ROOT/"results"/"walk_phase_peak_response.csv"
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    for n,pair in CASES:
        group=[r for r in results if r["L"]==n
               and (r["onsite_u"],r["onsite_v"])==pair]
        label=f"L={n}, ({pair[0]:g},{pair[1]:g})"
        ax.plot([r["phase_offset"] for r in group],
                [r["independently_measured_phase_peak_time_contrast"] for r in group],
                marker="o",label=label+" exact")
        ax.plot([r["phase_offset"] for r in group],
                [r["predicted_phase_peak_time_contrast"] for r in group],
                ls="--",marker="x",label=label+" finite walk")
    ax.set(xlabel="Initial bound-continuum phase offset |theta| [rad]",
           ylabel="Peak time (+theta) minus peak time (-theta)",
           title="No-diagonalization scattering walk predicts local phase shift")
    ax.axhline(0,color="gray",lw=.7)
    ax.legend(fontsize=7,ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"walk_phase_peak_response.png",dpi=190)
    plt.close(fig)
    print("A local phase-response formula, not a universal short-path law.")


if __name__=="__main__":
    main()
