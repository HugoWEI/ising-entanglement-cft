"""Predict first broad capacity-peak timing and height via symmetric Kubo response.

Uses only the symmetric defect (w,w) to construct predictions for the
reversed pair (w+delta/2,w-delta/2) and (w-delta/2,w+delta/2).
The exact asymmetric-defect dynamics are computed ONLY for validation.

Predictive formulas for an isolated smoothed reference maximum t0:
    Delta t_peak = -2*delta*chi_C'(t0)/C_0''(t0)+O(delta^3)
    Delta C_peak =  2*delta*chi_C(t0)+O(delta^3)

This is standard linear-response/implicit-function theory. It DOES NOT
solve strong-defect capacity revivals. A missing tracked peak is NaN.

Outputs:
  results/capacity_peak_response_benchmark.csv
  figures/capacity_peak_shift_prediction.png
  figures/capacity_peak_height_prediction.png

Run: python scripts/benchmark_capacity_peak_response.py
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
from capacity_asymmetry_response import DefectAsymmetryResponse
from capacity_peak_response import predict_reversed_peaks,tracked_peak

LENGTHS=(24,48,64)
MEANS=(.25,.775)
DELTAS=(.02,.1,.3,.85)
STEP=.2
MASS=2.5


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    rows=[]
    for n in LENGTHS:
        times=np.arange(0,1.7*n+.001,STEP)
        occupied=initial_modes(n,"staggered",stagger=MASS)
        for w in MEANS:
            response=DefectAsymmetryResponse(n,w,occupied).response_trajectory(times)
            for d in DELTAS:
                prediction=predict_reversed_peaks(
                    times,response[:,0],response[:,1],n,d)
                if prediction is None:
                    print(f"no reference peak L={n} w={w}")
                    continue
                # This section is a held-out independent validation.
                forward=trajectory(n,(w+d/2,w-d/2),times,occupied)[:,1]
                reverse=trajectory(n,(w-d/2,w+d/2),times,occupied)[:,1]
                plus=tracked_peak(times,forward,prediction["reference_peak_time"],n)
                minus=tracked_peak(times,reverse,prediction["reference_peak_time"],n)
                valid=plus is not None and minus is not None
                time_diff=(plus[0]-minus[0]) if valid else np.nan
                height_diff=(plus[1]-minus[1]) if valid else np.nan
                record=dict(
                    L=n,mean_potential=w,delta=d,
                    reference_smoothed_peak_time=prediction["reference_peak_time"],
                    reference_smoothed_peak_curvature=prediction["reference_peak_curvature"],
                    susceptibility_at_peak=prediction["susceptibility_at_peak"],
                    susceptibility_slope_at_peak=prediction["susceptibility_slope_at_peak"],
                    predicted_time_contrast=prediction["predicted_time_contrast"],
                    measured_time_contrast=time_diff,
                    abs_time_error=abs(time_diff-prediction["predicted_time_contrast"]),
                    predicted_height_contrast=prediction["predicted_height_contrast"],
                    measured_height_contrast=height_diff,
                    abs_height_error=abs(height_diff-prediction["predicted_height_contrast"]),
                    tracked_branch_exists=valid,
                    # Tracking is continuity-based. Larger deltas may
                    # switch peak branches and invalidate perturbation theory.
                )
                rows.append(record)
                print(f"L={n} w={w:g} delta={d:g}: predicted Δt="
                      f"{record['predicted_time_contrast']:+.6f},"
                      f" exact Δt={time_diff:+.6f}, valid={valid}")
    with (ROOT/"results"/"capacity_peak_response_benchmark.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)

    for metric,filename,title in (
        ("time","capacity_peak_shift_prediction.png","Predicted shift of a tracked broad capacity maximum"),
        ("height","capacity_peak_height_prediction.png","Predicted change of the same capacity peak's height")):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=False)
        for ax,w in zip(axes,MEANS):
            for n in LENGTHS:
                subset=[r for r in rows if r["L"]==n and
                        r["mean_potential"]==w and r["tracked_branch_exists"]]
                d=[r["delta"] for r in subset]
                actual=[r[f"measured_{metric}_contrast"] for r in subset]
                predicted=[r[f"predicted_{metric}_contrast"] for r in subset]
                ax.plot(d,actual,marker="o",label=f"L={n} exact")
                ax.plot(d,predicted,marker="x",linestyle="--",label=f"L={n} response")
            ax.axhline(0,color="gray",lw=.7)
            ax.set_xlabel("Defect asymmetry delta")
            ax.set_title(f"Mean potential w={w}")
        axes[0].set_ylabel(f"Peak {metric} contrast")
        axes[1].legend(fontsize=7,ncol=2)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(ROOT/"figures"/filename,dpi=180)
        plt.close(fig)
    print("Perturbative formula succeeds at weak asymmetry, fails at strong asymmetry.")
    print("A smoothed local maximum is an operational feature, not necessarily")
    print("the uniquely defined first quantum-information revival.")


if __name__=="__main__":
    main()
