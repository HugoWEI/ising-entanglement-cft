"""Predict reversed-defect modular-capacity contrasts using symmetric linear response.

Only the symmetric reference h(w,w) is used to COMPUTE the prediction:
  DeltaC_pred(t) = 2 delta chi_C(t),
  h(delta) = h(w,w) + delta (|a><a|-|b><b|)/2.

The exact dynamics of the +/-delta defect are subsequently used only
for validation. This is linear-response theory, not a new mechanism
or a universally valid first-revival formula.

Outputs:
  results/capacity_asymmetry_linear_response.csv
  results/capacity_asymmetry_example.csv
  figures/capacity_asymmetry_response.png
  figures/capacity_asymmetry_error_scaling.png
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
from capacity_asymmetry_response import DefectAsymmetryResponse
from revival_benchmark import gaussian_smooth

LENGTHS=(24,48,64)
MEAN_POTENTIALS=(.775,.25)
DELTAS=(.02,.1,.3,.85)
TIME_STEP=.2
INITIAL_STAGGER=2.5
SMOOTH_SIGMA=2.0


def write_csv(path,rows):
    if not rows:
        raise ValueError("empty records")
    with path.open("w",newline="") as dest:
        writer=csv.DictWriter(dest,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dominant_interior_contrast(times,signal,length):
    """Largest |smoothed contrast| within [0.65L,1.45L].

    If maximizing at either search-window boundary within 1.0 time unit,
    return NaN: this is NOT a resolved revival peak.
    """
    smoothed=gaussian_smooth(signal,SMOOTH_SIGMA/TIME_STEP)
    low=.65*length
    high=1.45*length
    window=np.where((times>=low)&(times<=high))[0]
    best=window[np.argmax(abs(smoothed[window]))]
    if times[best]-low<1.0 or high-times[best]<1.0:
        return np.nan,np.nan
    return float(times[best]),float(smoothed[best])


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    summary=[]
    example=[]
    for n in LENGTHS:
        times=np.arange(int(round(1.5*n/TIME_STEP))+1)*TIME_STEP
        initial=initial_modes(n,"staggered",stagger=INITIAL_STAGGER)
        window=(times>=.4*n)&(times<=1.5*n)
        for w in MEAN_POTENTIALS:
            model=DefectAsymmetryResponse(n,w,initial)
            response=model.response_trajectory(times)
            symmetric_capacity=response[:,0]
            chi=response[:,1]
            edge_weight=response[:,2]
            for delta in DELTAS:
                # Prediction is computed from symmetric reference ONLY.
                prediction=2*delta*chi
                # These two exact runs are held back until after prediction.
                forward=trajectory(n,(w+delta/2,w-delta/2),times,initial)[:,1]
                reverse=trajectory(n,(w-delta/2,w+delta/2),times,initial)[:,1]
                measured=forward-reverse
                residual=prediction-measured
                rmse=float(np.sqrt(np.mean(residual[window]**2)))
                null=float(np.sqrt(np.mean(measured[window]**2)))
                t_exact,h_exact=dominant_interior_contrast(times,measured,n)
                t_pred,h_pred=dominant_interior_contrast(times,prediction,n)
                record=dict(
                    L=n,mean_potential=w,asymmetry_delta=delta,
                    response_model="symmetric_reference_linear_response",
                    predicted_contrast_rmse=rmse,
                    zero_contrast_baseline_rmse=null,
                    relative_error_vs_null=rmse/null if null>0 else np.nan,
                    improvement_fraction=1-rmse/null if null>0 else np.nan,
                    max_near_edge_derivative_weight=float(max(edge_weight)),
                    interior_contrast_time_exact=t_exact,
                    interior_contrast_time_prediction=t_pred,
                    interior_contrast_value_exact=h_exact,
                    interior_contrast_value_prediction=h_pred,
                    # A contrast extremum is NOT a peak of C(t) itself.
                )
                summary.append(record)
                print(f"L={n}, w={w:.3g}, delta={delta:.3g}:"
                      f" linear RMSE={rmse:.6g}, null RMSE={null:.6g},"
                      f" improvement={record['improvement_fraction']:.1%}")
                if n==48 and w==.775 and delta in (.1,.85):
                    for j,t in enumerate(times):
                        example.append(dict(
                            L=n,mean_potential=w,asymmetry_delta=delta,
                            time=t,capacity_symmetric=symmetric_capacity[j],
                            susceptibility=chi[j],exact_difference=measured[j],
                            predicted_difference=prediction[j],
                            absolute_error=abs(residual[j]),
                            near_edge_derivative_weight=edge_weight[j],
                        ))
    write_csv(ROOT/"results"/"capacity_asymmetry_linear_response.csv",summary)
    write_csv(ROOT/"results"/"capacity_asymmetry_example.csv",example)

    fig,axes=plt.subplots(2,1,figsize=(9,7.4),sharex=True)
    for ax,delta in zip(axes,(.1,.85)):
        subset=[r for r in example if r["asymmetry_delta"]==delta]
        t=[r["time"] for r in subset]
        ax.plot(t,[r["exact_difference"] for r in subset],
                label="Exact contrast, opposite defects",lw=2)
        ax.plot(t,[r["predicted_difference"] for r in subset],
                ls="--",label="2 delta × analytic susceptibility")
        ax.axhline(0,color="gray",lw=.8)
        ax.axvspan(.4*48,1.5*48,alpha=.075,color="gray")
        ax.set_ylabel("Capacity contrast [nats²]")
        ax.set_title(f"L=48, w=.775, asymmetry delta={delta:g}")
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Time [inverse hopping]")
    fig.suptitle("A parameter-free linear-response prediction succeeds only for weak asymmetry")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"capacity_asymmetry_response.png",dpi=190)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.7))
    for n in LENGTHS:
        sub=[r for r in summary if r["L"]==n and r["mean_potential"]==.775]
        ax.loglog([r["asymmetry_delta"] for r in sub],
                  [r["predicted_contrast_rmse"] for r in sub],
                  marker="o",label=f"L={n}")
    ax.set(xlabel="Asymmetry delta",
           ylabel="RMSE of predicted capacity contrast",
           title="Perturbative linear response fails at large defect asymmetry")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"capacity_asymmetry_error_scaling.png",dpi=190)
    plt.close(fig)

    print("This is a general Kubo/Duhamel response baseline, not a new capacity-revival law.")


if __name__=="__main__":
    main()
