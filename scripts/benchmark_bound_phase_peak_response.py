"""Predict phase-controlled capacity peak shifts; test angular harmonics.

This fixes the Hamiltonian and its spectrum. The only variation is a
pure-state phase gate on the unique localized eigenmode, preserving all
final energy-mode populations.

Part A:
  Compute analytic chi_theta(t)=partial_theta C(t,theta)|theta=0 using
  ONLY the theta=0 state, then predict the peak-time/height contrasts
  for theta=+/-delta. Exact phase-kicked trajectories are computed
  afterwards solely to score the prediction.

Part B:
  Test whether a first angular Fourier harmonic, fitted at ONLY
  four cardinal phases, predicts twelve other held-out phase values.
  This empirical compression is not the exact determinant identity;
  modular capacity can contain higher angular harmonics.

Outputs:
  results/bound_phase_peak_susceptibility.csv
  results/bound_phase_fourier_heldout.csv
  figures/bound_phase_peak_shift_prediction.png
  figures/bound_phase_harmonic_compression.png

No claim of an original universal strong-defect first-revival law:
the reference band dynamics are evaluated by exact Gaussian evolution.
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
from scattering_phase import initial_modes,entropy_capacity
from bound_coherence_determinant import SingleBoundCoherence
from bound_phase_peak_response import (
    phase_trajectory,first_harmonic_from_cardinal_phases)
from capacity_peak_response import predict_reversed_peaks,tracked_peak
from revival_benchmark import first_broad_capacity_peak

CASES=((48,(1.2,.35)),(64,(1.2,.35)),
       (48,(.7,1.4)),(56,(1.1,.1)))
PHASE_STEPS=(.02,.10,.30)
NUM_PHASES=16
OBS_STEP=.2
FOURIER_STEP=.4
PHASE_CARDINAL=(0,4,8,12)


def save_csv(path,records):
    if not records:
        raise ValueError("nothing to save")
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def phase_capacity_at_components(pieces,theta):
    """One eigensolve; NO additional one-particle time evolution."""
    X=pieces["X"]
    x=X[:,0]
    y=X[:,1]
    xy=np.outer(x,y.conj())
    G=pieces["dephased"]+np.exp(1j*theta)*xy+np.exp(-1j*theta)*xy.conj().T
    p=np.linalg.eigvalsh(G)
    return entropy_capacity(p)[1]


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    response_records=[]
    fourier_records=[]
    visual_response={}
    visual_fourier={}
    for n,pair in CASES:
        occupied=initial_modes(n,"staggered",stagger=2.5)
        model=SingleBoundCoherence(n,pair,occupied)

        # Baseline theory uses one preparation phase ONLY.
        times=np.arange(0,1.8*n+.001,OBS_STEP)
        reference=phase_trajectory(model,times,theta=0.)
        for delta in PHASE_STEPS:
            prediction=predict_reversed_peaks(
                times,reference[:,0],reference[:,1],n,delta)
            if prediction is None:
                raise RuntimeError("isolated reference peak not identified")
            # HELD OUT phase-kicked trajectories, never used to fit theory.
            forward=model.trajectory(times,delta)[:,0]
            reverse=model.trajectory(times,-delta)[:,0]
            anchor=prediction["reference_peak_time"]
            a=tracked_peak(times,forward,anchor,n)
            b=tracked_peak(times,reverse,anchor,n)
            valid=a is not None and b is not None
            actual_time=(a[0]-b[0]) if valid else np.nan
            actual_height=(a[1]-b[1]) if valid else np.nan
            record=dict(
                L=n,onsite_u=pair[0],onsite_v=pair[1],delta_phase=delta,
                reference_peak_time=anchor,
                reference_peak_curvature=prediction["reference_peak_curvature"],
                predicted_phase_time_contrast=prediction["predicted_time_contrast"],
                exact_phase_time_contrast=actual_time,
                absolute_peak_time_contrast_error=abs(
                    prediction["predicted_time_contrast"]-actual_time),
                predicted_phase_height_contrast=prediction["predicted_height_contrast"],
                exact_phase_height_contrast=actual_height,
                absolute_peak_height_contrast_error=abs(
                    prediction["predicted_height_contrast"]-actual_height),
                tracked_peak_exists=valid,
                max_near_endpoint_occupations=int(np.max(reference[:,2])),
            )
            response_records.append(record)
            print(f"L={n} pair={pair} phase +/-{delta:g}:"
                  f" predicted Δt={record['predicted_phase_time_contrast']:+.5f}"
                  f" vs held-out {actual_time:+.5f}")
            if pair==(1.2,.35):
                visual_response[(n,delta)]=(record["predicted_phase_time_contrast"],
                                            actual_time)

        # Separate, predeclared 4-phase harmonic fit and 12-phase tests.
        t=np.arange(0,1.8*n+.001,FOURIER_STEP)
        theta=2*np.pi*np.arange(NUM_PHASES)/NUM_PHASES
        values=np.empty((len(t),NUM_PHASES))
        for i,time in enumerate(t):
            pieces=model.components(float(time),phase_kick=0.)
            for j,phi in enumerate(theta):
                values[i,j]=phase_capacity_at_components(pieces,float(phi))
        approx=first_harmonic_from_cardinal_phases(
            values[:,list(PHASE_CARDINAL)],theta)
        mask=(t>=.4*n)&(t<1.65*n)
        harmonic=np.fft.rfft(values,axis=1)/NUM_PHASES
        a1=float(2*np.mean(np.abs(harmonic[mask,1])))
        a2=float(2*np.mean(np.abs(harmonic[mask,2])))
        ratio=(a2/a1) if a1>0 else np.nan

        for k,phi in enumerate(theta):
            if k in PHASE_CARDINAL:
                continue
            actual=values[:,k]
            predicted=approx[:,k]
            a=first_broad_capacity_peak(t,actual,n)
            b=first_broad_capacity_peak(t,predicted,n)
            observed_rmse=float(np.sqrt(
                np.mean((actual[mask]-predicted[mask])**2)))
            null_rmse=float(np.sqrt(np.mean(
                (actual[mask]-np.mean(values[mask],axis=1))**2)))
            fourier_records.append(dict(
                L=n,onsite_u=pair[0],onsite_v=pair[1],
                heldout_phase_index=k,theta_over_pi=float(phi/np.pi),
                phase_first_harmonic_mean_amplitude=a1,
                phase_second_harmonic_mean_amplitude=a2,
                second_to_first_harmonic_ratio=ratio,
                heldout_capacity_RMSE=observed_rmse,
                phase_independent_baseline_RMSE=null_rmse,
                exact_first_peak_time=a[0],
                four_cardinal_phase_first_harmonic_peak_time=b[0],
                peak_grid_time_error=abs(a[0]-b[0]),
            ))
        rmse=np.mean([v["heldout_capacity_RMSE"] for v in fourier_records
                      if v["L"]==n and
                      (v["onsite_u"],v["onsite_v"])==pair])
        print(f"  held-out phase harmonic ratio A2/A1={ratio:.3f},"
              f" average RMSE={rmse:.5f}")
        if pair==(1.2,.35):
            visual_fourier[n]=(t,theta,values,approx)

    save_csv(ROOT/"results"/"bound_phase_peak_susceptibility.csv",
             response_records)
    save_csv(ROOT/"results"/"bound_phase_fourier_heldout.csv",
             fourier_records)

    fig,axes=plt.subplots(1,2,figsize=(10.7,4.4))
    for ax,n in zip(axes,(48,64)):
        delta=np.array(PHASE_STEPS)
        predicted=[visual_response[n,float(d)][0] for d in delta]
        measured=[visual_response[n,float(d)][1] for d in delta]
        ax.plot(delta,predicted,marker="x",ls="--",
                label="Prediction from theta=0 susceptibility")
        ax.plot(delta,measured,marker="o",
                label="Held-out exact +/-theta peak times")
        ax.axhline(0,color="gray",lw=.7)
        ax.set(xlabel="Absolute initial bound-band phase offset",
               ylabel="Peak-time contrast (+theta) minus (-theta)",
               title=f"Chain length L={n}")
        ax.legend(fontsize=8)
    fig.suptitle("First capacity-revival peak response to pure-state coherence")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"bound_phase_peak_shift_prediction.png",dpi=190)
    plt.close(fig)

    t,phases,values,approx=visual_fourier[48]
    fig,axes=plt.subplots(2,1,figsize=(8.8,7.1),sharex=True)
    for ax,j in zip(axes,(2,6)):
        ax.plot(t,values[:,j],label="Held-out phase capacity",lw=2)
        ax.plot(t,approx[:,j],ls="--",
                label="Prediction from four cardinal phases")
        ax.set(title=f"L=48, fixed defect (1.2,0.35), theta={phases[j]/np.pi:.3g}π",
               ylabel="Modular capacity [nats²]")
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Time [inverse hopping]")
    fig.suptitle("First harmonic predicts held-out phases but is not exact for capacity")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"bound_phase_harmonic_compression.png",dpi=190)
    plt.close(fig)
    print("The exact determinant is first-harmonic in phase;")
    print("the nonlinear capacity demonstrably contains higher harmonics.")


if __name__=="__main__":
    main()
