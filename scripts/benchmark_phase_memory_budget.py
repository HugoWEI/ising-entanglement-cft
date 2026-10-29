"""Empirically test the conserved bound-phase coherence budget in revivals.

Fixed post-quench defect + chosen initial staggered ground state:
rotate only the localized eigenmode phase theta, keeping h fixed.
For each initial mass/length/defect, record
  beta (bound occupation),
  alpha (bound probability in left half),
  exact globally conserved Frobenius coherence budget,
  actual reduced covariance trace difference and rigorous bound,
  actual modular-capacity difference and endpoint-safe upper bound,
  operational smoothed first revival peak time.

This test is intentionally MECHANISTIC and explores whether a
phase-memory norm budget is a useful descriptor. It does NOT assume
that peak timing is monotone in beta or that the bound is tight.
All Hamiltonian diagonalization here is strictly for validation /
mechanism analysis, NOT a claimed reduced-cost predictor.

Outputs:
 results/phase_memory_budget.csv
 results/phase_memory_time_samples.csv
 figures/phase_memory_occupation_and_peaks.png
 figures/phase_memory_capacity_bounds.png
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
from phase_memory_budget import (
    coherence_budget,capacity_difference_bounds,
    covariance_phase_distance)
from revival_benchmark import first_broad_capacity_peak

LENGTHS=(24,48,64)
DEFECTS=((1.2,.35),(.7,1.4))
STAGGER_MASSES=(.5,2.5,8.)
PHASES=(0.,np.pi/2,np.pi,3*np.pi/2)
STEP=.4


def save(name,rows):
    if not rows:
        raise ValueError("no measurements")
    with (ROOT/"results"/name).open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    rows=[]
    samples=[]
    for L in LENGTHS:
        times=np.arange(0,int(round(1.8*L/STEP))+1)*STEP
        for pair in DEFECTS:
            for mass in STAGGER_MASSES:
                f0=initial_modes(L,"staggered",stagger=mass)
                model=SingleBoundCoherence(L,pair,f0)
                beta=model.bound_weight
                alpha=float(np.linalg.norm(model.bound_vector[:L//2])**2)
                budget=coherence_budget(beta,np.pi,alpha)
                # Record all four phases, including the original preparation.
                curves={}
                for theta in PHASES:
                    curves[theta]=model.trajectory(times,phase_kick=theta)[:,0]
                initial_peak=first_broad_capacity_peak(times,curves[0.],L)[0]
                phases_peaks=[]
                for theta in PHASES:
                    cap=curves[theta]
                    peak_t,peak_h=first_broad_capacity_peak(times,cap,L)
                    phases_peaks.append(peak_t)
                    rows.append(dict(
                        L=L,u=pair[0],v=pair[1],initial_mass=mass,
                        theta_over_pi=theta/np.pi,
                        bound_occupation_beta=beta,
                        bound_subsystem_weight_alpha=alpha,
                        coherence_amplitude=np.sqrt(beta*(1-beta)),
                        pure_state_bound_phase_QFI=4*beta*(1-beta),
                        first_broad_peak_time=peak_t,
                        first_broad_peak_height=peak_h,
                        peak_time_shift_vs_theta_zero=peak_t-initial_peak,
                        global_phase_distance_fro_expected=
                            coherence_budget(beta,theta)["global_frobenius_exact"],
                    ))
                for t in (.45*L,.60*L,1.0*L):
                    base=model.components(t,phase_kick=0.)["full"]
                    rotated=model.components(t,phase_kick=np.pi)["full"]
                    diff=covariance_phase_distance(base,rotated)
                    upper=capacity_difference_bounds(
                        beta,alpha,np.pi,L//2)
                    observed=abs(
                        entropy_capacity(np.linalg.eigvalsh(base))[1]-
                        entropy_capacity(np.linalg.eigvalsh(rotated))[1])
                    if diff["trace"]>upper["subsystem_trace_upper"]+2e-9:
                        raise AssertionError("subsystem trace bound failed")
                    if observed>upper["capacity_holder_upper"]+2e-9:
                        raise AssertionError("capacity Hölder bound failed")
                    samples.append(dict(
                        L=L,u=pair[0],v=pair[1],initial_mass=mass,time=t,
                        beta=beta,alpha=alpha,
                        observed_subsystem_trace_distance=diff["trace"],
                        rigorous_subsystem_trace_upper=upper["subsystem_trace_upper"],
                        observed_capacity_difference=observed,
                        rigorous_capacity_holder_upper=upper["capacity_holder_upper"],
                        trace_bound_slack=
                            upper["subsystem_trace_upper"]-diff["trace"],
                    ))
                print(f"L={L}, defect={pair}, stagger={mass:g},"
                      f" beta={beta:.5f}, coherence={np.sqrt(beta*(1-beta)):.5f},"
                      f" phase-peak span={np.nanmax(phases_peaks)-np.nanmin(phases_peaks):.2f}")
    save("phase_memory_budget.csv",rows)
    save("phase_memory_time_samples.csv",samples)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    for pair in DEFECTS:
        key=[r for r in rows if r["L"]==48 and
             (r["u"],r["v"])==pair and r["theta_over_pi"]==0]
        betas=[r["bound_occupation_beta"] for r in key]
        masses=[r["initial_mass"] for r in key]
        axes[0].plot(masses,betas,marker="o",label=f"defect {pair}")
        spans=[]
        for mass in masses:
            sub=[r["first_broad_peak_time"] for r in rows
                 if r["L"]==48 and (r["u"],r["v"])==pair
                 and r["initial_mass"]==mass]
            spans.append(np.nanmax(sub)-np.nanmin(sub))
        axes[1].plot(betas,spans,marker="o",label=f"defect {pair}")
    axes[0].set(xlabel="Initial staggered mass",ylabel="Bound occupation beta",
                title="Preparation controls the coherence budget")
    axes[1].set(xlabel="Bound occupation beta",
                ylabel="Phase-dependent first-peak span",
                title="Peak timing is not fixed by beta alone")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"phase_memory_occupation_and_peaks.png",dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.5))
    selected=[x for x in samples if x["L"]==48]
    x=np.array([z["rigorous_subsystem_trace_upper"] for z in selected])
    y=np.array([z["observed_subsystem_trace_distance"] for z in selected])
    ax.scatter(x,y)
    maxval=max(x.max(),y.max())*1.06
    ax.plot([0,maxval],[0,maxval],ls="--",color="gray")
    ax.set(xlabel="Rigorous subsystem trace-distance upper bound",
           ylabel="Actual subsystem trace-distance",
           title="Rigorous bound vs physical interference (L=48)")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"phase_memory_capacity_bounds.png",dpi=180)
    plt.close(fig)
    print("The phase-budget bound is rigorous but may be loose.")
    print("It is a necessary constraint, not a predictive revival-time law.")


if __name__=="__main__":
    main()
