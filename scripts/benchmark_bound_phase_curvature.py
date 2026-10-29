"""Second-order pure-state phase control of strong-defect capacity peaks.

Theory computes C(t,0), chi_theta(t,0), and kappa_theta(t,0)
using ONE reference bound/continuum propagation, then predicts
capacity curves at +/-eta by the predeclared Taylor hierarchy

  C_lin=C0+eta*chi,
  C_quad=C0+eta*chi+eta²*kappa/2.

A nearby peak is predicted by maximizing each approximate
smoothed curve on a fixed preselected branch window.
An optional explicit t_peak=t0+t1*eta+t2*eta² law is also
computed using a local polynomial of the time derivatives.

The exact shifted-phase trajectories are used afterward ONLY
for evaluating prediction error; no parameters are fitted.

This experiment isolates nonlinear PHASE response at fixed
strong defect, not a novel scattering/short-return path law.

Output:
  results/bound_phase_second_order_peak.csv
  figures/bound_phase_quadratic_peak_errors.png
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
from bound_phase_curvature import (
    second_order_phase_trajectory,predict_peak_second_order_explicit)
from capacity_peak_response import tracked_peak,find_reference_peak

CASES=((48,(1.2,.35)),(64,(1.2,.35)),
       (48,(.7,1.4)),(56,(1.1,.1)))
OFFSETS=(.08,.3,.6)
TIME_STEP=.2


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    rows=[]
    for n,pair in CASES:
        times=np.arange(0,1.8*n+.01,TIME_STEP)
        modes=initial_modes(n,"staggered",stagger=2.5)
        model=SingleBoundCoherence(n,pair,modes)
        reference=second_order_phase_trajectory(model,times,theta=0.)
        C,chi,kappa=reference[:,0],reference[:,1],reference[:,2]
        origin=find_reference_peak(times,C,n)
        if origin is None:
            print(f"L={n},pair={pair}: missing reference isolated peak")
            continue
        t0=origin[0]
        coeff=predict_peak_second_order_explicit(
            times,C,chi,kappa,n)
        for eta in OFFSETS:
            for sign in (-1,1):
                signed=sign*eta
                predicted_lin=C+signed*chi
                predicted_quad=predicted_lin+.5*signed*signed*kappa
                lin=tracked_peak(times,predicted_lin,t0,n)
                quad=tracked_peak(times,predicted_quad,t0,n)
                measured=tracked_peak(
                    times,model.trajectory(times,phase_kick=signed)[:,0],t0,n)
                valid=lin is not None and quad is not None and measured is not None
                if coeff is None:
                    explicit_time=np.nan
                else:
                    explicit_time=t0+signed*coeff["t1"]+signed**2*coeff["t2"]
                row=dict(
                    L=n,u=pair[0],v=pair[1],
                    phase_offset=signed,
                    reference_peak=t0,
                    measured_time=measured[0] if measured else np.nan,
                    linear_time=lin[0] if lin else np.nan,
                    quadratic_time=quad[0] if quad else np.nan,
                    explicit_second_order_time=explicit_time,
                    linear_time_error=abs(lin[0]-measured[0]) if valid else np.nan,
                    quadratic_time_error=abs(quad[0]-measured[0]) if valid else np.nan,
                    explicit_time_error=abs(explicit_time-measured[0]) if valid else np.nan,
                    measured_height=measured[1] if measured else np.nan,
                    linear_height_error=abs(lin[1]-measured[1]) if valid else np.nan,
                    quadratic_height_error=abs(quad[1]-measured[1]) if valid else np.nan,
                    reference_second_phase_susceptibility_near_edge_count=
                        int(np.max(reference[:,3])),
                    tracked_branch_exists=bool(valid),
                )
                rows.append(row)
                print(f"L={n} defect={pair} phase={signed:+.2f}:"
                      f" exact={row['measured_time']:.5f},"
                      f" linear={row['linear_time']:.5f},"
                      f" quadratic={row['quadratic_time']:.5f},"
                      f" explicit={row['explicit_second_order_time']:.5f}")
    if not rows:
        raise RuntimeError("no phase-response records")
    with (ROOT/"results"/"bound_phase_second_order_peak.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)

    fig,axes=plt.subplots(1,2,figsize=(11.2,4.6))
    for n,pair in CASES:
        sub=[row for row in rows if row["L"]==n and
             (row["u"],row["v"])==pair]
        label=f"L={n},({pair[0]:g},{pair[1]:g})"
        axes[0].plot([r["phase_offset"] for r in sub],
                     [r["linear_time_error"] for r in sub],
                     marker="o",label=label)
        axes[1].plot([r["phase_offset"] for r in sub],
                     [r["quadratic_time_error"] for r in sub],
                     marker="o",label=label)
    axes[0].set(xlabel="Initial phase shift [radians]",
                ylabel="Absolute tracked peak-time error",
                title="First-order phase response")
    axes[1].set(xlabel="Initial phase shift [radians]",
                ylabel="Absolute tracked peak-time error",
                title="Second-order phase response")
    axes[0].legend(fontsize=7)
    fig.suptitle("Nonlinear phase-response expansion at fixed strong quantum defect")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"bound_phase_quadratic_peak_errors.png",dpi=190)
    plt.close(fig)
    valid=[r for r in rows if r["tracked_branch_exists"]]
    for offset in OFFSETS:
        sub=[r for r in valid if abs(r["phase_offset"])==offset]
        if sub:
            linear=float(np.mean([r["linear_time_error"] for r in sub]))
            quadratic=float(np.mean([r["quadratic_time_error"] for r in sub]))
            print(f"|phase|={offset:g}, mean peak-time error:"
                  f" first order={linear:.6g}, second order={quadratic:.6g}, "
                  f"improvement={(1-quadratic/linear):.1%}" if linear else
                  f"|phase|={offset:g}: zero first-order error")
    print("No universal improvement is guaranteed near flat or switching peaks.")


if __name__=="__main__":
    main()
