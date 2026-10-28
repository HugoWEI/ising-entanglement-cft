"""Pure-state bound-mode phase-kick control of finite capacity revivals.

For a fixed defect h with one localized mode b, apply at t=0
  F_theta(0)=[I+(exp(i theta)-1)|b><b|]F0,
then evolve under the SAME h. This is a valid pure Gaussian state
for every theta, with identical mode occupations and exactly the
same expectation of every function f(h), but different offdiagonal
bound-band coherences.

The bound-band rank-two covariance formula predicts
  G_A(t;theta) = G_deph(t)
     +exp(i theta)x(t)y(t)†+exp(-i theta)y(t)x(t)†.

Only the preparation phase changes; h, scattering, and E_b are fixed.
This is a controlled mechanism diagnostic, not a claim that
the original quench or experimental apparatus already includes
a tunable impurity-bound phase gate.

Outputs:
  results/bound_phase_kick_revivals.csv
  figures/bound_phase_kick_peak_times.png
  figures/bound_phase_kick_capacity_curves.png
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
from scattering_phase import initial_modes,tight_binding_hamiltonian
from bound_coherence_determinant import SingleBoundCoherence
from revival_benchmark import first_broad_capacity_peak

LENGTHS=(48,64)
PAIRS=((1.2,.35),(.7,1.4),(.5,.2))
PHASES=(0.,.5*np.pi,np.pi,1.5*np.pi)
DT=.2


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    rows=[]
    example={}
    for n in LENGTHS:
        times=np.arange(0,1.8*n+.01,DT)
        occupied=initial_modes(n,"staggered",stagger=2.5)
        for potentials in PAIRS:
            model=SingleBoundCoherence(n,potentials,occupied)
            h=tight_binding_hamiltonian(n,potentials)
            E0=float(np.trace(occupied.conj().T@h@occupied).real)
            b=model.bound_vector
            projector=np.outer(b,b.conj())
            for theta in PHASES:
                gate=np.eye(n)+(np.exp(1j*theta)-1)*projector
                kicked=gate@occupied
                gram_error=np.linalg.norm(kicked.conj().T@kicked-
                                           np.eye(n//2))
                if gram_error>1e-9:
                    raise RuntimeError("phase kick must be unitary")
                energy=float(np.trace(kicked.conj().T@h@kicked).real)
                if abs(energy-E0)>1e-9:
                    raise RuntimeError("phase kick changed energy populations")
                C=model.trajectory(times,phase_kick=theta)[:,0]
                first,height=first_broad_capacity_peak(times,C,n)
                rows.append(dict(
                    L=n,onsite_u=potentials[0],onsite_v=potentials[1],
                    bound_energy=model.bound_energy,
                    bound_mode_initial_occupation=model.bound_weight,
                    theta_in_pi_units=theta/np.pi,
                    first_broad_peak_time=first,
                    first_broad_peak_height=height,
                    initial_total_energy=energy,
                    max_kicked_orbital_orthogonality_error=gram_error,
                ))
                print(f"L={n}, defect={potentials}, theta/pi={theta/np.pi:g}: "
                      f"first broad peak={first:.2f} height={height:.5f}")
                if potentials==(1.2,.35):
                    example[(n,theta)]=(times,C)
    out=ROOT/"results"/"bound_phase_kick_revivals.csv"
    with out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)

    fig,axes=plt.subplots(1,2,figsize=(10.3,4.4))
    for ax,n in zip(axes,LENGTHS):
        for pair in PAIRS:
            subset=[r for r in rows if r["L"]==n and
                    (r["onsite_u"],r["onsite_v"])==pair]
            ax.plot([r["theta_in_pi_units"] for r in subset],
                    [r["first_broad_peak_time"] for r in subset],
                    marker="o",label=f"({pair[0]:g},{pair[1]:g})")
        ax.set(xlabel="Initial bound-band phase theta / pi",
               ylabel="First broad capacity peak time",
               title=f"L={n}, same Hamiltonian for all phases")
        ax.legend(fontsize=8)
    fig.suptitle("Initial spectral coherence controls finite-chain capacity-revival timing")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"bound_phase_kick_peak_times.png",dpi=190)
    plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(8.5,7.3),sharex=True)
    for ax,n in zip(axes,LENGTHS):
        for theta in PHASES:
            times,capacity=example[(n,theta)]
            ax.plot(times,capacity,label=f"theta/pi={theta/np.pi:g}")
        ax.set(ylabel="Capacity [nats²]",
               title=f"L={n}, fixed central potentials (1.2,0.35)")
        ax.legend(fontsize=8,ncol=2)
    axes[-1].set_xlabel("Time [inverse hopping]")
    fig.suptitle("Same scattering and bound spectrum, different initial coherence")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"bound_phase_kick_capacity_curves.png",dpi=190)
    plt.close(fig)
    print("This is a controlled pure-state preparation change,")
    print("not a claim of an original universal revival law.")


if __name__=="__main__":
    main()
