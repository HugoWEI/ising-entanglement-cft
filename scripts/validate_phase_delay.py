"""Known-physics benchmark: predict a wavepacket delay from reflection phases.

In the number-conserving nearest-neighbor tight-binding chain, two
reflection-related onsite defect pairs have identical transmission
amplitudes but different left-reflection phases. The stationary-phase
(Wigner) delay d(arg r)/dE predicts their reflected wavepacket arrival
time difference. This is an established scattering result, NOT a new
many-body capacity-revival theory.

Run:
  python scripts/validate_phase_delay.py

Produces: results/wavepacket_phase_delay.csv and
          figures/wavepacket_phase_delay.png
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from scattering_phase import tight_binding_hamiltonian, left_incident_scattering

LENGTH=240
ON_SITE_PAIRS=((1.2,.35),(.35,1.2))
K0=1.1
SIGMA_X=12.0
SOURCE=70
DETECTOR=55
TIMES=np.linspace(46.,79.,331)


def reflection_delay(k, pair, dk=1e-4):
    # E=-2 cos(k), with dE/dk=2 sin(k).
    r_minus,_=left_incident_scattering(k-dk,*pair)
    r_plus,_=left_incident_scattering(k+dk,*pair)
    derivative=np.angle(r_plus/r_minus)/(2*dk)
    return derivative/(2*np.sin(k))


def reflected_detector_profile(pair):
    x=np.arange(LENGTH)
    initial=np.exp(-(x-SOURCE)**2/(4*SIGMA_X**2))*np.exp(1j*K0*x)
    initial/=np.linalg.norm(initial)
    energies,vectors=np.linalg.eigh(tight_binding_hamiltonian(LENGTH,pair))
    weights=vectors.T@initial
    projected= vectors[DETECTOR,:]*weights
    amplitude=np.exp(-1j*TIMES[:,None]*energies[None,:])@projected
    return np.abs(amplitude)**2


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    profiles=[reflected_detector_profile(pair) for pair in ON_SITE_PAIRS]
    peak_times=[float(TIMES[np.argmax(p)]) for p in profiles]
    observed=peak_times[0]-peak_times[1]
    delays=[reflection_delay(K0,pair) for pair in ON_SITE_PAIRS]
    predicted=delays[0]-delays[1]
    assert abs(observed-predicted) < .3, "wavepacket phase-delay consistency failed"
    print(f"Predicted arrival-time difference from scattering phases: {predicted:.5f}")
    print(f"Observed detector peak-time difference: {observed:.5f}")
    import csv
    with (ROOT/"results"/"wavepacket_phase_delay.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("time","detector_prob_uv","detector_prob_vu"))
        for time,p,q in zip(TIMES,*profiles):
            writer.writerow((f"{time:.6g}",f"{p:.12g}",f"{q:.12g}"))
    fig,ax=plt.subplots(figsize=(7.5,4.3))
    for prof,pair in zip(profiles,ON_SITE_PAIRS):
        ax.plot(TIMES,prof,label=f"onsite defect {pair}")
    ax.set(xlabel="Evolution time [hbar/hopping]", ylabel="Probability at left detector",
           title=f"Known Wigner phase-delay benchmark: predicted {predicted:.3f}, measured {observed:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"wavepacket_phase_delay.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
