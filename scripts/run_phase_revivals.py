"""A controlled phase sensitivity pilot using reversed two-site scatterers.

This tests a NECESSARY ingredient of a phase-based explanation:
two distinct defect orientations have equal T(k) across all momenta,
different reflection phases, and potentially different transients.

It DOES NOT establish a predictive phase-delay theory or novelty.
The tight-binding model differs from the transverse-field Ising chain.
"""
from pathlib import Path
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from scattering_phase import left_incident_scattering, compare_reversed_defects

L=48
U,V=1.2,0.35
TIMES=np.linspace(0, 52, 105)
MOMENTA=np.linspace(.2, np.pi-.2, 161)


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    a,b=compare_reversed_defects(L,U,V,TIMES,"staggered")
    null_a,null_b=compare_reversed_defects(L,U,V,TIMES,"mirror_fock")
    assert np.max(np.abs(null_a-null_b)) < 1e-8
    scattering=[]
    for k in MOMENTA:
        r1,t1=left_incident_scattering(k,U,V)
        r2,t2=left_incident_scattering(k,V,U)
        assert abs(t1-t2)<1e-9
        scattering.append((k,abs(t1)**2,np.angle(r1),np.angle(r2)))
    with (ROOT/"results"/"phase_revival_pilot.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("time","S_uv","S_vu","C_uv","C_vu"))
        for time,obs_a,obs_b in zip(TIMES,a,b):
            writer.writerow((f"{time:.12g}",*map(lambda z: f"{z:.12g}",(obs_a[0],obs_b[0],obs_a[1],obs_b[1]))))
    with (ROOT/"results"/"two_site_scattering.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("k","T_equal","phase_reflection_uv","phase_reflection_vu"))
        writer.writerows(scattering)
    fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
    labels=("Entropy S_A [nats]","Modular capacity C_A [nats^2]")
    for idx,ax in enumerate(axes):
        ax.plot(TIMES,a[:,idx],label=f"(u,v)=({U},{V})")
        ax.plot(TIMES,b[:,idx],label=f"(v,u)=({V},{U})",linestyle="--")
        ax.set_ylabel(labels[idx])
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Evolution time [hbar/hopping]")
    fig.suptitle("Equal transmission, different reflection phases: staggered initial state")
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"phase_revival_pilot.png",dpi=180)
    plt.close(fig)
    residual=a-b
    print("Equal transmission amplitudes confirmed at",len(MOMENTA),"momenta")
    print("Max |Delta S|:",np.max(abs(residual[:,0])))
    print("Max |Delta C|:",np.max(abs(residual[:,1])))
    print("Reflection-symmetric initial state null test:",np.max(abs(null_a-null_b)))
    print("NOTE: this demonstrates phase sensitivity, not a phase-delay prediction.")
    

if __name__=="__main__":
    main()
