"""Study bound-state poles and convergence of repeated-defect scattering.

This produces a VALIDATED SCATTERING REPRESENTATION, not a new prediction
for many-body entanglement-capacity revival peaks.

Outputs:
- results/tmatrix_bound_poles.csv
- results/tmatrix_scattering_convergence.csv
- figures/tmatrix_scattering_poles.png
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
from defect_tmatrix import TwoSiteDefectResolvent
from scattering_phase import tight_binding_hamiltonian

LENGTHS=(48,64)
DEFECTS=((1.2,.35),(.7,1.4),(.5,.2))
ORDERS=(1,2,4,8,16,32,64)


def write_csv(name, rows):
    if not rows:
        raise ValueError("No results to save")
    with (ROOT/"results"/name).open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    poles=[]
    convergence=[]
    for n in LENGTHS:
        for uv in DEFECTS:
            model=TwoSiteDefectResolvent(n,uv)
            h=tight_binding_hamiltonian(n,uv)
            exact_spectrum=np.linalg.eigvalsh(h)
            ebs=model.bound_energies()
            exact_bound=exact_spectrum[np.abs(exact_spectrum)>2+1e-8]
            if not np.allclose(ebs,exact_bound,atol=5e-9):
                raise RuntimeError("2x2 secular roots failed full diagonalization check")
            for e in ebs:
                psi=model.bound_wavefunction(e)
                center=float(np.sum(np.abs(psi[n//2-4:n//2+4])**2))
                residual=float(np.linalg.norm(h@psi-e*psi))
                poles.append({
                    "L":n,"onsite_u":uv[0],"onsite_v":uv[1],
                    "bound_energy_from_2x2":e,
                    "bound_energy_full_diagonalization":exact_bound[np.argmin(abs(exact_bound-e))],
                    "central_eight_site_weight":center,
                    "wavefunction_residual_norm":residual,
                })
            test_energies={"away_from_pole":2.8+.5j}
            if ebs.size:
                test_energies["near_bound_pole"]=float(ebs[-1])+.05j
            for regime,z in test_energies.items():
                exact=model.t_matrix(z)
                radius=model.scattering_radius(z)
                for order in ORDERS:
                    approx=model.t_matrix(z,order)
                    relative=float(np.linalg.norm(approx-exact)/np.linalg.norm(exact))
                    convergence.append({
                        "L":n,"onsite_u":uv[0],"onsite_v":uv[1],
                        "energy_regime":regime,
                        "energy_real":z.real,"energy_imag":z.imag,
                        "number_scatterings":order,
                        "neumann_radius":radius,
                        "relative_tmatrix_error":relative,
                    })
            print(f"pair={uv}, L={n}, E_bound={ebs}, largest residual=",
                  max(p["wavefunction_residual_norm"] for p in poles if p["L"]==n))
    write_csv("tmatrix_bound_poles.csv",poles)
    write_csv("tmatrix_scattering_convergence.csv",convergence)
    model=TwoSiteDefectResolvent(48,DEFECTS[0])
    bound_energy=float(model.bound_energies()[0])
    grid=np.linspace(2.04,3.55,400)
    determinants=np.array([abs(model.secular_determinant(x)) for x in grid])
    fig,axes=plt.subplots(1,2,figsize=(11,4.35))
    axes[0].plot(grid,determinants,color="C0")
    axes[0].axvline(bound_energy,ls="--",color="C3",label=f"bound pole E={bound_energy:.5f}")
    axes[0].set(xlabel="Real spectral energy E",ylabel="|det(I - g_D(E)V_D)|",
                title="Bound mode as a two-site scattering pole")
    axes[0].legend(fontsize=9)
    for regime in ("away_from_pole","near_bound_pole"):
        sub=[r for r in convergence if r["L"]==48 and
             r["onsite_u"]==1.2 and r["onsite_v"]==.35 and
             r["energy_regime"]==regime]
        axes[1].semilogy([x["number_scatterings"] for x in sub],
                        [x["relative_tmatrix_error"] for x in sub],
                        marker="o",label=regime.replace("_"," "))
    axes[1].set(xlabel="Number of defect-scattering terms",
                ylabel="Relative error in two-site T matrix",
                title="Short multiple-scattering expansions fail near a pole")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"tmatrix_scattering_poles.png",dpi=200)
    plt.close(fig)
    print("The exact 2x2 resummation captures both scattering and bound poles.")
    print("This test does NOT predict modular-capacity revival peak timings.")


if __name__=="__main__":
    main()
