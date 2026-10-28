"""Validate exact parity-resummed resolvent contrast against full inversions.

All predicted resolvents use ONLY the reflection-symmetric reference
h_sym(w,w) and an exact 2x2 parity-channel resummation.

The held-out full asymmetric inverses are *validation references*.
No time-dependent capacity peak is predicted by this script.

Files:
 results/parity_resummation_benchmark.csv
 figures/parity_resummation_error.png
 figures/parity_channel_dressing.png
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
from parity_resummed_scattering import ParityResummedDefect
from scattering_phase import tight_binding_hamiltonian

LENGTHS=(24,48)
MEANS=(.775,.25)
DELTAS=(.02,.1,.3,.85,1.6)
ENERGIES=(.2+.4j,2.8+.35j)


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"figures").mkdir(exist_ok=True)
    records=[]
    for n in LENGTHS:
        for w in MEANS:
            model=ParityResummedDefect(n,w)
            for delta in DELTAS:
                plus=tight_binding_hamiltonian(
                    n,(w+delta/2,w-delta/2))
                minus=tight_binding_hamiltonian(
                    n,(w-delta/2,w+delta/2))
                for z in ENERGIES:
                    # The prediction below is computed without these
                    # asymmetric inversions; they are exact references only.
                    predict=model.contrast(z,delta)
                    linear=model.contrast(z,delta,linear=True)
                    measured=np.linalg.inv(z*np.eye(n)-plus)-np.linalg.inv(z*np.eye(n)-minus)
                    normalization=np.linalg.norm(measured)
                    err_full=float(np.linalg.norm(predict-measured)/normalization)
                    err_linear=float(np.linalg.norm(linear-measured)/normalization)
                    gp,gm,denominator,_=model.channel_data(z,delta)
                    records.append(dict(
                        L=n,mean_potential=w,delta=delta,
                        complex_energy_real=z.real,complex_energy_imag=z.imag,
                        relative_error_resummed=err_full,
                        relative_error_linear=err_linear,
                        abs_parity_denominator=abs(denominator),
                        norm_exact_operator_contrast=normalization,
                        abs_g_plus=abs(gp),
                        abs_g_minus=abs(gm),
                    ))
                    if err_full>2e-9:
                        raise AssertionError("nonperturbative parity identity failed")
                    print(f"L={n}, w={w:g}, delta={delta:g}, z={z}: "
                          f"exact err={err_full:.2e}, linear err={err_linear:.3g}")
    target=ROOT/"results"/"parity_resummation_benchmark.csv"
    with target.open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    pole_records=[]
    for n in LENGTHS:
        for w in MEANS:
            model=ParityResummedDefect(n,w)
            for delta in DELTAS:
                predicted=model.largest_eigenenergy(delta)
                perturbative=model.top_eigenenergy_quadratic(delta)
                h=tight_binding_hamiltonian(
                    n,(w+delta/2,w-delta/2))
                actual=float(np.linalg.eigvalsh(h)[-1])
                pole_records.append(dict(
                    L=n,mean_potential=w,delta=delta,
                    predicted_top_energy_from_parity=predicted,
                    exact_top_energy_validation=actual,
                    quadratic_weak_asymmetry_prediction=perturbative,
                    absolute_resummed_energy_error=abs(predicted-actual),
                    quadratic_energy_error=abs(perturbative-actual),
                    outside_propagating_band=bool(actual>2),
                ))
                if abs(predicted-actual)>1e-9:
                    raise AssertionError("parity secular pole prediction failed")
    with (ROOT/"results"/"parity_bound_energy_predictions.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(pole_records[0].keys()))
        writer.writeheader()
        writer.writerows(pole_records)

    fig,ax=plt.subplots(figsize=(8,4.8))
    for z in ENERGIES:
        sub=[x for x in records if x["L"]==24 and x["mean_potential"]==.775
             and x["complex_energy_real"]==z.real]
        ax.plot([x["delta"] for x in sub],
                [x["relative_error_linear"] for x in sub],
                marker="o",label=f"Linear response at z={z}")
        ax.plot([x["delta"] for x in sub],
                [x["relative_error_resummed"] for x in sub],
                marker="x",ls="--",label=f"Exact parity resummation at z={z}")
    ax.set(xlabel="Defect asymmetry delta",
           ylabel="Relative Frobenius error in resolvent contrast",
           yscale="log",
           title="Strong-asymmetry resolvent: parity resummation versus linear response")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"parity_resummation_error.png",dpi=180)
    plt.close(fig)

    model=ParityResummedDefect(48,.775)
    real=np.linspace(2.02,3.15,500)
    eps=.045
    fig,ax=plt.subplots(figsize=(8,4.5))
    for delta in (.1,.85,1.6):
        dressed=[]
        for energy in real:
            _,_,denominator,_=model.channel_data(complex(energy,eps),delta)
            dressed.append(abs(1/denominator))
        ax.plot(real,dressed,label=f"delta={delta:g}")
    ax.set(xlabel="Real part of complex energy",
           ylabel="|1/D(E+i epsilon)|",
           title="Even-in-asymmetry dressing near spectral resonances")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT/"figures"/"parity_channel_dressing.png",dpi=180)
    plt.close(fig)
    print("No time-domain modular-capacity revival prediction is claimed.")


if __name__=="__main__":
    main()
