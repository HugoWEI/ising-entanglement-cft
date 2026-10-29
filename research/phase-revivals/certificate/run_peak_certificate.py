"""Reproduce the source-only, conditional LOCAL peak-location enclosures.

Prediction: fourth-order CLEAN-Hamiltonian spectral / two-site Volterra solver.
The final defective Hamiltonian is NEVER diagonalized in this script.

The bounds hold as exact-arithmetic inequalities. Numpy/Scipy evaluations are
NOT outward-rounded numerical certificates. The peaks are for the explicitly
specified discrete Gaussian-kernel smoothing, not an earlier reflection-padded
filter. This certifies the full two-site solver, NOT the reduced Airy surrogate.
"""
import json
import numpy as np
from scipy.optimize import brentq
from high_order_volterra import solve_hermite_volterra, cubic_drive_error_majorant
from phase_revival_helper import initial_occupied
from peak_certificate import (weights_derivatives,capacity_polar,scalar_modulus_safe,
                              improved_strict_interval_diagnostic)


def run():
    L=48; h=.00625; T=40.; sigma=2.; local_halfwidth=.03
    times=np.arange(0,T+h/4,h)
    Q=initial_occupied(L)
    records=[]
    data=[]
    for pair, bracket in [((1.2,.35),(26.0,26.8)),((.35,1.2),(27.,27.7))]:
        F,Y,DY,details=solve_hermite_volterra(L,times,Q,pair,return_details=True)
        eps_amp=cubic_drive_error_majorant(L,times,Q,pair,F,Y,DY)
        eps_P=4*eps_amp
        eps_C=(L//2)*scalar_modulus_safe(eps_P)
        c=np.array([capacity_polar(x) for x in F])
        def deriv(x): return float(weights_derivatives(times,sigma,x)[1]@c)
        peak=float(brentq(deriv,*bracket,xtol=1e-12))
        bounds=improved_strict_interval_diagnostic(times,c,sigma,peak,local_halfwidth,eps_C,ngrids=801)
        rr={"L":L,"T":T,"dt":h,"pair":list(pair),"smoother":"normalized discrete Gaussian kernel", "sigma":sigma,
            "peak":peak,"source_amplitude_error_upper":float(eps_amp),"projector_error_upper":float(eps_P),
            "capacity_sample_error_upper":float(eps_C),"source_node_residuals":details,
            "peak_bound":bounds,"numerical_certification":"conditional exact-arithmetic; floating-point evaluation, not interval arithmetic"}
        records.append(rr);data.append((peak,c,pair))
        print(json.dumps({"pair":pair,"peak":peak,"amplitude_bound":eps_amp,"capacity_bound":eps_C,
                          "local_radius":bounds["radius_upper"],"local_uniqueness":bounds["robust_unique_root_by_analytic_majorants"]}),flush=True)
    with open('peak_certificate_results.json','w') as f: json.dump({"method":"source-only Hermite, conditional local certificate","cases":records},f,indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    for peak,c,pair in data:
        xp=np.linspace(peak-.11,peak+.11,101)
        deriv=np.array([weights_derivatives(times,sigma,x)[1]@c for x in xp])
        ax.plot(xp-peak,deriv,label=f"defect {tuple(pair)}")
        ax.axvspan(-local_halfwidth,local_halfwidth,alpha=.07)
    ax.axhline(0,linewidth=.9);ax.set_xlabel('Time relative to predicted local peak');ax.set_ylabel('Derivative of Gaussian-smoothed capacity');
    ax.set_title('Source-only local peak certification: current zero and bracketing window');ax.legend();fig.tight_layout()
    fig.savefig('figures/peak_current_local_windows.png',dpi=180);plt.close(fig)
    return records

if __name__=='__main__':run()
