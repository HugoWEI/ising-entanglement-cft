"""Sampled-to-continuous Gaussian-current envelope for Airy vs Bessel first returns.

Theorem is exact for input upper epsilon_j and a lower singular value s_j.
This implementation uses double-precision Bessel/Airy and singular values;
its numerical envelopes are conditional diagnostics, *not* outward-rounded proofs.
The common baseline is trapezoid Volterra with direct and two single-end images;
this does not validate the baseline against the exact defective evolution.
"""
from pathlib import Path
import os, sys, json
import numpy as np
from scipy.special import jv
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
CERT=(PROJECT/'certificate') if (PROJECT/'certificate').exists() else Path('/mnt/data/airy_certificate_work')
SRC=CERT/'src'
LIB=(PROJECT/'capacity_geometry') if (PROJECT/'capacity_geometry').exists() else Path('/mnt/data/gaussian_capacity_geometry_v1')
sys.path[:0]=[str(SRC),str(LIB),str(CERT)]
from phase_revival_research import hopping
from path_ablation import ablated_kernel
from airy_hybrid import leading_olver_j
from capacity_geometry import polar, capacity, capacity_bound
from peak_certificate import weights_derivatives
import volterra_scattering_revival as vol

L=48; M=L//2; dt=.025; T=36.; SIGMA=2.
times=np.arange(0,T+dt/4,dt)

def wavefront_differences():
    i=np.arange(L)[:,None]; j=np.arange(L)[None,:]; N=2*(L+1)
    n1=N+i-j; n2=N-i+j
    ns=np.arange(n1.min(),n1.max()+1)
    tab_b=jv(ns[:,None],2*times[None,:])
    tab_a=np.array([leading_olver_j(int(n),2*times) for n in ns])
    # Bessel and Airy evaluated separately using two implementations.
    phase1=(1j**n1)[None,:,:];phase2=(1j**n2)[None,:,:]
    B=phase1*tab_b[n1-ns[0]].transpose(2,0,1)+phase2*tab_b[n2-ns[0]].transpose(2,0,1)
    A=phase1*tab_a[n1-ns[0]].transpose(2,0,1)+phase2*tab_a[n2-ns[0]].transpose(2,0,1)
    return A,B

def amplitudes_baseline(pair,Q):
    saved=vol.image_bessel_kernel
    try:
        vol.image_bessel_kernel=lambda L,ts,shells,rows=None,cols=None:ablated_kernel(L,ts,shells,rows,cols,('direct','left','right'))
        return vol.defect_response(L,times,*pair,Q,1,return_full=True)
    finally:vol.image_bessel_kernel=saved

def source_envelope(base,WA,WB,Q,scale=1.0000000001):
    """epsilon_j bounds |C_B-C_A| if float D estimates are valid upper bounds.
    scale is NOT a rounding certificate and must not be interpreted as one.
    """
    eps=np.zeros(len(times)); a=np.zeros_like(eps); b=np.zeros_like(eps)
    residual=np.zeros_like(eps);singular_lower=np.zeros_like(eps)
    for k in range(len(times)):
        za=base[k]+WA[k]@Q;delta=(WB[k]-WA[k])@Q;zb=za+delta
        sv=np.linalg.svd(za,compute_uv=False)[-1]
        s=sv-np.linalg.norm(delta,ord=2)
        singular_lower[k]=s
        residual[k]=np.linalg.norm(delta,'fro')
        eps[k]=capacity_bound(residual[k]*scale,M,s)
        a[k]=capacity(polar(za));b[k]=capacity(polar(zb))
    return eps,a,b,residual,singular_lower

def peak_curves(C,low,high):
    def d(t,der=1):
        w,dw,ddw=weights_derivatives(times,SIGMA,t)
        return float((dw if der==1 else ddw if der==2 else w)@C)
    xs=np.linspace(low,high,251)
    roots=[brentq(lambda z:d(z),x,y,xtol=1e-11) for x,y in zip(xs[:-1],xs[1:]) if d(x)>0 and d(y)<0]
    return roots,d

def envelope_at(C,eps,peak,radius=.1,n=121):
    """Sampled diagnostics: continuous derivative envelope analytic expression,
    but sup/inf estimated on grid; must be replaced by interval majorants to certify.
    """
    xs=np.linspace(peak-radius,peak+radius,n)
    margins=[]
    for x in xs:
        w,dw,ddw=weights_derivatives(times,SIGMA,x)
        eta1=np.sum(np.abs(dw)*eps)
        eta2=np.sum(np.abs(ddw)*eps)
        curvature=ddw@C
        margins.append((float(eta1),float(eta2),float(curvature)))
    e1,e2,kappa=np.array(margins).T
    kappa_lo=float(np.min(-kappa));eta1=float(np.max(e1));eta2=float(np.max(e2))
    ld=float(weights_derivatives(times,SIGMA,peak-radius)[1]@C)
    rd=float(weights_derivatives(times,SIGMA,peak+radius)[1]@C)
    good=kappa_lo>eta2 and ld>eta1 and -rd>eta1
    return dict(radius=radius,eta1_grid=eta1,eta2_grid=eta2,kappa_grid=kappa_lo,
                left_current_minus_eta1=ld-eta1,right_current_minus_eta1=-rd-eta1,
                local_conditions_on_grid=bool(good),
                root_shift_grid_bound=eta1/(kappa_lo-eta2) if good else None)

def run():
    _,Q=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)));Q=Q[:,:M]
    WA,WB=wavefront_differences()
    output=[]
    for pair in [(1.2,.35),(.35,1.2),(.5,.2)]:
        print('Baseline',pair,flush=True)
        base=amplitudes_baseline(pair,Q)
        eps,ca,cb,res,sigma=source_envelope(base,WA,WB,Q)
        pka,da=peak_curves(ca,24,31)
        pkb,db=peak_curves(cb,24,31)
        if not pka or not pkb:raise ValueError('no peak')
        for p in pka:print('peak A',pair,p,flush=True)
        win=(times>=20)&(times<=34)
        d={"defect":pair,"n_samples":len(times),"dt":dt,
           "peak_airy":float(pka[0]),"peak_bessel":float(pkb[0]),
           "peak_shift_bessel_minus_airy":float(pkb[0]-pka[0]),
           "max_capacity_gap":float(np.max(abs(ca-cb))),
           "max_capacity_bound_all_samples":float(np.max(eps)),
           "max_capacity_bound_20_to_34":float(np.max(eps[win])),
           "max_occupation_amplitude_gap_20_to_34":float(np.max(res[win])),
           "min_singular_lower_all":float(np.min(sigma)),
           "bound_violations":int(np.sum(abs(ca-cb)>eps+1e-12)),
           "near_peak_diagnostics":[envelope_at(ca,eps,pka[0],r) for r in (.05,.1,.2,.4,.8)],
           "meaning":"Gaussian derivative bound exact for nodewise certified eps; here eps double-precision diagnostic only. Relative to Bessel surrogate with same baseline, NOT true defective-chain peak."}
        print(json.dumps(d,indent=2),flush=True)
        np.savez_compressed(ROOT/f'sampled_{pair[0]}_{pair[1]}.npz',t=times,airy=ca,bessel=cb,eps=eps,residual=res,sigma_lower=sigma)
        output.append(d)
    (ROOT/'RESULTS.json').write_text(json.dumps(output,indent=2))
    return output
if __name__=='__main__':run()
