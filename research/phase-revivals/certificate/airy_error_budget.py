"""Source-only Airy and Bessel-path component diagnostics at first return.

The Airy-Bessel component difference is calculated without the final defective
Hamiltonian. Does not yet provide a uniformly certified remainder or peak bound.
"""
from __future__ import annotations
import sys, json, time, os
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'src'))
from phase_revival_research import hopping
import analytic_first_return as ar
import volterra_scattering_revival as vol
from scattering_generator import polar_decomp


def run(L=48,dt=.1,T=30,pair=(1.2,.35)):
    t=np.arange(0,T+dt/4,dt)
    _,q=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)))
    Q=q[:,:L//2]
    orig=vol.image_bessel_kernel
    vol.image_bessel_kernel=ar.base_kernel
    try:base,Yb=vol.defect_response(L,t,*pair,Q,1,return_full=True,return_defect=True)
    finally:vol.image_bessel_kernel=orig
    wB=ar.wrap_matrix(L,t,'bessel')@Q
    wA=ar.wrap_matrix(L,t,'uniform_airy')@Q
    # both complete wavefronts; no rank compression
    DB=base+wB; DA=base+wA
    Am=np.array([np.linalg.norm(a-b,2) for a,b in zip(wB,wA)])
    smin=np.array([min(np.linalg.svd(x,compute_uv=False)[-1] for x in (a,b)) for a,b in zip(DB,DA)])
    projs=[]
    for a,b in zip(DB,DA):
        pa=polar_decomp(a)[0];pb=polar_decomp(b)[0]
        projs.append(np.linalg.norm(pa@pa.conj().T-pb@pb.conj().T,2))
    mask=(t>=.4*L)&(t<=.65*L)
    return dict(L=L,pair=pair,dt=dt,T=T,
                max_airy_amplitude_difference=float(Am[mask].max()),
                max_airy_projector_difference=float(np.max(np.array(projs)[mask])),
                min_singular_value=float(smin[mask].min()),
                component_bound=float(2*Am[mask].max()/smin[mask].min()),
                global_amplitude_difference=float(Am.max()),
                global_projector_difference=float(max(projs))),t,Am,np.array(projs)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--dt',type=float,default=.1);a.add_argument('--L',type=int,default=48);a.add_argument('--T',type=float,default=30.);a.add_argument('--u',type=float,default=1.2);a.add_argument('--v',type=float,default=.35);p=a.parse_args()
    out,*_=run(p.L,p.dt,p.T,(p.u,p.v))
    print(json.dumps(out,indent=2))
