"""State-dependent spectral interval capacity error, no defective H benchmark.

For a known operator-norm bound delta between physical restricted correlation
matrices, apply Weyl's inequality to every approximate eigenvalue. The exact
capacity value is enclosed by extrema of f on [p-delta,p+delta]∩[0,1].
Floating-point here is diagnostic, not directed-rounding certificate.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from high_order_volterra import solve_hermite_volterra,cubic_drive_error_majorant
from phase_revival_helper import initial_occupied
from peak_certificate import scalar_modulus_safe

pstar=brentq(lambda p:(1-2*p)*np.log((1-p)/p)-2.,1e-6,.49)

def f(p):
    p=np.asarray(p,dtype=float)
    pc=np.clip(p,1e-300,1-1e-15)
    ans=pc*(1-pc)*np.log((1-pc)/pc)**2
    return np.where((p<=0)|(p>=1),0.,ans)

def local_capacity_modulus(p,delta):
    p=np.clip(np.asarray(p),0.,1.)
    lo=np.maximum(0.,p-delta); hi=np.minimum(1.,p+delta)
    fv=f(p)
    lowers=np.minimum(f(lo),f(hi));uppers=np.maximum(f(lo),f(hi))
    for critical in [pstar,.5,1-pstar]:
        fc=float(f(critical));mask=(lo<=critical)&(critical<=hi)
        lowers=np.where(mask,np.minimum(lowers,fc),lowers)
        uppers=np.where(mask,np.maximum(uppers,fc),uppers)
    return float(np.sum(np.maximum(fv-lowers,uppers-fv)))

def calc(L=48,dt=.05,T=30.,pair=(1.2,.35)):
    t=np.arange(0,T+dt/4,dt)
    F,Y,DY=solve_hermite_volterra(L,t,initial_occupied(L),pair)
    E=cubic_drive_error_majorant(L,t,initial_occupied(L),pair,F,Y,DY)
    delta=4*E
    result=[]
    for k,tt in enumerate(t):
        x=F[k];ev,v=np.linalg.eigh(x.conj().T@x);x=x@((v*ev**(-.5))@v.conj().T)
        p=np.linalg.eigvalsh(x[:L//2]@x[:L//2].conj().T)
        result.append(local_capacity_modulus(p,delta))
    global_mod=(L//2)*scalar_modulus_safe(delta)
    return dict(L=L,dt=dt,T=T,pair=pair,E=float(E),delta=float(delta),global_bound=float(global_mod),
                local_max=float(np.max(result)),local_mean=float(np.mean(result)),
                local_peakwindow_max=float(np.max(np.asarray(result)[(t>24)&(t<29)])),
                local_min=float(np.min(result)),improvement_global_over_local=float(global_mod/np.max(result)),
                pstar=float(pstar))
if __name__=='__main__':
 import json,sys
 for pair in [(1.2,.35),(.35,1.2)]:
  for dt in [.10,.05,.025]:
   print(json.dumps(calc(dt=dt,pair=pair)),flush=True)
