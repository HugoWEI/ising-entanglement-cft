"""Exploratory transfer of pointwise wavefront bound to physical projector.

Inputs Z_base are computed from source-only direct + single-end scattering
Volterra scheme (second-order), *not* from final defective diagonalization.
The singular value and projector measurements use numpy floating point;
therefore this is not a formally certified projector bound.
"""
import sys,json
from pathlib import Path
import numpy as np
from scipy.special import jv
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
from path_ablation import ablated_kernel
import volterra_scattering_revival as vol
from phase_revival_research import hopping
from airy_hybrid import leading_olver_j
from run_finite_order_audit import comparison_one
from scipy.optimize import brentq
from scipy.integrate import simpson


def polar(X):
    u,_,v=np.linalg.svd(X,full_matrices=False)
    return u@v

def cap_f(p):
    p=np.asarray(p,float)
    inside=np.clip(p,1e-300,1-1e-15)
    return np.where((p<=0)|(p>=1),0.,inside*(1-inside)*np.log((1-inside)/inside)**2)

def local_modulus(p,d):
    critical=brentq(lambda x:(1-2*x)*np.log((1-x)/x)-2.,1e-6,.49)
    lo=np.maximum(0,p-d);hi=np.minimum(1,p+d)
    mid=cap_f(p); lowers=np.minimum(cap_f(lo),cap_f(hi));uppers=np.maximum(cap_f(lo),cap_f(hi))
    for z in (critical,.5,1-critical):
        mask=(lo<=z)&(z<=hi)
        lowers=np.where(mask,np.minimum(lowers,cap_f(z)),lowers)
        uppers=np.where(mask,np.maximum(uppers,cap_f(z)),uppers)
    return float(np.sum(np.maximum(mid-lowers,uppers-mid)))

def capacity_of_X(X):
    m=X.shape[1];p=np.linalg.eigvalsh(X[:m]@X[:m].conj().T)
    return float(np.sum(cap_f(p)))

def capacity_response(Za,Zb,num=81):
    """Numerical exact-path trace response to Bessel-vs-Airy drive.

    The integral is an identity for differentiable full-rank polar path;
    numeric quadrature and endpoint eigengaps remain unchecked.
    """
    Dz=Zb-Za;m=Za.shape[1]; L=Za.shape[0]
    vals=[]
    for lam in np.linspace(0,1,num):
        Z=Za+lam*Dz
        ev,v=np.linalg.eigh(Z.conj().T@Z)
        inv=(v*ev**(-.5))@v.conj().T
        X=Z@inv
        H=Dz@inv-X@(X.conj().T@Dz@inv)
        XA=X[:m,:]; HA=H[:m,:]
        p,U=np.linalg.eigh(XA@XA.conj().T)
        pc=np.clip(p,1e-13,1-1e-13)
        log=np.log((1-pc)/pc)
        fp=(1-2*pc)*log*log-2*log
        fpXA=U@(fp[:,None]*(U.conj().T@XA))
        vals.append(2*np.real(np.vdot(fpXA,HA)))
    lam=np.linspace(0,1,num)
    return {'integral':float(simpson(vals,x=lam)),'derivative_span':[float(min(vals)),float(max(vals))],
            'derivative_at_half':float(vals[(num-1)//2]),'num_nodes':num}

def one(pair,t_stop,dt=.1):
    L=48;m=L//2
    _,v=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)))
    Q=v[:,:m]
    times=np.arange(0,t_stop+dt/4,dt)
    t=times[-1]
    orig=vol.image_bessel_kernel
    try:
        vol.image_bessel_kernel=lambda L,times,shells,rows=None,cols=None: ablated_kernel(L,times,shells,rows,cols,('direct','left','right'))
        base=vol.defect_response(L,times,*pair,Q,1,return_full=True)[-1]
    finally:
        vol.image_bessel_kernel=orig
    i=np.arange(L)[:,None];j=np.arange(L)[None,:]
    N=2*(L+1);nplus=N+i-j;nminus=N-i+j
    Wb= (1j**nplus)*jv(nplus,2*t)+(1j**nminus)*jv(nminus,2*t)
    uniq=np.unique(np.r_[nplus.ravel(),nminus.ravel()]);tab=np.array([leading_olver_j(int(n),np.array([2*t]))[0] for n in uniq])
    Wa=(1j**nplus)*tab[np.searchsorted(uniq,nplus)]+(1j**nminus)*tab[np.searchsorted(uniq,nminus)]
    Zb=base+Wb@Q;Za=base+Wa@Q
    sigma_b=float(np.linalg.svd(Zb,compute_uv=False)[-1]);sigma_a=float(np.linalg.svd(Za,compute_uv=False)[-1])
    δ=float(np.linalg.svd((Zb-Za),compute_uv=False)[0])
    Xi=polar(Zb);Xa=polar(Za)
    actual=float(np.linalg.norm(Xi@Xi.conj().T-Xa@Xa.conj().T,2))
    intrinsic=comparison_one(L,str(round(t,4)))
    δupper=intrinsic['matrix_frobenius_upper']
    bound=4*δupper/(sigma_b+sigma_a)
    p=np.linalg.eigvalsh(Xa[:m]@Xa[:m].conj().T)
    capacity_upper=local_modulus(p,bound)
    actual_signed_capacity_change=capacity_of_X(Xi)-capacity_of_X(Xa)
    measured_cap_difference=abs(actual_signed_capacity_change)
    response=capacity_response(Za,Zb,81)
    response_fine=capacity_response(Za,Zb,161)
    return {'pair':pair,'L':L,'time':float(t),'sigma_bessel':sigma_b,'sigma_airy':sigma_a,
      'pointwise_wavefront_upper':δupper,'measured_occupied_amplitude_difference':δ,
      'measured_projector_difference':actual,'conditional_projector_upper':bound,
      'measured_capacity_difference':measured_cap_difference,'conditional_capacity_upper':capacity_upper,
      'signed_capacity_difference':actual_signed_capacity_change,'trace_response_integral':response['integral'],
      'trace_response_refinement_difference':abs(response['integral']-response_fine['integral']),
      'max_abs_trace_response_integrand':max(abs(x) for x in response['derivative_span']),
      'warning':'The singular values and Volterra baseline are evaluated in ordinary floating point; not a formal certificate.'}

if __name__=='__main__':
    cases=[((1.2,.35),26.4),((.35,1.2),27.3)]
    results=[one(pair,t) for pair,t in cases]
    for res in results:print(json.dumps(res),flush=True)
    (Path(__file__).parent/'projector_transfer.json').write_text(json.dumps(results,indent=2))
