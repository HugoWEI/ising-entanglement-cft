"""Physics-sourced Grassmannian generator for adding a coherent scattering path family.

Z(eps,t) = Z0(t) + eps*(Z1(t)-Z0(t)), where Z0/Z1 are
computed with Bessel-image/Dyson-Volterra kernels, independent of exact Hf eigenvectors.
This is an auxiliary homotopy among path approximants, NOT changing physical potential.
Polar factor X(eps)=Z(eps)(Z(eps)^dagger Z(eps))^-1/2.
The infinitesimal *projector* evolution has generator K=AX^dagger-XA^dagger
A=(I-XX^dagger)(dZ/deps)(Z^dagger Z)^-1/2.
"""
import numpy as np
from scipy.linalg import eigh
import volterra_scattering_revival as vol
from phase_revival_research import hopping,hfinal
from path_ablation import ablated_kernel
from physical_echo import f_cap

ORIGINAL=vol.image_bessel_kernel


def polar_decomp(z):
    g=z.conj().T@z
    d,v=eigh(g)
    if d.min()<1e-13: raise ValueError('rank deficiency')
    hinv=(v*d**(-.5))@v.conj().T
    x=z@hinv
    return x,hinv


def tangent_generator(z, dz, svd_cut=0):
    """Return exact horizontal generator at z (rank compression is optional)."""
    x,hinv=polar_decomp(z)
    a=dz@hinv
    a-=x@(x.conj().T@a)
    u,s,vh=np.linalg.svd(a,full_matrices=False)
    if svd_cut:
        r=np.sum(s>svd_cut*s[0])
    else:r=len(s)
    return x,u[:,:r],s[:r],vh[:r],a


def apply_rotation(y,x,u,s,vh,step):
    """Apply exp(step K) to any matrix y, with K=a x† - x a† and a=u diag(s) vh."""
    if not len(s):return y.copy()
    w=x@vh.conj().T
    ca=w.conj().T@y
    cb=u.conj().T@y
    c=np.cos(step*s)-1
    si=np.sin(step*s)
    return y+w@((c[:,None]*ca)-(si[:,None]*cb))+u@((si[:,None]*ca)+(c[:,None]*cb))


def projector(x):return x@x.conj().T


def projected_derivative(x,a):return a@x.conj().T+x@a.conj().T


def capacity(x):
    m=x.shape[1]
    return float(np.sum(f_cap(np.linalg.eigvalsh(x[:m]@x[:m].conj().T))))


def transport(z0,dz,nsteps=4,svd_cut=0):
    """Midpoint Lie-integrator for the projector path; no defective Hf diagonalization."""
    y,_=polar_decomp(z0)
    for i in range(nsteps):
        e=(i+.5)/nsteps
        x,u,s,vh,a=tangent_generator(z0+e*dz,dz,svd_cut)
        y=apply_rotation(y,x,u,s,vh,1/nsteps)
    return y


def construct_paths(L,t,pair,Q):
    vol.image_bessel_kernel=lambda length, tt, shells, rows=None, cols=None: ablated_kernel(length,tt,shells,rows,cols,('direct','left','right'))
    z0=vol.defect_response(L,t,*pair,Q,1,return_full=True)
    vol.image_bessel_kernel=ORIGINAL
    z1=vol.defect_response(L,t,*pair,Q,1,return_full=True)
    return z0,z1


def run(L=48,dt=.2,horizon=.75,pair=(1.2,.35),steps=(1,2,4,8),compute_curves=True):
    t=np.arange(0,L*horizon+dt/4,dt)
    _,v=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)))
    Q=v[:,:L//2]
    z0,z1=construct_paths(L,t,pair,Q)
    dz=z1-z0
    idx=round(.55*L/dt)
    Z=z0[idx]+.25*dz[idx]
    x,u,s,vh,a=tangent_generator(Z,dz[idx])
    eps=1e-5
    pplus=projector(polar_decomp(Z+eps*dz[idx])[0])
    pminus=projector(polar_decomp(Z-eps*dz[idx])[0])
    dpt=(pplus-pminus)/(2*eps)
    dpa=projected_derivative(x,a)
    diff=np.linalg.norm(dpa-dpt)/max(1,np.linalg.norm(dpt))
    vs=[]
    for n in steps:
        yy=transport(z0[idx],dz[idx],n)
        target=polar_decomp(z1[idx])[0]
        vs.append({'nsteps':n,'projector_error':float(np.linalg.norm(projector(yy)-projector(target))),
                   'orthonormal_error':float(np.linalg.norm(yy.conj().T@yy-np.eye(L//2))),
                   'capacity_error':float(capacity(yy)-capacity(target))})
    data={'L':L,'dt':dt,'pair':pair,'representative_t':float(t[idx]),
          'tangent_derivative_error':float(diff),'singular_values':s.tolist(),
          'path_delta_frobenius':float(np.linalg.norm(dz[idx])),
          'base_gram_error':float(np.linalg.norm(z0[idx].conj().T@z0[idx]-np.eye(L//2))),
          'target_gram_error':float(np.linalg.norm(z1[idx].conj().T@z1[idx]-np.eye(L//2))),
          'transport_checks':vs}
    if compute_curves:
        from scipy.signal import find_peaks
        c0=[];c1=[];trans={n:[] for n in steps}
        for a,b in zip(z0,z1):
            x0,_=polar_decomp(a);x1,_=polar_decomp(b)
            c0.append(capacity(x0));c1.append(capacity(x1))
            for n in steps:trans[n].append(capacity(transport(a,b-a,n)))
        p0=vol.peak(t,np.array(c0),L)
        p1=vol.peak(t,np.array(c1),L)
        data['base_peak']=p0;data['target_peak']=p1
        data['peak_approx']={n:vol.peak(t,np.array(trans[n]),L) for n in steps}
        data['capacity_mae']={n:float(np.mean(abs(np.array(trans[n])-c1))) for n in steps}
        data['curves']={'time':t.tolist(),'base':c0,'one_shell':c1,**{f'steps_{n}':v for n,v in trans.items()}}
    return data

if __name__=='__main__':
    import argparse,json
    ap=argparse.ArgumentParser()
    ap.add_argument('--length',type=int,default=48)
    ap.add_argument('--dt',type=float,default=.2)
    ap.add_argument('--u',type=float,default=1.2)
    ap.add_argument('--v',type=float,default=.35)
    ap.add_argument('--out',default='physical_echo_generator_results.json')
    a=ap.parse_args()
    result=run(a.length,a.dt,pair=(a.u,a.v))
    with open(a.out,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['curves','singular_values']},indent=2))
