"""Physicalize a time-dependent coherent path / Volterra approximate occupied-state map.

Requires full-chain amplitudes Z(t) for every occupied state. Polar isometry
Qhat(t)= Z(t)(Z(t)^dagger Z(t))^(-1/2) ensures Qhat^dagger Qhat=I.
It is an isometric repair, *not* the exact Hamiltonian propagation. Its
capacity current obeys dC/dt= J_h + 2 Re Tr[X_A^dagger f'(G_A) R_A]
with residual R=dQhat/dt + i*h*Qhat.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import volterra_scattering_revival as vol
ORIGINAL_KERNEL = vol.image_bessel_kernel
from path_ablation import ablated_kernel
from phase_revival_research import hopping, hfinal


def polar_modes(Z):
    """Canonical Frobenius-nearest column isometry for full-column-rank matrix Z."""
    M=(Z.conj().T @ Z)
    d,U=np.linalg.eigh((M+M.conj().T)/2)
    if d.min()<=0:raise ValueError('rank deficient approximate occupied modes')
    return (Z@U)*(d**(-.5))[None,:]@U.conj().T


def f_cap(p):
    p=np.asarray(p)
    if np.min(p)<-1e-8 or np.max(p)>1+1e-8:
        raise ValueError(f'unphysical occupations [{np.min(p)}, {np.max(p)}]')
    x=np.clip(p,0,1)
    y=np.zeros_like(x)
    sel=(x>0)&(x<1)
    y[sel]=x[sel]*(1-x[sel])*np.log((1-x[sel])/x[sel])**2
    return y


def f_prime(p, regularization=1e-11):
    x=np.clip(p,regularization,1-regularization)
    log=np.log((1-x)/x)
    return (1-2*x)*log**2-2*log


def point_cap_and_current(X,h):
    NA=len(X)
    N=X.shape[1]
    GA=X[:NA//2]@X[:NA//2].conj().T
    # X is full length L; M occupied columns; here L=2*M.
    p,V=np.linalg.eigh(GA)
    F=(V*f_prime(p))@V.conj().T
    GBA=X[NA//2:]@X[:NA//2].conj().T
    hAB=h[:NA//2,NA//2:]
    j=2*np.imag(np.trace(F@hAB@GBA))
    return float(np.sum(f_cap(p))),float(j),float(p.min()),float(p.max())


def amplitudes(L,t,pair,init,mode):
    if mode=='shell1':
        vol.image_bessel_kernel=ORIGINAL_KERNEL
        return vol.defect_response(L,t,*pair,init,shells=1,return_full=True)
    if mode=='first_echo_bessel':
        vol.image_bessel_kernel=lambda length,tt,shells,rows=None,cols=None: ablated_kernel(length,tt,shells,rows,cols,('direct','left','right','wrap_plus'))
        return vol.defect_response(L,t,*pair,init,shells=1,return_full=True)
    if mode=='base':
        vol.image_bessel_kernel=lambda length,tt,shells,rows=None,cols=None: ablated_kernel(length,tt,shells,rows,cols,('direct','left','right'))
        return vol.defect_response(L,t,*pair,init,shells=1,return_full=True)
    raise ValueError(mode)


def benchmark(L=48, dt=0.1, horizon=.75, pairs=((1.2,.35),(.35,1.2)),modes=('base','first_echo_bessel','shell1')):
    t=np.arange(0,L*horizon+1e-8,dt)
    _,U=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)))
    Q=U[:,:L//2]
    record=[]; curves={}
    for pair in pairs:
        hf=hfinal(L,*pair)
        E,V=np.linalg.eigh(hf); T=V.conj().T@Q
        Exact=np.einsum('ik,tkq->tiq',V,np.exp(-1j*t[:,None]*E[None,:])[:,:,None]*T[None,:,:],optimize=True)
        ce=np.array([point_cap_and_current(x,hf)[0] for x in Exact])
        p_exact=vol.peak(t,ce,L)
        for mode in modes:
            raw=amplitudes(L,t,pair,Q,mode)
            # use batched polar normalization
            iso=np.stack([polar_modes(z) for z in raw])
            ortho=np.max(np.linalg.norm(np.einsum('tij,tik->tjk',iso.conj(),iso)-np.eye(L//2),axis=(1,2)))
            defects=np.linalg.norm(np.einsum('tij,tik->tjk',raw.conj(),raw)-np.eye(L//2),axis=(1,2))
            rawmax=np.max([np.linalg.eigvalsh(x[:L//2]@x[:L//2].conj().T)[-1] for x in raw])
            cp=[]; curr=[]; lo=[];hi=[]
            for x in iso:
                c,j,mn,mx=point_cap_and_current(x,hf)
                cp.append(c);curr.append(j);lo.append(mn);hi.append(mx)
            cp=np.array(cp);curr=np.array(curr)
            # residual computed via centered finite differences away from boundaries
            residual=np.gradient(iso,dt,axis=0,edge_order=2)+1j*np.einsum('ij,tjq->tiq',hf,iso)
            residual_norm=np.linalg.norm(residual,axis=(1,2))
            correction=[]
            for x,r in zip(iso,residual):
                p,Vp=np.linalg.eigh(x[:L//2]@x[:L//2].conj().T)
                F=(Vp*f_prime(p))@Vp.conj().T
                correction.append(2*np.real(np.trace(x[:L//2].conj().T@F@r[:L//2])))
            correction=np.array(correction)
            drift=np.gradient(cp,dt,edge_order=2)
            mask=(t>=5)&(t<=L*.68)
            identity_error=np.max(np.abs(drift[mask] -curr[mask]- correction[mask]))
            first=vol.peak(t,cp,L)
            row=dict(L=L,dt=dt,pair=pair,mode=mode,ortho_error=float(ortho), raw_gram_max=float(np.max(defects)),raw_p_max=float(rawmax),iso_p_min=float(min(lo)),iso_p_max=float(max(hi)),
                     exact_peak_t=p_exact[0],predict_peak_t=first[0],peak_error=first[0]-p_exact[0],height_error=first[1]-p_exact[1],
                     cap_mae=float(np.mean(abs(cp-ce))),capacity_max_err=float(np.max(abs(cp-ce))),
                     residual_L2_median=float(np.median(residual_norm[mask])),residual_L2_max=float(np.max(residual_norm[mask])),
                     current_corr_med=float(np.median(abs(correction[mask]))),current_median=float(np.median(abs(curr[mask]))),
                     current_identity_discretization_error=float(identity_error))
            record.append(row)
            curves[f'{pair[0]}_{pair[1]}_{mode}']=dict(t=t,capacity=cp,reference=ce,current=curr,residual_correction=correction,drift=drift,gram=defects)
            print(row,flush=True)
    return record,curves

if __name__=='__main__':
    import sys,json,os
    L=int(sys.argv[1]) if len(sys.argv)>1 else 48
    dt=float(sys.argv[2]) if len(sys.argv)>2 else .1
    data,curves=benchmark(L,dt)
    base=os.path.dirname(__file__)
    with open(os.path.join(base,f'physical_echo_L{L}_dt{dt}.json'),'w') as f:json.dump(data,f,indent=2)
    np.savez_compressed(os.path.join(base,f'physical_echo_L{L}_dt{dt}.npz'),**{k+'_'+field:val for k,record in curves.items() for field,val in record.items()})
