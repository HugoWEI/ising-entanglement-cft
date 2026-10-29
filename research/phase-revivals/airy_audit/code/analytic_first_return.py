"""Analytical Airy reflected-image modes and a *physical* polar Gaussian state.

This uses the first two boundary-wrap images of the clean propagator only,
not the full defective propagator, to define a low-rank perturbation.
The baseline Dyson--Volterra solution is used for direct and single-end reflections.
There is NO fit to the exact capacity curve and NO final-defect diagonalization in prediction.
"""
import os,sys,time, json
import numpy as np
from scipy.special import jv
from scipy.linalg import svd
# Dependencies are packaged in this directory; run from any working directory.
import volterra_scattering_revival as vol
from path_ablation import ablated_kernel
from airy_hybrid import leading_olver_j
from phase_revival_research import hopping, hfinal
from scattering_generator import polar_decomp, capacity

ORIGINAL=vol.image_bessel_kernel

def base_kernel(L,times,shells,rows=None,cols=None):
    return ablated_kernel(L,times,shells,rows,cols,('direct','left','right'))

def wrap_matrix(L,t,method='uniform_airy',part='both'):
    t=np.asarray(t,float)
    i=np.arange(L)[:,None]; j=np.arange(L)[None,:]
    N=2*(L+1)
    orders=[]
    if part in ('both','plus'): orders.append(i-j+N)
    if part in ('both','minus'): orders.append(j-i+N)
    # All Bessel orders positive for site indices.
    ns=np.unique(np.stack(orders,axis=0))
    if method=='uniform_airy':
        tab=np.array([leading_olver_j(int(n),2*t) for n in ns])
    elif method=='turning_airy':
        from scipy.special import airy
        tab=np.array([(2/n)**(1/3)*airy(-2**(1/3)*(2*t-n)/n**(1/3))[0] for n in ns])
    elif method=='bessel':
        tab=jv(ns[:,None],2*t[None,:])
    else:raise ValueError(method)
    out=np.zeros((len(t),L,L),complex)
    for n in orders:
        ids=np.searchsorted(ns,n)
        out+=(1j**n)[None,:,:]*tab[ids].transpose(2,0,1)
    return out

def rank_approx(mats, rank=None, tol=None):
    """Truncate the *analytically constructed* wavefront, no full-Hamiltonian information."""
    results=np.empty_like(mats)
    diagnostics=[]
    for ix,m in enumerate(mats):
        u,s,vh=np.linalg.svd(m,full_matrices=False)
        if tol is not None:
            total=np.linalg.norm(s)
            r=next((r for r in range(0,len(s)+1) if np.linalg.norm(s[r:]) <= tol*total),len(s)) if total>1e-13 else 0
        else: r=len(s) if rank is None else min(rank,len(s))
        results[ix]=(u[:,:r]*s[:r])@vh[:r,:]
        diagnostics.append((r,float(s[r] if r<len(s) else 0.),float(np.linalg.norm(s[r:])),float(np.linalg.norm(s))))
    return results,np.array(diagnostics)

def run(L=48,dt=.1,end=None,pair=(1.2,.35),methods=('bessel','uniform_airy','turning_airy'),ranks=(None,4,6)):
    if end is None:end=.75*L
    t=np.arange(0,end+dt/4,dt)
    m=L//2
    eval0,Q=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)));Q=Q[:,:m]
    vol.image_bessel_kernel=base_kernel
    tic=time.time()
    base=vol.defect_response(L,t,*pair,Q,1,return_full=True)
    vol.image_bessel_kernel=ORIGINAL
    print('baseline built secs',round(time.time()-tic,2),flush=True)
    e,v=np.linalg.eigh(hfinal(L,*pair));vec=v.T@Q
    exact=np.array([v@(np.exp(-1j*e*tt)[:,None]*vec) for tt in t])
    ref=np.array([capacity(x) for x in exact]); p_exact=vol.peak(t,ref,L)
    data=[];curves={};cross=[]
    base_caps=np.array([capacity(polar_decomp(z)[0]) for z in base]);base_pk=vol.peak(t,base_caps,L)
    for method in methods:
        tic=time.time()
        W=wrap_matrix(L,t,method)
        # Analytic paths couple to initially occupied states; Volterra feedback from the paths
        # is intentionally omitted and tested against exact evolution.
        E=W@Q
        del W
        print(method,'WQ built secs',round(time.time()-tic,2),flush=True)
        for r in ranks:
            if r is None:
                A=E;diag=None
            else:
                A,diag=rank_approx(E,rank=r)
            amplitudes=base+A
            amps=[];unit=[];norm=[]
            for z in amplitudes:
                x,_=polar_decomp(z)
                amps.append(capacity(x))
                norm.append(np.linalg.svd(z,compute_uv=False)[-1])
                unit.append(np.linalg.norm(x.conj().T@x-np.eye(m)))
            c=np.asarray(amps)
            p=vol.peak(t,c,L)
            # low-rank residual measurement in peak region using analytic E alone
            row=dict(L=L,dt=dt,pair=pair,method=method,rank=r,exact_t=p_exact[0],exact_h=p_exact[1],baseline_t=base_pk[0],
                     peak_t=p[0],peak_h=p[1],time_error=p[0]-p_exact[0],height_error=p[1]-p_exact[1],
                     capacity_mae=float(np.mean(abs(c-ref))),peakwindow_mae=float(np.mean(abs(c[(t>.4*L)&(t<.65*L)]-ref[(t>.4*L)&(t<.65*L)]))),
                     min_sigma=float(min(norm)),max_orthonormal_error=float(max(unit)))
            if diag is not None:
                mask=(t>.48*L)&(t<.62*L)
                row.update(rank_tail_max_operator=float(np.max(diag[mask,1])),rank_tail_max_fro=float(np.max(diag[mask,2])),rank_fraction_energy=float(1-np.sum(diag[mask,2]**2)/np.sum(diag[mask,3]**2)))
            print(json.dumps(row),flush=True);data.append(row)
            curves[f'{method}_rank_{r}']=c
        del E
    return dict(t=t,ref=ref,base=base_caps,curves=curves,rows=data)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--L',type=int,default=48);ap.add_argument('--dt',type=float,default=.1);ap.add_argument('--u',type=float,default=1.2);ap.add_argument('--v',type=float,default=.35);ap.add_argument('--methods',nargs='+',default=['bessel','uniform_airy','turning_airy']);ap.add_argument('--ranks',nargs='+',default=['full','4','6']); a=ap.parse_args()
    results=run(a.L,a.dt,pair=(a.u,a.v),methods=a.methods,ranks=tuple(None if v=='full' else int(v) for v in a.ranks))
    label=f'L{a.L}_u{a.u}_v{a.v}_dt{a.dt}'
    root=os.getcwd()
    with open(os.path.join(root,label+'.json'),'w') as f:json.dump(results['rows'],f,indent=2)
    np.savez_compressed(os.path.join(root,label+'.npz'),t=results['t'],exact=results['ref'],base=results['base'],**results['curves'])
