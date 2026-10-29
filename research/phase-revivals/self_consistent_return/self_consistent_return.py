"""Direct insertion of first-return Airy wave into the two-impurity Volterra kernel.

Compared on a single Gaussian smoothing convention. No defective Hamiltonian
spectral decomposition enters prediction. Trapezoid solver is second order;
Hermite full-clean-source comparator is fourth-order.
"""
from __future__ import annotations
import sys, time, argparse, json
from pathlib import Path
import numpy as np
from scipy.special import jv
from scipy.signal import fftconvolve
from scipy.optimize import brentq

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
HOME=PHASE/'certificate' if (PHASE/'certificate').exists() else Path('/mnt/data/airy_certificate_work')
GEOM=PHASE/'capacity_geometry' if (PHASE/'capacity_geometry').exists() else Path('/mnt/data/gaussian_capacity_geometry_v1')
sys.path[:0]=[str(HOME),str(HOME/'src'),str(GEOM)]
from airy_hybrid import leading_olver_j
from phase_revival_helper import initial_occupied
from high_order_volterra import solve_hermite_volterra
from peak_certificate import weights_derivatives
from capacity_geometry import polar,capacity


def kernels(L,t,Q,mode='airy'):
    """Full-chain and two-site clean kernel from explicit images.

    base = direct + left reflection + right reflection;
    W = TWO leading wrap images (either Bessel or Olver-Air(y)).
    higher image: -J_{N+i+j+2}, tracked separately in 'full_bessel' mode.
    """
    i=np.arange(L)[:,None]; j=np.arange(L)[None,:];N=2*(L+1)
    # Tabulate all required orders once for entire grid.
    orders=[np.abs(i-j), i+j+2, np.abs(i+j+2-N),N+i-j,N-i+j,N+i+j+2]
    mx=max(np.max(n) for n in orders)
    basis=(1j**np.arange(mx+1))[:,None]*jv(np.arange(mx+1)[:,None],2*t[None,:])
    def val(n):return basis[n].transpose(2,0,1)
    base=val(orders[0])-val(orders[1])-val(orders[2])
    if mode=='base': clean=base
    elif mode in ('airy','bessel','full_bessel'):
        if mode=='airy':
            ns=np.arange(N-L+1,N+L)
            ta=np.asarray([leading_olver_j(int(n),2*t) for n in ns])
            phases=(1j**ns)[:,None]
            air=phases*ta
            wrap=air[orders[3]-ns[0]].transpose(2,0,1)+air[orders[4]-ns[0]].transpose(2,0,1)
        else:wrap=val(orders[3])+val(orders[4])
        clean=base+wrap
        if mode=='full_bessel':clean=clean-val(orders[5])
    else:raise ValueError(mode)
    D=np.array([L//2-1,L//2]);KDD=clean[:,D][:,:,D]
    KDA=clean[:,D];KAD=clean[:,:,D]
    baseF=clean@Q
    return dict(KDD=KDD,KDA=KDA,KAD=KAD,baseF=baseF,clean=clean)


def trapezoid_solve(ker,t,Q,pair):
    dt=t[1]-t[0];L=Q.shape[0];D=np.array([L//2-1,L//2]);V=np.diag(pair)
    KDD=ker['KDD'];KAD=ker['KAD'];F0=ker['baseF']; Y=np.zeros((len(t),2,Q.shape[1]),complex)
    Y[0]=Q[D]
    inv=np.linalg.inv(np.eye(2)+.5j*dt*KDD[0]@V)
    # All impurity insertions self-consistently summed. The wrap enters both source and kernel.
    for k in range(1,len(t)):
        interior=np.einsum('sij,sjq->iq',KDD[1:k],V@Y[1:k][::-1],optimize=False) if k>1 else 0
        rhs=F0[k,D]-1j*dt*(interior+.5*KDD[k]@V@Y[0]);Y[k]=inv@rhs
    VY=V@Y
    correction=np.zeros_like(F0)
    for d in range(2):
        correction+=fftconvolve(KAD[:,:,d][:,:,None],VY[:,d,:][:,None,:],axes=0,mode='full')[:len(t)]
        correction-=.5*KAD[:,:,d,None]*VY[0,d][None,None,:]
    correction-=.5*np.einsum('ab,tbq->taq',KAD[0],VY)
    F=F0-1j*dt*correction
    F[0]=Q
    return F,Y


def smoothed_peak(t,C,lo=24,hi=30):
    def d(x):return float(weights_derivatives(t,2.,x)[1]@C)
    xs=np.linspace(lo,hi,80)
    roots=[brentq(d,a,b) for a,b in zip(xs[:-1],xs[1:]) if d(a)>0 and d(b)<0]
    return float(roots[0]) if roots else float('nan')


def cap_curve(F):
    return np.array([capacity(polar(z)) for z in F])


def run(dt=.1,T=36.,pairs=((1.2,.35),(.35,1.2),(.5,.2)),L=48,save=True):
    t=np.arange(0,T+dt/4,dt);Q=initial_occupied(L)
    started=time.perf_counter()
    ks={key:kernels(L,t,Q,key) for key in ('base','airy','bessel','full_bessel')}
    print('Kernels constructed in %.2fs'%(time.perf_counter()-started),flush=True)
    allresults=[]
    for pair in pairs:
        start=time.perf_counter()
        out={};states={}
        for method,ker in ks.items():
            F,Y=trapezoid_solve(ker,t,Q,pair)
            c=cap_curve(F);states[method]=(F,Y,c)
            out[method]={'peak':smoothed_peak(t,c),'max_orthog_error':float(np.max([np.linalg.norm(polar(z).conj().T@polar(z)-np.eye(L//2)) for z in F[::max(1,len(t)//10)]]))}
        # Previous posthoc construction: base Volterra response + W Q, not selfconsistent.
        Fbase=states['base'][0]
        EA=ks['airy']['baseF']-ks['base']['baseF']
        Cpost=cap_curve(Fbase+EA)
        out['posthoc_airy']={'peak':smoothed_peak(t,Cpost)}
        Fh,Yh,DYh=solve_hermite_volterra(L,t,Q,pair)
        Ch=cap_curve(Fh)
        out['hermite_clean']={'peak':smoothed_peak(t,Ch)}
        p=out['hermite_clean']['peak']
        for key in ('posthoc_airy','airy','bessel','full_bessel','base'):
            out[key]['peak_error_to_hermite']=out[key]['peak']-p
        out['consistent_airy_vs_bessel_peak_difference']=out['airy']['peak']-out['bessel']['peak']
        out['feedback_peak_effect']=out['airy']['peak']-out['posthoc_airy']['peak']
        out['bessel_shell_vs_hermite_peak_difference']=out['full_bessel']['peak']-p
        out['peak_window_max_capacity_errors']={key:float(np.max(np.abs(states[key][2]-Ch))) for key in ks}
        out['defect']=pair;out['L']=L;out['dt']=dt;out['T']=T
        out['runtime_s']=time.perf_counter()-start
        print(json.dumps(out,indent=2),flush=True)
        allresults.append(out)
        if save:np.savez_compressed(HERE/f'curves_{L}_{pair[0]}_{pair[1]}_dt{dt}.npz',t=t,hermite=Ch,posthoc_airy=Cpost,**{k:states[k][2] for k in ks})
    if save:(HERE/f'RESULTS_dt{dt}.json').write_text(json.dumps(allresults,indent=2))
    return allresults

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,default=.1);p.add_argument('--T',type=float,default=36.);p.add_argument('--L',type=int,default=48)
    a=p.parse_args();run(dt=a.dt,T=a.T,L=a.L)
