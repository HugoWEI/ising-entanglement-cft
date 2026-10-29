"""Nonperturbative real-time two-site scattering via boundary image Bessel kernels.

NO diagonalization of the post-quench impurity Hamiltonian is used.
The initial staggered-mass eigenmodes are an input; exact final diagonalization
is used only by an independent benchmark.

The clean propagator for nearest-neighbor hopping -1 with open ends is
 U0[i,j](t) = sum_{r in Z} [i**abs(i-j+2r(L+1)) J_abs(...)(2t)
                            - i**abs(i+j+2+2r(L+1)) J_abs(...)(2t)].
The rank-two Volterra integral equation sums defect insertions to all orders.
Trapezoid product integration has an O(dt**2) discretization error.
"""
from __future__ import annotations
import numpy as np
from scipy.special import jv
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def image_bessel_kernel(L: int, times: np.ndarray, shells: int, rows=None, cols=None):
    if rows is None: rows=np.arange(L)
    if cols is None: cols=np.arange(L)
    i=np.asarray(rows, dtype=int)[:,None]
    j=np.asarray(cols,dtype=int)[None,:]
    t=np.asarray(times, dtype=float)
    out=np.zeros((len(t),len(i),len(cols)), dtype=complex)
    max_order=2*(shells+1)*(L+1)
    n_all=np.arange(max_order+1)
    # Tabulate each Bessel order once instead of recomputing for every site pair.
    basis=(1j**n_all)[:,None]*jv(n_all[:,None],2*t[None,:])
    for r in range(-shells,shells+1):
        n1=np.abs(i-j+2*r*(L+1)); n2=np.abs(i+j+2+2*r*(L+1))
        out+=basis[n1].transpose(2,0,1)
        out-=basis[n2].transpose(2,0,1)
    return out


def defect_response(L, times, u, v, initial_occupied, shells=1, return_full=False, return_defect=False):
    """Solve U_def(t) Q0 using only clean Bessel kernels and 2x2 Volterra convolution.

    n time samples, M occupied modes. Complex amps Y(t)= [U_def(t) Q0] at defect rows
       Y(t)= [U0(t)Q0]_B - i * [U0_BB * VY](t)
       X_A(t)= [U0(t)Q0]_A - i * [U0_AB * VY](t).
    Time integrals are solved by trapezoid-product integration including the implicit
    kernel at the current time; this approach resums all onsite defect interactions.
    """
    t=np.asarray(times,float)
    assert len(t)>1 and t[0]==0 and np.allclose(np.diff(t),t[1]-t[0])
    assert shells>=0
    dt=t[1]-t[0]
    B=np.array([L//2-1,L//2],dtype=int)
    A=np.arange(L) if return_full else np.arange(L//2)
    Q=np.asarray(initial_occupied,dtype=complex)
    assert Q.shape[0]==L
    KBB=image_bessel_kernel(L,t,shells,B,B)
    KBA=image_bessel_kernel(L,t,shells,B,np.arange(L))
    KAB=image_bessel_kernel(L,t,shells,A,B)
    KAA=image_bessel_kernel(L,t,shells,A,np.arange(L))
    baseB=KBA@Q
    baseA=KAA@Q
    V=np.diag([u,v])
    Y=np.zeros_like(baseB)
    Y[0]=baseB[0]
    solve=np.linalg.inv(np.eye(2,dtype=complex)+0.5j*dt*KBB[0]@V)
    # trapezoid: endpoint s=0 half, s=t half, interior sum.
    for n in range(1,len(t)):
        interior=np.einsum('kij,kjq->iq', KBB[1:n], (V@Y[1:n][::-1]), optimize=False) if n>1 else 0
        rhs=baseB[n]-1j*dt*(interior+0.5*KBB[n]@V@Y[0])
        Y[n]=solve@rhs
    VY=V@Y
    # Convolve across time only, sum over two defect channels and occupied modes.
    conv=np.zeros_like(baseA)
    for b in range(2):
        conv+=fftconvolve(KAB[:,:,b][:,:,None],VY[:,b,:][:,None,:],mode='full',axes=0)[:len(t)]
    conv-=0.5*KAB[:,:,0:1]*VY[0,0][None,None,:] + 0.5*KAB[:,:,1:2]*VY[0,1][None,None,:]
    conv-=0.5*np.einsum('ab,tbq->taq',KAB[0],VY)
    X=baseA-1j*dt*conv
    X[0]=Q[A,:]
    return (X,Y) if return_defect else X


def capacity_from_amplitudes(X, diagnostic=False):
    caps=[];maxocc=[];minocc=[]
    for B in X:
        eig=np.linalg.eigvalsh(B@B.conj().T)
        maxocc.append(float(np.max(eig)));minocc.append(float(np.min(eig)))
        if not diagnostic and (eig.max()>1+1e-3 or eig.min() < -1e-3):
            raise ValueError("Unphysical occupation eigenvalues: refine image shells / time step")
        pc=eig[(eig>1e-12)&(eig<1-1e-12)]
        cap=np.sum(pc*(1-pc)*np.log((1-pc)/pc)**2)
        caps.append(float(np.real(cap)))
    if diagnostic:return np.array(caps), np.array(maxocc),np.array(minocc)
    return np.array(caps)


def peak(t,y,L,sigma=2):
    smooth=gaussian_filter1d(y,sigma/(t[1]-t[0]),mode='reflect',truncate=4)
    inds,_=find_peaks(smooth)
    for k in inds:
        if k<t.size-1 and 0.4*L<=t[k]<1.65*L:
            rad=max(2,round(.15*L/(t[1]-t[0])))
            l=max(k-rad,0);r=min(k+rad+1,len(t))
            p=min(smooth[k]-np.min(smooth[l:k+1]),smooth[k]-np.min(smooth[k:r]))
            if p>=0.08:
                h=t[1]-t[0]
                den=smooth[k-1]-2*smooth[k]+smooth[k+1]
                off=0.5*(smooth[k-1]-smooth[k+1])/den if abs(den)>1e-15 else 0
                tp=t[k]+h*off
                hp=smooth[k]-.25*(smooth[k-1]-smooth[k+1])*off
                return float(tp),float(hp)
    return np.nan,np.nan

if __name__=='__main__':
    from phase_revival_research import initial_projector, exact_capacity_series
    for L in [24,48]:
        from phase_revival_research import hopping
        e,q=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)))
        Q=q[:,:L//2]
        times=np.arange(0,1.7*L,0.2)
        for potentials in [(1.2,.35),(.35,1.2)]:
            exact=exact_capacity_series(L,2.5,*potentials,times)
            for shells in [0,1,2]:
                X=defect_response(L,times,*potentials,Q,shells)
                approx,maxocc,minocc=capacity_from_amplitudes(X,True)
                print('L',L,'pot',potentials,'shells',shells,'MAE',np.mean(abs(approx-exact)),
                      'MaxErr',max(abs(approx-exact)),'eig max over 1',max(0,max(maxocc)-1),
                      'peak',peak(times,approx,L), 'exactpeak',peak(times,exact,L))
