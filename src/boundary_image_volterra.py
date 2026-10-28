"""Two-site Dyson/Volterra integral equation from boundary image paths.

THE EXACT IDENTITY (before quadrature):
 F(t) = U0(t)F(0) - i integral_0^t U0(t-s)[:,D] V_D Y(s) ds,
 Y(t) = F(t)[D,:].

Taking defect-site rows gives a CLOSED 2xM Volterra equation:
 Y(t)=Y0(t)-i integral_0^t K(t-s)V_D Y(s) ds,
 K(t)=U0(t)[D,D].

We use trapezoidal time quadrature and a truncated coherent image sum
for U0(t). This is an exploratory numerical scattering-path model,
not yet a closed-form or efficient new theory of modular-capacity peaks.

IMPORTANT PHYSICALITY CAVEAT: finite image and time quadrature errors
can break the occupied-orbital isometry. When 'physicality=polar', we
explicitly replace F(t) by its nearest column-isometric matrix in
Frobenius norm, F(F†F)^(-1/2). This makes occupations physical but
introduces a documented correction, NOT an original physical law.
For 'physicality=none', observations with occupations outside [0,1]
by >1e-7 are returned as NaN instead of silently clipping them.
"""
from __future__ import annotations
import numpy as np
from boundary_images import finite_clean_image_propagator
from scattering_phase import entropy_capacity


def volterra_trajectory(length, onsite, occupied_modes, step=0.2,
                        final_time=12.0, image_order=2,
                        fft_points=4096, observe_every=1,
                        physicality="polar"):
    n=int(length)
    if n < 4 or n%2 or step<=0 or final_time<0 or observe_every<1:
        raise ValueError("invalid length or time parameters")
    if physicality not in ("polar","none"):
        raise ValueError("physicality must be 'polar' or 'none'")
    v=np.diag(np.asarray(onsite,dtype=float))
    if v.shape!=(2,2):
        raise ValueError("onsite must have two potentials")
    occupied=np.asarray(occupied_modes,dtype=complex)
    if occupied.shape!=(n,n//2):
        raise ValueError("occupied modes shape mismatch")
    if not np.allclose(occupied.conj().T@occupied,np.eye(n//2),atol=1e-8):
        raise ValueError("occupied modes must be orthonormal")
    nsteps=int(round(final_time/step))
    times=np.arange(nsteps+1)*step
    defect=np.array([n//2-1,n//2],dtype=int)
    nfilled=n//2

    kernels_d=np.empty((nsteps+1,2,2),complex)
    kernels_x=np.empty((nsteps+1,n,2),complex)
    sources_d=np.empty((nsteps+1,2,nfilled),complex)
    sources_x=np.empty((nsteps+1,n,nfilled),complex)
    for j,time in enumerate(times):
        clean=finite_clean_image_propagator(
            n,float(time),image_order,fft_points)
        kernels_d[j]=clean[np.ix_(defect,defect)]
        kernels_x[j]=clean[:,defect]
        sources_d[j]=clean[defect,:]@occupied
        sources_x[j]=clean@occupied

    y=np.empty((nsteps+1,2,nfilled),complex)
    y[0]=occupied[defect,:]
    vy=np.empty_like(y)
    vy[0]=v@y[0]
    implicit=np.linalg.inv(np.eye(2)+.5j*step*kernels_d[0]@v)
    for j in range(1,nsteps+1):
        integral=.5*kernels_d[j]@vy[0]
        if j>1:
            integral+=np.einsum('mab,mbp->ap',
                                kernels_d[1:j][::-1],vy[1:j],
                                optimize=True)
        y[j]=implicit@(sources_d[j]-1j*step*integral)
        vy[j]=v@y[j]

    out_times=[];observables=[];eigen_bounds=[];raw_isometry_errors=[]
    for j in range(0,nsteps+1,observe_every):
        if j:
            integral=.5*kernels_x[j]@vy[0]+.5*kernels_x[0]@vy[j]
            if j>1:
                integral+=np.einsum('mab,mbp->ap',
                                    kernels_x[1:j][::-1],vy[1:j],
                                    optimize=True)
            f=sources_x[j]-1j*step*integral
        else:
            f=sources_x[0].copy()
        gram=f.conj().T@f
        raw_error=float(np.linalg.norm(gram-np.eye(nfilled)))
        if physicality=="polar":
            eig,q=np.linalg.eigh(gram)
            if eig.min()<1e-10:
                raise ValueError("rank-deficient approximate orbital matrix")
            f=f@(q@(np.diag(1/np.sqrt(eig))@q.conj().T))
        left=f[:n//2,:]
        p=np.linalg.eigvalsh(left@left.conj().T)
        if physicality=="none" and (p.min() < -1e-7 or p.max()>1+1e-7):
            sc=(np.nan,np.nan)
        else:
            sc=entropy_capacity(p)
        out_times.append(times[j])
        observables.append(sc)
        eigen_bounds.append((p.min(),p.max()))
        raw_isometry_errors.append(raw_error)
    return {
        "times":np.asarray(out_times),
        "entropy_capacity":np.asarray(observables),
        "occupation_bounds":np.asarray(eigen_bounds),
        "raw_isometry_error":np.asarray(raw_isometry_errors),
    }
