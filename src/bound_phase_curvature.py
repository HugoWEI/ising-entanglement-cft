"""Exact second phase response of free-fermion modular capacity.

At fixed final defect h, vary the PURE initial bound-band phase
U_theta=P_c+exp(i theta)P_b. The left covariance is

  G(theta)=D+exp(i theta)x y†+exp(-i theta)y x†.

Define H=partial_theta G and K=partial_theta^2 G:
  H=i(exp(i theta)x y†-exp(-i theta)y x†),
  K= -(exp(i theta)x y†+exp(-i theta)y x†).

For f(p)=p(1-p)log²((1-p)/p),
  C'(theta)=Tr[f'(G)H],
  C''(theta)=Tr[f'(G)K]
       +SUM_{ij} f'^[1](p_i,p_j) |H_ij|²,

where f'^[1](a,b)=(f'(a)-f'(b))/(a-b) with diagonal
limit f''(a). This is standard spectral matrix-function
differentiation and is EXACT when the spectrum is in (0,1).

The second-order local capacity model around theta0 is

 C(t,theta0+eta)=C0(t)+eta*chi0(t)
                 +0.5*eta²*kappa0(t)+O(eta³).

If a smoothed peak t0 is nondegenerate and smooth, the
corresponding individual peak time obeys

 t_peak(eta)=t0+eta*t1+eta²*t2+O(eta³)
 t1=-chi_t/C_tt,
 t2=-(0.5*C_ttt*t1² + chi_tt*t1 +0.5*kappa_t)/C_tt

with all derivatives evaluated at (t0,theta0), after applying
the SAME fixed linear temporal smoothing to C,chi,kappa.

A quadratic-in-phase *capacity curve* can be independently
maximized locally to predict a peak without estimating C_ttt.

ENDPOINT CAVEAT: f' and f'' are singular near 0 and 1.
We clip eigenvalues ONLY when evaluating the spectral
derivative and return near-edge diagnostics. No rigorous
uniform susceptibility/curvature bound is claimed.
"""
from __future__ import annotations

import numpy as np
from bound_coherence_determinant import SingleBoundCoherence
from bound_phase_peak_response import (
    phase_susceptibility_from_components)
from scattering_phase import entropy_capacity
from capacity_peak_response import (
    find_reference_peak,tracked_peak)


def capacity_scalar_second_derivative(p,clip=1e-10):
    """Analytical second derivative of the scalar capacity function."""
    if not 0 < clip < 0.5:
        raise ValueError("clip must be in (0,1/2)")
    p=np.clip(np.asarray(p,float),clip,1-clip)
    log=np.log1p(-p)-np.log(p)
    inverse=1/(p*(1-p))
    return (-2*log**2-2*(1-2*p)*log*inverse+2*inverse)


def second_phase_susceptibility_from_components(
        pieces,theta,derivative_clip=1e-10,degeneracy_tol=1e-7):
    """Exact C_theta_theta from eigenvalue divided differences.

    Returns (second derivative, number of near-endpoint eigenvalues).
    Degenerate eigenvalues are treated through the f'' limit.
    """
    theta=float(theta)
    if not np.isfinite(theta) or not 0<derivative_clip<.5 or degeneracy_tol<=0:
        raise ValueError("invalid phase or numerical derivative parameters")
    G=np.asarray(pieces["full"],complex)
    x=np.asarray(pieces["X"][:,0],complex)
    y=np.asarray(pieces["X"][:,1],complex)
    if G.ndim!=2 or G.shape[0]!=G.shape[1] or len(x)!=len(G):
        raise ValueError("invalid components")
    a=np.exp(1j*theta)
    cross=a*np.outer(x,y.conj())
    H=1j*(cross-cross.conj().T)
    K=-(cross+cross.conj().T)
    eig,U=np.linalg.eigh(G)
    clipped=np.clip(eig,derivative_clip,1-derivative_clip)
    log=np.log1p(-clipped)-np.log(clipped)
    fp=(1-2*clipped)*log**2-2*log
    spectrum_weight=np.real(np.diag(U.conj().T@K@U))
    first=float(np.dot(fp,spectrum_weight))

    # Stable Hermitian divided differences; eigendecomposition
    # avoids differentiating near-degenerate eigenvectors.
    pi=clipped[:,None]
    pj=clipped[None,:]
    diff=pi-pj
    diff_fp=fp[:,None]-fp[None,:]
    divided=np.empty_like(diff)
    away=np.abs(diff)>degeneracy_tol
    np.divide(diff_fp,diff,out=divided,where=away)
    mids=(pi+pj)/2
    divided[~away]=capacity_scalar_second_derivative(
        mids[~away],clip=derivative_clip)

    H_energy=U.conj().T@H@U
    second=float(np.sum(divided*abs(H_energy)**2))
    near=int(np.count_nonzero(
        (eig<=derivative_clip)|(eig>=1-derivative_clip)))
    return first+second,near


def second_order_phase_trajectory(model:SingleBoundCoherence,times,theta=0.,
                                   derivative_clip=1e-10):
    """Columns [capacity, phase derivative, second derivative, edge count]."""
    t=np.asarray(times,float)
    if t.ndim!=1 or not len(t) or np.any(~np.isfinite(t)) or np.min(t)<0:
        raise ValueError("invalid times")
    out=np.empty((len(t),4))
    for i,time in enumerate(t):
        parts=model.components(float(time),phase_kick=theta)
        p=np.linalg.eigvalsh(parts["full"])
        if p.min()<-1e-8 or p.max()>1+1e-8:
            raise RuntimeError("unphysical covariance")
        out[i,0]=entropy_capacity(p)[1]
        out[i,1],count1=phase_susceptibility_from_components(
            parts,theta,derivative_clip)
        out[i,2],count2=second_phase_susceptibility_from_components(
            parts,theta,derivative_clip)
        out[i,3]=max(count1,count2)
    return out


def predict_individual_peak(model,times,delta_phase,base_phase=0.,
                            derivative_clip=1e-10,
                            branch_radius=3.):
    """Peak time/height prediction, one reference phase only.

    The linear and quadratic predicted curves are maximized
    independently with the same local branch tracker, anchored
    to the reference maximum. No shifted-phase curves are used.
    """
    t=np.asarray(times,float)
    base=second_order_phase_trajectory(
        model,t,theta=base_phase,derivative_clip=derivative_clip)
    anchor=find_reference_peak(t,base[:,0],model.length)
    if anchor is None:
        return None
    d=float(delta_phase)
    linear=base[:,0]+d*base[:,1]
    quadratic=linear+.5*d*d*base[:,2]
    t0=anchor[0]
    a=tracked_peak(t,linear,t0,model.length,window_radius=branch_radius)
    b=tracked_peak(t,quadratic,t0,model.length,window_radius=branch_radius)
    return {
        "reference_time":float(t0),
        "reference_height":float(anchor[1]),
        "linear_time":float(a[0]) if a else np.nan,
        "quadratic_time":float(b[0]) if b else np.nan,
        "linear_height":float(a[1]) if a else np.nan,
        "quadratic_height":float(b[1]) if b else np.nan,
        "maximum_edge_sensitive_modes":int(np.max(base[:,3])),
    }


def predict_peak_second_order_explicit(
        times,capacity,susceptibility,second_susceptibility,
        length,sigma_time=2.0):
    """Explicit t1 and t2 from local polynomial temporal derivatives.

    This is a derivative-based alternative to maximizing the
    quadratic approximate capacity curve. Report the curvature
    and reject unstable flat peaks.
    """
    from revival_benchmark import gaussian_smooth
    t=np.asarray(times,float)
    c=np.asarray(capacity,float)
    chi=np.asarray(susceptibility,float)
    kappa=np.asarray(second_susceptibility,float)
    if any(x.shape!=t.shape for x in (c,chi,kappa)) or len(t)<20:
        raise ValueError("incompatible arrays")
    ref=find_reference_peak(t,c,length,sigma_time=sigma_time)
    if ref is None:
        return None
    t0,c0,_,i=ref
    dt=float(t[1]-t[0])
    sm=[gaussian_smooth(x,sigma_time/dt) for x in (c,chi,kappa)]
    span=5
    if i-span<0 or i+span>=len(t):
        return None
    grid=t[i-span:i+span+1]-t0
    # Degree four captures the third time derivative while
    # suppressing small 0.2-grid interpolation artifacts.
    coeff=[np.polynomial.polynomial.polyfit(grid,
             data[i-span:i+span+1],4) for data in sm]
    ctt=2*coeff[0][2]
    if not ctt < -1e-5:
        return None
    cttt=6*coeff[0][3]
    chit=coeff[1][1]
    chitt=2*coeff[1][2]
    kappat=coeff[2][1]
    first=-chit/ctt
    second=-(.5*cttt*first*first+
             chitt*first+.5*kappat)/ctt
    return dict(reference_time=t0,
                reference_curvature=float(ctt),
                t1=float(first),t2=float(second),
                # individual branch: t0+eta*t1+eta²*t2
                # both signs share eta²*t2
                )
