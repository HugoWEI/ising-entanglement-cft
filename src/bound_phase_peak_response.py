"""Exact phase susceptibility and locally predictive capacity peak response.

A fixed Hamiltonian has one localized final eigenstate. The optional
pure-state preparation gate P_c + exp(i theta) P_b changes only the
relative phase of bound and extended components, preserving all final
energy-mode occupations.

For subsystem covariance
    G(t,theta)=D(t)+exp(i theta)x(t)y(t)^dag+
                       exp(-i theta)y(t)x(t)^dag
where D is a valid mixed Gaussian reference, define
    f(p)=p(1-p)*log^2((1-p)/p).

At finite t, for a differentiable entanglement spectrum in (0,1),
    chi_theta(t,theta) := partial_theta C(t,theta)
        =Tr[f'(G) partial_theta G]
        =-2 Im[exp(i theta)*y^dag f'(G)x],
    f'(p)=(1-2p)log^2((1-p)/p)-2log((1-p)/p).

This is exact, NONPERTURBATIVE in defect strength and bound-band
coherence, and requires only ONE reference bound/continuum evolution.
It does NOT imply small-coherence perturbation.

For a nondegenerate isolated smoothed capacity maximum t0 at theta0,
implicit differentiation predicts nearby phase-controlled peak changes:
    t_peak(theta0+dtheta)
      =t0-dtheta*chi'_theta(t0)/C''(t0)+O(dtheta^2).
The two-sided time and height contrasts have cubic remainders for
smooth stable peak branches. This is standard implicit-function theory.

The matrix determinant lemma gives a further EXACT phase identity:
    det(zI-G_theta)/det(zI-D)
      =1-det(M)
         -exp(i theta)*M[1,0]-exp(-i theta)*M[0,1],
    M=X^dag (zI-D)^-1 X, X=[x,y].
Thus the CHARACTERISTIC-POLYNOMIAL RATIO is first harmonic in theta
for EVERY spectral z, even though the nonlinear capacity can contain
higher theta harmonics. This distinction matters.

Prior-art/novelty: standard Gaussian matrix-function response and
rank-two determinant algebra, not a new general theorem. Close
physics prior art: Donelli et al., PRA 113, 013311 (2026),
DOI 10.1103/5kt1-5m83, on initial coherence near a localized defect.
That work focuses on different observables and geometry.

Endpoint caveat: f' diverges logarithmically near p=0 and p=1.
We report spectral clipping weight and validate by finite differences.
A global analytic phase-derivative error theorem is NOT claimed.
"""
from __future__ import annotations

import numpy as np
from bound_coherence_determinant import SingleBoundCoherence
from capacity_peak_response import predict_reversed_peaks
from scattering_phase import entropy_capacity


def capacity_scalar_derivative(p,derivative_clip=1e-12):
    if not 0<derivative_clip<.5:
        raise ValueError("invalid derivative_clip")
    p=np.clip(np.asarray(p,float),derivative_clip,1-derivative_clip)
    q=np.log1p(-p)-np.log(p)
    return (1-2*p)*q*q-2*q


def phase_susceptibility_from_components(pieces,theta,derivative_clip=1e-12):
    """Exact phase derivative of Gaussian capacity for rank-two update."""
    theta=float(theta)
    if not np.isfinite(theta):
        raise ValueError("phase must be finite")
    g=np.asarray(pieces["full"],complex)
    x=np.asarray(pieces["X"][:,0],complex)
    y=np.asarray(pieces["X"][:,1],complex)
    p,vecs=np.linalg.eigh(g)
    fp=capacity_scalar_derivative(p,derivative_clip)
    # x and y are the theta-independent physical vectors.
    # f'(G)*x = vecs @ [ f'(p) (vecs† x) ].
    fx=vecs@(fp*(vecs.conj().T@x))
    result=-2*np.imag(np.exp(1j*theta)*np.vdot(y,fx))
    near=(p<derivative_clip)|(p>1-derivative_clip)
    return float(result),int(np.count_nonzero(near))


def phase_trajectory(model: SingleBoundCoherence,times,theta=0.,
                     derivative_clip=1e-12):
    """Capacity C(t,theta) and exact first phase derivative chi(t)."""
    t=np.asarray(times,float)
    if t.ndim!=1 or len(t)==0 or np.any(~np.isfinite(t)) or np.any(t<0):
        raise ValueError("invalid times")
    out=np.empty((len(t),3))
    for i,time in enumerate(t):
        pieces=model.components(float(time),phase_kick=theta)
        p=np.linalg.eigvalsh(pieces["full"])
        if p.min() < -1e-8 or p.max()>1+1e-8:
            raise RuntimeError("nonphysical Gaussian covariance")
        out[i,0]=entropy_capacity(p)[1]
        out[i,1],out[i,2]=phase_susceptibility_from_components(
            pieces,theta,derivative_clip)
    return out


def phase_peak_prediction(model: SingleBoundCoherence,times,delta_theta,
                          base_phase=0.,derivative_clip=1e-12):
    """Predict peak shifts for +/-dtheta from ONE base-phase calculation.

    The method returns None when an isolated reference peak cannot
    be identified by the predeclared smoothing/prominence protocol.
    It does NOT use the +/-dtheta trajectories for its prediction.
    """
    result=phase_trajectory(model,times,base_phase,derivative_clip)
    prediction=predict_reversed_peaks(
        times,result[:,0],result[:,1],model.length,float(delta_theta))
    if prediction is None:
        return None
    prediction["base_phase"]=float(base_phase)
    prediction["maximum_near_endpoint_eigenvalues"]=int(max(result[:,2]))
    return prediction


def first_harmonic_characteristic_ratio(model,time,z,theta):
    """Exact 2x2 determinant ratio, manifestly first harmonic in theta."""
    pieces=model.components(float(time),phase_kick=theta)
    D=pieces["dephased"]
    X=pieces["X"]
    z=complex(z)
    M=X.conj().T@np.linalg.solve(z*np.eye(len(D))-D,X)
    theta=float(theta)
    return complex(1-np.linalg.det(M)
                   -np.exp(1j*theta)*M[1,0]
                   -np.exp(-1j*theta)*M[0,1])


def first_harmonic_from_cardinal_phases(cardinal_capacities,angles):
    """Predict held-out phases using only 0,pi/2,pi,3pi/2 capacities.

    Values have shape (N_times,4) in that order. A standard Fourier
    interpolation, NOT a fundamental scattering-only approximation:
    it still requires FOUR rank-two capacity evaluations per time.
    """
    c=np.asarray(cardinal_capacities,float)
    theta=np.asarray(angles,float)
    if c.ndim!=2 or c.shape[1]!=4 or theta.ndim!=1:
        raise ValueError("expected (N_times,4) data and 1d phase samples")
    mean=np.mean(c,axis=1)
    cosine=(c[:,0]-c[:,2])/2
    sine=(c[:,1]-c[:,3])/2
    return mean[:,None]+cosine[:,None]*np.cos(theta)+sine[:,None]*np.sin(theta)
