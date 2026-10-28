"""Bound-pole-exact Chebyshev/Bessel WALK expansion of strong-defect capacity.

This is an established Chebyshev propagation technique (Tal-Ezer &
Kosloff, 1984) combined with the already implemented exact
clean-chain scattering bound pole. It is NOT a new universal analytical
strong-defect revival law; its advantage is a rigorously controlled
finite expansion without a complex-energy contour or asymmetric
Hamiltonian eigendecomposition.

Model class: open even hopping chain (-1), nonnegative central onsite
potentials, *exactly one* eigenstate E_b>2 above the propagating
band [-2,2], and no negative bound states.

Let h be the final single-particle Hamiltonian, P_b its exact
rank-one bound projector (reconstructed from a clean-chain two-site
Birman-Schwinger scattering pole without diagonalization), P_c=I-P_b. Then

  U(t)F0 =
      exp(-i E_b t) P_b F0
    + J_0(2t) P_c F0
    + 2 sum_{m=1}^infty (-i)^m J_m(2t)
                            T_m(h/2) P_c F0.         [EXACT]

Truncating after m=M, the spectral theorem and |T_m(x)|<=1 for
x in [-1,1] give

  ||(U_M-U)F0||_2 <= 2 sum_{m=M+1}^infty |J_m(2t)|.

For n=M+1>2t, the Fourier-contour Bessel estimate yields
  |J_n(2t)| <= exp[-n acosh(n/(2t)) + sqrt(n^2-4t^2)].
As the logarithmic bound decreases with derivative
-acosh(n/(2t)), summing a geometric majorant gives

  EPS(t,M) =
       2 exp[-n acosh(n/(2t))+sqrt(n^2-4t^2)]
         / [1-exp(-acosh(n/(2t)))].

This bound is conservative and valid for the *analytic
truncation*, not for floating-point roundoff or errors in
the numerical pole projector.

NUMERICAL STABILITY: the full h/2 has an eigenvalue E_b/2>1,
where T_m grows exponentially. An imperfect initial projection
alone is NOT enough. The recurrence uses the *projected*
operator P_c(h/2)P_c at EVERY step to suppress bound leakage.
Its spectral radius on the continuum is <=1.

We evaluate Bessel coefficients with an FFT of exp[-2it cos(k)].
The discrete Fourier alias contribution is bounded by
  2*EPS(t,N_fft-M-1), assuming N_fft-M>2t.
This alias estimate is additional to analytic truncation.

At each observation the approximate occupied orbitals are
explicitly restored to a column isometry by polar projection.
For analytic spectral-norm error epsilon<1, the nearest-isometry
correction changes the orbitals by at most epsilon, so the
corrected orbitals are at most 2epsilon from exact, neglecting
floating-point errors. The reduced correlation matrices are
then at most 4epsilon apart.

For f(p)=p(1-p)ln^2((1-p)/p) and a subsystem of size d,
an elementary endpoint modulus-of-continuity bound gives
  |C_M-C| <= 2*d*s*[ln(1/s)+2]^2,   s=4epsilon<=1/2.

These are deterministic analytic bounds in exact arithmetic and
for exact P_b; the benchmark separately measures numerical error.
No clipping of nonphysical occupations is necessary after polar
orthonormalization (apart from the safe entropy function's
roundoff handling).

The m-th term corresponds to an m-step polynomial in the hopping
Hamiltonian and thus includes coherent walks up to that order.
The expansion is NOT a small-number-of-reflections closed form.
"""
from __future__ import annotations

import numpy as np

from scattering_phase import entropy_capacity
from defect_tmatrix import TwoSiteDefectResolvent


def bessel_tail_bound(time: float, degree: int) -> float:
    """Conservative bound for 2*sum_{n>degree}|J_n(2*time)|.

    Returns +inf when degree+1 <= 2*time (the simple exponential
    majorant then does not apply). Returns zero at time zero.
    """
    t=float(time)
    if not np.isfinite(t) or t<0 or not isinstance(degree,int) or degree<0:
        raise ValueError("invalid time or polynomial degree")
    if t==0:
        return 0.0
    n=degree+1
    if n<=2*t:
        return float("inf")
    a=np.arccosh(n/(2*t))
    log_first=-n*a+np.sqrt((n-2*t)*(n+2*t))
    return float(2*np.exp(log_first)/(-np.expm1(-a)))


def degree_for_tolerance(max_time: float, target: float=1e-12) -> int:
    """Choose M before running any asymmetric reference trajectory."""
    max_time=float(max_time)
    target=float(target)
    if not np.isfinite(max_time) or max_time<0:
        raise ValueError("invalid max_time")
    if not (0<target<1):
        raise ValueError("target must be in (0,1)")
    if max_time==0:
        return 0
    m=int(np.floor(2*max_time))
    while bessel_tail_bound(max_time,m)>target:
        m+=1
        if m>100000:
            raise RuntimeError("could not find polynomial degree")
    return m


def _capacity_error_bound(epsilon: float, subsystem_size: int) -> float:
    """Endpoint-safe modulus bound after nearest-isometry correction."""
    eps=float(epsilon)
    if eps<0 or subsystem_size<1:
        raise ValueError("invalid error bound")
    s=4*eps
    if s<=0:
        return 0.0
    if s>=0.5:
        return float("inf")
    return float(2*subsystem_size*s*(np.log(1/s)+2)**2)


def _one_positive_bound_from_clean_edge(length,u,v):
    """Birman-Schwinger 2x2 count of energies above bulk edge E=2.

    For V>=0, eigenvalue count above E=2 of h0+V equals the
    number of eigenvalues >1 of sqrt(V) (2I-h0)^-1 sqrt(V).
    The latter is computed analytically for the two central sites.
    No diagonalization of the *defective* Hamiltonian is needed.
    """
    if min(u,v)<0:
        raise ValueError("only nonnegative onsite potentials supported")
    a=length//2
    b=a+1
    side=length+1
    g=np.array([
        [a*(side-a)/side, -a*(side-b)/side],
        [-a*(side-b)/side,b*(side-b)/side],
    ])
    sqrt_v=np.sqrt([u,v])
    k=sqrt_v[:,None]*g*sqrt_v[None,:]
    count=int(np.sum(np.linalg.eigvalsh(k)>1+1e-9))
    if count!=1:
        raise ValueError("requires exactly one eigenvalue above band edge")
    return count


class BoundProjectedChebyshev:
    """Analytically bounded finite-walk predictor with exact bound pole."""

    def __init__(self,length,mean_potential,delta,occupied_modes):
        n=int(length)
        if n<4 or n%2:
            raise ValueError("even length >=4 required")
        d=float(delta)
        if not np.isfinite(d) or d==0:
            raise ValueError("requires finite nonzero asymmetry")
        w=float(mean_potential)
        u,v=w+d/2,w-d/2
        _one_positive_bound_from_clean_edge(n,u,v)
        initial=np.asarray(occupied_modes,dtype=complex)
        if initial.shape!=(n,n//2):
            raise ValueError("initial orbitals must have shape L x (L/2)")
        if not np.allclose(initial.conj().T@initial,np.eye(n//2),atol=1e-9):
            raise ValueError("occupied orbitals must be orthonormal")
        self.length=n
        self.potentials=(u,v)
        self.occupied=initial
        # The clean open-chain sine spectrum is available analytically:
        # this step diagonalizes NO symmetric or asymmetric Hamiltonian.
        # For V>=0, the 2x2 Birman-Schwinger eigenvalues of
        # sqrt(V)*g_clean(E)*sqrt(V) are monotone decreasing for E>2.
        # Its largest eigenvalue crosses one at the unique bound pole.
        self.scattering=TwoSiteDefectResolvent(n,(u,v))
        weights=np.sqrt([u,v])
        def birman_schwinger_largest(energy):
            g=self.scattering.defect_green(energy).real
            k=weights[:,None]*g*weights[None,:]
            return float(np.linalg.eigvalsh(k)[-1])
        lo=2.
        if birman_schwinger_largest(lo)<=1:
            raise ValueError("no positive out-of-band pole")
        hi=2+max(3.,u+v+2.)
        for _ in range(60):
            if birman_schwinger_largest(hi)<1:
                break
            hi=2+2*(hi-2)
        else:
            raise RuntimeError("could not bracket clean-chain scattering pole")
        for _ in range(110):
            mid=(lo+hi)/2
            if hi-lo<3e-13:
                break
            if birman_schwinger_largest(mid)>1:
                lo=mid
            else:
                hi=mid
        E=(lo+hi)/2
        b=self.scattering.bound_wavefunction(E)
        self.b=np.real_if_close(b).astype(float)
        self.bound_energy=float(E)
        self.bound_initial=np.outer(self.b,self.b@self.occupied)
        self.continuum_initial=self.occupied-self.bound_initial
        self._moments=None
        self._max_degree=-1

    def _apply_continuum_h_scaled(self,x):
        """Projected (h/2) action; 2-site sparse local hopping stencil."""
        u,v=self.potentials
        n=self.length
        result=np.zeros_like(x)
        result[1:]-=x[:-1]
        result[:-1]-=x[1:]
        result[n//2-1]+=u*x[n//2-1]
        result[n//2]+=v*x[n//2]
        result*=.5
        result-=np.outer(self.b,self.b@result)
        return result

    def moments(self,degree:int):
        """Precompute coherent walk moments T_m(Pc h Pc/2)Pc F0."""
        if not isinstance(degree,int) or degree<0:
            raise ValueError("degree must be nonnegative integer")
        if degree<=self._max_degree:
            return self._moments[:degree+1]
        # From scratch ensures caller never receives stale pointers.
        moments=np.empty((degree+1,self.length,self.length//2),complex)
        moments[0]=self.continuum_initial
        if degree:
            moments[1]=self._apply_continuum_h_scaled(moments[0])
            for m in range(2,degree+1):
                moments[m]=2*self._apply_continuum_h_scaled(
                    moments[m-1])-moments[m-2]
        self._moments=moments
        self._max_degree=degree
        return moments

    def _coefficients(self,times,degree,fft_points=None):
        """Analytic Jacobi-Anger coefficients via alias-controlled FFT."""
        t=np.asarray(times,dtype=float)
        if t.ndim!=1 or not len(t) or np.any(~np.isfinite(t)) or np.any(t<0):
            raise ValueError("times must be a nonempty nonnegative vector")
        tmax=float(t.max())
        if fft_points is None:
            minimum=max(1024,int(2*(degree+2*tmax+32)))
            fft_points=1<<int(np.ceil(np.log2(minimum)))
        if not isinstance(fft_points,int) or fft_points<=degree+2*tmax+1:
            raise ValueError("FFT grid cannot control aliasing")
        k=2*np.pi*np.arange(fft_points)/fft_points
        values=np.exp(-2j*t[:,None]*np.cos(k)[None,:])
        coeff=np.fft.fft(values,axis=1)[:,:degree+1]/fft_points
        coeff[:,1:]*=2
        alias=2*bessel_tail_bound(tmax,fft_points-degree-1)
        return coeff,float(alias),fft_points

    def trajectory(self,times,degree=None,target=1e-12,
                   polar=True,fft_points=None):
        """Return [entropy,capacity] and transparent error diagnostics.

        If degree is omitted, select it by an analytic a-priori
        tolerance for the largest requested time. Numerical
        errors/pole-residue error are NOT covered by the bound.
        """
        times=np.asarray(times,dtype=float)
        if times.ndim!=1 or not len(times) or np.any(~np.isfinite(times)) or np.any(times<0):
            raise ValueError("invalid times")
        if degree is None:
            degree=degree_for_tolerance(float(times.max()),target)
        moments=self.moments(degree)
        coeff,alias,nfft=self._coefficients(times,degree,fft_points)
        expanded=(coeff@moments.reshape(degree+1,-1)).reshape(
            len(times),self.length,self.length//2)
        expanded+=np.exp(-1j*self.bound_energy*times)[:,None,None]*self.bound_initial[None,:,:]
        capacities=np.empty((len(times),2))
        raw_error=0.
        max_polar_change=0.
        for i,fraw in enumerate(expanded):
            gram=fraw.conj().T@fraw
            raw_error=max(raw_error,float(np.linalg.norm(gram-np.eye(self.length//2))))
            if polar:
                eig,q=np.linalg.eigh(gram)
                if min(eig)<1e-12:
                    raise RuntimeError("truncated orbitals too rank-deficient for polar repair")
                fix=q@(np.diag(1/np.sqrt(eig))@q.conj().T)
                f=fraw@fix
                max_polar_change=max(max_polar_change,float(np.linalg.norm(f-fraw)))
            else:
                f=fraw
            p=np.linalg.eigvalsh(f[:self.length//2]@f[:self.length//2].conj().T)
            if p.min() < -1e-8 or p.max()>1+1e-8:
                raise RuntimeError("nonphysical Gaussian covariance")
            capacities[i]=entropy_capacity(p)
        trunc=bessel_tail_bound(float(times.max()),degree)
        epsilon=trunc+alias
        return {
            "times":times,
            "values":capacities,
            "degree":int(degree),
            "fft_points":int(nfft),
            "bound_energy":self.bound_energy,
            "analytic_walk_truncation_bound":float(trunc),
            "analytic_FFT_alias_bound":float(alias),
            "formal_amplitude_error_bound":float(epsilon),
            "formal_capacity_error_bound":_capacity_error_bound(
                epsilon,self.length//2) if polar and epsilon<1 else np.inf,
            "max_raw_occupied_isometry_error":float(raw_error),
            "max_polar_change_fro":float(max_polar_change),
        }
