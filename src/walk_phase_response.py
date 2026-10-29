"""Phase-dependent revival prediction from POLE-EXACT FINITE WALKS.

This combines:
 (a) exact bound-state energy/vector from a clean-chain 2x2
     Birman-Schwinger scattering equation (NO full Hamiltonian diag);
 (b) Bessel/Chebyshev finite walks for the band occupied orbitals,
     with a priori chosen polynomial degree;
 (c) exact two-vector bound–band phase derivative of modular capacity.

Nothing here fits the +/-phase test trajectories or diagonalizes the
defective chain for prediction. Thus this is a scattering/walk-based
implementation of the LOCAL pure-phase revival-peak law

    dt_peak/dtheta = -[d/dt chi_theta(t0)]/[d²/dt² C(t0)].

Crucial limitation: bessel_tail_bound certifies the band ORBITAL
propagator in exact arithmetic and with exact bound pole. Here we
DO NOT claim a certified bound for chi_theta or peak-time
derivatives (f' is singular near occupation 0 and 1).
We independently validate the phase response and peak shifts
against exact Gaussian evolution. We also refuse insufficiently
unitary truncated orbitals rather than silently clipping them.

The finite-walk model order grows roughly with 2*max_time, so
it is not a handful-of-scattering-path asymptotic theory.
"""
from __future__ import annotations

import numpy as np
from bound_projected_chebyshev import (
    BoundProjectedChebyshev,bessel_tail_bound,degree_for_tolerance)
from bound_phase_peak_response import phase_susceptibility_from_components
from bound_phase_curvature import second_phase_susceptibility_from_components
from capacity_peak_response import find_reference_peak,tracked_peak
from capacity_peak_response import predict_reversed_peaks
from scattering_phase import entropy_capacity


class WalkPhaseResponse:
    def __init__(self,length,mean_potential,asymmetry,occupied_modes):
        self.walk=BoundProjectedChebyshev(
            length,mean_potential,asymmetry,occupied_modes)
        self.length=int(length)

    def trajectory(self,times,phase=0.,degree=None,target=1e-12,
                   isometry_tolerance=2e-7,include_curvature=False):
        times=np.asarray(times,float)
        phase=float(phase)
        if (times.ndim!=1 or not len(times) or
            np.any(~np.isfinite(times)) or np.any(times<0) or
            not np.isfinite(phase)):
            raise ValueError("invalid time grid or phase")
        solver=self.walk
        M=(degree_for_tolerance(float(times.max()),target)
           if degree is None else int(degree))
        moments=solver.moments(M)
        coeff,alias,nfft=solver._coefficients(times,M)
        n=self.length
        fc=(coeff@moments.reshape(M+1,-1)).reshape(
            len(times),n,n//2)
        b=solver.b
        q=b@solver.occupied
        beta=float(np.vdot(q,q).real)
        values=np.empty((len(times),4 if include_curvature else 3))
        max_error=0.
        for i,time in enumerate(times):
            A=fc[i,:n//2]
            x=np.exp(-1j*solver.bound_energy*time)*b[:n//2]
            y=A@q.conj()
            D=A@A.conj().T+beta*np.outer(x,x.conj())
            X=np.column_stack([x,y])
            G=D+np.exp(1j*phase)*np.outer(x,y.conj())+(
                np.exp(-1j*phase)*np.outer(y,x.conj()))
            full_orbitals=fc[i]+np.exp(1j*phase)*(
                np.exp(-1j*solver.bound_energy*time)
                *solver.bound_initial)
            gram_error=float(np.linalg.norm(
                full_orbitals.conj().T@full_orbitals-np.eye(n//2)))
            max_error=max(max_error,gram_error)
            if gram_error>isometry_tolerance:
                raise RuntimeError("truncated walk is insufficiently unitary")
            p=np.linalg.eigvalsh(G)
            if p.min() < -isometry_tolerance or p.max()>1+isometry_tolerance:
                raise RuntimeError("truncated walk gave nonphysical occupations")
            values[i,0]=entropy_capacity(p)[1]
            chi,near=phase_susceptibility_from_components(
                {"full":G,"X":X},phase)
            values[i,1]=chi
            if include_curvature:
                second,second_near=second_phase_susceptibility_from_components(
                    {"full":G,"X":X},phase)
                values[i,2]=second
                values[i,3]=max(near,second_near)
            else:
                values[i,2]=near
        return {
            "times":times,"values":values,
            "degree":M,
            "bound_energy":float(solver.bound_energy),
            "a_priori_orbital_tail_bound":float(
                bessel_tail_bound(float(times.max()),M)+alias),
            "max_occupied_isometry_error":max_error,
            "fft_points":nfft,
        }

    def peak_shift(self,times,delta_phase,base_phase=0.,target=1e-12):
        data=self.trajectory(times,phase=base_phase,target=target)
        result=predict_reversed_peaks(
            times,data["values"][:,0],data["values"][:,1],
            self.length,delta_phase)
        if result is None:
            return None
        result["polynomial_degree"]=data["degree"]
        result["a_priori_orbital_tail_bound"]=data[
            "a_priori_orbital_tail_bound"]
        result["max_walk_isometry_error"]=data[
            "max_occupied_isometry_error"]
        return result


    def quadratic_peak(self,times,phase_offset,base_phase=0.,
                       target=1e-12,window_radius=3.):
        """Predict individual peak times to second phase order using walks.

        All reference C,chi,kappa are computed from one fixed
        preparation phase with NO full-chain Hamiltonian diagonalization.
        The shifted preparations are NOT used for this prediction.
        """
        t=np.asarray(times,float)
        data=self.trajectory(t,phase=base_phase,target=target,
                             include_curvature=True)
        C,chi,kappa=(data["values"][:,i] for i in range(3))
        reference=find_reference_peak(t,C,self.length)
        if reference is None:
            return None
        eta=float(phase_offset)
        first=C+eta*chi
        second=first+.5*eta*eta*kappa
        t0=reference[0]
        a=tracked_peak(t,first,t0,self.length,
                       window_radius=window_radius)
        b=tracked_peak(t,second,t0,self.length,
                       window_radius=window_radius)
        return {
            "reference_peak_time":float(t0),
            "linear_peak_time":float(a[0]) if a else np.nan,
            "quadratic_peak_time":float(b[0]) if b else np.nan,
            "linear_peak_height":float(a[1]) if a else np.nan,
            "quadratic_peak_height":float(b[1]) if b else np.nan,
            "polynomial_degree":int(data["degree"]),
            "a_priori_orbital_tail_bound":
                data["a_priori_orbital_tail_bound"],
            "max_walk_isometry_error":
                data["max_occupied_isometry_error"],
            "max_near_edge_occupations":int(np.max(data["values"][:,3])),
        }
