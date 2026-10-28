"""Parameter-free linear response of modular capacity to defect asymmetry.

For an even open hopping chain, let
  h(delta)=h_sym + delta W,
  h_sym=h_hopping + w (|a><a|+|b><b|),
  W=(|a><a|-|b><b|)/2.
The two reversed defects are delta and -delta.

The first-order orbital derivative is calculated analytically from
Duhamel's formula using the *symmetric* reference Hamiltonian only:

  dU/d(delta)|0 = -i int_0^t U_sym(t-s) W U_sym(s) ds.

In the energy eigenbasis of h_sym (eigenvalues E_m), this is
  [dU]_{mn} = W_mn * (exp(-i E_m t)-exp(-i E_n t))/(E_m-E_n),
with the diagonal value -i*t*exp(-i E_m t)*W_mm.

For G_A=F_A F_A^dagger, differentiate to dG_A. Then
  capacity=Tr f(G_A),
  f(p)=p(1-p) log^2((1-p)/p),
  f'(p)=(1-2p)log^2((1-p)/p)-2log((1-p)/p),
  chi_C(t)=Tr[f'(G_A,sym(t)) dG_A(t)].

The prediction for the *contrast* is
  C_A(t;delta)-C_A(t;-delta)=2*delta*chi_C(t)+O(delta^3)
provided the relevant eigenvalues stay strictly inside (0,1)
and the capacity is differentiable through the expansion.

This is Kubo/Duhamel perturbation theory plus Gaussian-state
functional calculus. It is established mathematical physics,
not a claim of novel quantum capacity physics.

At very small / nearly-one eigenvalues f' diverges. We clip ONLY when
evaluating the first derivative, report a near-edge diagnostic,
and validate against finite differences. This is not an error bound.
"""
from __future__ import annotations

import numpy as np

from scattering_phase import tight_binding_hamiltonian, entropy_capacity


def capacity_derivative(p: np.ndarray, clip_at: float=1e-12):
    """Derivative of the one-mode modular-capacity function."""
    if not (0<clip_at<.5):
        raise ValueError("clip_at must lie in (0,1/2)")
    q=np.clip(np.asarray(p,dtype=float),clip_at,1-clip_at)
    log=np.log((1-q)/q)
    return (1-2*q)*log*log-2*log


class DefectAsymmetryResponse:
    def __init__(self, length, mean_potential, occupied_modes,
                 derivative_clip=1e-12):
        if length < 4 or length%2:
            raise ValueError("length must be even and >=4")
        f0=np.asarray(occupied_modes,dtype=complex)
        if f0.shape!=(length,length//2):
            raise ValueError("occupied_modes have invalid shape")
        if not np.allclose(f0.conj().T@f0,np.eye(length//2),atol=1e-9):
            raise ValueError("occupied_modes must be orthonormal")
        if not (0<derivative_clip<.5):
            raise ValueError("invalid derivative clip")
        self.length=int(length)
        self.mean_potential=float(mean_potential)
        self.derivative_clip=float(derivative_clip)
        symmetric=tight_binding_hamiltonian(
            length,(self.mean_potential,self.mean_potential))
        self.energies,self.eigenvectors=np.linalg.eigh(symmetric)
        self.initial_amplitudes=self.eigenvectors.conj().T@f0
        w=np.zeros((length,length),float)
        a,b=length//2-1,length//2
        w[a,a]=.5
        w[b,b]=-.5
        self.w_energy=self.eigenvectors.conj().T@w@self.eigenvectors
        self.energy_gap=self.energies[:,None]-self.energies[None,:]
        self.nonzero_gap=abs(self.energy_gap)>1e-11

    def at_time(self,time):
        """Return (symmetric capacity, dCapacity/dDelta, edge sensitivity)."""
        time=float(time)
        if time < 0:
            raise ValueError("time must be nonnegative")
        phase=np.exp(-1j*self.energies*time)
        divided=np.zeros_like(self.energy_gap,dtype=complex)
        np.divide(phase[:,None]-phase[None,:],self.energy_gap,
                  out=divided,where=self.nonzero_gap)
        # The spectrum of a connected open Jacobi chain is nondegenerate;
        # this diagonal fill is its exact coincident-energy limit.
        np.fill_diagonal(divided,-1j*time*phase)
        f=self.eigenvectors@(phase[:,None]*self.initial_amplitudes)
        df=self.eigenvectors@((divided*self.w_energy)@self.initial_amplitudes)
        left=f[:self.length//2]
        derivative_left=df[:self.length//2]
        g=left@left.conj().T
        dg=derivative_left@left.conj().T+left@derivative_left.conj().T
        p,vecs=np.linalg.eigh(g)
        derivative_modes=np.real(np.diag(vecs.conj().T@dg@vecs))
        fp=capacity_derivative(p,self.derivative_clip)
        susceptibility=float(np.dot(fp,derivative_modes))
        near=(p<=self.derivative_clip)|(p>=1-self.derivative_clip)
        edge_weight=float(np.sum(abs(derivative_modes[near])))
        capacity=entropy_capacity(p)[1]
        return capacity,susceptibility,edge_weight

    def response_trajectory(self,times):
        """Each row: [symmetric capacity, susceptibility, edge weight]."""
        times=np.asarray(times,dtype=float)
        if times.ndim != 1 or np.any(times<0):
            raise ValueError("times must be a 1d nonnegative array")
        return np.array([self.at_time(t) for t in times],
                        dtype=float).reshape(-1,3)

    def predict_reversal_contrast(self,delta,times):
        """First-order prediction WITHOUT simulating either reversed defect."""
        series=self.response_trajectory(times)
        return 2*float(delta)*series[:,1]
