"""Exact rank-two bound-continuum coherence update of Gaussian entanglement.

SCOPE: a number-conserving even open hopping chain with EXACTLY ONE
out-of-band single-particle eigenstate. This is a standard Gaussian
spectral decomposition + matrix determinant lemma, not a novel
strong-defect first-revival formula.

With h eigenvectors b (bound) and V_c (propagating):
  F(t)=F_c(t)+F_b(t), F_b(t)=exp(-i E_b t) b q,
  q=b^dagger F(0), with q shape (N_occupied,).
Restrict to the left subsystem:
  A=F_c(t)[left,:],
  x=exp(-i E_b t) b[left],
  y=A q^*.
The reduced Gaussian covariance is EXACTLY
  G_full = A A^dag + ||q||^2 x x^dag + x y^dag+y x^dag.
Define the valid, mixed Gaussian reference covariance
  G_deph = A A^dag + ||q||^2 x x^dag.
Then
  G_full=G_deph+X J X^dag, X=[x,y], J=[[0,1],[1,0]].

The determinant lemma gives an EXACT two-by-two entanglement-spectrum
correction at any complex lambda off the spectrum:
  det(lambda I-G_full)/det(lambda I-G_deph)
    = det(I_2 - J X^dag (lambda I-G_deph)^-1 X).

The reference G_deph is a valid 0<=G<=I covariance, arising from
discarding bound-band covariance blocks in the final h eigenbasis.
It is NOT the evolution of the SAME initial pure state and not a
physical defect with its bound state removed. Its Gaussian capacity
is a controlled mathematically valid *dephased reference*.

The difference C_full-C_deph cannot be interpreted as an additive
capacity of a bound state. Capacity is nonlinear and the changes
include all correlations through the determinant update.
"""
from __future__ import annotations
import numpy as np
from scattering_phase import tight_binding_hamiltonian,entropy_capacity


class SingleBoundCoherence:
    def __init__(self,length,potentials,occupied_modes,band_edge=2.0,
                 tolerance=1e-8):
        if length<4 or length%2:
            raise ValueError("length must be even and >=4")
        f0=np.asarray(occupied_modes,dtype=complex)
        if f0.shape!=(length,length//2):
            raise ValueError("occupied modes must be L by L/2")
        if not np.allclose(f0.conj().T@f0,np.eye(length//2),atol=1e-9):
            raise ValueError("occupied modes must be orthonormal")
        e,v=np.linalg.eigh(tight_binding_hamiltonian(length,potentials))
        bound=np.abs(e)>band_edge+tolerance
        if np.count_nonzero(bound)!=1:
            raise ValueError("requires exactly one out-of-band bound mode")
        self.length=int(length)
        self.bound_energy=float(e[bound][0])
        self.bound_vector=v[:,bound][:,0]
        self.bound_overlap=self.bound_vector.conj()@f0
        self.extended_energies=e[~bound]
        self.extended_modes=v[:,~bound]
        self.extended_amplitudes=self.extended_modes.conj().T@f0
        self.bound_weight=float(np.vdot(self.bound_overlap,
                                         self.bound_overlap).real)

    def components(self,time):
        """Exact covariance and two-column update, without modification."""
        time=float(time)
        if time<0:
            raise ValueError("time must be nonnegative")
        n=self.length//2
        extended=self.extended_modes[:n,:]@(
            np.exp(-1j*self.extended_energies*time)[:,None]*
            self.extended_amplitudes)
        x=np.exp(-1j*self.bound_energy*time)*self.bound_vector[:n]
        y=extended@self.bound_overlap.conj()
        g_band=extended@extended.conj().T
        g_deph=g_band+self.bound_weight*np.outer(x,x.conj())
        cross=np.outer(x,y.conj())+np.outer(y,x.conj())
        g_full=g_deph+cross
        X=np.column_stack([x,y])
        J=np.array([[0.,1.],[1.,0.]])
        return {
            "full":g_full,
            "dephased":g_deph,
            "band_only":g_band,
            "cross":cross,
            "X":X,
            "J":J,
        }

    def capacity_at(self,time):
        """Full, dephased-reference, band-only capacities, plus controls."""
        pieces=self.components(time)
        for name in ("full","dephased","band_only"):
            eig=np.linalg.eigvalsh(pieces[name])
            if eig.min() < -2e-9 or eig.max()>1+2e-9:
                raise RuntimeError(name+" covariance violates fermionic bounds")
        full=entropy_capacity(np.linalg.eigvalsh(pieces["full"]))[1]
        deph=entropy_capacity(np.linalg.eigvalsh(pieces["dephased"]))[1]
        band=entropy_capacity(np.linalg.eigvalsh(pieces["band_only"]))[1]
        return np.array([full,deph,band,
                         np.linalg.norm(pieces["cross"],"fro")])

    def trajectory(self,times):
        times=np.asarray(times,dtype=float)
        if times.ndim!=1 or np.any(times<0):
            raise ValueError("times must be a 1d nonnegative array")
        return np.array([self.capacity_at(t) for t in times]).reshape(-1,4)

    def entanglement_determinant_ratio(self,time,lam):
        """Two-by-two determinant ratio, without forming full covariance."""
        pieces=self.components(time)
        lam=complex(lam)
        n=self.length//2
        Q=np.linalg.solve(lam*np.eye(n)-pieces["dephased"],pieces["X"])
        return np.linalg.det(
            np.eye(2)-pieces["J"]@pieces["X"].conj().T@Q)

    def direct_determinant_ratio(self,time,lam):
        """Independent full covariance determinant, for validation only."""
        pieces=self.components(time)
        lam=complex(lam)
        n=self.length//2
        return (np.linalg.det(lam*np.eye(n)-pieces["full"])/
                np.linalg.det(lam*np.eye(n)-pieces["dephased"]))
