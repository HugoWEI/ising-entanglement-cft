"""Unitary rank-two implicit-midpoint evolution for the open hopping chain.

A second-order *numerical method*, NOT an analytical prediction for
modular-capacity revivals or a new physical law.

The one-particle Hamiltonian is h=h0+P_D^T V_D P_D; h0 has uniform
nearest-neighbor hopping -1 and V_D=diag(u,v) acts on two central sites.
Writing h in clean sine modes gives h=diag(e)+B V_D B^T.

The Cayley step is:
  z(t+dt) = [I+i dt h/2]^-1 [I-i dt h/2] z(t).

Woodbury reduces each non-diagonal inverse to a 2x2 solve. The Cayley
transform is exactly unitary for Hermitian h, so occupied orbitals remain
orthonormal and restricted Gaussian-state occupations stay physical.
The global time-discretization error is O(dt^2).

This is a time-domain method based on finite-rank scattering algebra,
related to the defect T-matrix. It does not yet give a *simpler physical
theory* for revival timing or amplitudes.
"""
from __future__ import annotations
import numpy as np


class RankTwoCayley:
    def __init__(self, length: int, onsite=(1.2, 0.35), step: float = 0.2):
        if length < 4 or length % 2:
            raise ValueError("length must be even and >= 4")
        if step <= 0:
            raise ValueError("step must be positive")
        self.length = length
        self.step = float(step)
        self.onsite = tuple(float(x) for x in onsite)
        if len(self.onsite) != 2:
            raise ValueError("onsite must have two entries")
        q = np.pi*np.arange(1,length+1)/(length+1)
        self.e = -2*np.cos(q)
        j = np.arange(1,length+1)[:,None]
        self.Q = np.sqrt(2/(length+1))*np.sin(j*q)
        self.B = self.Q[[length//2-1,length//2],:].T
        self.V = np.diag(self.onsite)
        self.alpha = 0.5j*self.step
        self.dplus_inv = 1/(1+self.alpha*self.e)
        self.dminus = 1-self.alpha*self.e
        self.dinv_B = self.dplus_inv[:,None]*self.B
        self.K = self.alpha*self.V
        self.g = self.B.T@self.dinv_B
        self.small_inverse = np.linalg.inv(np.eye(2)+self.K@self.g)
        self.Qhalf = self.Q[:length//2,:]

    def to_clean_basis(self, occupied_modes: np.ndarray) -> np.ndarray:
        occupied_modes = np.asarray(occupied_modes, dtype=complex)
        if occupied_modes.shape != (self.length, self.length//2):
            raise ValueError("occupied_modes shape mismatch")
        if not np.allclose(occupied_modes.conj().T@occupied_modes,
                           np.eye(self.length//2),atol=1e-8):
            raise ValueError("occupied modes must be orthonormal")
        return self.Q.T@occupied_modes

    def one_step(self, z: np.ndarray) -> np.ndarray:
        a = self.B.T@z
        rhs = self.dminus[:,None]*z-self.B@(self.K@a)
        y = self.dplus_inv[:,None]*rhs
        return y-self.dinv_B@(self.small_inverse@(self.K@(self.B.T@y)))

    def observation(self, z: np.ndarray) -> tuple[float,float]:
        from scattering_phase import entropy_capacity
        left=self.Qhalf@z
        p=np.linalg.eigvalsh(left@left.conj().T)
        return entropy_capacity(p)

    def trajectory(self, initial_modes, times, return_unitarity_error=False):
        times=np.asarray(times,dtype=float)
        if times.ndim!=1 or len(times)<1 or np.any(times<0):
            raise ValueError("times must be a nonempty nonnegative vector")
        steps_float=times/self.step
        if not np.allclose(steps_float,np.rint(steps_float),rtol=0,atol=1e-8):
            raise ValueError("times must align with the time step")
        steps=np.rint(steps_float).astype(int)
        if np.any(np.diff(steps)<0):
            raise ValueError("times must be sorted")
        z=self.to_clean_basis(initial_modes)
        result=np.empty((len(times),2))
        error=0.0
        current=0
        for i,end in enumerate(steps):
            for _ in range(end-current):
                z=self.one_step(z)
            current=end
            result[i]=self.observation(z)
            if return_unitarity_error:
                error=max(error,float(np.linalg.norm(z.conj().T@z-np.eye(z.shape[1]))))
        return (result,error) if return_unitarity_error else result
