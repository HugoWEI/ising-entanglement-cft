"""Exactly unitary, second-order clean-propagation/defect-kick propagator.

This is a *numerical splitting method*, NOT a closed-form physical law
for capacity revival times or amplitudes.

In the clean sine basis h0=diag(e), B=Q^T[:,defect_sites],
V_D=diag(u,v), with B^T B = I_2. The time step is

U_split(dt) = exp(-i h0 dt/2) exp(-i B V_D B^T dt)
              exp(-i h0 dt/2).

For the rank-two defect,
exp(-i B V_D B^T dt)
= I + B[exp(-i V_D dt)-I]B^T,

so each step involves clean diagonal phases and a 2x2 defect kick.
It preserves occupied-mode orthonormality (hence physical Gaussian
entanglement spectra) exactly up to roundoff and treats *clean band
dispersion exactly*. Global splitting error is O(dt^2), governed by
commutators between h0 and the defect.

The defect's bound-state contribution is included implicitly in the
finite-step dynamics; this does not explicitly isolate bound poles.
"""
from __future__ import annotations
import numpy as np
from rank2_cayley import RankTwoCayley


class CleanDefectStrang(RankTwoCayley):
    def __init__(self, length: int, onsite=(1.2,.35), step: float=.2):
        super().__init__(length, onsite, step)
        self.clean_half=np.exp(-.5j*step*self.e)
        self.defect_delta=np.exp(-1j*step*np.asarray(self.onsite))-1

    def one_step(self, z: np.ndarray) -> np.ndarray:
        half=self.clean_half[:,None]*z
        kicked=half+self.B@(self.defect_delta[:,None]*(self.B.T@half))
        return self.clean_half[:,None]*kicked
