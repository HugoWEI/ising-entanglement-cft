"""Spectral diagnostics of localized bound-state effects on modular capacity.

This is an *analysis of an exactly solvable, number-conserving tight-binding
chain*, not a new approximation to many-body scattering.

For h with bulk spectrum in [-2,2], classify eigenmodes with |E|>2+tol
as out-of-band impurity-bound modes. Decompose evolved occupied orbitals
F(t) = F_band(t)+F_bound(t).

The restricted correlation matrix decomposes EXACTLY as
G_A = G_cc + G_bb + G_cb + G_bc,
where G_xy=P_A F_x F_y^dag P_A. Cross terms need not be positive.

The *counterfactual* replaces each bound-state energy with its nearest
band edge (+2 or -2), keeping eigenvectors and all band energies fixed.
This is unitary evolution under an artificial, generally NONLOCAL
Hamiltonian. It preserves the Gaussian state and permits a controlled
sensitivity test; it is not removal of bound states from a physical defect.
"""
from __future__ import annotations

import numpy as np
from scattering_phase import tight_binding_hamiltonian, entropy_capacity


def defective_hamiltonian(length: int, onsite=(0.0, 0.0),
                          central_hopping: float = 1.0) -> np.ndarray:
    """Nearest-neighbor hopping -1 apart from the central hopping -g."""
    if central_hopping <= 0:
        raise ValueError("central_hopping must be positive")
    h = tight_binding_hamiltonian(length, onsite)
    center = length // 2
    h[center-1,center] = -central_hopping
    h[center,center-1] = -central_hopping
    return h


def spectral_partition(hamiltonian: np.ndarray, band_edge: float = 2.0,
                       tol: float = 1e-8):
    """Return energies, orthonormal eigenvectors and out-of-band mask."""
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("Hamiltonian must be square")
    if not np.allclose(hamiltonian, hamiltonian.conj().T):
        raise ValueError("Hamiltonian must be Hermitian")
    if band_edge <= 0 or tol < 0:
        raise ValueError("invalid edge/tolerance")
    e, v = np.linalg.eigh(hamiltonian)
    bound = np.abs(e) > band_edge + tol
    return e, v, bound


def bound_localization(vectors: np.ndarray, bound: np.ndarray,
                       central_radius: int = 4) -> np.ndarray:
    """Per-mode weight on the 2*central_radius central sites."""
    n = vectors.shape[0]
    if central_radius < 1 or 2*central_radius > n:
        raise ValueError("invalid central_radius")
    mid=n//2
    return np.sum(np.abs(vectors[mid-central_radius:mid+central_radius, bound])**2,
                  axis=0)


def capacity_spectral_diagnostics(hamiltonian: np.ndarray,
                                  occupied_modes: np.ndarray,
                                  times, band_edge: float = 2.0):
    """Return time-series diagnostics and bound-mode metadata.

    Columns of time series:
      entropy_full, capacity_full,
      entropy_bound_edge, capacity_bound_edge,
      fro_norm_bb, fro_norm_band_bound_cross, fro_norm_band_band.

    Entropy/capacity in the bound-edge column comes from an artificial
    *unitary* time evolution; it is not a physical bound-free impurity.

    Frobenius norms of matrix terms are not additive contributions to
    capacity. No statement that C_full-C_counterfactual is a "bound-state
    fraction" is implied.

    The initial occupied orbitals are identical in both calculations.
    """
    e, v, bound = spectral_partition(hamiltonian,band_edge)
    n=len(e)
    occupied=np.asarray(occupied_modes,dtype=complex)
    times=np.asarray(times,dtype=float)
    if n%2 or occupied.shape!=(n,n//2):
        raise ValueError("expected even chain and half-filled occupied_modes")
    if times.ndim!=1 or np.any(times<0):
        raise ValueError("times must be a vector of nonnegative values")
    if not np.allclose(occupied.conj().T@occupied,np.eye(n//2),atol=1e-9):
        raise ValueError("occupied orbitals must be orthonormal")
    amplitudes=v.conj().T@occupied
    vc=v[:,~bound]
    vb=v[:,bound]
    bc=amplitudes[~bound]
    bb=amplitudes[bound]
    e_c=e[~bound]
    e_b=e[bound]
    n_a=n//2
    out=np.empty((len(times),7))
    for i,t in enumerate(times):
        f_c=vc@(np.exp(-1j*e_c*t)[:,None]*bc)
        f_b=vb@(np.exp(-1j*e_b*t)[:,None]*bb)
        f_full=f_c+f_b
        f_edge=f_c+vb@(np.exp(-1j*np.sign(e_b)*band_edge*t)[:,None]*bb)
        a_c=f_c[:n_a,:]
        a_b=f_b[:n_a,:]
        a_full=f_full[:n_a,:]
        a_edge=f_edge[:n_a,:]
        g_cc=a_c@a_c.conj().T
        g_bb=a_b@a_b.conj().T
        cross=a_c@a_b.conj().T
        cross=cross+cross.conj().T
        g_full=a_full@a_full.conj().T
        # Spectral decomposition should agree at matrix level before
        # attempting a nonlinear functional such as modular capacity.
        if not np.allclose(g_full,g_cc+g_bb+cross,atol=5e-10):
            raise RuntimeError("spectral covariance decomposition failed")
        out[i,0:2]=entropy_capacity(np.linalg.eigvalsh(g_full))
        out[i,2:4]=entropy_capacity(np.linalg.eigvalsh(a_edge@a_edge.conj().T))
        out[i,4]=np.linalg.norm(g_bb,ord="fro")
        out[i,5]=np.linalg.norm(cross,ord="fro")
        out[i,6]=np.linalg.norm(g_cc,ord="fro")
    return {
        "values":out,
        "bound_energies":e_b,
        "central_weights":bound_localization(v,bound),
        "num_bound":int(np.sum(bound)),
    }
