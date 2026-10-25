"""Ground-state entanglement of an open transverse-field Ising chain.

H = -sum_j J_j X_j X_{j+1} - h sum_j Z_j.
Under the Jordan-Wigner map H=(i/4) gamma^T A gamma, where A is real
antisymmetric. The restricted ground-state Majorana covariance gives
the von Neumann entropy of a spin block.
"""
from __future__ import annotations
import numpy as np


def majorana_matrix(length: int, field: float = 1.0, bonds: np.ndarray | None = None) -> np.ndarray:
    """Return the 2L-by-2L antisymmetric Majorana coupling matrix."""
    if length < 2:
        raise ValueError("length must be >= 2")
    if field <= 0:
        raise ValueError("field must be positive")
    if bonds is None:
        bonds = np.ones(length - 1, dtype=float)
    bonds = np.asarray(bonds, dtype=float)
    if bonds.shape != (length - 1,):
        raise ValueError("expected length-1 bond strengths")
    a = np.zeros((2 * length, 2 * length), dtype=float)
    for j in range(length):
        a[2*j, 2*j+1] = 2.0 * field
        a[2*j+1, 2*j] = -2.0 * field
    for j, strength in enumerate(bonds):
        a[2*j+1, 2*j+2] = 2.0 * strength
        a[2*j+2, 2*j+1] = -2.0 * strength
    return a


def ground_covariance(a: np.ndarray, tol: float = 1e-11) -> np.ndarray:
    """Return Gamma=-A/sqrt(-A^2) for a finite, nonzero-gap ground state."""
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] % 2:
        raise ValueError("A must be square and even dimensional")
    if not np.allclose(a, -a.T, atol=1e-10):
        raise ValueError("A must be antisymmetric")
    e, v = np.linalg.eigh(1j * a)
    if np.min(np.abs(e)) < tol:
        raise ValueError("Zero-energy mode: ground-state parity may be ambiguous")
    sign_operator = (v * np.sign(e)) @ v.conj().T
    gamma = np.real(1j * sign_operator)
    return 0.5 * (gamma - gamma.T)


def entropy_from_covariance(gamma: np.ndarray, sites: int) -> float:
    """Von Neumann entropy (natural-log units) of the first 'sites' spins."""
    if sites < 1 or 2 * sites >= gamma.shape[0]:
        raise ValueError("subsystem must be a nontrivial prefix of the chain")
    sub = gamma[:2*sites, :2*sites]
    eig = np.linalg.eigvalsh(1j * sub)
    positive = np.sort(eig)[sites:]
    p = np.clip((1 + positive) / 2, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p*np.log(p), 0) - np.where(p < 1, (1-p)*np.log1p(-p), 0)
    return float(np.sum(h))


def half_chain_entropy(length: int, defect: float = 1.0, field: float = 1.0) -> float:
    """Entropy across the central bond in an even open chain."""
    if length % 2:
        raise ValueError("length must be even")
    if defect < 0:
        raise ValueError("defect must be nonnegative")
    bonds = np.ones(length - 1, dtype=float)
    bonds[length//2 - 1] = defect
    a = majorana_matrix(length, field, bonds)
    return entropy_from_covariance(ground_covariance(a), length//2)


def fit_effective_c(lengths, entropies):
    """Fit S(L/2) = (c_eff / 6) ln(L) + b (finite-size estimate only)."""
    lengths = np.asarray(lengths, dtype=float)
    entropies = np.asarray(entropies, dtype=float)
    if lengths.shape != entropies.shape or lengths.size < 2:
        raise ValueError("need at least two matched lengths and entropies")
    slope, intercept = np.polyfit(np.log(lengths), entropies, 1)
    return float(6*slope), float(intercept)
