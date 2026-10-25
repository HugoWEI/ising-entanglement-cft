"""Gaussian state quench evolution and modular entropy/capacity for Ising chains.

Use 'defect' for the central bond strength; 'time' for physical evolution
time (units: hbar = 1). Quench the bulk transverse field while keeping
the interface bond fixed before and after the quench.
"""
from __future__ import annotations

import numpy as np

from ising import majorana_matrix, ground_covariance


def defect_bonds(length: int, defect: float) -> np.ndarray:
    if length < 2 or length % 2:
        raise ValueError("length must be even and at least 2")
    if defect < 0:
        raise ValueError("defect must be nonnegative")
    bonds = np.ones(length - 1, dtype=float)
    bonds[length // 2 - 1] = float(defect)
    return bonds


def entropy_and_capacity(gamma: np.ndarray, sites: int) -> tuple[float, float]:
    """Compute (S_A, C_A) from restricted Majorana covariance.

    Eigenvalues of i Gamma_A occur in +/-nu pairs. For each nonnegative
    nu, the associated fermionic probability is p=(1+nu)/2. Pure modes
    with p exactly 0 or 1 contribute zero. Units: nats, nats squared.
    """
    if sites < 1 or 2 * sites >= gamma.shape[0]:
        raise ValueError("sites must be a nonempty proper prefix")
    sub = gamma[:2 * sites, :2 * sites]
    nu = np.linalg.eigvalsh(1j * sub)[sites:]
    p = np.clip((1 + nu) / 2, 0.0, 1.0)
    interior = (p > 1e-14) & (p < 1 - 1e-14)
    q = p[interior]
    s = float(np.sum(-q * np.log(q) - (1 - q) * np.log1p(-q)))
    c = float(np.sum(q * (1 - q) * (np.log1p(-q) - np.log(q)) ** 2))
    return s, c


def quench_trajectory(
    length: int,
    defect: float,
    field_initial: float,
    field_final: float,
    times,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve an initial Gaussian ground state after a global field quench.

    H_initial has transverse field field_initial; H_final has field_final.
    Other parameters, including the defect bond, are held fixed.

    With H=(i/4) gamma^T A gamma, Majoranas rotate as exp(A_final*time);
    thus Gamma(time) = U Gamma(0) U^T. Numerical validation against direct
    spin-basis exact diagonalization is in tests/test_dynamics.py.
    """
    if field_initial <= 0 or field_final <= 0:
        raise ValueError("fields must be positive")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or np.any(times < 0):
        raise ValueError("times must be a 1D sequence of nonnegative values")
    bonds = defect_bonds(length, defect)
    gamma0 = ground_covariance(majorana_matrix(length, field_initial, bonds))
    a_final = majorana_matrix(length, field_final, bonds)
    spectrum, eigenvectors = np.linalg.eigh(1j * a_final)

    entropy, capacity = [], []
    for time in times:
        u = (eigenvectors * np.exp(-1j * spectrum * time)) @ eigenvectors.conj().T
        u = np.real_if_close(u, tol=1000).real
        gamma_t = u @ gamma0 @ u.T
        gamma_t = 0.5 * (gamma_t - gamma_t.T)
        s, c = entropy_and_capacity(gamma_t, length // 2)
        entropy.append(s)
        capacity.append(c)
    return np.asarray(entropy), np.asarray(capacity)
