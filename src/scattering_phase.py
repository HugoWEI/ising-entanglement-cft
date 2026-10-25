"""Phase-sensitive, number-conserving free-fermion chain pilot.

Important: this tight-binding hopping model is a RELATED but DIFFERENT model
from the transverse-field Ising model in src/ising.py. It is motivated by
Capizzi & Eisler (SciPost Physics 2023), not an Ising-chain derivation.

The central defect consists of two neighboring onsite potentials, (u,v)
or its spatial reverse (v,u). Reversal gives the same transmission amplitude
for all scattering momenta but different left-reflection phases.
"""
from __future__ import annotations

import numpy as np


def tight_binding_hamiltonian(length: int, central_potentials=(0.0, 0.0)) -> np.ndarray:
    """Uniform nearest-neighbor hopping -1 and two onsite defect potentials."""
    if length < 4 or length % 2:
        raise ValueError("length must be even and at least 4")
    u, v = map(float, central_potentials)
    ham = np.diag(np.zeros(length, dtype=float))
    indexes = np.arange(length - 1)
    ham[indexes, indexes + 1] = -1.0
    ham[indexes + 1, indexes] = -1.0
    ham[length//2 - 1, length//2 - 1] = u
    ham[length//2, length//2] = v
    return ham


def left_incident_scattering(momentum: float, u: float, v: float) -> tuple[complex, complex]:
    """(reflection, transmission) amplitudes of adjacent onsite impurities.

    Infinite-chain tight binding: E=-2*cos(k), hop=-1. Sites 0 and 1
    carry potentials u and v. Incoming left wave exp(i*k*j),
    reflected wave r*exp(-i*k*j), transmitted wave t*exp(i*k*j).
    The lead/boundary conventions are identical for both orientations.
    """
    k = float(momentum)
    if not (0 < k < np.pi):
        raise ValueError("momentum must be in (0,pi)")
    energy = -2.0 * np.cos(k)
    transfer0 = np.array([[u - energy, -1.0], [1.0, 0.0]])
    transfer1 = np.array([[v - energy, -1.0], [1.0, 0.0]])
    transfer = transfer1 @ transfer0
    z = np.exp(1j*k)
    column_r = transfer @ np.array([1, z])
    column_t = -np.array([z*z, z])
    rhs = -transfer @ np.array([1, 1/z])
    reflection, transmission = np.linalg.solve(
        np.column_stack([column_r, column_t]), rhs
    )
    return complex(reflection), complex(transmission)


def initial_modes(length: int, kind="staggered", stagger=2.5) -> np.ndarray:
    """Columns are occupied orthonormal one-particle modes at half filling.

    staggered: ground state of a gapped, sublattice-staggered Hamiltonian,
    with no final central impurity (not reflection symmetric).

    mirror_fock: explicitly mirror-symmetric product Fock state, useful as
    a null control: reflection-related final Hamiltonians MUST give the
    same bipartition entropy and capacity for this preparation.
    """
    if kind == "staggered":
        ham0 = tight_binding_hamiltonian(length)
        ham0 += np.diag(stagger * (-1.0)**np.arange(length))
        _, vecs = np.linalg.eigh(ham0)
        return vecs[:, :length//2]
    if kind == "mirror_fock":
        if length % 4:
            raise ValueError("mirror_fock preparation requires length divisible by four")
        occupied = [j for j in range(length) if j % 4 in (1, 2)]
        return np.eye(length)[:, occupied]
    raise ValueError("kind must be staggered or mirror_fock")


def entropy_capacity(occupations: np.ndarray) -> tuple[float, float]:
    """Entanglement entropy and modular capacity for number-conserving Gaussian states."""
    p = np.clip(np.asarray(occupations, dtype=float), 0.0, 1.0)
    p = p[(p > 1e-12) & (p < 1 - 1e-12)]
    entropy = float(np.sum(-p*np.log(p) - (1-p)*np.log1p(-p)))
    capacity = float(np.sum(p*(1-p) * np.log((1-p)/p)**2))
    return entropy, capacity


def trajectory(length, potentials, times, occupied_modes) -> np.ndarray:
    """Columns [S_A(t), C_A(t)] across the middle cut."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or np.any(times < 0):
        raise ValueError("times must be a 1D array of nonnegative values")
    modes = np.asarray(occupied_modes, dtype=complex)
    if modes.shape != (length, length//2):
        raise ValueError("occupied_modes must be length x (length/2)")
    if not np.allclose(modes.conj().T @ modes, np.eye(length//2), atol=1e-9):
        raise ValueError("occupied modes must be orthonormal")
    ham = tight_binding_hamiltonian(length, potentials)
    energies, vectors = np.linalg.eigh(ham)
    transformed = vectors.conj().T @ modes
    results = np.empty((times.size, 2))
    for i, time in enumerate(times):
        evolved_modes = vectors @ (np.exp(-1j * energies * time)[:, None] * transformed)
        block = evolved_modes[:length//2, :]
        restricted_correlation = block @ block.conj().T
        occupations = np.linalg.eigvalsh(restricted_correlation)
        results[i, :] = entropy_capacity(occupations)
    return results


def compare_reversed_defects(length, u, v, times, preparation="staggered"):
    """Return (forward defect result, reflected defect result), same initial state."""
    modes = initial_modes(length, kind=preparation)
    return (trajectory(length, (u,v), times, modes),
            trajectory(length, (v,u), times, modes))
