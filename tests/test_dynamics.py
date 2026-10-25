"""Independent checks of capacity and real-time Majorana evolution."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamics import defect_bonds, entropy_and_capacity, quench_trajectory
from ising import majorana_matrix, ground_covariance

X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])
I = np.eye(2)


def operators(length, actions):
    product = np.array([[1.]])
    for j in range(length):
        product = np.kron(product, actions.get(j, I))
    return product


def spin_hamiltonian(length, field, defect):
    h = np.zeros((2 ** length, 2 ** length))
    for j in range(length):
        h -= field * operators(length, {j: Z})
    for j in range(length - 1):
        coupling = defect if j == length // 2 - 1 else 1.
        h -= coupling * operators(length, {j: X, j + 1: X})
    return h


def exact_spin_entropy_capacity(length, defect, initial_field, final_field, time):
    _, initial_vectors = np.linalg.eigh(spin_hamiltonian(length, initial_field, defect))
    initial_state = initial_vectors[:, 0]
    e, vectors = np.linalg.eigh(spin_hamiltonian(length, final_field, defect))
    evolved = vectors @ (np.exp(-1j * e * time) * (vectors.conj().T @ initial_state))
    schmidt = np.linalg.svd(
        evolved.reshape(2 ** (length // 2), 2 ** (length // 2)),
        compute_uv=False,
    )
    probabilities = schmidt ** 2
    probabilities = probabilities[probabilities > 1e-14]
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    modular_energy = -np.log(probabilities)
    capacity = float(np.sum(probabilities * modular_energy ** 2) - entropy ** 2)
    return entropy, capacity


def test_vs_independent_exact_diagonalization():
    for defect in (0.0, 0.3, 1.0):
        for time in (0., 0.15, 0.7, 1.4):
            entropy, capacity = quench_trajectory(4, defect, 2., 1., [time])
            exact_s, exact_c = exact_spin_entropy_capacity(4, defect, 2., 1., time)
            assert abs(entropy[0] - exact_s) < 2e-10
            assert abs(capacity[0] - exact_c) < 2e-10


def test_disconnected_halves_stay_unentangled():
    entropy, capacity = quench_trajectory(8, 0., 2., 1., np.linspace(0., 4., 8))
    assert np.max(np.abs(entropy)) < 1e-8
    assert np.max(np.abs(capacity)) < 1e-8


def test_no_quench_is_stationary():
    entropy, capacity = quench_trajectory(8, 0.7, 1., 1., np.linspace(0., 4., 8))
    assert np.ptp(entropy) < 1e-10
    assert np.ptp(capacity) < 1e-10


def test_majorana_ground_covariance_is_pure():
    matrix = majorana_matrix(6, 1., defect_bonds(6, 0.5))
    gamma = ground_covariance(matrix)
    assert np.allclose(gamma @ gamma, -np.eye(12), atol=1e-11)


def test_maximally_mixed_subsystem_has_zero_capacity():
    # A pair of maximally entangled fermionic modes has S=2 ln2, C=0.
    gamma = np.zeros((8, 8))
    for j in range(2):
        gamma[2*j, 2*j+4] = 1.
        gamma[2*j+4, 2*j] = -1.
    entropy, capacity = entropy_and_capacity(gamma, 2)
    assert abs(entropy - 2*np.log(2)) < 1e-12
    assert capacity < 1e-12
