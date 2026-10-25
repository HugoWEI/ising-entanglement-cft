"""Physical sanity checks, including independent spin-basis diagonalization."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ising import majorana_matrix, ground_covariance, half_chain_entropy, fit_effective_c

X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])
I = np.eye(2)


def _kron_ops(n, actions):
    out = np.array([[1.]])
    for j in range(n):
        out = np.kron(out, actions.get(j, I))
    return out


def _exact_entropy(n, t, h=1.):
    """Independent many-body Hamiltonian in the 2**n spin basis."""
    ham = np.zeros((2**n, 2**n))
    for j in range(n):
        ham -= h * _kron_ops(n, {j: Z})
    for j in range(n-1):
        strength = t if j == n//2-1 else 1.
        ham -= strength * _kron_ops(n, {j: X, j+1: X})
    _, vec = np.linalg.eigh(ham)
    psi = vec[:, 0].reshape(2**(n//2), 2**(n//2))
    p = np.linalg.svd(psi, compute_uv=False)**2
    p = p[p > 0]
    return float(-np.sum(p*np.log(p)))


def test_majorana_antisymmetry():
    a = majorana_matrix(6)
    assert np.allclose(a, -a.T)


def test_pure_covariance():
    gamma = ground_covariance(majorana_matrix(8))
    assert np.allclose(gamma @ gamma, -np.eye(16), atol=1e-10)


def test_zero_bond():
    assert abs(half_chain_entropy(12, 0.0)) < 1e-9


def test_exact_diagonalization():
    for t in [0.2, 0.5, 1.0]:
        assert abs(half_chain_entropy(6, t) - _exact_entropy(6, t)) < 1e-10


def test_clean_cft_slope():
    lengths = [32, 48, 64, 96]
    entropies = [half_chain_entropy(L, 1.0) for L in lengths]
    c_estimate, _ = fit_effective_c(lengths, entropies)
    assert abs(c_estimate - 0.5) < 0.03
