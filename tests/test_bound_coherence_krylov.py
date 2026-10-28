"""Independent projected-capacity and controlled truncation tests."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_phase import initial_modes,entropy_capacity
from bound_coherence_determinant import SingleBoundCoherence
from bound_coherence_krylov import (
    block_krylov_basis,krylov_capacity_correction,predict_capacity_from_dephased)


def test_completed_krylov_subspace_reproduces_exact_nonlinear_capacity():
    n=12
    model=SingleBoundCoherence(n,(1.2,.35),initial_modes(n,"staggered"))
    for t in (0.,2.,7.,12.):
        pieces=model.components(t)
        result,dim=predict_capacity_from_dephased(
            pieces["dephased"],pieces["X"],pieces["J"],num_steps=6)
        full=entropy_capacity(np.linalg.eigvalsh(pieces["full"]))[1]
        assert result==pytest.approx(full,abs=2e-8)
        assert dim<=n//2


def test_block_krylov_subspace_contains_interference_vectors():
    n=48
    model=SingleBoundCoherence(n,(1.2,.35),initial_modes(n,"staggered"))
    pieces=model.components(26.4)
    A,X=pieces["dephased"],pieces["X"]
    for steps in (1,2,4,8):
        Q=block_krylov_basis(A,X,steps)
        residual=np.linalg.norm(X-Q@(Q.conj().T@X))
        assert residual<2e-10
        assert Q.shape[1]<=min(n//2,2*steps)
        assert np.allclose(Q.conj().T@Q,np.eye(Q.shape[1]),atol=1e-9)


def test_eight_steps_recover_first_broad_rev_power_for_original_pair():
    from revival_benchmark import first_broad_capacity_peak
    n=48
    f0=initial_modes(n,"staggered")
    t=np.arange(0,1.8*n+.01,.2)
    for pair in ((1.2,.35),(.35,1.2)):
        model=SingleBoundCoherence(n,pair,f0)
        reference=model.trajectory(t)
        predicted=np.array([
            predict_capacity_from_dephased(
                p["dephased"],p["X"],p["J"],8)[0]
            for p in (model.components(tt) for tt in t)
        ])
        peak_exact=first_broad_capacity_peak(t,reference[:,0],n)[0]
        peak_pred=first_broad_capacity_peak(t,predicted,n)[0]
        assert peak_pred==pytest.approx(peak_exact,abs=.21)
        mask=(t>=.4*n)&(t<1.65*n)
        assert np.sqrt(np.mean((predicted[mask]-reference[mask,0])**2))<.05


def test_very_short_krylov_is_not_a_uniformly_reliable_approximation():
    n=48
    model=SingleBoundCoherence(n,(1.2,.35),initial_modes(n,"staggered"))
    t=np.arange(.4*n,1.65*n,.8)
    exact=[]
    short=[]
    for time in t:
        p=model.components(time)
        exact.append(entropy_capacity(np.linalg.eigvalsh(p["full"]))[1])
        short.append(predict_capacity_from_dephased(
            p["dephased"],p["X"],p["J"],1)[0])
    error=np.sqrt(np.mean((np.array(short)-np.array(exact))**2))
    assert error>.2


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        block_krylov_basis(np.eye(8),np.zeros((8,2)),num_steps=0)
    with pytest.raises(ValueError):
        block_krylov_basis(np.eye(8),np.zeros((8,3)))
    with pytest.raises(ValueError):
        block_krylov_basis(np.eye(8),np.zeros((8,2)))
    with pytest.raises(ValueError):
        predict_capacity_from_dephased(np.eye(8)*1.2,
                                      np.ones((8,2)),np.eye(2))
