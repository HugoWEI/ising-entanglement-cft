"""No-diagonalization walk-based quadratic phase response controls."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_phase import initial_modes
from walk_phase_response import WalkPhaseResponse
from bound_coherence_determinant import SingleBoundCoherence
from bound_phase_curvature import second_order_phase_trajectory
from capacity_peak_response import tracked_peak


@pytest.mark.parametrize("n,pair",[
    (12,(1.2,.35)),(24,(.7,1.4)),(48,(1.2,.35)),
    (56,(1.1,.1))])
def test_walk_based_second_phase_susceptibility_matches_exact_reference(n,pair):
    f0=initial_modes(n,"staggered")
    predictor=WalkPhaseResponse(n,(pair[0]+pair[1])/2,
                                pair[0]-pair[1],f0)
    times=np.array([.5*n,.6*n,1.1*n])
    output=predictor.trajectory(times,target=1e-12,
                                include_curvature=True)
    exact=second_order_phase_trajectory(
        SingleBoundCoherence(n,pair,f0),times)
    assert np.allclose(output["values"][:,:3],exact[:,:3],
                       atol=2e-5,rtol=2e-4)
    assert output["max_occupied_isometry_error"]<1e-8


def test_walk_based_quadratic_peak_is_accurate_at_finite_phase():
    n=48
    pair=(1.2,.35)
    f0=initial_modes(n,"staggered")
    t=np.arange(0,1.8*n+.01,.2)
    predictor=WalkPhaseResponse(n,.775,.85,f0)
    exact=SingleBoundCoherence(n,pair,f0)
    p=predictor.quadratic_peak(t,+.3)
    assert p is not None
    actual=tracked_peak(t,exact.trajectory(t,.3)[:,0],
                        p["reference_peak_time"],n)
    assert actual is not None
    assert abs(p["quadratic_peak_time"]-actual[0])<.01
    assert abs(p["quadratic_peak_time"]-actual[0]) < (
        abs(p["linear_peak_time"]-actual[0]))
    assert p["a_priori_orbital_tail_bound"]<1e-12


def test_no_full_chain_hamiltonian_diagonalization_in_quadratic_prediction(
        monkeypatch):
    n=24
    f0=initial_modes(n,"staggered")
    original=np.linalg.eigh
    def guarded(x,*args,**kwargs):
        if np.asarray(x).shape==(n,n):
            raise AssertionError("full Hamiltonian diagonalization prohibited")
        return original(x,*args,**kwargs)
    t=np.arange(0,1.8*n+.01,.4)
    with monkeypatch.context() as patch:
        patch.setattr(np.linalg,"eigh",guarded)
        solver=WalkPhaseResponse(n,.775,.85,f0)
        result=solver.quadratic_peak(t,.1,target=1e-12)
    assert result is not None
    assert np.isfinite(result["quadratic_peak_time"])
