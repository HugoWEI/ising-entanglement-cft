"""Verify phase-response predictions from a pole-exact finite walk solver."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from walk_phase_response import WalkPhaseResponse
from bound_coherence_determinant import SingleBoundCoherence
from scattering_phase import initial_modes
from capacity_peak_response import tracked_peak


@pytest.mark.parametrize("n,pair,phase",[
    (12,(1.2,.35),0.),(24,(.7,1.4),.7),
    (48,(1.2,.35),0.),(56,(1.1,.1),.5),
])
def test_walk_phase_capacity_and_susceptibility_match_exact_spectral_reference(
        n,pair,phase):
    initial=initial_modes(n,"staggered")
    model=WalkPhaseResponse(n,(pair[0]+pair[1])/2,
                            pair[0]-pair[1],initial)
    t=np.array([0.,.6*n,1.1*n])
    result=model.trajectory(t,phase=phase,target=1e-12)
    reference=SingleBoundCoherence(n,pair,initial)
    from bound_phase_peak_response import phase_trajectory
    exact=phase_trajectory(reference,t,theta=phase)
    assert np.allclose(result["values"][:,:2],exact[:,:2],
                       atol=3e-6,rtol=1e-5)
    assert result["max_occupied_isometry_error"]<1e-8
    assert result["a_priori_orbital_tail_bound"]<1e-12


@pytest.mark.parametrize("n,pair",[
    (48,(1.2,.35)),(64,(1.2,.35)),(48,(.7,1.4)),
    (56,(1.1,.1))])
def test_scattering_walk_predicts_local_phase_peak_shift_without_phase_fits(
        n,pair):
    initial=initial_modes(n,"staggered")
    times=np.arange(0,1.8*n+.01,.2)
    model=WalkPhaseResponse(n,(pair[0]+pair[1])/2,
                            pair[0]-pair[1],initial)
    result=model.peak_shift(times,.1,target=1e-12)
    assert result is not None
    reference=SingleBoundCoherence(n,pair,initial)
    base=result["reference_peak_time"]
    plus=tracked_peak(times,reference.trajectory(times,.1)[:,0],
                      base,n)
    minus=tracked_peak(times,reference.trajectory(times,-.1)[:,0],
                       base,n)
    assert plus is not None and minus is not None
    assert abs(result["predicted_time_contrast"]-(plus[0]-minus[0]))<.006
    assert result["a_priori_orbital_tail_bound"]<1e-12


def test_phase_prediction_uses_no_full_chain_eigendecomposition(monkeypatch):
    n=24
    initial=initial_modes(n,"staggered")
    exact_eigh=np.linalg.eigh
    def prohibit_full_eigh(a,*args,**kwargs):
        if np.asarray(a).shape==(n,n):
            raise AssertionError("full-chain Hamiltonian diagonalization forbidden")
        return exact_eigh(a,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(np.linalg,"eigh",prohibit_full_eigh)
        solver=WalkPhaseResponse(n,.775,.85,initial)
        result=solver.trajectory([0.,3.,9.],target=1e-12)
    assert np.isfinite(result["values"]).all()


def test_invalid_inputs_are_rejected():
    n=12
    solver=WalkPhaseResponse(n,.775,.85,initial_modes(n,"staggered"))
    with pytest.raises(ValueError):
        solver.trajectory([-1.])
    with pytest.raises(ValueError):
        solver.trajectory([1.],phase=np.nan)
    with pytest.raises(ValueError):
        solver.trajectory([])
