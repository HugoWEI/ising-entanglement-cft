"""Independent tests of rank-two bound-band capacity updates."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_phase import initial_modes,trajectory,entropy_capacity
from bound_coherence_determinant import SingleBoundCoherence


@pytest.mark.parametrize("n,pair",[
    (12,(1.2,.35)),(24,(.7,1.4)),(48,(1.2,.35)),
    (48,(.5,.2)),(64,(1.2,.35)),
])
def test_full_capacity_matches_independent_exact_gaussian_evolution(n,pair):
    f0=initial_modes(n,"staggered")
    t=np.array([0.,1.,5.,.6*n,1.1*n])
    model=SingleBoundCoherence(n,pair,f0)
    out=model.trajectory(t)
    exact=trajectory(n,pair,t,f0)[:,1]
    assert np.allclose(out[:,0],exact,atol=2e-9)
    assert model.bound_energy>2.


@pytest.mark.parametrize("n,pair",[
    (12,(1.2,.35)),(24,(.7,1.4)),(48,(.5,.2)),
])
def test_two_by_two_characteristic_determinant_identity(n,pair):
    model=SingleBoundCoherence(n,pair,initial_modes(n,"staggered"))
    for time in (0.,2.,.8*n):
        for lam in (.2+.4j,1.7+.12j,-.6+.3j):
            via_small=model.entanglement_determinant_ratio(time,lam)
            via_full=model.direct_determinant_ratio(time,lam)
            assert np.allclose(via_small,via_full,atol=5e-11,rtol=5e-11)


def test_bound_band_dephasing_is_a_valid_gaussian_covariance():
    n=48
    model=SingleBoundCoherence(n,(1.2,.35),initial_modes(n,"staggered"))
    for t in (0.,4.,26.4,48.,70.):
        comp=model.components(t)
        for key in ("full","dephased","band_only"):
            p=np.linalg.eigvalsh(comp[key])
            assert min(p)>=-2e-10
            assert max(p)<=1+2e-10
        assert np.allclose(comp["full"],comp["dephased"]+comp["cross"],
                           atol=3e-12)


def test_dephased_covariance_is_average_over_pure_relative_phases():
    n=24
    model=SingleBoundCoherence(n,(1.2,.35),initial_modes(n,"staggered"))
    pieces=model.components(13.)
    x,y=pieces["X"].T
    avg=np.zeros_like(pieces["full"])
    for theta in (0.,.5*np.pi,np.pi,1.5*np.pi):
        coh=(np.exp(1j*theta)*np.outer(x,y.conj())
             +np.exp(-1j*theta)*np.outer(y,x.conj()))
        state=pieces["dephased"]+coh
        p=np.linalg.eigvalsh(state)
        assert p.min()>-1e-9 and p.max()<1+1e-9
        avg+=state/4
    assert np.allclose(avg,pieces["dephased"],atol=3e-12)


def test_original_reversal_peak_shift_disappears_on_dephasing():
    n=48
    times=np.arange(0,1.8*n+.01,.2)
    f0=initial_modes(n,"staggered")
    from revival_benchmark import first_broad_capacity_peak
    a=SingleBoundCoherence(n,(1.2,.35),f0).trajectory(times)
    b=SingleBoundCoherence(n,(.35,1.2),f0).trajectory(times)
    t_full_a=first_broad_capacity_peak(times,a[:,0],n)[0]
    t_full_b=first_broad_capacity_peak(times,b[:,0],n)[0]
    t_deph_a=first_broad_capacity_peak(times,a[:,1],n)[0]
    t_deph_b=first_broad_capacity_peak(times,b[:,1],n)[0]
    assert t_full_a==pytest.approx(26.4,abs=.01)
    assert t_full_b==pytest.approx(27.4,abs=.01)
    assert t_deph_a==pytest.approx(27.8,abs=.01)
    assert t_deph_b==pytest.approx(27.8,abs=.01)


def test_no_bound_mode_is_not_mislabeled_as_single_bound_problem():
    n=12
    with pytest.raises(ValueError):
        SingleBoundCoherence(n,(0.,0.),initial_modes(n,"staggered"))


def test_bad_input_rejected():
    n=12
    with pytest.raises(ValueError):
        SingleBoundCoherence(13,(1.2,.35),np.eye(13)[:,:6])
    with pytest.raises(ValueError):
        SingleBoundCoherence(n,(1.2,.35),np.zeros((n,n//2)))
    model=SingleBoundCoherence(n,(1.2,.35),initial_modes(n,"staggered"))
    with pytest.raises(ValueError):
        model.components(-1)
    with pytest.raises(ValueError):
        model.trajectory([-1])
