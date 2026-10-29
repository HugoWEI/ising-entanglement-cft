"""Phase response, peak displacement, and determinant harmonic checks."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_phase import initial_modes,trajectory
from bound_coherence_determinant import SingleBoundCoherence
from bound_phase_peak_response import (
    capacity_scalar_derivative,phase_susceptibility_from_components,
    phase_trajectory,phase_peak_prediction,
    first_harmonic_characteristic_ratio,first_harmonic_from_cardinal_phases)
from capacity_peak_response import tracked_peak


@pytest.mark.parametrize("n,pair,time,theta",[
    (12,(1.2,.35),4.,0.),(24,(1.2,.35),12.,.3),
    (48,(1.2,.35),26.4,1.1),(24,(.7,1.4),18.,2.),
])
def test_analytic_phase_susceptibility_against_independent_finite_difference(
        n,pair,time,theta):
    model=SingleBoundCoherence(n,pair,initial_modes(n,"staggered"))
    pieces=model.components(time,phase_kick=theta)
    analytic,near=phase_susceptibility_from_components(pieces,theta)
    eps=1e-4
    plus=model.capacity_at(time,theta+eps)[0]
    minus=model.capacity_at(time,theta-eps)[0]
    numeric=(plus-minus)/(2*eps)
    assert np.isclose(analytic,numeric,atol=2e-5,rtol=1e-3)
    assert near>=0


@pytest.mark.parametrize("n,pair,theta",[
    (12,(1.2,.35),0.5),(24,(.7,1.4),1.7),
    (48,(1.2,.35),2.4)])
def test_exact_phase_characteristic_ratio_has_only_first_harmonic(n,pair,theta):
    model=SingleBoundCoherence(n,pair,initial_modes(n,"staggered"))
    for time in (0.,.55*n):
        for z in (.2+.4j,1.5+.2j):
            direct=model.direct_determinant_ratio(time,z,theta)
            small=model.entanglement_determinant_ratio(time,z,theta)
            harmonic=first_harmonic_characteristic_ratio(
                model,time,z,theta)
            assert np.allclose(harmonic,direct,atol=8e-11,rtol=8e-11)
            assert np.allclose(small,direct,atol=8e-11,rtol=8e-11)


@pytest.mark.parametrize("n,pair",[
    (24,(1.2,.35)),(48,(1.2,.35)),
    (48,(.7,1.4)),(64,(1.2,.35))])
def test_analytic_first_peak_phase_contrast_is_predictive_when_phase_small(
        n,pair):
    times=np.arange(0,1.8*n+.01,.2)
    model=SingleBoundCoherence(n,pair,initial_modes(n,"staggered"))
    prediction=phase_peak_prediction(model,times,.02)
    assert prediction is not None
    ref=prediction["reference_peak_time"]
    plus=tracked_peak(times,model.trajectory(times,.02)[:,0],ref,n)
    minus=tracked_peak(times,model.trajectory(times,-.02)[:,0],ref,n)
    assert plus is not None and minus is not None
    contrast=plus[0]-minus[0]
    assert abs(contrast-prediction["predicted_time_contrast"])<.002


def test_opposite_direction_phase_shift_between_two_system_sizes():
    for n,expected_sign in ((48,-1),(64,1)):
        model=SingleBoundCoherence(
            n,(1.2,.35),initial_modes(n,"staggered"))
        times=np.arange(0,1.8*n+.01,.2)
        p=phase_peak_prediction(model,times,.1)
        assert p is not None
        assert np.sign(p["predicted_time_contrast"])==expected_sign


def test_cardinal_interpolation_is_exact_on_pure_first_harmonic():
    t=np.linspace(0,5,12)
    theta=np.arange(16)*2*np.pi/16
    mean=2.3+.01*t
    a=.1+.02*np.cos(t)
    b=.03*np.sin(t)
    card=np.column_stack((mean+a,mean+b,mean-a,mean-b))
    predicted=first_harmonic_from_cardinal_phases(card,theta)
    exact=mean[:,None]+a[:,None]*np.cos(theta)+b[:,None]*np.sin(theta)
    assert np.allclose(predicted,exact,atol=1e-12)


def test_phase_derivative_scalar_and_invalid_input():
    p=np.array([.0832,.2,.5,.9168])
    assert np.all(np.isfinite(capacity_scalar_derivative(p)))
    with pytest.raises(ValueError):
        capacity_scalar_derivative(p,derivative_clip=0)
    model=SingleBoundCoherence(12,(1.2,.35),
                               initial_modes(12,"staggered"))
    with pytest.raises(ValueError):
        phase_trajectory(model,[-1.])
    with pytest.raises(ValueError):
        phase_susceptibility_from_components(
            model.components(3.),np.nan)
    with pytest.raises(ValueError):
        first_harmonic_from_cardinal_phases(
            np.zeros((6,3)),np.arange(12))
