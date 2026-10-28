"""Independent finite-difference tests for capacity asymmetry response."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_phase import initial_modes,trajectory
from capacity_asymmetry_response import DefectAsymmetryResponse,capacity_derivative


def test_capacity_derivative_matches_scalar_finite_difference():
    from scattering_phase import entropy_capacity
    for p in (.01,.2,.5,.81,.98):
        eps=1e-6
        numerical=(entropy_capacity(np.array([p+eps]))[1]-
                   entropy_capacity(np.array([p-eps]))[1])/(2*eps)
        assert abs(numerical-capacity_derivative(np.array([p]))[0])<2e-7


def test_analytic_orbital_response_matches_finite_difference():
    n=12
    w=.775
    modes=initial_modes(n,"staggered")
    model=DefectAsymmetryResponse(n,w,modes)
    for time in (1.,2.5,5.0,8.0):
        _,chi,_=model.at_time(time)
        eps=1e-4
        cplus=trajectory(n,(w+eps/2,w-eps/2),[time],modes)[0,1]
        cminus=trajectory(n,(w-eps/2,w+eps/2),[time],modes)[0,1]
        numerical=(cplus-cminus)/(2*eps)
        assert np.isclose(chi,numerical,atol=5e-5,rtol=5e-4)


def test_odd_reversal_contrast_has_cubic_remainder_at_fixed_time():
    n=12
    w=.775
    t=4.
    modes=initial_modes(n,"staggered")
    model=DefectAsymmetryResponse(n,w,modes)
    chi=model.at_time(t)[1]
    errors=[]
    for d in (.01,.02):
        forward=trajectory(n,(w+d/2,w-d/2),[t],modes)[0,1]
        reverse=trajectory(n,(w-d/2,w+d/2),[t],modes)[0,1]
        errors.append(abs(forward-reverse-2*d*chi))
    assert errors[0]>1e-10
    assert 4*errors[0] < errors[1] < 12*errors[0]


def test_reflection_symmetric_preparation_null_response():
    n=12
    modes=initial_modes(n,"mirror_fock")
    model=DefectAsymmetryResponse(n,.775,modes)
    for t in (.2,2.,5.,10.):
        assert abs(model.at_time(t)[1])<2e-7
        a=trajectory(n,(1.0,.55),[t],modes)[0,1]
        b=trajectory(n,(.55,1.0),[t],modes)[0,1]
        assert abs(a-b)<1e-10


def test_initial_time_response_is_zero():
    modes=initial_modes(12,"staggered")
    model=DefectAsymmetryResponse(12,.775,modes)
    t0=model.at_time(0.)
    assert t0[1]==pytest.approx(0,abs=1e-12)
    assert t0[0]>=0


def test_small_asymmetry_prediction_improves_over_null():
    n=24
    w=.775
    d=.02
    modes=initial_modes(n,"staggered")
    t=np.arange(.4*n,1.4*n,.4)
    model=DefectAsymmetryResponse(n,w,modes)
    prediction=model.predict_reversal_contrast(d,t)
    forward=trajectory(n,(w+d/2,w-d/2),t,modes)[:,1]
    reverse=trajectory(n,(w-d/2,w+d/2),t,modes)[:,1]
    measured=forward-reverse
    error=np.sqrt(np.mean((prediction-measured)**2))
    null=np.sqrt(np.mean(measured**2))
    assert error < .1*null


def test_bad_inputs_rejected():
    modes=initial_modes(12,"staggered")
    with pytest.raises(ValueError):
        DefectAsymmetryResponse(13,.775,modes)
    with pytest.raises(ValueError):
        DefectAsymmetryResponse(12,.775,np.zeros((12,6)))
    model=DefectAsymmetryResponse(12,.775,modes)
    with pytest.raises(ValueError):
        model.at_time(-1)
    with pytest.raises(ValueError):
        model.response_trajectory(np.array([0.,-1.]))
    with pytest.raises(ValueError):
        capacity_derivative(np.array([.2]),clip_at=0)
