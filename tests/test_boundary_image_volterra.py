"""Independent physicality, image-path and Volterra-scattering tests."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from boundary_images import (
    finite_clean_image_propagator,exact_clean_sine_propagator)
from boundary_image_volterra import volterra_trajectory
from scattering_phase import initial_modes,trajectory


def test_open_chain_image_sum_matches_independent_sine_propagator():
    for length,time in ((12,2.),(16,14.),(48,38.4),(48,67.2)):
        approximate=finite_clean_image_propagator(length,time,2)
        exact=exact_clean_sine_propagator(length,time)
        assert np.allclose(approximate,exact,atol=1e-8)


def test_boundary_returns_require_more_images_at_later_times():
    length=48
    time=67.2
    exact=exact_clean_sine_propagator(length,time)
    one=finite_clean_image_propagator(length,time,1)
    two=finite_clean_image_propagator(length,time,2)
    err_one=np.linalg.norm(one-exact)/np.linalg.norm(exact)
    err_two=np.linalg.norm(two-exact)/np.linalg.norm(exact)
    assert err_one>.1
    assert err_two<1e-10


def test_no_defect_volterra_is_exact_clean_evolution():
    n=12
    initial=initial_modes(n,"staggered")
    result=volterra_trajectory(n,(0.,0.),initial,step=.4,
                              final_time=8.,image_order=2)
    exact=trajectory(n,(0.,0.),result["times"],initial)
    assert np.allclose(result["entropy_capacity"],exact,atol=2e-10)
    assert max(result["raw_isometry_error"])<1e-10


def test_defect_volterra_converges_against_independent_exact_trajectory():
    n=12
    initial=initial_modes(n,"staggered")
    errors=[]
    for step in (.4,.2,.1):
        result=volterra_trajectory(n,(1.2,.35),initial,step=step,
                                  final_time=8.,image_order=2)
        exact=trajectory(n,(1.2,.35),result["times"],initial)
        errors.append(float(np.sqrt(np.mean(
            (result["entropy_capacity"][:,1]-exact[:,1])**2))))
        assert np.max(result["occupation_bounds"][:,1])<=1+1e-9
        assert np.min(result["occupation_bounds"][:,0])>=-1e-9
    assert errors[2]<errors[1]<errors[0]
    assert errors[2]<.02


def test_physicality_correction_is_explicit_not_silent_clipping():
    n=24
    occupied=initial_modes(n,"staggered")
    result=volterra_trajectory(n,(1.2,.35),occupied,step=.2,
                              final_time=1.6*n,image_order=1,
                              observe_every=5,physicality="polar")
    assert np.max(result["raw_isometry_error"])>.1
    assert np.max(result["occupation_bounds"][:,1])<1+1e-9
    assert np.min(result["occupation_bounds"][:,0])>-1e-9
    uncorrected=volterra_trajectory(n,(1.2,.35),occupied,step=.2,
                                   final_time=1.6*n,image_order=1,
                                   observe_every=5,physicality="none")
    assert np.isnan(uncorrected["entropy_capacity"]).any()


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        finite_clean_image_propagator(12,8.,max_image_order=6,fft_points=32)
    with pytest.raises(ValueError):
        finite_clean_image_propagator(12,-1)
    with pytest.raises(ValueError):
        volterra_trajectory(11,(1.2,.35),np.eye(11),step=.2)
