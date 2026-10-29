"""Validate exact second-phase capacity response and local peak extrapolation."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_phase import initial_modes
from bound_coherence_determinant import SingleBoundCoherence
from bound_phase_curvature import (
    capacity_scalar_second_derivative,
    second_phase_susceptibility_from_components,
    second_order_phase_trajectory,predict_individual_peak,
    predict_peak_second_order_explicit)
from capacity_peak_response import tracked_peak


def test_scalar_curvature_against_finite_difference():
    from bound_phase_peak_response import capacity_scalar_derivative
    for p in (.035,.15,.5,.81,.95):
        h=1e-5
        numeric=(capacity_scalar_derivative(p+h)-
                 capacity_scalar_derivative(p-h))/(2*h)
        analytic=capacity_scalar_second_derivative(p)
        assert np.isclose(numeric,analytic,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize("n,pair,time,theta",[
    (12,(1.2,.35),4.,.1),
    (24,(1.2,.35),14.,.6),
    (48,(1.2,.35),26.4,.2),
    (24,(.7,1.4),18.,1.1),
])
def test_exact_phase_curvature_matches_two_phase_difference(n,pair,time,theta):
    model=SingleBoundCoherence(n,pair,initial_modes(n,"staggered"))
    pieces=model.components(time,phase_kick=theta)
    analytic,near=second_phase_susceptibility_from_components(pieces,theta)
    h=.002
    exact=(model.capacity_at(time,theta+h)[0]
           -2*model.capacity_at(time,theta)[0]
           +model.capacity_at(time,theta-h)[0])/h**2
    assert np.isclose(analytic,exact,rtol=.007,atol=.007)
    assert near>=0


@pytest.mark.parametrize("n,pair",[
    (24,(1.2,.35)),(48,(1.2,.35)),(64,(1.2,.35))])
def test_individual_phase_peak_quadratic_expansion_small_offset(n,pair):
    t=np.arange(0,1.8*n+.01,.2)
    model=SingleBoundCoherence(n,pair,initial_modes(n,"staggered"))
    guess=predict_individual_peak(model,t,.08)
    assert guess is not None
    exact=tracked_peak(t,model.trajectory(t,.08)[:,0],
                       guess["reference_time"],n)
    assert exact is not None
    # It is an asymptotic expansion, not a fitted phase curve.
    # At sufficiently small phase, both models should find same branch.
    assert abs(guess["quadratic_time"]-exact[0])<.08
    assert abs(guess["quadratic_height"]-exact[1])<.008


def test_second_order_explicit_coefficients_are_finite_at_reference_peak():
    n=48
    t=np.arange(0,1.8*n+.01,.2)
    model=SingleBoundCoherence(n,(1.2,.35),initial_modes(n,"staggered"))
    data=second_order_phase_trajectory(model,t)
    law=predict_peak_second_order_explicit(t,data[:,0],data[:,1],
                                           data[:,2],n)
    assert law is not None
    assert np.isfinite(law["t1"]) and np.isfinite(law["t2"])
    assert law["reference_curvature"]<0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        capacity_scalar_second_derivative(np.array([.1]),clip=0)
    n=12
    model=SingleBoundCoherence(n,(1.2,.35),initial_modes(n,"staggered"))
    with pytest.raises(ValueError):
        second_order_phase_trajectory(model,[-1.])
    with pytest.raises(ValueError):
        second_phase_susceptibility_from_components(
            model.components(4.),np.nan)
