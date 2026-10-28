"""Tests of pole-exact Chebyshev/Bessel formula and analytical bounds."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_phase import initial_modes,trajectory,tight_binding_hamiltonian
from bound_projected_chebyshev import (
    BoundProjectedChebyshev,bessel_tail_bound,degree_for_tolerance,
    _capacity_error_bound,_one_positive_bound_from_clean_edge,
)
from revival_benchmark import first_broad_capacity_peak


@pytest.mark.parametrize("time",[0.1,3.,25.,80.])
def test_bessel_tail_certificate_bounds_independent_fft_coefficients(time):
    for degree in (int(2*time)+1,int(2*time)+20,int(2*time)+40):
        N=4096
        angles=2*np.pi*np.arange(N)/N
        coefficients=np.fft.fft(np.exp(-2j*time*np.cos(angles)))/N
        # The far FFT coefficients are negligible at this oversampling.
        sampled_tail=2*float(np.sum(abs(coefficients[degree+1:N//2])))
        upper=bessel_tail_bound(time,degree)
        assert sampled_tail<=upper+1e-12


@pytest.mark.parametrize("time",[0.,.3,8.,36.,110.])
def test_selected_degree_certifies_requested_error(time):
    for tol in (1e-5,1e-9,1e-12):
        M=degree_for_tolerance(time,tol)
        assert bessel_tail_bound(time,M)<=tol
        if time>0 and M>0:
            assert bessel_tail_bound(time,M-1)>tol


def test_analytic_pole_classification_and_rejection():
    for n,u,v in ((48,1.2,.35),(64,.35,1.2),(56,1.1,.1)):
        assert _one_positive_bound_from_clean_edge(n,u,v)==1
    with pytest.raises(ValueError):
        _one_positive_bound_from_clean_edge(48,0.,0.)
    with pytest.raises(ValueError):
        _one_positive_bound_from_clean_edge(48,-.2,.8)
    # Two large positive onsite potentials produce two out-of-band modes.
    with pytest.raises(ValueError):
        _one_positive_bound_from_clean_edge(48,4.,4.)


@pytest.mark.parametrize("n,w,d",[
    (12,.775,.85),(24,1.05,-.7),(48,.775,.85),
    (48,.775,-.85),(56,.6,1.),(64,.775,.85),
])
def test_walk_capacity_matches_independently_diagonalized_defect(n,w,d):
    times=np.array([0.,2.,.55*n,1.25*n])
    f0=initial_modes(n,"staggered")
    predictor=BoundProjectedChebyshev(n,w,d,f0)
    result=predictor.trajectory(times,target=1e-12)
    exact=trajectory(n,(w+d/2,w-d/2),times,f0)
    assert np.allclose(result["values"],exact,atol=3e-8,rtol=3e-8)
    assert result["max_raw_occupied_isometry_error"]<3e-8
    assert result["analytic_walk_truncation_bound"]<1e-12
    assert result["formal_capacity_error_bound"]<2e-6


def test_orbital_operator_error_respects_a_priori_truncation_bound():
    n=24
    w=.775
    d=.85
    f0=initial_modes(n,"staggered")
    model=BoundProjectedChebyshev(n,w,d,f0)
    times=np.array([4.,12.,18.])
    h=tight_binding_hamiltonian(n,(w+d/2,w-d/2))
    e,q=np.linalg.eigh(h)
    initial=q.T@f0
    for degree in (30,55,75):
        result=model.trajectory(times,degree=degree)
        assert result["max_raw_occupied_isometry_error"]>=0.
        # A finite degree below the ballistic light cone need not
        # enjoy the explicit geometric tail bound.
        if not np.isfinite(result["formal_amplitude_error_bound"]):
            continue
        moments=model.moments(degree)
        coeff,alias,_=model._coefficients(times,degree)
        orbital=(coeff@moments.reshape(degree+1,-1)).reshape(
            len(times),n,n//2)
        orbital+=np.exp(-1j*model.bound_energy*times)[:,None,None]*model.bound_initial
        for time,trial in zip(times,orbital):
            exact=q@(np.exp(-1j*e*time)[:,None]*initial)
            err=np.linalg.norm(trial-exact,2)
            assert err<=result["formal_amplitude_error_bound"]+2e-10


def test_first_peak_predicted_on_grid_without_contour_quadrature():
    n=48
    times=np.arange(0,1.8*n+.01,.4)
    f0=initial_modes(n,"staggered")
    for d,target_time in ((.85,26.4),(-.85,27.2)):
        solver=BoundProjectedChebyshev(n,.775,d,f0)
        predicted=solver.trajectory(times,target=1e-12)
        exact=trajectory(n,(.775+d/2,.775-d/2),times,f0)
        p=first_broad_capacity_peak(times,predicted["values"][:,1],n)
        q=first_broad_capacity_peak(times,exact[:,1],n)
        assert abs(p[0]-target_time)<.21
        assert p[0]==pytest.approx(q[0],abs=1e-10)
        assert p[1]==pytest.approx(q[1],abs=3e-8)


def test_end_point_capacity_bound_is_finite_without_global_lipschitz():
    assert _capacity_error_bound(1e-12,32)>0
    assert _capacity_error_bound(1e-12,32)<1e-5
    assert _capacity_error_bound(1.,32)==float("inf")


def test_bad_parameters_and_fft_alias_guard():
    n=12
    f0=initial_modes(n,"staggered")
    with pytest.raises(ValueError):
        BoundProjectedChebyshev(n,.775,0.,f0)
    with pytest.raises(ValueError):
        BoundProjectedChebyshev(n,.0,.85,f0)
    with pytest.raises(ValueError):
        degree_for_tolerance(1.,0.)
    with pytest.raises(ValueError):
        bessel_tail_bound(-1.,40)
    model=BoundProjectedChebyshev(n,.775,.85,f0)
    with pytest.raises(ValueError):
        model.trajectory([-1.])
    with pytest.raises(ValueError):
        model.trajectory([4.],degree=30,fft_points=64)
