"""Proof checks for the first-broad-capacity-peak grid certificate."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from first_peak_certificate import certify_first_broad_grid_peak
from revival_benchmark import first_broad_capacity_peak
from scattering_phase import initial_modes
from bound_projected_chebyshev import BoundProjectedChebyshev


def test_isolated_gaussian_peak_has_robust_discrete_certificate():
    t=np.arange(0,80.01,.4)
    cap=.3+np.exp(-.5*((t-26.4)/2.8)**2)
    error=1e-6
    report=certify_first_broad_grid_peak(t,cap,48,error)
    assert report["certified"]
    assert report["peak_time"]==pytest.approx(26.4)
    assert report["local_max_margin"]>0
    assert report["prominence_margin"]>0
    rng=np.random.default_rng(212)
    for _ in range(15):
        alt=cap+rng.uniform(-error,error,size=len(t))
        true_peak=first_broad_capacity_peak(t,alt,48)[0]
        assert true_peak==report["peak_time"]


def test_insufficient_accuracy_produces_explicitly_uncertified_peak():
    t=np.arange(0,80.01,.4)
    cap=.3+np.exp(-.5*((t-26.4)/2.8)**2)
    report=certify_first_broad_grid_peak(t,cap,48,.04)
    assert not report["certified"]


def test_exactish_walk_certificate_avoids_reference_trajectory():
    n=48
    t=np.arange(0,1.8*n+.01,.4)
    f0=initial_modes(n,"staggered")
    result=BoundProjectedChebyshev(n,.775,.85,f0).trajectory(
        t,target=1e-12)
    cert=certify_first_broad_grid_peak(
        t,result["values"][:,1],n,
        result["formal_capacity_error_bound"])
    assert cert["peak_time"]==pytest.approx(26.4,abs=.21)
    assert isinstance(cert["certified"],bool)
    if cert["certified"]:
        assert cert["local_max_margin"]>0
        assert cert["prominence_margin"]>0


def test_invalid_bounds_rejected():
    t=np.arange(0,80.01,.4)
    c=np.ones(len(t))
    with pytest.raises(ValueError):
        certify_first_broad_grid_peak(t,c,48,-1)
    with pytest.raises(ValueError):
        certify_first_broad_grid_peak(t[:-1],c,48,1e-6)
