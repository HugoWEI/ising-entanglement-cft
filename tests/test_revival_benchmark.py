"""Numerical and adversarial tests of the first-revival benchmark machinery."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from revival_benchmark import (wigner_reflection_delay,
                               capacity_weighted_delay_difference,
                               first_broad_capacity_peak)


def test_phase_delay_is_stable_under_difference_step():
    potentials=(1.2,0.35)
    d1=wigner_reflection_delay(1.1,potentials,dk=1e-4)
    d2=wigner_reflection_delay(1.1,potentials,dk=2e-5)
    assert abs(d1-d2)<1e-5


def test_weighted_delay_uses_scattering_only():
    k,delta=capacity_weighted_delay_difference((1.2,.35))
    assert 0<k<np.pi
    assert np.isfinite(delta)
    assert delta<0


def test_identical_reflected_pair_predicts_zero_shift():
    k,delta=capacity_weighted_delay_difference((.4,.4))
    assert np.isfinite(k)
    assert abs(delta)<1e-10


def test_peak_detector_on_synthetic_two_revivals():
    time=np.arange(0,90.01,.1)
    signal=(0.2+2*np.exp(-(time-27)**2/(2*3**2))+
            1.5*np.exp(-(time-65)**2/(2*4**2)))
    first,height=first_broad_capacity_peak(time,signal,48)
    assert abs(first-27)<.11
    assert height>1


def test_detector_reports_no_peak_on_monotonic_signal():
    time=np.arange(0,90.01,.1)
    first,height=first_broad_capacity_peak(time,time/90,48)
    assert np.isnan(first) and np.isnan(height)
