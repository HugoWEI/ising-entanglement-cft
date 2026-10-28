"""Independent peak-response validation against exact Gaussian defect evolutions."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_phase import initial_modes, trajectory
from capacity_asymmetry_response import DefectAsymmetryResponse
from capacity_peak_response import (
    smooth_signal,find_reference_peak,predict_reversed_peaks,tracked_peak)


def _system(n,w,dt=.2):
    times=np.arange(0,1.7*n+.01,dt)
    initial=initial_modes(n,"staggered",stagger=2.5)
    response=DefectAsymmetryResponse(n,w,initial).response_trajectory(times)
    return times,initial,response


@pytest.mark.parametrize("length,w",[
    (24,.25),(24,.775),(48,.25),(48,.775),(64,.25),(64,.775)])
def test_small_asymmetry_peak_shift_from_symmetric_reference(length,w):
    times,initial,response=_system(length,w)
    delta=.02
    predicted=predict_reversed_peaks(
        times,response[:,0],response[:,1],length,delta)
    assert predicted is not None
    assert predicted["reference_peak_curvature"]<0
    a=trajectory(length,(w+delta/2,w-delta/2),times,initial)[:,1]
    b=trajectory(length,(w-delta/2,w+delta/2),times,initial)[:,1]
    plus=tracked_peak(times,a,predicted["reference_peak_time"],length)
    minus=tracked_peak(times,b,predicted["reference_peak_time"],length)
    assert plus is not None and minus is not None
    actual_shift=plus[0]-minus[0]
    actual_height=plus[1]-minus[1]
    assert abs(actual_shift-predicted["predicted_time_contrast"])<.001
    assert abs(actual_height-predicted["predicted_height_contrast"])<.0002


def test_peak_contrast_expansion_is_not_valid_for_strong_defect():
    length,w,delta=48,.775,.85
    times,initial,response=_system(length,w)
    prediction=predict_reversed_peaks(
        times,response[:,0],response[:,1],length,delta)
    a=trajectory(length,(w+delta/2,w-delta/2),times,initial)[:,1]
    b=trajectory(length,(w-delta/2,w+delta/2),times,initial)[:,1]
    plus=tracked_peak(times,a,prediction["reference_peak_time"],length)
    minus=tracked_peak(times,b,prediction["reference_peak_time"],length)
    assert plus is not None and minus is not None
    actual=plus[0]-minus[0]
    assert actual<0 and prediction["predicted_time_contrast"]>0
    assert abs(actual-prediction["predicted_time_contrast"])>1.


def test_synthetic_parabolic_peak_and_linear_shift():
    t=np.arange(0,40.01,.2)
    c=2-.025*(t-17.13)**2
    chi=.3+.4*(t-17.13)
    # Gaussian smoothing only shifts the *height* of a quadratic, not
    # the interior peak curvature/center away from the padded edges.
    out=predict_reversed_peaks(t,c,chi,24,.02,minimum_curvature=1e-5)
    assert out is not None
    assert abs(out["reference_peak_time"]-17.13)<1e-5
    assert abs(out["predicted_time_contrast"]-.32)<1e-5


def test_flat_or_monotonic_signals_are_not_presented_as_revivals():
    t=np.arange(0,55.01,.2)
    assert find_reference_peak(t,t/55,24) is None
    assert predict_reversed_peaks(t,t/55,np.zeros_like(t),24,.2) is None
    assert tracked_peak(t,t/55,20,24) is None


def test_symmetric_initial_state_has_zero_asymmetry_peak_response():
    n=24
    initial=initial_modes(n,"mirror_fock")
    times=np.arange(0,1.7*n+.01,.2)
    response=DefectAsymmetryResponse(n,.775,initial).response_trajectory(times)
    prediction=predict_reversed_peaks(times,response[:,0],response[:,1],n,.05)
    if prediction is not None:
        assert abs(prediction["predicted_time_contrast"])<1e-6
        assert abs(prediction["predicted_height_contrast"])<1e-6


def test_bad_grids_rejected():
    times=np.array([0.,.2,.4,.6,.81,1.0])
    with pytest.raises(ValueError):
        smooth_signal(times,np.ones(len(times)))
    with pytest.raises(ValueError):
        smooth_signal(np.arange(10),np.ones(8))
    with pytest.raises(ValueError):
        smooth_signal(np.arange(10),np.ones(10),sigma_time=0)
