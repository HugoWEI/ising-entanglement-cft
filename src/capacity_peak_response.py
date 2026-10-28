"""Predict isolated capacity-peak time/height changes from analytic susceptibility.

Given the *symmetric* reference capacity C0(t) and asymmetry
susceptibility chi(t) = d C_delta(t)/d(delta)|0, consider an isolated
nondegenerate local maximum t0 with C0'(t0)=0, C0''(t0)<0.

Implicit differentiation of C_delta'(t_peak(delta))=0 yields
  t_peak(delta)=t0 - delta*chi'(t0)/C0''(t0)+O(delta^2).

For two spatially reversed defects delta and -delta:
  t_peak(+) - t_peak(-) =
      -2*delta*chi'(t0)/C0''(t0)+O(delta^3),
  C_peak(+) - C_peak(-)=2*delta*chi(t0)+O(delta^3).

The odd-difference cubic remainder assumes sufficient smoothness.
The expression also applies after the same *fixed linear smoothing*
is applied to both the capacity and the susceptibility.

This is a standard implicit-function/linear-response calculation:
a useful predictive benchmark, NOT an original universal law and
not a reliable strong-asymmetry formula.

Numerical maxima use a three-sample local quadratic interpolation.
A missing nearby maximum is reported as undefined rather than
replaced with a window boundary. A flat peak is excluded.
"""
from __future__ import annotations
import numpy as np
from revival_benchmark import gaussian_smooth


def _check_grid(times,signal):
    times=np.asarray(times,float)
    signal=np.asarray(signal,float)
    if times.ndim!=1 or signal.shape!=times.shape or len(times)<5:
        raise ValueError("expected >=5 matched 1d observations")
    dt=np.diff(times)
    if np.any(dt<=0) or not np.allclose(dt,dt[0],atol=1e-10,rtol=1e-7):
        raise ValueError("times must form uniform ascending grid")
    if not np.isfinite(signal).all():
        raise ValueError("nonfinite observations")
    return times,signal,float(dt[0])


def smooth_signal(times,signal,sigma_time: float=2.0):
    times,signal,dt=_check_grid(times,signal)
    if sigma_time<=0:
        raise ValueError("smoothing width must be positive")
    return gaussian_smooth(signal,sigma_time/dt)


def _parabolic_maximum(times,y,index):
    """Return (t*,height,actual second derivative, fitted grid index)."""
    dt=float(times[1]-times[0])
    i=int(index)
    if not (1<=i<len(times)-1):
        return None
    minus,center,plus=map(float,y[i-1:i+2])
    derivative=(plus-minus)/(2*dt)
    curvature=(plus-2*center+minus)/dt**2
    if curvature>=0:
        return None
    displacement=-derivative/curvature
    if abs(displacement)>dt:
        return None
    return (float(times[i]+displacement),
            float(center+derivative*displacement+.5*curvature*displacement**2),
            float(curvature), i)


def find_reference_peak(times,values,length,sigma_time=2.0,
                        minimum_prominence=.08,minimum_curvature=1e-4):
    """First prominent smoothed local maximum in [0.4L,1.65L].

    Prominence assessed on both sides over +/-0.15L.
    This is an operational *first broad post-ballistic maximum*, not
    a universal first-revival definition.
    """
    times,values,dt=_check_grid(times,values)
    sm=smooth_signal(times,values,sigma_time)
    reach=max(2,int(round(.15*length/dt)))
    for i in range(1,len(times)-1):
        if not (.4*length<=times[i]<1.65*length):
            continue
        if not (sm[i]>sm[i-1] and sm[i]>=sm[i+1]):
            continue
        lo=max(0,i-reach)
        hi=min(len(times),i+reach+1)
        prominence=min(sm[i]-np.min(sm[lo:i+1]),
                       sm[i]-np.min(sm[i:hi]))
        candidate=_parabolic_maximum(times,sm,i)
        if candidate is not None and prominence>=minimum_prominence and (
                -candidate[2]>=minimum_curvature):
            return candidate
    return None


def predict_reversed_peaks(times,reference_capacity,susceptibility,
                           length,delta,sigma_time=2.0,
                           minimum_curvature=1e-4):
    """Use ONLY C0, chi from the symmetric reference to predict peaks."""
    times,capacity,dt=_check_grid(times,reference_capacity)
    _,chi,_=_check_grid(times,susceptibility)
    ref=find_reference_peak(times,capacity,length,sigma_time,
                            minimum_curvature=minimum_curvature)
    if ref is None:
        return None
    t0,c0,curvature,i=ref
    smchi=smooth_signal(times,chi,sigma_time)
    offset=t0-times[i]
    chi_prime_center=(smchi[i+1]-smchi[i-1])/(2*dt)
    chi_curvature=(smchi[i+1]-2*smchi[i]+smchi[i-1])/dt**2
    chi_at_peak=(smchi[i]+chi_prime_center*offset
                 +.5*chi_curvature*offset**2)
    chi_prime=chi_prime_center+chi_curvature*offset
    d=float(delta)
    shift_one=-d*chi_prime/curvature
    return {
        "reference_peak_time":t0,
        "reference_peak_height":c0,
        "reference_peak_curvature":curvature,
        "susceptibility_at_peak":float(chi_at_peak),
        "susceptibility_slope_at_peak":float(chi_prime),
        "predicted_plus_time":t0+shift_one,
        "predicted_minus_time":t0-shift_one,
        "predicted_time_contrast":2*shift_one,
        "predicted_plus_height":c0+d*chi_at_peak,
        "predicted_minus_height":c0-d*chi_at_peak,
        "predicted_height_contrast":2*d*chi_at_peak,
    }


def tracked_peak(times,values,reference_time,length,
                 sigma_time=2.0,window_radius=3.0):
    """Nearest smoothed local maximum to a reference branch, not global max.

    Returns None when no local maximum lies inside the tracking window.
    Do not infer a peak from an artificial search-window boundary.
    """
    times,values,_=_check_grid(times,values)
    sm=smooth_signal(times,values,sigma_time)
    candidates=[]
    for i in range(1,len(times)-1):
        if abs(times[i]-reference_time)>window_radius:
            continue
        if sm[i]>sm[i-1] and sm[i]>=sm[i+1]:
            result=_parabolic_maximum(times,sm,i)
            if result is not None and abs(result[0]-reference_time)<=window_radius:
                candidates.append(result)
    if not candidates:
        return None
    return min(candidates,key=lambda r:abs(r[0]-reference_time))
