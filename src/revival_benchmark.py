"""Exploratory first capacity-revival benchmark and a deliberately simple delay model.

This module does NOT implement a coherent many-body scattering theory.
It tests (and can falsify) the heuristic claim that a single capacity-weighted
reflection Wigner delay predicts the first broad peak displacement of C_A.

The free number-conserving hopping chain here differs from src/ising.py.
"""
from __future__ import annotations
import numpy as np
from scattering_phase import left_incident_scattering


def wigner_reflection_delay(k: float, potentials, dk: float = 1e-4) -> float:
    """d arg(r_L)/dE for E=-2 cos k, using a fixed scattering reference convention.

    Use phase *ratios* instead of subtracting individually wrapped angles.
    Invalid near an exact zero of the reflection amplitude.
    """
    k = float(k)
    if not (dk < k < np.pi - dk):
        raise ValueError("k must lie inside (dk, pi-dk)")
    r_minus, _ = left_incident_scattering(k-dk, *potentials)
    r_plus, _ = left_incident_scattering(k+dk, *potentials)
    if abs(r_minus) < 1e-9 or abs(r_plus) < 1e-9:
        raise ValueError("reflection amplitude vanishes: delay undefined")
    dphi_dk = np.angle(r_plus/r_minus)/(2*dk)
    return float(dphi_dk/(2*np.sin(k)))


def capacity_weighted_delay_difference(potentials, initial_stagger: float = 2.5):
    """Heuristic, parameter-free selector of a representative quench momentum.

    Weight w(k) = v(k)*C_occupation(k)*T(k)*[1-T(k)]. The factor T(1-T)
    is motivated by possible reflected/transmitted path interference.
    There is NO derivation that this selects the correct many-body capacity
    revival momentum, and competing weighting choices are possible.

    Returns (k_star, delta_Wigner) comparing (u,v) to the reverse (v,u).
    """
    if initial_stagger <= 0:
        raise ValueError("initial_stagger must be positive")
    u,v = map(float,potentials)
    k = np.linspace(0.05, np.pi-0.05, 1001)
    velocity = 2*np.sin(k)
    q = (2*np.cos(k))/np.sqrt(initial_stagger**2 + (2*np.cos(k))**2)
    occupation = (1+q)/2
    c_mode = occupation*(1-occupation)*np.log((1-occupation)/occupation)**2
    transmission = np.array([abs(left_incident_scattering(float(kk),u,v)[1])**2 for kk in k])
    transmission = np.clip(transmission,0,1)
    weights = velocity*c_mode*transmission*(1-transmission)
    if np.max(weights) <= 0:
        raise ValueError("no nonzero-capacity contributing momentum")
    k_star = float(k[int(np.argmax(weights))])
    delta = wigner_reflection_delay(k_star,(u,v))-wigner_reflection_delay(k_star,(v,u))
    return k_star, float(delta)


def gaussian_smooth(y, sigma_samples: float) -> np.ndarray:
    """Gaussian convolution implemented with NumPy only; reflected boundaries."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or sigma_samples <= 0:
        raise ValueError("expected 1d signal and positive sigma")
    radius = max(1,int(np.ceil(4*sigma_samples)))
    z = np.arange(-radius,radius+1,dtype=float)
    weights = np.exp(-0.5*(z/sigma_samples)**2)
    weights /= weights.sum()
    return np.convolve(np.pad(y,radius,mode="reflect"),weights,mode="valid")


def first_broad_capacity_peak(times, capacities, length, sigma_time=2.0,
                              min_prominence=0.08):
    """First broad envelope maximum after the ballistic onset.

    Explicit operational definition (NOT universal): smooth with Gaussian
    width 2 hopping-time units; consider 0.4L < time < 1.65L, and require
    >= min_prominence compared with both sides over a +/- 0.15L window.
    Returns (time, height) or (nan,nan) if no well-defined peak.
    """
    times=np.asarray(times,dtype=float)
    capacities=np.asarray(capacities,dtype=float)
    if times.ndim != 1 or capacities.shape!=times.shape or len(times)<10:
        raise ValueError("times and capacities must be matched 1d arrays")
    dt=np.diff(times)
    if np.any(dt<=0) or not np.allclose(dt,dt[0],rtol=1e-6,atol=1e-9):
        raise ValueError("time grid must be uniformly spaced and increasing")
    smooth=gaussian_smooth(capacities,sigma_time/dt[0])
    candidates=np.where((smooth[1:-1]>smooth[:-2])&
                        (smooth[1:-1]>=smooth[2:]))[0]+1
    candidates=candidates[(times[candidates]>=0.4*length)&
                          (times[candidates]<1.65*length)]
    around=max(2,int(np.round(0.15*length/dt[0])))
    for ix in candidates:
        low=max(0,ix-around)
        high=min(len(times),ix+around+1)
        prominence=min(smooth[ix]-np.min(smooth[low:ix+1]),
                       smooth[ix]-np.min(smooth[ix:high]))
        if prominence >= min_prominence:
            return float(times[ix]),float(smooth[ix])
    return float("nan"),float("nan")
