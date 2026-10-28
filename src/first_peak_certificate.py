"""A posteriori first-BROAD-peak certificate from an a priori error bound.

Suppose a predicted sampled capacity sequence c_hat[j] obeys
  |c_true[j]-c_hat[j]| <= eta
for every sampled time. A *fixed Gaussian smoothing with reflection
padding* is a positive, row-stochastic linear operator; therefore its
sampled smoothed output also differs by at most eta.

The repository's operational peak is the FIRST local maximum
in [0.4L,1.65L] whose prominence over +/-0.15L exceeds 0.08.

One can certify the predicted first-peak GRID INDEX as unchanged
under ANY unknown exact signal satisfying the error bound if:
  (i) the predicted peak beats both neighbors by more than 2eta;
  (ii) its prominence exceeds the threshold by more than 2eta;
  (iii) every earlier grid sample is either provably NOT a local
        maximum (at least one neighbor exceeds it by >2eta),
        OR its maximum POSSIBLE prominence is below threshold
        (predicted prominence +2eta < threshold).

This proves stability of the operational grid index conditional
on the stated uniform error bound. It does NOT prove a
continuous-time peak, sub-grid timing, uniqueness, an error estimate
for numerical roundoff or validity of the physical model.
"""
from __future__ import annotations
import numpy as np
from revival_benchmark import gaussian_smooth, first_broad_capacity_peak


def certify_first_broad_grid_peak(times,approximate_capacity,length,
                                  uniform_capacity_error,
                                  sigma_time=2.0,min_prominence=.08):
    times=np.asarray(times,float)
    cap=np.asarray(approximate_capacity,float)
    eta=float(uniform_capacity_error)
    if (times.ndim!=1 or cap.shape!=times.shape or len(times)<10 or
        np.any(~np.isfinite(times)) or np.any(~np.isfinite(cap)) or
        np.any(np.diff(times)<=0) or not np.allclose(
            np.diff(times),np.diff(times)[0],atol=1e-9,rtol=1e-6) or
        not np.isfinite(eta) or eta<0 or sigma_time<=0 or
        min_prominence<=0 or length<4):
        raise ValueError("invalid sampled signal or error bound")
    dt=float(times[1]-times[0])
    peak,_=first_broad_capacity_peak(times,cap,length,
                        sigma_time=sigma_time,min_prominence=min_prominence)
    if not np.isfinite(peak):
        return {"certified":False,"peak_time":float("nan"),
                "reason":"no operational first broad peak"}
    S=gaussian_smooth(cap,sigma_time/dt)
    i=int(np.argmin(abs(times-peak)))
    radius=max(2,int(round(.15*length/dt)))

    def prominence(j):
        lo=max(0,j-radius)
        hi=min(len(times),j+radius+1)
        return float(min(S[j]-np.min(S[lo:j+1]),
                         S[j]-np.min(S[j:hi])))

    neighbor_gap=float(min(S[i]-S[i-1],S[i]-S[i+1]))
    prominence_margin=prominence(i)-min_prominence
    if neighbor_gap<=2*eta:
        return {"certified":False,"peak_time":peak,
                "reason":"peak may cease to be a discrete local maximum",
                "local_max_margin":neighbor_gap-2*eta}
    if prominence_margin<=2*eta:
        return {"certified":False,"peak_time":peak,
                "reason":"peak may cease to satisfy prominence",
                "prominence_margin":prominence_margin-2*eta}
    for j in range(1,i):
        if not (.4*length<=times[j]<1.65*length):
            continue
        # A true local max at j is impossible if either neighbor
        # already exceeds it by >2eta.
        if max(S[j-1]-S[j],S[j+1]-S[j])>2*eta:
            continue
        if prominence(j)+2*eta<min_prominence:
            continue
        return {
            "certified":False,
            "peak_time":peak,
            "reason":"an earlier local maximum cannot be ruled out",
            "ambiguous_earlier_time":float(times[j]),
        }
    return {
        "certified":True,"peak_time":peak,
        "reason":"first prominent discrete peak index unchanged under uniform error",
        "local_max_margin":neighbor_gap-2*eta,
        "prominence_margin":prominence_margin-2*eta,
    }
