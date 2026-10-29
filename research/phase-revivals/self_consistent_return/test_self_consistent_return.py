"""Independent tests for image accounting and Volterra feedback. Run pytest -q."""
import sys
from pathlib import Path
import numpy as np
import pytest
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
sys.path.insert(0,str((P.parent/'certificate'/'src') if (P.parent/'certificate'/'src').exists() else Path('/mnt/data/airy_certificate_work/src')))
from self_consistent_return import kernels,trapezoid_solve,smoothed_peak,cap_curve
from phase_revival_research import hopping
import volterra_scattering_revival as vol


def setup(L=12,tmax=7,dt=.2):
    t=np.arange(0,tmax+dt/4,dt)
    _,q=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)))
    return t,q[:,:L//2]


def test_bessel_shell_identity():
    for L in (12,20):
        t,Q=setup(L)
        k=kernels(L,t,Q,'full_bessel')
        target=vol.image_bessel_kernel(L,t,1)
        assert np.max(np.abs(k['clean']-target))<5e-14
        assert np.max(np.abs(k['baseF']-target@Q))<5e-14


def test_volterra_solver_independent_implementation():
    t,Q=setup(L=12,tmax=8,dt=.1)
    k=kernels(12,t,Q,'full_bessel')
    F,Y=trapezoid_solve(k,t,Q,(1.2,.35))
    reference,Yr=vol.defect_response(12,t,1.2,.35,Q,1,return_full=True,return_defect=True)
    assert np.max(np.abs(F-reference))<2e-13
    assert np.max(np.abs(Y-Yr))<2e-13


def test_selfconsistent_wrap_changes_impurity_drive():
    t,Q=setup(L=12,tmax=14,dt=.1)
    base=kernels(12,t,Q,'base')
    wrapped=kernels(12,t,Q,'bessel')
    F0,Y0=trapezoid_solve(base,t,Q,(.9,.3))
    F1,Y1=trapezoid_solve(wrapped,t,Q,(.9,.3))
    # When the two wraps are present they affect both the state and the two-site response.
    assert np.max(np.abs(Y1-Y0))>1e-12
    assert np.max(np.abs(F1-(F0+wrapped['baseF']-base['baseF'])))>1e-9


def test_small_window_airy_bessel_agreement():
    t,Q=setup(L=24,tmax=16,dt=.1)
    a=kernels(24,t,Q,'airy');b=kernels(24,t,Q,'bessel')
    Fa,_=trapezoid_solve(a,t,Q,(1.2,.35));Fb,_=trapezoid_solve(b,t,Q,(1.2,.35))
    # Near turning region, leading Olver term has a small but nonzero discrepancy.
    assert np.max(np.abs(Fa-Fb))<1e-3
    assert np.linalg.norm(Fa[0]-Q)<1e-14
    assert np.isfinite(cap_curve(Fa)).all()
