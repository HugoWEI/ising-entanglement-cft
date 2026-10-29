"""Independent algebra, physicality and Gaussian-capacity budget tests."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_phase import initial_modes,tight_binding_hamiltonian,trajectory,entropy_capacity
from phase_memory_budget import (
    bound_mode_occupation,coherence_budget,pure_phase_gate,
    covariance_phase_distance,capacity_endpoint_holder_bound,
    capacity_spectral_gap_bound,capacity_difference_bounds)


@pytest.mark.parametrize("n,pair",[
    (12,(1.2,.35)),(24,(.7,1.4)),(48,(1.2,.35))])
@pytest.mark.parametrize("theta,phi",[
    (0.,.3),(.2,1.8),(0.,np.pi),(np.pi/3,5*np.pi/3)])
def test_global_coherence_norm_exact_and_conserved_in_time(n,pair,theta,phi):
    h=tight_binding_hamiltonian(n,pair)
    e,U=np.linalg.eigh(h)
    b=U[:,-1]
    f0=initial_modes(n,"staggered",stagger=2.5)
    beta=bound_mode_occupation(b,f0)
    fth,_=pure_phase_gate(b,f0,theta)
    fph,_=pure_phase_gate(b,f0,phi)
    expected=coherence_budget(beta,theta-phi)
    for t in (0.,.31*n,.83*n):
        phases=np.exp(-1j*e*t)
        a=U@(phases[:,None]*(U.conj().T@fth))
        c=U@(phases[:,None]*(U.conj().T@fph))
        G1=a@a.conj().T
        G2=c@c.conj().T
        norm=covariance_phase_distance(G1,G2)
        assert np.isclose(norm["frobenius"],
                          expected["global_frobenius_exact"],atol=2e-10)
        assert np.isclose(norm["trace"],
                          expected["global_trace_exact"],atol=2e-10)
        d=n//2
        restricted=covariance_phase_distance(G1[:d,:d],G2[:d,:d])
        alpha=float(np.sum(abs(b[:d])**2))
        bud=coherence_budget(beta,theta-phi,alpha)
        assert restricted["trace"]<=bud["subsystem_trace_upper"]+2e-9
        caps=[entropy_capacity(np.linalg.eigvalsh(g[:d,:d]))[1]
              for g in (G1,G2)]
        observed=abs(caps[0]-caps[1])
        upper=capacity_difference_bounds(beta,alpha,theta-phi,d)
        assert observed<=upper["capacity_holder_upper"]+3e-9
        eigs=np.concatenate([np.linalg.eigvalsh(g[:d,:d]) for g in (G1,G2)])
        gap=min(eigs.min(),1-eigs.max())
        if gap>1e-7:
            bound=capacity_difference_bounds(
                beta,alpha,theta-phi,d,spectral_gap=.99*gap)
            assert observed<=bound["capacity_gap_upper"]+3e-8


@pytest.mark.parametrize("beta_value",[0.,.5,1.])
def test_pure_slater_orbitals_can_saturate_coherence_budget(beta_value):
    n=12
    h=tight_binding_hamiltonian(n,(1.2,.35))
    e,U=np.linalg.eigh(h)
    b=U[:,-1]
    # Construct one occupied mode that mixes bound with a continuum mode,
    # and the other five purely in orthogonal continuum states.
    q=np.sqrt(beta_value)*b+np.sqrt(1-beta_value)*U[:,0]
    f0=np.column_stack([q,U[:,1:6]])
    assert np.allclose(f0.conj().T@f0,np.eye(6),atol=1e-12)
    beta=bound_mode_occupation(b,f0)
    assert beta==pytest.approx(beta_value,abs=1e-12)
    plus,_=pure_phase_gate(b,f0,0.)
    minus,_=pure_phase_gate(b,f0,np.pi)
    ca=plus@plus.conj().T
    cb=minus@minus.conj().T
    observed=covariance_phase_distance(ca,cb)
    expect=coherence_budget(beta,np.pi)
    assert observed["trace"]==pytest.approx(
        expect["global_trace_exact"],abs=2e-10)
    if beta_value in (0.,1.):
        assert observed["trace"]<1e-12


def test_zero_or_full_bound_occupation_is_phase_invisible():
    n=12
    h=tight_binding_hamiltonian(n,(1.2,.35))
    _,U=np.linalg.eigh(h)
    b=U[:,-1]
    occupied_cases=[U[:,:n//2],np.column_stack([U[:,:n//2-1],b])]
    for f in occupied_cases:
        beta=bound_mode_occupation(b,f)
        assert beta==pytest.approx(round(beta),abs=1e-12)
        p0,_=pure_phase_gate(b,f,0.)
        p1,_=pure_phase_gate(b,f,2.7)
        assert np.allclose(p0@p0.conj().T,
                           p1@p1.conj().T,atol=1e-12)


def test_capacity_half_holder_bound_on_scalar_grid_including_endpoints():
    p=np.concatenate(([0.],np.geomspace(1e-13,.5,100),[1.]))
    # Include close to one and adversarial differences near p=0,1.
    p=np.unique(np.concatenate((p,1-p)))
    F=lambda x: x*(1-x)*np.log((1-x)/x)**2 if 0<x<1 else 0.
    for a in p[::4]:
        for b in p[::4]:
            assert abs(F(a)-F(b))<=12*np.sqrt(abs(a-b))+1e-10


def test_bounds_inputs_and_occupations():
    with pytest.raises(ValueError):
        coherence_budget(-.1,0.)
    with pytest.raises(ValueError):
        coherence_budget(.5,np.nan)
    with pytest.raises(ValueError):
        coherence_budget(.5,.1,subsystem_weight=1.2)
    with pytest.raises(ValueError):
        capacity_endpoint_holder_bound(0,1)
    with pytest.raises(ValueError):
        capacity_spectral_gap_bound(0,1)
    with pytest.raises(ValueError):
        pure_phase_gate(np.ones(4),np.eye(4),.5)
    n=12
    b=np.linalg.eigh(tight_binding_hamiltonian(n,(1.2,.35)))[1][:,-1]
    with pytest.raises(ValueError):
        pure_phase_gate(b,initial_modes(n,"staggered"),np.inf)
