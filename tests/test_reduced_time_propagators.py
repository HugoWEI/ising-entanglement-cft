"""Independent validation of reduced rank-two unitary time propagators."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from rank2_cayley import RankTwoCayley
from clean_defect_strang import CleanDefectStrang
from scattering_phase import initial_modes, tight_binding_hamiltonian, trajectory


def test_cayley_one_step_matches_dense_implicit_midpoint():
    n=12
    dt=.2
    pair=(1.2,.35)
    f0=initial_modes(n,"staggered")
    model=RankTwoCayley(n,pair,dt)
    h=tight_binding_hamiltonian(n,pair)
    dense=np.linalg.solve(np.eye(n)+.5j*dt*h,
                          (np.eye(n)-.5j*dt*h)@f0)
    z=model.one_step(model.to_clean_basis(f0))
    assert np.allclose(model.Q@z,dense,atol=2e-12)


def test_strang_step_matches_independent_dense_exponentials():
    n=12
    dt=.2
    pair=(1.2,.35)
    f0=initial_modes(n,"staggered")
    model=CleanDefectStrang(n,pair,dt)
    h0=tight_binding_hamiltonian(n,(0.,0.))
    e,q=np.linalg.eigh(h0)
    u0=q@(np.exp(-.5j*dt*e)[:,None]*q.T)
    kick=np.eye(n,dtype=complex)
    kick[n//2-1,n//2-1]=np.exp(-1j*dt*pair[0])
    kick[n//2,n//2]=np.exp(-1j*dt*pair[1])
    dense=u0@kick@u0@f0
    z=model.one_step(model.to_clean_basis(f0))
    assert np.allclose(model.Q@z,dense,atol=2e-12)


def test_both_methods_preserve_orthonormal_occupied_orbitals():
    n=16
    f0=initial_modes(n,"staggered")
    times=np.arange(0,10.1,.1)
    for cls in (RankTwoCayley,CleanDefectStrang):
        traj,unitarity=cls(n,(1.2,.35),.1).trajectory(
            f0,times,return_unitarity_error=True)
        assert np.isfinite(traj).all()
        assert unitarity<2e-10
        assert np.all(traj>=-1e-10)


def test_strang_exact_when_defect_is_absent():
    n=12
    dt=.4
    f0=initial_modes(n,"staggered")
    times=np.arange(0,10.01,dt)
    model=CleanDefectStrang(n,(0.,0.),dt)
    approx=model.trajectory(f0,times)
    exact=trajectory(n,(0.,0.),times,f0)
    assert np.allclose(approx,exact,atol=4e-10)


def test_strang_refinement_matches_exact_dynamics():
    n=12
    f0=initial_modes(n,"staggered")
    target_time=8.0
    exact=trajectory(n,(1.2,.35),np.array([target_time]),f0)[0]
    errors=[]
    for dt in (.4,.2,.1):
        approx=CleanDefectStrang(n,(1.2,.35),dt).trajectory(
            f0,[target_time])[0]
        errors.append(np.linalg.norm(approx-exact))
    assert errors[2]<errors[1]<errors[0]
    assert errors[1]/errors[2]>3.0


def test_invalid_time_inputs_are_rejected():
    import pytest
    model=RankTwoCayley(12,(1.2,.35),step=.2)
    f0=initial_modes(12,"staggered")
    for times in ([.1],[.4,.2],[-.2],[]):
        with pytest.raises(ValueError):
            model.trajectory(f0,times)
    with pytest.raises(ValueError):
        RankTwoCayley(5)
    with pytest.raises(ValueError):
        CleanDefectStrang(12,step=0)


def test_approximation_at_zero_time_exact_initial_entanglement():
    n=16
    f0=initial_modes(n,"staggered")
    reference=trajectory(n,(1.2,.35),[0.0],f0)
    for cls in (RankTwoCayley,CleanDefectStrang):
        approx=cls(n,(1.2,.35),.2).trajectory(f0,[0.])
        assert np.allclose(approx,reference,atol=1e-12)
