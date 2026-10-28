"""Compare 2x2 T-matrix and pole eigenvectors to independent dense methods."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from defect_tmatrix import TwoSiteDefectResolvent
from scattering_phase import tight_binding_hamiltonian


def test_exact_green_matches_direct_inverse():
    for n in (12,48):
        for u,v in ((1.2,.35),(.7,1.4),(.5,.2),(-.4,-.7)):
            model=TwoSiteDefectResolvent(n,(u,v))
            h=tight_binding_hamiltonian(n,(u,v))
            for z in (.35+.8j,2.8+.5j):
                exact=np.linalg.inv(z*np.eye(n)-h)
                t_green=model.green(z)
                assert np.allclose(t_green,exact,atol=1e-10,rtol=1e-10)


def test_clean_chain_sine_spectrum():
    n=16
    model=TwoSiteDefectResolvent(n,(0.,0.))
    h=tight_binding_hamiltonian(n,(0.,0.))
    assert np.allclose(np.linalg.eigvalsh(h),np.sort(model.clean_energies))
    assert np.allclose(model.clean_vectors.T@model.clean_vectors,np.eye(n))


def test_bound_energy_and_wavefunction_are_predicted_from_two_by_two_pole():
    n=48
    for u,v in ((1.2,.35),(.7,1.4),(.5,.2),(-.6,-.5)):
        model=TwoSiteDefectResolvent(n,(u,v))
        h=tight_binding_hamiltonian(n,(u,v))
        e,vectors=np.linalg.eigh(h)
        bound=e[np.abs(e)>2+1e-8]
        predicted=model.bound_energies()
        assert len(predicted)==len(bound)
        assert np.allclose(predicted,bound,atol=3e-9)
        for energy in predicted:
            psi=model.bound_wavefunction(energy)
            assert np.isclose(np.linalg.norm(psi),1,atol=1e-10)
            assert np.linalg.norm(h@psi-energy*psi)<3e-8


def test_reversing_defect_preserves_spectral_poles():
    a=TwoSiteDefectResolvent(48,(1.2,.35))
    b=TwoSiteDefectResolvent(48,(.35,1.2))
    assert np.allclose(a.bound_energies(),b.bound_energies(),atol=3e-10)


def test_no_bound_state_without_defect():
    model=TwoSiteDefectResolvent(48,(0.,0.))
    assert model.bound_energies().size==0
    for z in (1.5+.5j,3.+.1j):
        assert np.allclose(model.green(z),model.clean_green(z),atol=1e-12)


def test_truncated_multiple_scattering_converges_away_from_pole():
    model=TwoSiteDefectResolvent(48,(1.2,.35))
    z=2.8+.5j
    exact=model.t_matrix(z)
    errors=[np.linalg.norm(model.t_matrix(z,m)-exact) for m in (1,2,4,8,16)]
    assert model.scattering_radius(z)<.7
    assert np.all(np.diff(errors)<0)
    assert errors[-1]/np.linalg.norm(exact)<.001


def test_near_bound_pole_short_series_is_not_reliable():
    model=TwoSiteDefectResolvent(48,(1.2,.35))
    eb=model.bound_energies()[0]
    z=eb+.05j
    exact=model.t_matrix(z)
    short=model.t_matrix(z,16)
    relative=np.linalg.norm(exact-short)/np.linalg.norm(exact)
    assert model.scattering_radius(z)>.95
    assert relative>.5


def test_bad_arguments_and_nonpole_state_rejected():
    import pytest
    with pytest.raises(ValueError):
        TwoSiteDefectResolvent(3)
    model=TwoSiteDefectResolvent(12,(1.2,.35))
    with pytest.raises(ValueError):
        model.t_matrix(2.7+.3j,0)
    with pytest.raises(ValueError):
        model.bound_wavefunction(3.5)
