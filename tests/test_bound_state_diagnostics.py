"""Independent tests for bound-state decomposition and controlled counterfactual."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_phase import initial_modes, trajectory
from bound_state_diagnostics import (
    defective_hamiltonian, spectral_partition, bound_localization,
    capacity_spectral_diagnostics,
)


def test_out_of_band_modes_are_localized_near_defect():
    h=defective_hamiltonian(48,(1.2,.35))
    e,v,mask=spectral_partition(h)
    assert np.sum(mask)==1
    assert e[mask][0]>2
    assert bound_localization(v,mask)[0] > .9


def test_spectrum_and_localization_are_identical_under_reflection():
    h1=defective_hamiltonian(48,(1.2,.35))
    h2=defective_hamiltonian(48,(.35,1.2))
    e1,v1,m1=spectral_partition(h1)
    e2,v2,m2=spectral_partition(h2)
    assert np.allclose(e1,e2,atol=1e-11)
    assert np.allclose(bound_localization(v1,m1),bound_localization(v2,m2),
                       atol=1e-10)


def test_full_diagnostics_agree_with_independent_trajectory_solver():
    n=12
    pair=(1.2,.35)
    times=np.array([0.,.25,2.,5.,12.])
    occupied=initial_modes(n,"staggered")
    diag=capacity_spectral_diagnostics(defective_hamiltonian(n,pair),occupied,times)
    reference=trajectory(n,pair,times,occupied)
    assert np.allclose(diag["values"][:,:2],reference,atol=5e-11)
    assert np.allclose(diag["values"][0,:2],diag["values"][0,2:4],
                       atol=1e-11)


def test_phase_counterfactual_is_sensitive_to_bound_energy():
    n=48
    times=np.linspace(0,8,21)
    occupied=initial_modes(n,"staggered")
    diag=capacity_spectral_diagnostics(
        defective_hamiltonian(n,(1.2,.35)),occupied,times)
    assert diag["num_bound"]==1
    assert np.max(np.abs(diag["values"][:,1]-diag["values"][:,3]))>.01
    assert np.max(diag["values"][:,5])>.01


def test_weak_bond_has_no_out_of_band_modes_and_no_change():
    n=48
    times=np.array([0.,1.,3.,7.,23.])
    occupied=initial_modes(n,"staggered")
    h=defective_hamiltonian(n,(0,0),central_hopping=.5)
    diag=capacity_spectral_diagnostics(h,occupied,times)
    assert diag["num_bound"]==0
    assert diag["bound_energies"].size==0
    assert np.allclose(diag["values"][:,:2],diag["values"][:,2:4],atol=1e-11)
    assert np.max(np.abs(diag["values"][:,4:6]))<1e-12


def test_invalid_hopping_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        defective_hamiltonian(12,(0,0),central_hopping=0)
