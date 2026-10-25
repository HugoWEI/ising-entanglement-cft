"""Tests emphasizing proper controls against spurious phase novelty claims."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scattering_phase import (
    left_incident_scattering,
    tight_binding_hamiltonian,
    initial_modes,
    trajectory,
    compare_reversed_defects,
)


def test_mirror_defect_hamiltonians_are_reflection_related():
    n = 12
    a = tight_binding_hamiltonian(n, (1.2, .35))
    b = tight_binding_hamiltonian(n, (.35, 1.2))
    assert np.allclose(a[::-1, ::-1], b)


def test_reversed_defects_have_equal_transmission_amplitudes():
    for k in np.linspace(.3, np.pi-.3, 13):
        ra, ta = left_incident_scattering(k, 1.2, .35)
        rb, tb = left_incident_scattering(k, .35, 1.2)
        assert abs(ta - tb) < 1e-11
        assert abs(abs(ra)**2 + abs(ta)**2 - 1) < 1e-11
        assert abs(abs(rb)**2 + abs(tb)**2 - 1) < 1e-11
    ra, _ = left_incident_scattering(1.1, 1.2, .35)
    rb, _ = left_incident_scattering(1.1, .35, 1.2)
    assert abs(ra - rb) > .1  # reflection phase is genuinely different


def test_mirror_symmetric_initial_state_has_identical_entanglement():
    t = np.linspace(0, 12, 13)
    a, b = compare_reversed_defects(12, 1.2, .35, t, "mirror_fock")
    assert np.max(abs(a-b)) < 1e-9


def test_same_asymmetric_initial_state_reveals_difference():
    t = np.array([0., 2., 4.])
    a, b = compare_reversed_defects(12, 1.2, .35, t, "staggered")
    assert abs(a[0,0]-b[0,0]) < 1e-10
    assert abs(a[0,1]-b[0,1]) < 1e-10
    assert abs(a[1,1]-b[1,1]) > .05


def test_identical_defects_have_identical_trajectories():
    t=np.array([0., .5, 2., 4.])
    a, b = compare_reversed_defects(12, .8, .8, t, "staggered")
    assert np.allclose(a, b, atol=1e-12)


def test_occupied_modes_and_initial_entropy():
    modes = initial_modes(12, "staggered")
    assert np.allclose(modes.T @ modes, np.eye(6), atol=1e-11)
    a = trajectory(12,(1.2,.35),[0.], modes)
    b = trajectory(12,(.35,1.2),[0.], modes)
    assert np.allclose(a,b,atol=1e-10)
