"""Independent tests of the all-delta symmetric-reference contour capacity."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from parity_contour_capacity import ContourCapacity
from scattering_phase import initial_modes,trajectory,tight_binding_hamiltonian


@pytest.mark.parametrize("n,w,d,times",[
    (12,.775,.85,[0.,2.,8.,10.]),
    (24,.25,1.6,[0.,4.,18.,30.]),
    (48,.775,.85,[0.,2.,26.,38.,60.]),
])
def test_contour_capacity_matches_independent_asymmetric_eigensolver(n,w,d,times):
    f0=initial_modes(n,"staggered")
    solver=ContourCapacity(n,w,f0)
    actual=trajectory(n,(w+d/2,w-d/2),times,f0)
    predicted=solver.capacity_trajectory(d,times,quadrature_nodes=768)
    assert np.allclose(predicted["values"],actual,atol=2e-7,rtol=2e-7)
    assert predicted["max_isometry_error"]<2e-7
    assert predicted["max_occupation"]<1+2e-7


def test_contour_orbitals_match_independent_full_matrix_propagator():
    n=12
    w=.775
    delta=.85
    times=[0.,2.,9.]
    f0=initial_modes(n,"staggered")
    solver=ContourCapacity(n,w,f0)
    predicted=solver.propagate(delta,times,quadrature_nodes=512)
    e,v=np.linalg.eigh(tight_binding_hamiltonian(
        n,(w+delta/2,w-delta/2)))
    amplitudes=v.T@f0
    for t,f in zip(times,predicted):
        exact=v@(np.exp(-1j*e*t)[:,None]*amplitudes)
        assert np.allclose(f,exact,atol=1e-8)


def test_zero_asymmetry_exact_reference_independent_of_contour_nodes():
    n=12
    f0=initial_modes(n,"staggered")
    solver=ContourCapacity(n,.775,f0)
    times=[0.,1.,7.,14.]
    for nodes in (64,256):
        predicted=solver.capacity_trajectory(0,times,quadrature_nodes=nodes)
        reference=trajectory(n,(.775,.775),times,f0)
        assert np.allclose(predicted["values"],reference,atol=1e-9)


def test_contour_respects_reflection_symmetric_preparation():
    n=12
    f0=initial_modes(n,"mirror_fock")
    solver=ContourCapacity(n,.775,f0)
    times=[0.,2.,8.]
    a=solver.capacity_trajectory(.85,times,quadrature_nodes=512)
    b=solver.capacity_trajectory(-.85,times,quadrature_nodes=512)
    assert np.allclose(a["values"],b["values"],atol=1e-8)


def test_bad_geometry_and_parameters_rejected():
    n=12
    solver=ContourCapacity(n,.775,initial_modes(n,"staggered"))
    with pytest.raises(ValueError):
        solver.propagate(.85,[1],quadrature_nodes=16)
    with pytest.raises(ValueError):
        solver.propagate(.85,[-1])
    with pytest.raises(ValueError):
        solver.propagate(.85,[2.],real_semiaxis=2.0)
    with pytest.raises(ValueError):
        solver.propagate(.85,[2.],imag_semiaxis=-1)
    with pytest.raises(ValueError):
        solver.propagate(np.nan,[2.])
