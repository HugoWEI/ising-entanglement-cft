"""Independent validation of scattering-only bound-pole/Krylov capacity."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scattering_pole_krylov_capacity import ScatteringPoleKrylovCapacity
from scattering_phase import initial_modes,trajectory


@pytest.mark.parametrize("n,w,d",[(12,.775,.85),(24,.775,.85),
                                    (24,1.05,-.7)])
def test_full_subspace_scattering_formula_matches_exact_asymmetric_solver(n,w,d):
    initial=initial_modes(n,"staggered")
    times=[0.,2.,.6*n,1.2*n]
    model=ScatteringPoleKrylovCapacity(n,w,initial)
    out=model.trajectory(d,times,krylov_blocks=n//2,
                         quadrature_nodes=768)
    reference=trajectory(n,(w+d/2,w-d/2),times,initial)[:,1]
    assert np.allclose(out["capacities"][:,0],reference,atol=2e-7)
    assert np.allclose(out["capacities"][:,2],reference,atol=2e-7)
    assert out["maximum_contour_isometry_error"]<2e-7
    assert out["bound_energy"]>2.


def test_eight_krylov_blocks_capture_strong_defect_capacity_at_late_times():
    n=48
    w=.775
    d=.85
    t=np.array([0.,20.,26.4,27.4,36.,50.,62.])
    initial=initial_modes(n,"staggered")
    out=ScatteringPoleKrylovCapacity(n,w,initial).trajectory(
        d,t,krylov_blocks=8,quadrature_nodes=768)
    reference=trajectory(n,(w+d/2,w-d/2),t,initial)[:,1]
    assert out["maximum_krylov_dimension"]<=16
    assert np.sqrt(np.mean((out["capacities"][:,0]-reference)**2))<.08
    assert np.max(abs(out["capacities"][:,2]-reference))<2e-7


def test_symmetry_of_scattering_model_for_mirror_symmetric_initial_state():
    n=12
    f0=initial_modes(n,"mirror_fock")
    t=[0.,4.,11.]
    model=ScatteringPoleKrylovCapacity(n,.775,f0)
    a=model.trajectory(.85,t,krylov_blocks=6)["capacities"][:,0]
    b=model.trajectory(-.85,t,krylov_blocks=6)["capacities"][:,0]
    assert np.allclose(a,b,atol=2e-7)


def test_zero_asymmetry_is_not_conflated_with_one_bound_dressing():
    n=12
    model=ScatteringPoleKrylovCapacity(n,.775,initial_modes(n,"staggered"))
    with pytest.raises(ValueError):
        model.trajectory(0.,[1.])
