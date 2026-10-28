"""Independent exact-matrix tests of parity-resummed scattering identity."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from parity_resummed_scattering import ParityResummedDefect
from scattering_phase import tight_binding_hamiltonian


def direct_green(n,w,d,z):
    h=tight_binding_hamiltonian(n,(w+d/2,w-d/2))
    return np.linalg.inv(z*np.eye(n)-h)


@pytest.mark.parametrize("n,w,d",[
    (12,.775,.02),(12,.775,.85),(24,.775,1.6),
    (24,.25,.3),(24,-.5,.85),(48,.775,.85)])
@pytest.mark.parametrize("z",[.2+.4j,2.8+.35j,-1.2+.6j])
def test_exact_contrast_matches_two_full_direct_inverses(n,w,d,z):
    model=ParityResummedDefect(n,w)
    actual=direct_green(n,w,d,z)-direct_green(n,w,-d,z)
    predicted=model.contrast(z,d)
    assert np.allclose(predicted,actual,atol=3e-12,rtol=3e-11)
    assert np.allclose(model.green(z,d),direct_green(n,w,d,z),
                       atol=3e-12,rtol=3e-11)


def test_reversal_and_parity_are_exact():
    n=24
    model=ParityResummedDefect(n,.775)
    assert model.parity_diagnostic()<1e-13
    z=1.2+.7j
    delta=.85
    c=model.contrast(z,delta)
    assert np.allclose(c,-model.contrast(z,-delta))
    assert np.allclose(c,-c[::-1,::-1],atol=1e-12)
    assert np.allclose(c,c.T,atol=1e-12)
    assert np.allclose(model.green(z,-delta),
                       model.green(z,delta)[::-1,::-1],atol=1e-12)


def test_resummation_beats_linear_response_at_large_asymmetry():
    model=ParityResummedDefect(24,.775)
    z=2.8+.35j
    delta=1.6
    exact=direct_green(24,.775,delta,z)-direct_green(24,.775,-delta,z)
    corrected=model.contrast(z,delta)
    linear=model.contrast(z,delta,linear=True)
    err_exact=np.linalg.norm(corrected-exact)/np.linalg.norm(exact)
    err_linear=np.linalg.norm(linear-exact)/np.linalg.norm(exact)
    assert err_exact<1e-11
    assert err_linear>.2


def test_even_channel_denominator_and_zero_asymmetry():
    model=ParityResummedDefect(12,.775)
    for z in (.1+.4j,3+.25j):
        delta=.85
        gp,gm,D,_=model.channel_data(z,delta)
        assert D==pytest.approx(1-delta**2*gp*gm/4)
        assert D==pytest.approx(model.channel_data(z,-delta)[2])
        assert np.max(abs(model.contrast(z,0)))==0
        assert np.allclose(model.green(z,0),direct_green(12,.775,0,z))


def test_small_delta_linear_remainder_is_cubic_at_fixed_complex_energy():
    model=ParityResummedDefect(24,.775)
    z=.2+.4j
    errs=[]
    for delta in (.02,.04):
        errs.append(np.linalg.norm(model.contrast(z,delta)-
                                   model.contrast(z,delta,linear=True)))
    assert 7 < errs[1]/errs[0] < 9


def test_bad_input_rejected():
    with pytest.raises(ValueError):
        ParityResummedDefect(5,.3)
    model=ParityResummedDefect(12,.775)
    with pytest.raises(ValueError):
        model.green(complex(model.e[-1]),.85)


@pytest.mark.parametrize("n,w,d",[
    (12,.775,.02),(24,.775,.1),(48,.775,.85),
    (48,.775,1.6),(24,.25,.3),(24,-.5,.85)])
def test_highest_energy_from_parity_pole_matches_direct_eigenvalue(n,w,d):
    model=ParityResummedDefect(n,w)
    result=model.largest_eigenenergy(d)
    h=tight_binding_hamiltonian(n,(w+d/2,w-d/2))
    actual=np.linalg.eigvalsh(h)[-1]
    assert result==pytest.approx(actual,abs=1e-10)
    assert result==pytest.approx(model.largest_eigenenergy(-d),abs=1e-10)
    assert result>=model.e[-1]


def test_highest_energy_even_quadratic_small_delta():
    model=ParityResummedDefect(48,.775)
    actual=model.largest_eigenenergy(.02)
    quadratic=model.top_eigenenergy_quadratic(.02)
    assert abs(actual-quadratic)<2e-7
    assert model.largest_eigenenergy(0)==pytest.approx(model.e[-1])


def test_top_energy_pole_is_actually_bound_in_original_case():
    model=ParityResummedDefect(48,.775)
    root=model.largest_eigenenergy(.85)
    assert root>2
    assert root==pytest.approx(2.39874791578603,abs=1e-8)


@pytest.mark.parametrize("n,w,d",[
    (12,.775,.3),(24,.775,.85),(48,.775,.85),
    (24,-.5,.85),(48,.25,1.6)])
def test_top_pole_residue_reconstructs_exact_eigenprojector(n,w,d):
    model=ParityResummedDefect(n,w)
    energy,prediction=model.largest_eigenprojector_from_residue(d)
    h=tight_binding_hamiltonian(n,(w+d/2,w-d/2))
    e,vecs=np.linalg.eigh(h)
    reference=np.outer(vecs[:,-1],vecs[:,-1])
    assert abs(energy-e[-1])<2e-10
    assert np.linalg.norm(prediction-reference)<2e-9
    assert np.linalg.norm(prediction@prediction-prediction)<2e-9
    assert abs(np.trace(prediction)-1)<2e-9
    assert np.linalg.norm(h@prediction-energy*prediction)<3e-9


def test_original_bound_residue_localizes_to_central_sites():
    model=ParityResummedDefect(48,.775)
    energy,projector=model.largest_eigenprojector_from_residue(.85)
    center=np.arange(48//2-4,48//2+4)
    localization=np.trace(projector[np.ix_(center,center)])
    assert energy>2
    assert abs(localization-.9930661718984761)<1e-8


def test_residue_projector_reflects_under_asymmetry_reversal():
    model=ParityResummedDefect(24,.775)
    _,positive=model.largest_eigenprojector_from_residue(.85)
    _,negative=model.largest_eigenprojector_from_residue(-.85)
    assert np.allclose(positive[::-1,::-1],negative,atol=1e-10)
