import numpy as np
from hermite_airy_solver import (
    propagator_block,hermite_shapes,solve_hermite_image,
    initial_occupied,solve_hermite_volterra
)
from self_consistent_return import kernels

def test_existing_kernel():
    L=24;t=np.array([0.,.4,3.,14.]);Q=initial_occupied(L)
    for mode in ("base","airy","bessel","full_bessel"):
        np.testing.assert_allclose(
            propagator_block(L,t,np.arange(L),np.arange(L),mode),
            kernels(L,t,Q,mode)["clean"],atol=2e-13,rtol=0)

def test_hermite_shape_endpoints():
    H=hermite_shapes(np.array([0.,1.]))
    np.testing.assert_allclose(H[:4],[[1,0],[0,0],[0,1],[0,0]],atol=1e-15)
    np.testing.assert_allclose(H[4:],[[0,0],[1,0],[0,0],[0,1]],atol=1e-15)

def test_zero_defect():
    L=16;t=np.arange(0,4.001,.1);Q=initial_occupied(L)
    F,Y,_,d=solve_hermite_image(L,t,Q,(0.,0.),"bessel",return_details=True)
    expected=propagator_block(L,t,np.arange(L),np.arange(L),"bessel")@Q
    np.testing.assert_allclose(F,expected,atol=2e-13,rtol=0)
    np.testing.assert_allclose(Y,F[:,[L//2-1,L//2]],atol=1e-13,rtol=0)
    assert d["max_node_residual"]<1e-12

def test_independent_fourth_order_bessel_solver():
    L=24;t=np.arange(0,11.001,.1);Q=initial_occupied(L)
    F,_,_,d=solve_hermite_image(L,t,Q,(1.2,.35),"bessel",return_details=True)
    reference,_,_=solve_hermite_volterra(L,t,Q,(1.2,.35))
    assert np.max(abs(F-reference))<5e-11
    assert d["max_node_residual"]<1e-12

def test_heldout_pair_before_wavefront():
    L=32;t=np.arange(0,10.0001,.2);Q=initial_occupied(L)
    a=solve_hermite_image(L,t,Q,(.5,.2),"airy")[0]
    b=solve_hermite_image(L,t,Q,(.5,.2),"bessel")[0]
    assert max(np.linalg.norm(x-y) for x,y in zip(a,b))<1e-8
