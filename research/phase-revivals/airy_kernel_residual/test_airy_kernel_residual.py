import numpy as np
from scipy.special import jv
from airy_kernel_residual import olver_value_derivative,image_kernel,clean_residual,residual_integral_diagnostic

def test_analytic_derivative():
    for n in (51,74,104):
        for z in (.83,.97499,.97501,.9999,1.,1.0001,1.02499,1.02501,1.17):
            x=n*z;h=2e-4
            _,d=olver_value_derivative(n,[x])
            vp,_=olver_value_derivative(n,[x+h])
            vm,_=olver_value_derivative(n,[x-h])
            assert abs(d[0]-(vp[0]-vm[0])/(2*h))<2e-7

def test_image_derivative():
    for mode in ("airy","bessel"):
        for t in (3.,12.4,15.7):
            h=1e-4
            d=image_kernel(24,[t],mode=mode,derivative=True)[0]
            fd=(image_kernel(24,[t+h],mode=mode)[0]-image_kernel(24,[t-h],mode=mode)[0])/(2*h)
            assert np.linalg.norm(d-fd,2)<4e-6

def test_clean_bessel_residual():
    R=clean_residual(24,[2.,5.,8.],mode="full_bessel")
    assert np.linalg.norm(R[0],2)<1e-10

def test_source_residual_vs_clean_reference():
    L=24;t,bound=residual_integral_diagnostic(L,18.,.05)
    k=np.arange(1,L+1)
    phi=np.sqrt(2/(L+1))*np.sin(np.pi*np.outer(k,k)/(L+1))
    e=-2*np.cos(np.pi*k/(L+1))
    for ti in (17.9,18.):
        U=(phi*np.exp(-1j*e*ti))@phi.T
        Ka=image_kernel(L,[ti],mode="airy")[0]
        assert np.linalg.norm(Ka-U,2)<np.interp(ti,t,bound)+1e-8
