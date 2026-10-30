"""Independent validation of moments, Hermite coefficients and Gauss remainder."""
import numpy as np
from math import factorial
from scipy.integrate import quad
from exact_moment_witness import (hermite_power_coefficients,exponential_moments,
  moment_tail_upper,witness_moment_nodes)
from quadrature_certificate import gauss_coefficient,bound_clean_convolution
from clean_witness import witness_nodes,sine_modes
from phase_revival_helper import initial_occupied

def test_gauss_constant():
    x,w=np.polynomial.legendre.leggauss(2);x=(x+1)/2;w=w/2
    assert abs(abs(np.sum(w*x**4)-.2)-24*gauss_coefficient(2))<1e-14

def test_analytic_moments():
    for z in (0j,.1j,-.17j):
        vals=exponential_moments(np.array([z]),15)[:,0]
        for k in range(4):
            f=lambda u:np.exp(z*u)*u**k
            exact=quad(lambda u:f(u).real,0,1)[0]+1j*quad(lambda u:f(u).imag,0,1)[0]
            assert abs(vals[k]-exact)<2e-14

def test_hermite():
    h=.15
    y0=np.array([[1+1j]]);dy0=np.array([[2-1j]])
    y1=np.array([[3-2j]]);dy1=np.array([[-1+2j]])
    c=hermite_power_coefficients(y0,dy0,y1,dy1,h)
    assert np.allclose(c[0],y0)
    assert np.allclose(sum(c),y1)
    assert np.allclose(c[1]/h,dy0)
    assert np.allclose((c[1]+2*c[2]+3*c[3])/h,dy1)

def test_against_12_point_gauss():
    L=16;h=.1;t=np.arange(0,2.0001,h);Q=initial_occupied(L)
    rng=np.random.default_rng(15)
    Y=.2*rng.normal(size=(len(t),2,L//2))+.1j*rng.normal(size=(len(t),2,L//2))
    DY=.2*rng.normal(size=Y.shape)+.1j*rng.normal(size=Y.shape)
    pair=(1.2,.35)
    F,r=witness_moment_nodes(L,t,Q,pair,Y,DY,return_bound=True)
    G=witness_nodes(L,t,Q,pair,Y,DY,n_gauss=12)
    assert np.max(abs(F-G))<2e-13
    assert r["Taylor_truncation_bound"]<1e-20
    assert bound_clean_convolution(Y,DY,h,n=12,b=1.2)["uniform_all_nodes_bound"]>0

def test_zero_defect():
    L=16;t=np.arange(0,2.0001,.1);Q=initial_occupied(L)
    Y=np.zeros((len(t),2,L//2),complex)
    F=witness_moment_nodes(L,t,Q,(0,0),Y,Y)
    lam,S=sine_modes(L)
    G=np.einsum("ik,tk,kq->tiq",S,np.exp(-1j*t[:,None]*lam),S.T@Q)
    assert np.max(abs(F-G))<2e-13
