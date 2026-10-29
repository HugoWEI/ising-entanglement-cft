"""Rigorous (exact-arithmetic) image-tail and two-site Volterra stability majorants.

No final-Hamiltonian eigenvalues, exact solution data, or fitted parameters used.
The scalar bounds follow by Cauchy's integral formula applied to
exp(t(z-z**(-1))) = sum_{n in Z} J_n(2t) z**n.
Floating-point evaluation of bounds is not interval-verified.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import jv


def image_tail_entry_bound(L: int, shells: int, T: float, rho: float) -> float:
    """Uniform entrywise omitted-image bound for 0 <= t <= T."""
    assert L >= 2 and shells >= 0 and T >= 0 and rho > 1
    q = rho ** (-2 * (L + 1))
    # direct q=+/-n with n >= S+1, and reflected q=+n/-n terms
    factor = 2 * rho ** (L - 1) + rho ** (2 * L) + rho ** (-2)
    return math.exp(T * (rho - 1/rho)) * q ** (shells+1) * factor / (1-q)


def image_tail_op_bound(L: int, shells: int, T: float, rho: float) -> float:
    """The spectral norm is at most L times the largest entry modulus."""
    return L * image_tail_entry_bound(L,shells,T,rho)


def volterra_amplitude_error_bound(L: int, shells: int, T: float,
                                   vnorm: float, rho: float) -> float:
    """Error on U_def(t)Q vs continuous-time shell-S Volterra solution,
    assuming Q has orthonormal columns. Uses unitarity of exact U_def.
    No time discretization and no Airy replacement included.
    """
    eps = image_tail_op_bound(L,shells,T,rho)
    x = vnorm*T
    den = 1 - eps * x * (1+x)
    if den <= 0:
        return float('inf')
    return eps * (1+x)**2 / den


def kernel_direct(L: int, t: float, shells: int) -> np.ndarray:
    i=np.arange(L)[:,None];j=np.arange(L)[None,:]
    acc=np.zeros((L,L),dtype=complex)
    for q in range(-shells,shells+1):
        d=np.abs(i-j+2*q*(L+1))
        r=np.abs(i+j+2+2*q*(L+1))
        acc+=(1j)**d*jv(d,2*t)-(1j)**r*jv(r,2*t)
    return acc


def test_image_bound():
    for L,S,T,rho in [(8,0,3.,2.),(12,1,9.,2.),(48,1,30.,3.),(64,1,40.,3.)]:
        bound=image_tail_op_bound(L,S,T,rho)
        for t in [0.,T/2,T]:
            A=kernel_direct(L,t,S)
            B=kernel_direct(L,t,S+3)
            err=np.linalg.norm(A-B,2)
            assert err <= bound+1e-14, (L,t,err,bound)
    assert image_tail_op_bound(48,1,30.,3.) < 1e-10
    assert volterra_amplitude_error_bound(48,1,30.,1.2,3.) < 1e-7


if __name__=='__main__':
    test_image_bound()
    for L,T,vnorm,rho in [(48,30.,1.2,3.),(64,40.,1.2,3.),(80,50.,1.2,3.)]:
        e=image_tail_op_bound(L,1,T,rho)
        a=volterra_amplitude_error_bound(L,1,T,vnorm,rho)
        print(f'L={L} T={T:g} rho={rho:g} image_tail_op={e:.9g} volterra_amplitude={a:.9g}')
