"""Pilot: replace long-distance Bessel wavefronts in image kernel with leading
uniform Olver-Air(y) approximation, without altering defect Volterra solution.
Formula from DLMF 10.20.2-4, with an analytic turning-point limit near z=1.
"""
import numpy as np
from scipy.special import jv, airy
import volterra_scattering_revival as vol


def leading_olver_j(n, x):
    """Leading uniform large-order approximation for J_n(x), n>=1, x>=0."""
    x=np.asarray(x,float)
    result=np.zeros_like(x)
    pos=x>0
    z=x[pos]/float(n)
    res=np.zeros_like(z)
    near=np.abs(z-1.)<.0001
    # linearized zeta; error O((z-1)^2) and small relative to leading asymptotic terms
    if np.any(near):
        res[near]=(2/n)**(1/3)*airy(-2**(1/3)*(x[pos][near]-n)/n**(1/3))[0]
    low=(z<1)&~near
    if np.any(low):
        q=np.sqrt(1-z[low]*z[low]); eta=np.log((1+q)/z[low])-q
        zeta=(1.5*eta)**(2/3)
        amp=(4*zeta/(1-z[low]**2))**.25/n**(1/3)
        res[low]=amp*airy(n**(2/3)*zeta)[0]
    high=(z>1)&~near
    if np.any(high):
        q=np.sqrt(z[high]*z[high]-1); eta=q-np.arccos(1/z[high])
        zeta=-(1.5*eta)**(2/3)
        amp=(4*zeta/(1-z[high]**2))**.25/n**(1/3)
        res[high]=amp*airy(n**(2/3)*zeta)[0]
    result[pos]=res
    return result


def hybrid_kernel(L,times,shells,rows=None,cols=None,order_cutoff=32):
    if rows is None: rows=np.arange(L)
    if cols is None: cols=np.arange(L)
    i=np.asarray(rows,dtype=int)[:,None];j=np.asarray(cols,dtype=int)[None,:]
    t=np.asarray(times,float)
    out=np.zeros((len(t),len(i),len(cols)),dtype=complex)
    n_all=np.arange(2*(shells+1)*(L+1)+1)
    basis=np.empty((len(n_all),len(t)),dtype=complex)
    for n in n_all:
        val=(jv(n,2*t) if n<order_cutoff else leading_olver_j(n,2*t))
        basis[n]=(1j**n)*val
    for r in range(-shells,shells+1):
        n1=np.abs(i-j+2*r*(L+1));n2=np.abs(i+j+2+2*r*(L+1))
        out+=basis[n1].transpose(2,0,1)
        out-=basis[n2].transpose(2,0,1)
    return out


def use_hybrid(cutoff):
    def fun(L,times,shells,rows=None,cols=None):
        return hybrid_kernel(L,times,shells,rows,cols,order_cutoff=cutoff)
    vol.image_bessel_kernel=fun
    return fun
