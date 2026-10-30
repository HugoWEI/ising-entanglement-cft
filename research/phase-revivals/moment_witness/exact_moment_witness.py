"""Source-only clean-witness reconstruction using cubic-Hermite exponential moments.

Exact-arithmetic series tail is bounded. Does not certify floating-point rounding.
"""
from math import factorial
import numpy as np

def hermite_power_coefficients(y0,dy0,y1,dy1,h):
    d=y1-y0
    return np.stack([y0,h*dy0,3*d-h*(2*dy0+dy1),-2*d+h*(dy0+dy1)])

def exponential_moments(z,degree=14):
    z=np.asarray(z,dtype=complex)
    powers=np.ones_like(z)
    out=np.zeros((4,)+z.shape,dtype=complex)
    for q in range(degree+1):
        for k in range(4): out[k]+=powers/(q+k+1)
        powers=powers*z/(q+1)
    return out

def moment_tail_upper(radius,k=0,degree=14):
    if radius<0 or radius>=degree+2: raise ValueError("invalid radius")
    q=degree+1
    return radius**q/(factorial(q)*(q+k+1)*(1-radius/(q+1)))

def witness_moment_nodes(L,t,Q,pair,Y,DY,degree=14,return_bound=False):
    from clean_witness import sine_modes
    t=np.asarray(t);h=float(t[1]-t[0])
    if not np.allclose(np.diff(t),h): raise ValueError("nonuniform grid")
    lam,S=sine_modes(L)
    D=[L//2-1,L//2];V=np.diag(pair);Sd=S[D,:]
    phase=np.exp(-1j*lam*h)
    moments=exponential_moments(1j*lam*h,degree=degree)
    phat=S.T@Q
    F=np.zeros((len(t),L,Q.shape[1]),dtype=complex);F[0]=S@phat
    b=max(map(abs,pair))
    tails=np.array([moment_tail_upper(2*h,k,degree) for k in range(4)])
    bound=0.
    for j in range(len(t)-1):
        c=hermite_power_coefficients(Y[j],DY[j],Y[j+1],DY[j+1],h)
        src=np.einsum("sa,kam->ksm",Sd.T@V,c,optimize=True)
        influence=np.einsum("ks,ksm->sm",moments,src,optimize=True)
        phat=phase[:,None]*(phat-1j*h*influence)
        F[j+1]=S@phat
        bound+=h*b*sum(tails[k]*np.linalg.norm(c[k],ord=2) for k in range(4))
    if return_bound:
        return F,dict(degree=degree,Taylor_truncation_bound=float(bound),
            scope="exact-arithmetic series truncation only, not floating-point error")
    return F
