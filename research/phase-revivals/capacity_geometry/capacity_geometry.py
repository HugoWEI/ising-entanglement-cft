"""Endpoint-regular modular-capacity calculus on pure fermionic Gaussian subspaces.

Numerical computations are floating point; analytic bounds require certified inputs.
"""
import numpy as np

def f(p):
    p=np.clip(np.asarray(p,dtype=float),0,1)
    ans=np.zeros_like(p)
    inside=(p>0)&(p<1)
    q=p[inside]; l=np.log1p(-q)-np.log(q)
    ans[inside]=q*(1-q)*l*l
    return ans

def weight(p):
    """sqrt(p*(1-p)) times f'(p), with its continuous endpoint extension."""
    p=np.clip(np.asarray(p,dtype=float),0,1)
    ans=np.zeros_like(p)
    inside=(p>0)&(p<1)
    q=p[inside]; l=np.log1p(-q)-np.log(q)
    ans[inside]=np.sqrt(q*(1-q))*((1-2*q)*l*l-2*l)
    return ans

def capacity(X,left_sites=None):
    a=X.shape[1] if left_sites is None else left_sites
    p=np.linalg.eigvalsh(X[:a].conj().T@X[:a])
    return float(sum(f(p)))

def gradient_norm(X,left_sites=None):
    a=X.shape[1] if left_sites is None else left_sites
    p=np.linalg.eigvalsh(X[:a].conj().T@X[:a])
    return 2*float(np.linalg.norm(weight(p)))

def polar(Z):
    u,_,v=np.linalg.svd(Z,full_matrices=False)
    return u@v

def horizontal(Z,Delta):
    evals,U=np.linalg.eigh(Z.conj().T@Z)
    inv=(U*evals**(-.5))@U.conj().T
    X=Z@inv
    T=Delta@inv
    return X,T-X@(X.conj().T@T)

def derivative(X,H,left_sites=None):
    a=X.shape[1] if left_sites is None else left_sites
    p,V=np.linalg.eigh(X[:a].conj().T@X[:a])
    XX=X@V; HH=H@V
    dp=2*np.real(np.sum(XX[:a].conj()*HH[:a],axis=0))
    out=0.
    for x,d in zip(p,dp):
        if x<=1e-13 or x>=1-1e-13: continue
        l=np.log1p(-x)-np.log(x)
        out+=((1-2*x)*l*l-2*l)*d
    return float(out)

def capacity_bound(delta_fro,m,sigma_lower,M=1.22):
    if sigma_lower<=0: raise ValueError("nonpositive singular-value lower bound")
    return 2*M*np.sqrt(m)*delta_fro/sigma_lower
