"""Smooth leading Olver Airy first-return kernel and its Schrödinger residual.

Numerical residual integrals are NOT certified quadratures. The turning-point
Taylor polynomial is not accompanied by a formally certified remainder.
"""
import numpy as np
from scipy.special import airy,jv

A=2**(1/3)
COEFF=np.array([1.,3/10,32/175,1037/7875,103727/1010625])

def olver_value_derivative(n,x):
    """J_n(nz) leading Olver approximation and analytic d/dx."""
    if n<1:raise ValueError("positive order required")
    xx=np.asarray(x,float);shape=xx.shape;x=xx.reshape(-1)
    val=np.zeros(len(x));der=np.zeros(len(x));pos=x>0
    if not np.any(pos):return val.reshape(shape),der.reshape(shape)
    z=x[pos]/n
    zeta=np.empty_like(z);amp=np.empty_like(z)
    zp=np.empty_like(z);ap=np.empty_like(z)
    near=abs(1-z)<.025
    if np.any(near):
        w=1-z[near]
        P=np.polynomial.polynomial.polyval(w,COEFF)
        Pd=np.polynomial.polynomial.polyval(w,np.arange(1,len(COEFF))*COEFF[1:])
        a=A*(P/(1-w/2))**.25
        zeta[near]=A*w*P;amp[near]=a
        zp[near]=-A*(P+w*Pd)
        ap[near]=-a/4*(Pd/P+.5/(1-w/2))
    low=(~near)&(z<1)
    if np.any(low):
        v=z[low];s=np.sqrt(1-v*v)
        eta=np.log((1+s)/v)-s
        zz=(1.5*eta)**(2/3);a=(4*zz/(1-v*v))**.25
        dz=-s/(v*np.sqrt(zz))
        zeta[low]=zz;amp[low]=a;zp[low]=dz
        ap[low]=a/4*(dz/zz+2*v/(1-v*v))
    high=(~near)&(z>1)
    if np.any(high):
        v=z[high];s=np.sqrt(v*v-1)
        eta=s-np.arccos(1/v)
        zz=-(1.5*eta)**(2/3);a=(4*zz/(1-v*v))**.25
        dz=-s/(v*np.sqrt(-zz))
        zeta[high]=zz;amp[high]=a;zp[high]=dz
        ap[high]=a/4*(dz/zz+2*v/(1-v*v))
    arg=n**(2/3)*zeta
    ai,aip,_,_=airy(arg)
    val[pos]=n**(-1/3)*amp*ai
    der[pos]=n**(-4/3)*(ap*ai+amp*n**(2/3)*zp*aip)
    return val.reshape(shape),der.reshape(shape)

def image_kernel(L,t,rows=None,cols=None,mode="airy",derivative=False):
    """Open-chain image approximation; derivative is w.r.t. physical time."""
    t=np.atleast_1d(np.asarray(t,float))
    rows=np.arange(L) if rows is None else np.asarray(rows,int)
    cols=np.arange(L) if cols is None else np.asarray(cols,int)
    i=rows[:,None];j=cols[None,:];N=2*(L+1)
    ns=[abs(i-j),i+j+2,abs(i+j+2-N),N+i-j,N-i+j,N+i+j+2]
    allorders=np.unique(np.concatenate([v.ravel() for v in ns]))
    bessel={}
    for n in allorders:
        n=int(n)
        v=(jv(n-1,2*t)-jv(n+1,2*t)) if derivative else jv(n,2*t)
        bessel[n]=1j**n*v
    airyvals={}
    if mode=="airy":
        for n in np.unique(np.r_[ns[3].ravel(),ns[4].ravel()]):
            n=int(n)
            v,d=olver_value_derivative(n,2*t)
            airyvals[n]=1j**n*(2*d if derivative else v)
    def part(orders,replace=False):
        tbl=airyvals if replace else bessel
        vals=np.stack([tbl[int(n)] for n in orders.ravel()],axis=1)
        return vals.reshape(len(t),*orders.shape)
    K=part(ns[0])-part(ns[1])-part(ns[2])
    if mode in ("airy","bessel","full_bessel"):
        K+=part(ns[3],mode=="airy")+part(ns[4],mode=="airy")
        if mode=="full_bessel":K-=part(ns[5])
    elif mode!="base":raise ValueError(mode)
    return K

def clean_residual(L,t,mode="airy"):
    """dot(K)+i*h0*K, h0=-nearest-neighbor hopping."""
    K=image_kernel(L,t,mode=mode)
    Kdot=image_kernel(L,t,mode=mode,derivative=True)
    hK=np.zeros_like(K)
    hK[:,1:]-=K[:,:-1]
    hK[:,:-1]-=K[:,1:]
    return Kdot+1j*hK

def residual_integral_diagnostic(L,T,dt=.1):
    """Sampled trapezoid, NOT validated outward-rounded quadrature."""
    t=np.arange(0,T+dt/4,dt)
    R=clean_residual(L,t)
    norm=np.array([np.linalg.norm(r,2) for r in R])
    return t,np.r_[0.,np.cumsum((norm[:-1]+norm[1:])*dt/2)]
