"""Cubic-Hermite two-site Volterra solver with an Airy first return."""
import numpy as np
import sys
from pathlib import Path
from scipy.special import jv

ROOT=Path(__file__).resolve().parent.parent
for sub in ["certificate","certificate/src","capacity_geometry","self_consistent_return"]:
    sys.path.insert(0,str(ROOT/sub))
from airy_hybrid import leading_olver_j
from phase_revival_helper import initial_occupied
from high_order_volterra import solve_hermite_volterra
from self_consistent_return import cap_curve,smoothed_peak


def propagator_block(L,t,rows,cols,mode="airy"):
    t=np.atleast_1d(np.asarray(t,float))
    i=np.asarray(rows,int)[:,None]
    j=np.asarray(cols,int)[None,:]
    N=2*(L+1)
    ns=[abs(i-j),i+j+2,abs(i+j+2-N),N+i-j,N-i+j,N+i+j+2]
    orders=np.unique(np.concatenate([x.ravel() for x in ns]))
    data={int(n):(1j**int(n))*jv(int(n),2*t) for n in orders}
    def part(a):
        return np.stack([data[int(n)] for n in a.ravel()],axis=1).reshape(len(t),*a.shape)
    result=part(ns[0])-part(ns[1])-part(ns[2])
    if mode in ("airy","bessel","full_bessel"):
        if mode=="airy":
            for n in np.unique(np.r_[ns[3].ravel(),ns[4].ravel()]):
                data[int(n)]=(1j**int(n))*leading_olver_j(int(n),2*t)
        result+=part(ns[3])+part(ns[4])
        if mode=="full_bessel":
            result-=part(ns[5])
    elif mode!="base":
        raise ValueError(mode)
    return result


def hermite_shapes(x):
    x=np.asarray(x)
    return np.array([2*x**3-3*x**2+1,x**3-2*x**2+x,
                    -2*x**3+3*x**2,x**3-x**2,
                    6*x*x-6*x,3*x*x-4*x+1,
                    -6*x*x+6*x,3*x*x-2*x])


def convolution_weights(L,t,D,mode="airy",n_gauss=8,full_rows=True):
    h=t[1]-t[0]
    z,w=np.polynomial.legendre.leggauss(n_gauss)
    x=(z+1)/2
    w=w/2
    scale=np.array([h,h*h,h,h*h,1,h,1,h])
    shapes=hermite_shapes(x)*scale[:,None]
    rows=np.arange(L) if full_rows else D
    W=np.zeros((len(t),8,len(rows),len(D)),complex)
    for n in range(1,len(t)):
        K=propagator_block(L,(n-x)*h,rows,D,mode)
        W[n]=np.einsum("p,ap,pij->aij",w,shapes,K,optimize=True)
    return W


def source_derivative(L,t,Q,D,mode,eps=.001):
    times=np.stack([t-2*eps,t-eps,t+eps,t+2*eps])
    near=t<2*eps
    times[:,near]=np.stack([t[near],t[near]+eps,t[near]+2*eps,t[near]+3*eps])
    source=np.stack([
        propagator_block(L,s,D,np.arange(L),mode)@Q for s in times
    ])
    d=(source[0]-8*source[1]+8*source[2]-source[3])/(12*eps)
    if np.any(near):
        d[near]=(-11*source[0,near]+18*source[1,near]-
                 9*source[2,near]+2*source[3,near])/(6*eps)
    return d


def solve_hermite_image(L,t,Q,pair,mode="airy",n_gauss=8,return_details=False):
    t=np.asarray(t,float)
    n=len(t)-1
    m=Q.shape[1]
    D=np.array([L//2-1,L//2])
    V=np.diag(pair)
    initial=propagator_block(L,t,np.arange(L),np.arange(L),mode)@Q
    g=initial[:,D]
    gp=source_derivative(L,t,Q,D,mode)
    W=convolution_weights(L,t,D,mode,n_gauss,full_rows=False)
    K=propagator_block(L,t,D,D,mode)

    mat=np.block([[np.eye(2)+1j*W[1,2]@V,1j*W[1,3]@V],
                  [1j*W[1,6]@V,np.eye(2)+1j*W[1,7]@V]])
    Y=np.zeros((n+1,2,m),complex)
    DY=np.zeros_like(Y)
    Y[0]=Q[D]
    DY[0]=gp[0]-1j*K[0]@V@Y[0]
    for k in range(1,n+1):
        weights=W[1:k+1]
        pieces=np.zeros((8,2,m),complex)
        for a,values in ((0,Y[:k][::-1]),(1,DY[:k][::-1]),
                         (4,Y[:k][::-1]),(5,DY[:k][::-1])):
            pieces[a]=np.einsum("nij,njq->iq",weights[:,a]@V,values,optimize=True)
        if k>1:
            for a,values in ((2,Y[1:k][::-1]),(3,DY[1:k][::-1]),
                             (6,Y[1:k][::-1]),(7,DY[1:k][::-1])):
                pieces[a]=np.einsum("nij,njq->iq",weights[1:,a]@V,values,optimize=True)
        rhs=np.vstack([g[k]-1j*np.sum(pieces[:4],axis=0),
                       gp[k]-1j*(K[k]@V@Y[0]+np.sum(pieces[4:],axis=0))])
        solution=np.linalg.solve(mat,rhs)
        Y[k]=solution[:2]
        DY[k]=solution[2:]

    WF=convolution_weights(L,t,D,mode,n_gauss,full_rows=True)
    F=initial.copy()
    VY=V@Y
    VDY=V@DY
    for k in range(1,n+1):
        weights=WF[1:k+1]
        conv=(np.einsum("nij,njq->iq",weights[:,0],VY[:k][::-1],optimize=True)
             +np.einsum("nij,njq->iq",weights[:,1],VDY[:k][::-1],optimize=True)
             +np.einsum("nij,njq->iq",weights[:,2],VY[1:k+1][::-1],optimize=True)
             +np.einsum("nij,njq->iq",weights[:,3],VDY[1:k+1][::-1],optimize=True))
        F[k]-=1j*conv
    F[0]=Q
    if return_details:
        return F,Y,DY,dict(
            matrix_condition=float(np.linalg.cond(mat)),
            max_node_residual=float(np.max(np.linalg.norm(Y-F[:,D],axis=(1,2)))))
    return F,Y,DY


def validate(L=48,h=.1,T=36,pairs=((1.2,.35),(.35,1.2),(.5,.2))):
    t=np.arange(0,T+h/4,h)
    Q=initial_occupied(L)
    result=[]
    for pair in pairs:
        entry=dict(L=L,h=h,pair=pair)
        for mode in ("airy","bessel"):
            F,_,_=solve_hermite_image(L,t,Q,pair,mode)
            entry[mode+"_peak"]=smoothed_peak(t,cap_curve(F),
                                  lo=24 if L==48 else 33,
                                  hi=30 if L==48 else 41)
        Fref,_,_=solve_hermite_volterra(L,t,Q,pair)
        entry["reference_peak"]=smoothed_peak(t,cap_curve(Fref),
                                  lo=24 if L==48 else 33,
                                  hi=30 if L==48 else 41)
        result.append(entry)
        print(entry,flush=True)
    return result


if __name__=="__main__":
    validate()
