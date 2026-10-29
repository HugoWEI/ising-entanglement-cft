"""Source-only defect-drive a posteriori residual for free-fermion quench.

Construct a continuous piecewise-linear impurity drive from arbitrary complex
nodal values Y_k and use the clean open-chain eigenbasis (known analytically)
to compute its *exact in time* Duhamel response at nodes, with phi functions.

The exact-arithmetic theorem for any continuous drive y(t):
 F_y(t)=U0(t)Q-i int_0^t U0(t-s)B V y(s)ds,
 r(t)=y(t)-D F_y(t),
 ||F_exact(t)-F_y(t)||_2 <= ||V|| int_0^t ||r(s)||_2 ds.

No full defective-chain eigendecomposition is used.
Numerical quadrature of ||r|| is diagnostic, NOT a certified interval bound.
"""
from __future__ import annotations
import numpy as np
from scipy.special import jv


def clean_modes(L):
    j=np.arange(1,L+1)[:,None]
    k=np.arange(1,L+1)[None,:]
    S=np.sqrt(2/(L+1))*np.sin(np.pi*j*k/(L+1))
    energy=-2*np.cos(np.pi*np.arange(1,L+1)/(L+1))
    return energy,S


def phi12(z):
    z=np.asarray(z,dtype=complex)
    out1=np.empty_like(z)
    out2=np.empty_like(z)
    small=np.abs(z)<0.1
    if np.any(~small):
        out1[~small]=np.expm1(z[~small])/z[~small]
        out2[~small]=(np.expm1(z[~small])-z[~small])/z[~small]**2
    if np.any(small):
        # Absolute error below 1e-15 for |z|<=0.1 and 20 terms
        a=z[small]
        p1=np.ones_like(a)
        p2=np.ones_like(a)*.5
        f1=np.ones_like(a)
        f2=np.ones_like(a)*.5
        for k in range(1,21):
            f1=f1*a/(k+1)
            f2=f2*a/(k+2)
            p1+=f1
            p2+=f2
        out1[small]=p1
        out2[small]=p2
    return out1,out2


def defect_drive_response(L,times,Q,potentials,Y,all_times=None):
    """Exact clean forced solution for piecewise-linear y(s) at t-grid nodes.

    For arbitrary interior times, all_times must be a sorted array including t0 and
    consecutive values within the original grid; linear interpolation is used.
    """
    t=np.asarray(times,float)
    Q=np.asarray(Q,complex)
    Y=np.asarray(Y,complex)
    assert np.allclose(np.diff(t),t[1]-t[0])
    assert Y.shape==(len(t),2,Q.shape[1])
    D=np.array([L//2-1,L//2]); V=np.diag(potentials)
    eps,S=clean_modes(L)
    S_D=S[D,:]
    drive_eig=S_D.T @ V
    Qhat=S.T@Q
    b=np.zeros((L,Q.shape[1]),complex)
    F=np.zeros((len(t),L,Q.shape[1]),complex)
    F[0]=Q
    for ix in range(1,len(t)):
        h=t[ix]-t[ix-1]
        z=-1j*eps*h
        p1,p2=phi12(z)
        fprev=drive_eig@Y[ix-1]
        fcur=drive_eig@Y[ix]
        b=np.exp(z)[:,None]*b+h*(p1[:,None]*fprev+p2[:,None]*(fcur-fprev))
        F[ix]=S@(np.exp(-1j*eps*t[ix])[:,None]*Qhat-1j*b)
    r=Y-F[:,D,:]
    return F,r


def diagnostic(L=48,dt=.1,T=30.,pair=(1.2,.35)):
    import sys,os
    sys.path.insert(0,os.path.join(os.path.dirname(__file__),'src'))
    import volterra_scattering_revival as vol
    from phase_revival_research import hopping
    t=np.arange(0,T+dt/4,dt)
    h0=hopping(L)
    _, vec=np.linalg.eigh(h0+np.diag(2.5*(-1.)**np.arange(L)))
    Q=vec[:,:L//2]
    Z,Y=vol.defect_response(L,t,*pair,Q,shells=1,return_full=True,return_defect=True)
    F,r=defect_drive_response(L,t,Q,pair,Y)
    # Source-only F - Z checks how well trapezoid approximates exact convolution.
    errnorms=np.array([np.linalg.norm(ri,2) for ri in r])
    capabs=np.array([np.linalg.norm(f-z,2) for f,z in zip(F,Z)])
    # Physical benchmark diagnostic, not part of the certificate.
    hf=h0.copy(); hf[L//2-1,L//2-1]+=pair[0];hf[L//2,L//2]+=pair[1]
    eig,U=np.linalg.eigh(hf); M=U.T@Q
    errors=np.array([np.linalg.norm(f-U@(np.exp(-1j*eig*ti)[:,None]*M),2) for f,ti in zip(F,t)])
    integral=np.trapezoid(errnorms,t)*max(abs(pair[0]),abs(pair[1]))
    return dict(L=L,dt=dt,T=T,pair=pair,node_residual_max=float(errnorms.max()),node_residual_l1_trap=float(np.trapezoid(errnorms,t)),source_only_bound_diagnostic=float(integral),measured_full_error_max=float(errors.max()),numerical_convolution_disagreement=float(capabs.max()),ratio=float(integral/errors.max())), t,errnorms,errors

if __name__=='__main__':
    import json
    for pair in [(1.2,.35),(.35,1.2)]:
        for dt in [.2,.1,.05]:
            row,*_=diagnostic(dt=dt,pair=pair)
            print(json.dumps(row),flush=True)


def hermite_residual_integral_majorant(L,t,Q,pair,Y,F):
    """Exact-arithmetic bound for integral_0^T ||r(t)|| dt for linear-drive clean response.

    Uses cubic Hermite interpolation of the two-site residual on each time cell,
    with a fourth-derivative bound. Provided nodal F is the *exact clean forced*
    response. Numerical evaluated value is NOT rigorous unless those nodes and
    norms are interval-enclosed.
    """
    from numpy.linalg import norm
    h0=np.diag(-np.ones(L-1),1)+np.diag(-np.ones(L-1),-1)
    D=np.array([L//2-1,L//2]);Vmat=np.zeros((L,L))
    Vmat[D,D]=pair
    b=max(abs(pair[0]),abs(pair[1]))
    h=t[1]-t[0]
    vals=Y-F[:,D,:]
    ydiff=(Y[1:]-Y[:-1])/h
    fdot=-1j*(h0@F+Vmat[:,D]@Y)
    # rhs dimensions: fdot=(n,L,M); Vmat[:,D]@Y uses numpy batch broadcast
    integrals=[]
    for i in range(len(t)-1):
        slope=ydiff[i]
        rp0=slope-fdot[i,D,:]
        rp1=slope-fdot[i+1,D,:]
        y_max=max(norm(Y[i],2),norm(Y[i+1],2))
        f_max=norm(F[i],2)+h*b*y_max
        m4=16*f_max+8*b*y_max+8*b*norm(slope,2)
        integrals.append(h*(norm(vals[i],2)+norm(vals[i+1],2))/2 +
                         h*h*(norm(rp0,2)+norm(rp1,2))/12 +
                         h**5*m4/384)
    return b*sum(integrals)
