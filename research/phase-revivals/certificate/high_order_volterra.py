"""Fourth-order cubic-Hermite collocation of the TWO-SITE impurity equation.

Uses exact clean-chain sine transform, exponential phi-functions, and a fixed
4x4 complex linear solve per time step, with M=L/2 right-hand sides.
Not an eigen-decomposition of the defective Hamiltonian.

Exact arithmetic: piecewise-cubic drive makes nodal two-site residual and its
first derivative vanish. Source-only Hermite remainder bounds the integrated
residual on the entire time interval.
"""
from __future__ import annotations
import numpy as np
import math
from defect_residual import clean_modes


def phi1_to_4(z):
    z=np.asarray(z,complex)
    out=[]
    for j in range(1,5):
        term=np.ones_like(z)/math.factorial(j)
        p=term.copy()
        for k in range(1,50):
            term=term*z/(j+k)
            p+=term
        out.append(p)
    return out


def hermite_weights(z):
    p1,p2,p3,p4=phi1_to_4(z)
    return (p1-6*p3+12*p4, p2-4*p3+6*p4,
            6*p3-12*p4, -2*p3+6*p4)


def solve_hermite_volterra(L,times,Q,pair,return_details=False):
    t=np.asarray(times,float)
    assert len(t)>1 and t[0]==0 and np.allclose(np.diff(t),t[1]-t[0])
    dt=t[1]-t[0]
    Q=np.asarray(Q,complex)
    m=Q.shape[1]
    e,S=clean_modes(L)
    D=np.array([L//2-1,L//2]);V=np.diag(pair);SD=S[D,:]
    drive=SD.T@V  # S.T @ B V
    z=-1j*e*dt
    w00,w10,w01,w11=hermite_weights(z)
    P=np.exp(z)
    A1=dt*w01[:,None]*drive
    A2=dt*dt*w11[:,None]*drive
    D_A1=SD@A1; D_A2=SD@A2
    D_HA1=SD@(e[:,None]*A1); D_HA2=SD@(e[:,None]*A2)
    K=np.block([[np.eye(2)+1j*D_A1,1j*D_A2],
                [D_HA1+1j*V,np.eye(2)+D_HA2]])
    eigF= S.T@Q
    Y=np.zeros((len(t),2,m),complex); DY=np.zeros_like(Y)
    Y[0]=Q[D]
    DY[0]=-1j*(SD@(e[:,None]*eigF)+V@Y[0])
    frames=np.zeros((len(t),L,m),complex);frames[0]=Q
    max_node_r=0;max_node_dr=0
    for k in range(1,len(t)):
        baseFhat=P[:,None]*eigF-1j*(dt*w00[:,None]*(drive@Y[k-1]) +dt**2*w10[:,None]*(drive@DY[k-1]))
        rhs=np.vstack([SD@baseFhat,-1j*SD@(e[:,None]*baseFhat)])
        sol=np.linalg.solve(K,rhs)
        Y[k]=sol[:2];DY[k]=sol[2:]
        eigF=baseFhat-1j*(A1@Y[k]+A2@DY[k])
        frames[k]=S@eigF
        max_node_r=max(max_node_r,float(np.linalg.norm(Y[k]-SD@eigF,2)))
        max_node_dr=max(max_node_dr,float(np.linalg.norm(DY[k]+1j*(SD@(e[:,None]*eigF)+V@Y[k]),2)))
    if return_details: return frames,Y,DY,dict(node_r=max_node_r,node_dr=max_node_dr)
    return frames,Y,DY


def cell_coefficients(y0,y1,dy0,dy1,h):
    return [y0,h*dy0,-3*y0-2*h*dy0+3*y1-h*dy1,
            2*y0+h*dy0-2*y1+h*dy1]


def cubic_drive_error_majorant(L,t,Q,pair,F,Y,DY):
    """Exact-arithmetic continuous time source-residual bound given exact nodal
    exponential convolution values and collocation data; returns float evaluation.
    """
    from numpy.linalg import norm
    h=float(t[1]-t[0]);b=max(abs(pair[0]),abs(pair[1]));total=0.0
    for k in range(len(t)-1):
        cs=cell_coefficients(Y[k],Y[k+1],DY[k],DY[k+1],h)
        y_max=max(norm(Y[k],2),norm(Y[k+1],2)) +h/4*max(norm(DY[k],2),norm(DY[k+1],2))
        yp_max=(norm(cs[1],2)+2*norm(cs[2],2)+3*norm(cs[3],2))/h
        ypp_max=(2*norm(cs[2],2)+6*norm(cs[3],2))/h**2
        yppp_max=6*norm(cs[3],2)/h**3
        fmax=norm(F[k],2)+h*b*y_max
        m4=16*fmax+8*b*y_max+8*b*yp_max+2*b*ypp_max+b*yppp_max
        # endpoints of r and r' should vanish in exact arithmetic; residuals are
        # numerical and therefore not interval-certified here
        total+=h**5/384*m4
    return b*total


def diagnostic(L=48,dt=.2,T=30,pair=(1.2,.35)):
    from phase_revival_helper import initial_occupied,exact_error_check
    t=np.arange(0,T+dt/4,dt)
    Q=initial_occupied(L)
    F,Y,DY,detail=solve_hermite_volterra(L,t,Q,pair,True)
    bound=cubic_drive_error_majorant(L,t,Q,pair,F,Y,DY)
    measured=exact_error_check(L,t,Q,pair,F)
    return dict(L=L,dt=dt,T=T,pair=pair,source_majorant=float(bound),measured_error=float(measured),**detail)

if __name__=='__main__':
    import json
    for pair in [(1.2,.35),(.35,1.2)]:
        for dt in [.2,.1,.05,.025]:
            print(json.dumps(diagnostic(dt=dt,pair=pair)),flush=True)
