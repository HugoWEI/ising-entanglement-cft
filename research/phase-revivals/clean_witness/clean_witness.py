"""Source-only residual and between-node analytic majorant for a two-site cubic-Hermite drive.

The drive is computed by the Airy approximation, but the *verification witness*
uses the exactly defined clean sine-mode propagator U0(t), not Airy/Bessel
images or the defective-chain solution.

Proof is in RESEARCH_NOTE.md. Numeric quadrature/nodal norms are floating-point
and NOT outward-rounded certificates.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/data')
sys.path[:0]=[str(ROOT/'hermite_airy_update'), str(ROOT/'airy_kernel_residual_release')]
from hermite_airy_solver import solve_hermite_image, hermite_shapes
from phase_revival_helper import initial_occupied
from scipy.linalg import eigh_tridiagonal


def sine_modes(L):
    k=np.arange(1,L+1);i=np.arange(1,L+1)
    s=np.sqrt(2/(L+1))*np.sin(np.pi*i[:,None]*k[None,:]/(L+1))
    return -2*np.cos(np.pi*k/(L+1)),s


def witness_nodes(L,t,Q,pair,Y,DY,n_gauss=12):
    """Compute F_Y(t_j)=U0(t_j)Q-i U0*B VY using Gauss-Hermite cell quadrature.

    The integral is smooth/entire but finite Gauss quadrature remains a numerical
    approximation; the mathematical witness assumes the exact integral.
    """
    t=np.asarray(t); h=float(t[1]-t[0]);n=len(t)-1
    D=np.array([L//2-1,L//2]);V=np.diag(pair)
    eig,S=sine_modes(L)
    SQ=S.T@Q
    orig=(S[None,:,:]*np.exp(-1j*t[:,None,None]*eig[None,None,:]))@SQ
    x,w=np.polynomial.legendre.leggauss(n_gauss)
    x=(x+1)/2; w=w/2
    H=hermite_shapes(x)[:4]*np.array([h,h*h,h,h*h])[:,None]
    W=np.zeros((n+1,4,L,2),complex)
    Sd=S[D,:]
    for lag in range(1,n+1):
        phases=np.exp(-1j*((lag-x)*h)[:,None]*eig[None,:])
        # K(t)_{all,D}=S @ diag(phases) @ Sd.T
        B=np.einsum('ik,pk,dk->pid',S,phases,Sd,optimize=True)
        W[lag]=np.einsum('p,ap,pid->aid',w,H,B,optimize=True)
    F=orig.copy()
    VY=V@Y;VDY=V@DY
    for k in range(1,n+1):
        ws=W[1:k+1]
        F[k]-=1j*(np.einsum('nij,njq->iq',ws[:,0],VY[:k][::-1],optimize=True)
               +np.einsum('nij,njq->iq',ws[:,1],VDY[:k][::-1],optimize=True)
               +np.einsum('nij,njq->iq',ws[:,2],VY[1:k+1][::-1],optimize=True)
               +np.einsum('nij,njq->iq',ws[:,3],VDY[1:k+1][::-1],optimize=True))
    return F


def majorant(L,t,pair,Y,DY,F):
    """Exact-arithmetic *formulas*, with floating point inputs and norms here.

    Per-cell integral of ||q|| ≤ h/2(||q_j||+||q_{j+1}||)
     + h²/12(||q'_j||+||q'_{j+1}||)+h^5 M4_j/720.
    """
    h=float(t[1]-t[0]);D=np.array([L//2-1,L//2]);b=max(map(abs,pair))
    hF=np.zeros_like(F)
    hF[:,1:]-=F[:,:-1]
    hF[:,:-1]-=F[:,1:]
    q=F[:,D]-Y
    qdot=-1j*hF[:,D]-1j*np.diag(pair)@Y-DY
    def op(x):return float(np.linalg.norm(x,ord=2))
    qnorm=np.array([op(x) for x in q]);dqn=np.array([op(x) for x in qdot])
    rows=[]
    for j in range(len(t)-1):
        y0,y1=Y[j:j+2];dy0,dy1=DY[j:j+2]
        # Exact cubic Hermite endpoint derivatives.
        ypp0=(6*(y1-y0)/h**2)-(4*dy0+2*dy1)/h
        ypp1=(-6*(y1-y0)/h**2)+(2*dy0+4*dy1)/h
        yppp=(-12*(y1-y0)/h**3)+(6*(dy0+dy1)/h**2)
        # Y'' affine on the cell; the norm is convex, maximal at an endpoint.
        M2=max(op(ypp0),op(ypp1));M3=op(yppp)
        M1=op(dy0)+h*M2
        M0=op(y0)+h*op(dy0)+.5*h*h*M2
        # local source-state envelope: unitarity-free, only ||h0|| ≤2.
        MF=(op(F[j])+h*b*M0)/(1-2*h)
        M4=16*MF+8*b*M0+4*b*M1+2*b*M2+b*M3
        node=h/2*(qnorm[j]+qnorm[j+1])+h*h/12*(dqn[j]+dqn[j+1])
        remainder=h**5*M4/720
        rows.append((node,remainder,M4,MF,M0,M1,M2,M3))
    cells=np.array(rows)
    return {'q_nodes_max':float(qnorm.max()),'dq_nodes_max':float(dqn.max()),
      'q_node_l1_bound':float(np.sum(cells[:,0])),
      'q_hermite_l1_bound':float(np.sum(cells[:,1])),
      'q_total_l1_bound':float(np.sum(cells[:,0:2])),
      'state_error_l1_bound':float(b*np.sum(cells[:,0:2])),
      'max_M4':float(np.max(cells[:,2])),
      'max_F_cell_bound':float(np.max(cells[:,3])),
      'nodes':qnorm,'dq_nodes':dqn,'cell':cells}


def run(L=48,T=30,dt=.1,pairs=((1.2,.35),(.35,1.2)),n_gauss=12):
    import json
    t=np.arange(0,T+dt/4,dt);Q=initial_occupied(L)
    results=[]
    for pair in pairs:
        FA,Y,DY=solve_hermite_image(L,t,Q,pair,'airy',n_gauss=8)
        F0=witness_nodes(L,t,Q,pair,Y,DY,n_gauss)
        ans=majorant(L,t,pair,Y,DY,F0)
        obs={'L':L,'T':T,'dt':dt,'pair':pair,'n_gauss':n_gauss,
          'max_airykernel_amplitude_discrepancy_at_nodes':float(max(np.linalg.norm(FA[j]-F0[j],2) for j in range(len(t)))),
          **{k:v for k,v in ans.items() if not isinstance(v,np.ndarray)}}
        results.append(obs);print(json.dumps(obs),flush=True)
    return results
if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,default=.1);p.add_argument('--T',type=float,default=30);p.add_argument('--gauss',type=int,default=12);a=p.parse_args()
    r=run(dt=a.dt,T=a.T,n_gauss=a.gauss)
    (HERE/'RESULTS.json').write_text(json.dumps(r,indent=2))
