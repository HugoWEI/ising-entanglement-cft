"""Measure midpoint self-consistency errors of cubic-Hermite Volterra modes.

A diagnostic, NOT a uniform-in-cell or formally certified quadrature bound.
Requires the sibling hermite_airy solver previously committed to this repo.
"""
from pathlib import Path
import sys,json
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hermite_airy'))
from airy_kernel_residual import olver_value_derivative,image_kernel
import hermite_airy_solver as solver

def install():
    solver.leading_olver_j=lambda n,x:olver_value_derivative(n,x)[0]
    solver.smooth_source_derivative=lambda L,t,Q,mode,D,eps=.001: image_kernel(
        L,t,rows=D,cols=np.arange(L),mode=mode,derivative=True)@Q
    return solver

def hermite_interpolation(Y,DY,j,x,h):
    x=np.asarray(x)
    p0=2*x**3-3*x**2+1
    p1=x**3-2*x**2+x
    p2=-2*x**3+3*x**2
    p3=x**3-x**2
    return (p0[...,None,None]*Y[j]
           +p1[...,None,None]*h*DY[j]
           +p2[...,None,None]*Y[j+1]
           +p3[...,None,None]*h*DY[j+1])

def intercell_defects(L,t,Q,pair,Y,DY,n_gauss=12):
    h=t[1]-t[0];cells=len(t)-1;D=np.array([L//2-1,L//2])
    V=np.diag(pair)
    z,w=np.polynomial.legendre.leggauss(n_gauss)
    q_mid=[]
    for k in range(cells):
        tm=(k+.5)*h
        j=np.arange(k+1)
        lengths=np.ones(k+1);lengths[-1]=.5
        x=lengths[:,None]*(z[None,:]+1)/2
        wt=lengths[:,None]*w[None,:]*h/2
        ys=hermite_interpolation(Y,DY,j[:,None],x,h)
        lags=(tm-h*(j[:,None]+x)).reshape(-1)
        K=image_kernel(L,lags,rows=D,cols=D,mode='airy').reshape(k+1,n_gauss,2,2)
        convolution=np.einsum('jp,jpab,jpbq->aq',wt,K,V@ys,optimize=True)
        g=image_kernel(L,[tm],rows=D,cols=np.arange(L),mode='airy')[0]@Q
        ymid=hermite_interpolation(Y,DY,k,.5,h)
        q_mid.append(g-1j*convolution-ymid)
    return np.array(q_mid)

def run(L=48,T=30.,dts=(.2,.1),pairs=((1.2,.35),(.35,1.2))):
    install()
    from phase_revival_helper import initial_occupied
    Q=initial_occupied(L); rows=[]
    for dt in dts:
        t=np.arange(0,T+dt/4,dt)
        for pair in pairs:
            F,Y,DY=solver.solve_hermite_image(L,t,Q,pair,'airy')
            qmid=intercell_defects(L,t,Q,pair,Y,DY)
            norm=np.array([np.linalg.norm(q,2) for q in qmid])
            rows.append(dict(L=L,T=T,dt=dt,pair=pair,
                max_midpoint_q=float(norm.max()),
                midpoint_L1_estimate=float(norm.sum()*dt),
                induced_state_L1_estimate=float(max(map(abs,pair))*norm.sum()*dt),
                max_nodal_q=float(max(np.linalg.norm(
                    Y[k]-F[k,L//2-1:L//2+1],2) for k in range(len(t))))))
            print(rows[-1],flush=True)
    (HERE/'intercell_defect_results.json').write_text(json.dumps(rows,indent=2))
    return rows

if __name__=='__main__':run()
