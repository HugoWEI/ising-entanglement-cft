import numpy as np
from defect_residual import clean_modes

def initial_occupied(L,m=2.5):
    h0=np.diag(-np.ones(L-1),-1)+np.diag(-np.ones(L-1),1)
    _,Q=np.linalg.eigh(h0+np.diag(m*(-1.)**np.arange(L)))
    return Q[:,:L//2]

def exact_error_check(L,t,Q,pair,F):
    h0=np.diag(-np.ones(L-1),-1)+np.diag(-np.ones(L-1),1)
    h=h0.copy();h[L//2-1,L//2-1]+=pair[0];h[L//2,L//2]+=pair[1]
    e,U=np.linalg.eigh(h); M=U.T@Q
    return max(np.linalg.norm(f-U@(np.exp(-1j*e*ti)[:,None]*M),2) for f,ti in zip(F,t))
