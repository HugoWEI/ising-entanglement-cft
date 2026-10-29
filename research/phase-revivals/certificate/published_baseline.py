"""Capacity predicted by differentiating Eq.(64) of Capizzi & Eisler (2023).

IMPORTANT: This is for their scale-invariant conformal hopping defect, not a
prediction for the present two-site, momentum-dependent onsite-potential defect.
We use the paper's |v_q|=|sin(q)| and t_paper=2*t_ours, mu_paper=m_ours/2.
C_A(t)-C_A(0) is estimated; finite-size initial offset is not part of Eq.(64).
"""
import numpy as np

def mode_capacity(p):
    x=np.clip(np.asarray(p,float),1e-15,1-1e-15)
    return x*(1-x)*np.log((1-x)/x)**2

def occupation_from_product(q):
    """Select >=1/2 root, which leaves entropy and capacity unchanged."""
    return .5*(1+np.sqrt(np.maximum(0,1-4*q)))

def capizzi_eisler_capacity(Ltotal,mass,physical_times,lam=1.,Nq=8192):
    assert Ltotal%2==0 and 0<=lam<=1 and mass>=0
    half=Ltotal//2
    q=-np.pi+(np.arange(Nq)+.5)*2*np.pi/Nq
    mu=mass/2
    n=.5*(1+np.cos(q)/np.sqrt(np.cos(q)**2+mu**2))
    mix=n*(1-n)
    n1=occupation_from_product(lam**2*mix)
    n2=occupation_from_product(4*lam**2*(1-lam**2)*mix)
    c1=mode_capacity(n1); c2=mode_capacity(n2)
    v=np.abs(np.sin(q))
    t=np.asarray(physical_times,float)
    x=v[None,:]*2*t[:,None]/(2*half)
    fractional=x-np.floor(x)
    weight1=2*half*np.minimum(fractional,1-fractional)
    y=v[None,:]*2*t[:,None]/(4*half)
    fy=y-np.floor(y)
    weight2=half*np.maximum(0,1-np.abs(4*fy-2))
    return np.mean(c1[None,:]*weight1+c2[None,:]*weight2,axis=1)

def test_limits():
    t=np.linspace(0,25,101)
    zero=capizzi_eisler_capacity(48,2.5,t,lam=0)
    assert max(abs(zero))<1e-10
    homo=capizzi_eisler_capacity(48,2.5,t,lam=1)
    assert abs(homo[0])<1e-12
    # At lambda=1, second term vanishes, first is strictly positive at small times
    assert homo[5]>0
    return float(max(homo))

if __name__=='__main__':
    from scipy.ndimage import gaussian_filter1d
    import json
    from defect_residual import clean_modes
    L=48;m=2.5;t=np.arange(0,48*.8+.0001,.2)
    h0=np.diag(-np.ones(L-1),1)+np.diag(-np.ones(L-1),-1)
    hi=h0+np.diag(m*(-1.)**np.arange(L))
    _,Q=np.linalg.eigh(hi);Q=Q[:,:L//2]
    eps,S=clean_modes(L)
    D=S.T@Q
    exact=[]
    for tt in t:
        F=S@(np.exp(-1j*eps*tt)[:,None]*D)
        C=F[:L//2]@F[:L//2].conj().T
        exact.append(mode_capacity(np.linalg.eigvalsh(C)).sum())
    exact=np.array(exact)
    pred=capizzi_eisler_capacity(L,m,t,lam=1)
    diff=exact-exact[0]
    print(json.dumps(dict(initial_capacity=float(exact[0]),max_abs_homogeneous_capacity_change_difference=float(np.max(abs(pred-diff))),homogeneous_capacity_RMSE=float(np.sqrt(np.mean((pred-diff)**2))),published_lambda_zero_bound=test_limits()),indent=2))
