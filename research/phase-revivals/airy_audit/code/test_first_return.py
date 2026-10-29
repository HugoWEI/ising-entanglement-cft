import numpy as np
from scipy.linalg import expm
from analytic_first_return import wrap_matrix, rank_approx
from scattering_generator import polar_decomp, projector, tangent_generator, projected_derivative


def test_uniform_airy_vs_exact_bessel_at_return():
    t=np.array([26.4,27.3])
    exact=wrap_matrix(48,t,method='bessel')
    approx=wrap_matrix(48,t,method='uniform_airy')
    for a,b in zip(exact,approx):
        assert np.linalg.norm(a-b,2)/np.linalg.norm(a,2)<3e-4


def test_rank_bound_and_physicality():
    rng=np.random.default_rng(7)
    L=48;Q,_=np.linalg.qr(rng.standard_normal((L,L//2)))
    Z0=Q.astype(complex)
    E=wrap_matrix(L,np.array([26.4]),'uniform_airy')[0]@Q
    E4,_=rank_approx(E[None],rank=4);E4=E4[0]
    X,_=polar_decomp(Z0+E)
    Xr,_=polar_decomp(Z0+E4)
    sigma=np.linalg.svd(E,compute_uv=False)[4]
    den=np.linalg.svd(Z0+E,compute_uv=False)[-1]+np.linalg.svd(Z0+E4,compute_uv=False)[-1]
    diff=np.linalg.norm(projector(X)-projector(Xr),2)
    assert diff <=4*sigma/den+1e-10
    assert np.linalg.norm(Xr.conj().T@Xr-np.eye(L//2))<1e-9
    assert np.linalg.eigvalsh(Xr[:L//2]@Xr[:L//2].conj().T).max() <=1+1e-10


def test_scattering_generator_identity():
    rng=np.random.default_rng(11)
    L=24
    Q,_=np.linalg.qr(rng.standard_normal((L,L//2)))
    E=wrap_matrix(L,np.array([14.0]),'uniform_airy')[0]@Q
    Z=Q+0.3*E
    x,uu,s,vh,a=tangent_generator(Z,E)
    h=1e-5
    xp,_=polar_decomp(Z+h*E)
    xm,_=polar_decomp(Z-h*E)
    dp=(projector(xp)-projector(xm))/(2*h)
    assert np.linalg.norm(dp-projected_derivative(x,a))<2e-7
    K=a@x.conj().T-x@a.conj().T
    assert np.linalg.norm(K+K.conj().T)<1e-10
    assert np.linalg.norm(K@projector(x)-projector(x)@K-dp)<2e-7
