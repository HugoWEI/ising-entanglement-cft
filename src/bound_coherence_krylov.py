"""Controlled block-Krylov hierarchy for a rank-two covariance update.

Given a VALID dephased fermionic covariance A (0<=A<=I),
the exact bound-continuum coherence update is A+X J X^dag,
where X has two columns and J=[[0,1],[1,0]].
The capacity correction is

    Delta C = Tr f(A+X J X†) - Tr f(A),
    f(p)=p(1-p)log^2((1-p)/p).

Block Krylov builds the subspace
    K_m(A,X)=span{X,AX,...,A^(m-1)X}
with dimension at most 2*m. With an orthonormal basis Q_m, a
nonperturbative but APPROXIMATE correction is

    Delta C_m=Tr f(Q†A Q + (Q†X)J(Q†X)†) - Tr f(Q†A Q).

It retains the COMPLETE finite-rank coherence interaction inside the
selected subspace. Contributions from the orthogonal complement,
including its dynamical coupling to Krylov modes, are omitted.

The approximation is EXACT if span Q_m is the entire subsystem, or
if the chosen subspace reduces both A and A+X J X†. Convergence of
capacity as m increases is empirical, not automatic/monotone.

This is standard projected matrix-function / Krylov physics, not
a new compact first-revival law. Knowledge of A and X is still needed
from a propagation model; our present tests use an exact spectral
reference to isolate approximation errors.
"""
from __future__ import annotations
import numpy as np
from scattering_phase import entropy_capacity


def _validated_matrices(covariance,X,J):
    A=np.asarray(covariance,complex)
    X=np.asarray(X,complex)
    J=np.asarray(J,complex)
    if A.ndim!=2 or A.shape[0]!=A.shape[1]:
        raise ValueError("covariance must be square")
    if X.ndim!=2 or X.shape[0]!=len(A) or X.shape[1]!=2:
        raise ValueError("X must have two columns")
    if J.shape!=(2,2):
        raise ValueError("J must be 2x2")
    if not np.allclose(A,A.conj().T,atol=1e-10):
        raise ValueError("covariance must be Hermitian")
    if not np.allclose(J,J.conj().T,atol=1e-10):
        raise ValueError("J must be Hermitian")
    return A,X,J


def block_krylov_basis(covariance,X,num_steps=8,tolerance=1e-11):
    """Double-reorthogonalized, rank-revealing block-Arnoldi basis.

    max dimension = min(subsystem_size, 2*num_steps). Orthogonality is
    validated rather than assumed. The block span uses [X,A X,...].
    """
    A=np.asarray(covariance,complex)
    X=np.asarray(X,complex)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or X.shape!=(len(A),2):
        raise ValueError("invalid A/X dimensions")
    if not isinstance(num_steps,int) or num_steps<1 or tolerance<=0:
        raise ValueError("invalid steps/tolerance")
    blocks=[]
    source=X.copy()
    for _ in range(num_steps):
        for _ in range(2):
            for Qprev in blocks:
                source-=Qprev@(Qprev.conj().T@source)
        U,s,_=np.linalg.svd(source,full_matrices=False)
        rank=int(np.count_nonzero(s>tolerance))
        if not rank:
            break
        current=U[:,:rank]
        blocks.append(current)
        if sum(block.shape[1] for block in blocks)>=len(A):
            break
        source=A@current
    if not blocks:
        raise ValueError("Krylov starting block has zero rank")
    Q=np.hstack(blocks)
    if not np.allclose(Q.conj().T@Q,np.eye(Q.shape[1]),atol=1e-9):
        raise RuntimeError("Krylov subspace has lost orthogonality")
    return Q


def krylov_capacity_correction(covariance,X,J,num_steps=8):
    """Return a nonlinear projected coherence capacity correction.

    Returns correction and retained dimension. It does not compute
    Tr f(A), which can be provided independently by a Gaussian solver.
    """
    A,X,J=_validated_matrices(covariance,X,J)
    Q=block_krylov_basis(A,X,num_steps)
    H=Q.conj().T@A@Q
    Z=Q.conj().T@X
    Hp=H+Z@J@Z.conj().T
    for Htest in (H,Hp):
        p=np.linalg.eigvalsh(Htest)
        if p.min() < -2e-9 or p.max()>1+2e-9:
            raise RuntimeError("projected state has invalid fermionic occupations")
    c0=entropy_capacity(np.linalg.eigvalsh(H))[1]
    c1=entropy_capacity(np.linalg.eigvalsh(Hp))[1]
    return float(c1-c0),int(Q.shape[1])


def predict_capacity_from_dephased(covariance,X,J,num_steps=8):
    """C_deph + Krylov rank-two correction; no full G eigensolver."""
    A,X,J=_validated_matrices(covariance,X,J)
    eig=np.linalg.eigvalsh(A)
    if eig.min() < -2e-9 or eig.max()>1+2e-9:
        raise RuntimeError("dephased covariance not physical")
    base=entropy_capacity(eig)[1]
    correction,dim=krylov_capacity_correction(A,X,J,num_steps)
    return float(base+correction),dim
