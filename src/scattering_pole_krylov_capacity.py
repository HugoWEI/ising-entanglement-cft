"""Strong-defect capacity predictions from parity scattering + bound pole + Krylov.

NO ASYMMETRIC HAMILTONIAN DIAGONALIZATION is used here.

Inputs:
  symmetric-reference eigensystem (in parity_contour_capacity.py),
  exact rank-two scattering resolvent T_delta(z),
  bound-state pole E_b and projector P_b from a scalar secular equation,
  initial pure Gaussian occupied orbitals F0.

1. Integrate the exact finite-rank scattering contour to obtain
   approximate F_full(t)=U_delta(t) F0 without asymmetric eigenvectors.
2. Subtract the exact bound contribution
      F_bound(t)=exp(-i E_b t) P_b F0
   to obtain the continuum occupied orbitals F_c(t).
3. Form the valid dephased Gaussian covariance G_deph, and the
   rank-two bound-continuum coherence vectors X=[x,y].
4. Predict C(t) from
      C_pred(t)=Tr f(G_deph)+
        Tr f(Q_m†(G_deph+X J X†)Q_m)-Tr f(Q_m†G_deph Q_m)
   where Q_m spans the block Krylov space {X,G_deph X,...}.

This is a *fully nonperturbative in defect strength* approximation to
the CAPACITY CORRECTION, but uses an expensive (near-exact) contour
evaluation of the band propagator. It is NOT a compact or
computationally optimal original peak law. A future analytical
approximation to the continuum propagator is still necessary.

Current implementation assumes exactly one positive bound state E_b>2
and no additional out-of-band eigenmodes for the tested defect class.
"""
from __future__ import annotations
import numpy as np
from parity_contour_capacity import ContourCapacity
from bound_coherence_krylov import predict_capacity_from_dephased
from scattering_phase import entropy_capacity


class ScatteringPoleKrylovCapacity:
    def __init__(self,length,mean_potential,occupied_modes):
        self.length=int(length)
        self.contour=ContourCapacity(length,mean_potential,occupied_modes)
        self.f0=np.asarray(occupied_modes,complex)

    def trajectory(self,delta,times,krylov_blocks=8,
                   quadrature_nodes=768,isometry_tolerance=2e-6):
        delta=float(delta)
        times=np.asarray(times,float)
        r=self.contour.reference
        energy,projector=r.largest_eigenprojector_from_residue(delta)
        if energy<=2:
            raise ValueError("expected positive out-of-band localized mode")
        if abs(np.trace(projector)-1)>2e-8:
            raise RuntimeError("bound-pole projector is not rank-one normalized")
        idx=int(np.argmax(np.real(np.diag(projector))))
        diag=float(np.real(projector[idx,idx]))
        if diag<1e-9:
            raise RuntimeError("cannot reconstruct pole eigenvector")
        b=projector[:,idx]/np.sqrt(diag)
        if not np.allclose(np.outer(b,b.conj()),projector,atol=2e-8):
            raise RuntimeError("bound projector reconstruction failed")
        q=b.conj()@self.f0
        beta=float(np.vdot(q,q).real)
        band_overlap=projector@self.f0

        # THIS is the expensive scattering-based continuum input.
        ffull=self.contour.propagate(
            delta,times,quadrature_nodes=quadrature_nodes)
        vals=np.empty((len(times),3))
        raw_errors=np.empty(len(times))
        dim=np.empty(len(times),dtype=int)
        J=np.array([[0.,1.],[1.,0.]])
        a_len=self.length//2
        for it,(t,f) in enumerate(zip(times,ffull)):
            raw_errors[it]=np.linalg.norm(f.conj().T@f-np.eye(a_len))
            if raw_errors[it]>isometry_tolerance:
                raise RuntimeError("contour evolution not sufficiently unitary")
            phase=np.exp(-1j*energy*t)
            fbound=phase*band_overlap
            fc=f-fbound
            A=fc[:a_len,:]
            x=phase*b[:a_len]
            y=A@q.conj()
            gdeph=A@A.conj().T+beta*np.outer(x,x.conj())
            X=np.column_stack([x,y])
            approx,kept=predict_capacity_from_dephased(
                gdeph,X,J,num_steps=krylov_blocks)
            dim[it]=kept
            cdeph=entropy_capacity(np.linalg.eigvalsh(gdeph))[1]
            # Only diagnostic: exact contour capacity, NOT used in
            # calculation of approx/predicted capacity.
            physical=f[:a_len,:]@f[:a_len,:].conj().T
            cpure=entropy_capacity(np.linalg.eigvalsh(physical))[1]
            vals[it,:]=[approx,cdeph,cpure]
        return {
            "times":times,
            "capacities":vals,
            "bound_energy":float(energy),
            "maximum_krylov_dimension":int(np.max(dim)),
            "maximum_contour_isometry_error":float(np.max(raw_errors)),
        }
