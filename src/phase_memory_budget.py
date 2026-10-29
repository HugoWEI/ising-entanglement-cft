"""A conserved initial-coherence budget for bound-mode phase control.

Model: number-conserving pure Gaussian fermions; an arbitrary fixed
Hermitian single-particle Hamiltonian h with a normalized simple bound
eigenvector b. The pure initial correlation projector is P0=F0 F0†.
Define the bound occupation beta=<b|P0|b> in [0,1].

The phase gate U_theta=I+(exp(i theta)-1)|b><b| commutes with h.
Under identical subsequent evolution, its global covariance difference
has EXACT constant Frobenius norm

 ||G_full(theta,t)-G_full(phi,t)||_F
      =2 sqrt(2) |sin((theta-phi)/2)| sqrt(beta*(1-beta)).

Its trace norm is likewise
      =4 |sin((theta-phi)/2)| sqrt(beta*(1-beta)).

The derivation uses v=(I-|b><b|)P0 b. For P0^2=P0,
 ||v||^2=beta(1-beta); the global difference is a rank-two
off-diagonal operator connecting b and v. It has eigenvalues
 +/-2|sin((theta-phi)/2)| sqrt(beta*(1-beta)).

Under projection to a spatial subsystem A, let
 alpha_A=||P_A b||^2 and y=P_A e^(-i h t)(I-Pb)P0 b.
Then ||y||<=sqrt(beta(1-beta)) and
  ||G_A(theta)-G_A(phi)||_1
       <=4 |sin((theta-phi)/2)| sqrt(alpha_A beta(1-beta)).
This is a rigorous sensitivity ceiling that depends on initial
coherence AND bound-state weight in A, but NOT on the system size
directly. The actual response can be much smaller through interference.

For modular capacity C=Tr f(G_A),
  f(p)=p(1-p)ln^2[(1-p)/p],
we have a global scalar 1/2-Hölder bound
  |f(p)-f(q)| <=12 sqrt(|p-q|), for p,q in [0,1],
from |f'(p)|<=4/sqrt(min(p,1-p)). Combining with the
Hermitian eigenvalue trace-norm inequality gives
  |C_A(theta)-C_A(phi)|
       <=12 sqrt(d_A * ||G_A(theta)-G_A(phi)||_1).
It is conservative but endpoint-safe (no spectral gap assumption).

If both *subsystem* spectra lie in [eps,1-eps], one obtains the
stronger (conditional) dimension-independent bound
  |Delta C| <= [L_eps^2+2 L_eps] * ||Delta G_A||_1,
  L_eps=ln((1-eps)/eps).
The Gaussian capacity itself is not monotone in initial coherence.

This is elementary projector algebra and matrix perturbation theory,
not a new theorem of quantum mechanics. Its role is an actionable
mechanism constraint on any proposed phase-sensitive revival law.
"""
from __future__ import annotations
import numpy as np


def bound_mode_occupation(bound_orbital,occupied_modes):
    """Return beta=<b|F F†|b> and test pure-state normalization."""
    b=np.asarray(bound_orbital,complex)
    f=np.asarray(occupied_modes,complex)
    if b.ndim!=1 or f.ndim!=2 or f.shape[0]!=len(b):
        raise ValueError("invalid orbital dimensions")
    if not np.allclose(np.vdot(b,b),1,atol=1e-9):
        raise ValueError("bound orbital must be normalized")
    if not np.allclose(f.conj().T@f,np.eye(f.shape[1]),atol=1e-9):
        raise ValueError("occupied orbitals must be orthonormal")
    beta=float(np.linalg.norm(b.conj()@f)**2)
    if beta< -1e-9 or beta>1+1e-9:
        raise RuntimeError("invalid bound occupation")
    # A projector expectation can round to 1 - O(machine epsilon);
    # saturate numerically indistinguishable endpoint occupations.
    if beta<1e-12:
        return 0.0
    if 1-beta<1e-12:
        return 1.0
    return min(1.,max(0.,beta))


def coherence_budget(beta,phase_difference,subsystem_weight=1.):
    """Exact GLOBAL norms and conservative SUBSYSTEM trace-norm bound.

    phase_difference = theta-phi in radians; alpha=||P_A b||^2.
    No knowledge of time, band energies or scattering phases required.
    """
    beta=float(beta)
    alpha=float(subsystem_weight)
    d=float(phase_difference)
    if not (np.isfinite(beta) and -1e-10<=beta<=1+1e-10
            and np.isfinite(alpha) and -1e-10<=alpha<=1+1e-10
            and np.isfinite(d)):
        raise ValueError("invalid occupation, subsystem weight or phase")
    beta=np.clip(beta,0.,1.)
    alpha=np.clip(alpha,0.,1.)
    coh=np.sqrt(beta*(1-beta))
    phase=abs(np.sin(d/2))
    return {
        "global_frobenius_exact":float(2*np.sqrt(2)*phase*coh),
        "global_trace_exact":float(4*phase*coh),
        "subsystem_trace_upper":float(4*phase*np.sqrt(alpha)*coh),
        "beta":float(beta),
        "alpha":float(alpha),
    }


def capacity_endpoint_holder_bound(dimension,trace_norm_bound):
    """Uniform capacity bound for any subsystem spectra in [0,1]."""
    d=int(dimension)
    trace=float(trace_norm_bound)
    if d<1 or not np.isfinite(trace) or trace<0:
        raise ValueError("invalid dimension or trace norm")
    return float(12*np.sqrt(d*trace))


def capacity_spectral_gap_bound(gap,trace_norm_bound):
    """Dimension-INDEPENDENT bound if both spectra in [gap,1-gap]."""
    gap=float(gap)
    trace=float(trace_norm_bound)
    if not (0<gap<=.5) or not np.isfinite(trace) or trace<0:
        raise ValueError("invalid spectral gap or trace norm")
    L=np.log((1-gap)/gap)
    return float((L**2+2*L)*trace)


def pure_phase_gate(bound_orbital,occupied_modes,theta):
    """Pure, fixed-energy phase preparation of initially occupied modes."""
    b=np.asarray(bound_orbital,complex)
    f=np.asarray(occupied_modes,complex)
    beta=bound_mode_occupation(b,f)
    theta=float(theta)
    if not np.isfinite(theta):
        raise ValueError("invalid phase")
    kicked=f+np.expm1(1j*theta)*np.outer(b,b.conj()@f)
    return kicked,beta


def covariance_phase_distance(covariance_a,covariance_b):
    """Return 1-norm, Frobenius, operator norm of Hermitian contrast."""
    a=np.asarray(covariance_a,complex)
    b=np.asarray(covariance_b,complex)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=a.shape[1]:
        raise ValueError("incompatible covariance matrices")
    d=a-b
    if not np.allclose(d,d.conj().T,atol=1e-9):
        raise ValueError("contrast is not Hermitian")
    eig=np.linalg.eigvalsh(d)
    return {
        "trace":float(np.sum(np.abs(eig))),
        "frobenius":float(np.linalg.norm(eig)),
        "operator":float(np.max(np.abs(eig))),
    }


def capacity_difference_bounds(beta,alpha,phase_difference,
                               subsystem_dimension,spectral_gap=None):
    """Budget-based capacity bounds; gap bound optional and conditional."""
    budget=coherence_budget(beta,phase_difference,alpha)
    bound=capacity_endpoint_holder_bound(
        subsystem_dimension,budget["subsystem_trace_upper"])
    output={**budget,"capacity_holder_upper":bound}
    if spectral_gap is not None:
        output["capacity_gap_upper"]=capacity_spectral_gap_bound(
            spectral_gap,budget["subsystem_trace_upper"])
    return output



def pure_slater_phase_overlap(beta,phase_difference):
    """Exact many-body overlap for bound-number phase gates.

    The many-body unitary is exp(i theta n_b) with n_b=b†b in {0,1}.
    For ANY initial pure Slater determinant with initial occupation beta,
      <Psi_theta|Psi_phi> = 1-beta + beta*exp(i*(phi-theta)).
    This overlap is conserved under any common subsequent Hamiltonian
    evolution; if the generator commutes with h it also preserves
    each spectral occupation and moment.

    We return the complex overlap in the convention phi-theta.
    """
    beta=float(beta)
    diff=float(phase_difference)
    if not np.isfinite(beta) or beta < -1e-10 or beta >1+1e-10 or (
            not np.isfinite(diff)):
        raise ValueError("invalid beta or phase difference")
    beta=float(np.clip(beta,0,1))
    return complex(1-beta+beta*np.exp(1j*diff))


def pure_slater_phase_fidelity(beta,phase_difference):
    """Exact |<Psi_theta|Psi_phi>|^2=1-4 beta(1-beta) sin^2(d/2)."""
    overlap=pure_slater_phase_overlap(beta,phase_difference)
    return float(abs(overlap)**2)


def bound_number_phase_quantum_fisher(beta):
    """Pure-state phase QFI F_Q=4 Var(n_b)=4 beta(1-beta).

    This is standard pure-state quantum metrology, not a new
    capacity-revival law. It gives an operational interpretation
    of the conserved phase-coherence budget, F_Q=4*B^2.
    """
    beta=float(beta)
    if not np.isfinite(beta) or not 0<=beta<=1:
        raise ValueError("invalid occupation")
    return float(4*beta*(1-beta))
