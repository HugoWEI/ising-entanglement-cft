# Dynamics of entanglement entropy and capacity across an Ising defect

**Status (25 Oct 2026): Literature-screened research question; preliminary numerical implementation. Not verified original research.**

## Precise question

For an open transverse-field Ising chain with a central bond defect of strength
`d`, what happens to the *time dependence* of entropy `S_A(time)` and
modular capacity `C_A(time)` following a global field quench? Does the defect
suppress the growth of the two observables by the same relative factor?

We use `d` for defect strength and `tau` for evolution time to avoid the
ambiguous `t` used for both in much of the literature.

## Protocol implemented

1. Set even chain length L, J=1, open boundaries, central bond J_(L/2)=d.
2. Prepare the ground state with transverse field h_i=2 (gapped initial state).
3. At tau=0, change ONLY the field to h_f=1 (bulk critical point); keep the
   defect fixed. This is a *global field quench*, NOT a joining/local quench.
4. Evolve the Majorana covariance under the post-quench Hamiltonian:
   Gamma(tau)=U(tau) Gamma(0) U(tau)^T, U=exp(A_f tau).
5. Restrict to half the chain and calculate entropy AND capacity using its
   fermionic occupation probabilities p_k.

    S = sum_k [-p_k ln p_k - (1-p_k) ln(1-p_k)]
    C = sum_k p_k(1-p_k) [ln((1-p_k)/p_k)]²

Capacity is the *variance of the modular Hamiltonian* and is not measurement
error or generic 'entanglement-generating capacity'.

## Validation (implemented)

- Direct spin-basis exact diagonalization independently agrees with the
  Majorana method at L=4 for several defect values and evolution times.
- With d=0, the disconnected halves remain unentangled.
- Without a quench, both quantities are stationary.
- Ground-state covariance obeys purity condition Gamma²=-I.
- A maximally entangled bipartition has nonzero entropy and zero capacity.

Run `python -m pytest -q` and `python scripts/run_quench.py`.

## Key prior work and remaining uncertainty

- Wen, Wang & Ryu (2018), arXiv:1711.02126: entropy dynamics across a
  conformal interface, including global and local quenches.
- Capizzi & Eisler (2023), arXiv:2209.03297: entropy evolution after
  a global quench across a conformal defect.
- Arias et al. (2023), arXiv:2301.02117: capacity dynamics after
  global quenches, predominantly homogeneous systems.
- Bonsignori & Eisler (2026), arXiv:2508.19406: entanglement Hamiltonian
  dynamics after a local *joining* quench.
- Kusuki, Suzuyama & Tang (2026), arXiv:2610.02309: equilibrium
  entropy/capacity at conformal interfaces. Their discussion explicitly
  proposes studying how interfaces filter capacity growth after quenches.

**Critical literature qualification (added after examining the full replica method):**
Wen, Wang & Ryu (2018), arXiv:1711.02126, Sec. II, show that the *replica
partition functions* for ground-state and several quench geometries map to
the same interface strip problem, with a time-dependent strip width W(tau).
Since capacity is the *second derivative* of log Tr(rho_A^n) at n=1, their
method already strongly constrains (and may implicitly determine) the leading
capacity dynamics in the universal CFT regime. We must NOT advertise that
leading continuum relationship as a new discovery.

Our narrower tentative question is how *specific lattice global-field
quenches* differ from the ideal CFT/replica prediction, especially when
initial quasiparticle occupations are not perfectly thermal and the defect
filters momenta differently. Capizzi & Eisler (2023) already observed such
continuum-vs-lattice differences in *entropy* slopes; we would explore whether
capacity shows a distinct discrepancy and whether it is predictable.

No directly matching quantitative study of this specific lattice-capacity
deviation was identified in the searches completed so far. This is NOT
proof of novelty; examine the full Rényi derivations of the cited works
before writing claims about original research.

## Candidate measurement (not yet validated)

If a clear time window of near-linear growth exists for both quantities,
compare v_S(d)=dS/dtau and v_C(d)=dC/dtau, normalized by clean-chain rates.
A diagnostic ratio is R(d)=[v_C(d)/v_C(1)]/[v_S(d)/v_S(1)] where denominators
are nonzero. A generic finite-time slope does NOT constitute a universal
CFT growth coefficient.

We must systematically vary L, h_i, the time window and d; distinguish
transients and revivals; check the quench matches prior-work assumptions;
test finite-size convergence; and assess any analytical quasiparticle
prediction before interpreting apparent deviations.

**Warning:** the equilibrium coefficient inequality c_cap >= c_eff is
not an inequality C_A(tau) >= S_A(tau) at arbitrary times.

## Next milestones

1. Verify Fig. 1 pilot trajectories at L=24 against L=32, 48, 64.
2. Inspect if there is any stable linear-growth window before reflections.
3. Test different h_i and whether conclusions depend on the quench protocol.
4. Analyze derivative of full Renyi entropy and possible existing solutions.
5. Only then update the scientific manuscript and formulate any novelty claim.
