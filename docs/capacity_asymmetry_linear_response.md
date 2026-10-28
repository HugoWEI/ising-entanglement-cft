# Analytical first-order prediction of capacity contrast under defect reversal

**Research note, 28 October 2026. Status: tested linear-response baseline, not a novel many-body revival law.**

## Why this problem?

The exact Gaussian time evolution and our path-based Volterra approximation
can reproduce a finite quantum chain's modular capacity, but they still
require detailed time evolution. We therefore ask a strictly more
controlled *predictive* question:

> Can the capacity difference caused by reversing a weakly asymmetric
> two-site defect be predicted using only the symmetric defect?

This yields a falsifiable response formula without fitting or evolving
either of the two reversed *test* Hamiltonians to obtain the prediction.

**This application uses standard Duhamel/Kubo linear response.**
First-order perturbations of the capacity of entanglement were already
studied, including in [Aspects of Capacity of Entanglement,
Phys. Rev. D 99, 066012 (2019)](https://doi.org/10.1103/PhysRevD.99.066012).
We do not claim the general linear-response identity is original.

## 1. Model and the reversal parameter

This experiment uses the number-conserving tight-binding chain, not the
transverse-field Ising chain. In units with hopping = hbar = 1,

    H = sum_ij h_ij c_i^dagger c_j.

For even L with open boundaries, nearest-neighbor hopping is -1.
The two central onsite potentials are parameterized as

    (u,v) = (w+delta/2, w-delta/2),
    h(delta) = h_sym + delta W,
    W = (|a><a| - |b><b|)/2.

The reference h_sym has equal onsite potentials (w,w) at the central
sites. Reversal (u,v)->(v,u) is exactly delta->-delta.
The pair has identical transmission probabilities at every real k.

The initial pure half-filled Gaussian state F0 is the staggered-field
ground state (mass m=2.5) of a clean open chain, kept IDENTICAL under
both final Hamiltonians. This preparation breaks reflection symmetry.
A reflection-symmetric Fock preparation is an exact zero-response control.

## 2. Exact derivative of the propagator

For

    U_delta(t)=exp[-i(h_sym+delta W)t],

Duhamel's identity gives, without approximation,

    dot U(t) := dU_delta/d(delta)|_{delta=0}
             = -i integral_0^t U_sym(t-s) W U_sym(s) ds.

Let h_sym=Q diag(E_m) Q^dagger, and W^e=Q^dagger W Q.
Then the matrix elements of the derivative in the energy eigenbasis are

    dot U_mn(t) = W^e_mn D_mn(t),

where for E_m != E_n,

    D_mn(t) =
       [exp(-i E_m t)-exp(-i E_n t)]/(E_m-E_n),

and for E_m=E_n the analytic limiting value is

    D_mm(t) = -i t exp(-i E_m t).

We implement this directly as a spectral divided difference.
Only the symmetric Hamiltonian is diagonalized for the prediction.
No asymmetric-defect trajectory is needed to compute dot U(t).

The factor D_mn is complex, retains coherent interference across
energies, and automatically includes both finite-boundary propagation
and any impurity-bound eigenmodes already present in h_sym.

## 3. Gaussian covariance and modular capacity

Let F_delta(t)=U_delta(t)F0, where F0 contains L/2 orthonormal
occupied orbitals. For the left half A define

    G_A(delta,t)=F_delta(t)[A,:] F_delta(t)[A,:]^dagger.

Differentiating at delta=0,

    dot F(t)=dot U(t)F0,
    dot G_A(t)=dot F_A F_A^dagger+F_A dot F_A^dagger.

For eigenvalues 0<=p_j<=1 of G_A,

    C_A=Tr f(G_A)
       =sum_j p_j(1-p_j) log^2[(1-p_j)/p_j],

with scalar derivative on (0,1)

    f'(p)=(1-2p) log^2[(1-p)/p]
           -2 log[(1-p)/p].

For a differentiable Hermitian matrix function,
the trace derivative (including degenerate eigenvalues) obeys

    d Tr f(G_A)/d(delta)|0
       = Tr[ f'(G_A(0,t)) dot G_A(t) ].

This gives an explicitly computable *capacity susceptibility*:

    chi_C(t)=Tr[f'(G_A,sym(t)) dot G_A(t)].

The odd-in-delta capacity contrast is therefore

    Delta C_A(t;delta)
      =C_A(t;delta)-C_A(t;-delta)
      =2 delta chi_C(t)+O(delta^3)

at fixed finite t whenever the capacity is sufficiently smooth in
delta. The O(delta^2) term cancels by antisymmetrizing the two defects.

**Endpoint warning:** f'(p) diverges as p->0 or p->1. At t=0 or
for nearly disentangled modes the derivative requires careful limiting
analysis, and uniform O(delta^3) control over very large time/size
ranges is NOT established. Our numerical implementation clips only
the eigenvalues passed to the derivative f', reports the total
near-endpoint diagonal weight, and tests the result against independent
finite differences. It does NOT silently clip covariance eigenvalues
to force agreement.

## 4. What is predicted, and what is not?

The analytical expression predicts the entire *difference trajectory*
Delta C_A(t) to first order using the SYMMETRIC reference.

The exact trajectories of h(delta) and h(-delta) are computed
*afterwards*, only to validate the prediction.

This is not a closed-form expression for the capacity of an arbitrary
defect, and does not by itself predict the first individual capacity
revival peak. Maxima of a smoothed contrast are NOT the same thing as
capacity peaks.

Nor does this method provide a computationally optimal solution:
diagonalizing h_sym is still required. Its value here is a controlled,
falsifiable perturbation expansion and a baseline against which
genuinely simplified scattering-path or delay models can be judged.

## 5. Independent numerical tests

The code independently verifies:
1. Analytic scalar f'(p) against centered finite differences.
2. Capacity susceptibility against independent full-chain
   finite differences in delta at multiple times.
3. Cubic remainder scaling in the odd delta contrast at fixed time.
4. Exactly zero first-order susceptibility for an explicitly
   reflection-symmetric initial state.
5. Zero response at t=0.
6. A measurable advantage over the Delta C = 0 null for small delta.
7. Rejection of invalid inputs.

## 6. Quantitative pilot: where the prediction works and fails

Initial stagger m=2.5, L=48, mean onsite w=.775. Error measured as
RMSE of Delta C(t) in the prespecified 0.4L <= t <= 1.5L window.

| asymmetry delta | analytic response RMSE | zero-contrast null RMSE | verdict |
|---:|---:|---:|---|
| .02 | ~0.000131 | ~0.006 | perturbation works |
| .10 | ~0.00598 | ~0.03236 | substantial improvement |
| .30 | ~0.06316 | ~0.06186 | no improvement |
| .85 | ~0.30019 | ~0.15316 | perturbation fails |

At small asymmetries and fixed times, the residual approximately
scales like delta^3. At the original widely asymmetric pair
(u,v)=(1.20,.35), delta=.85, first-order response overpredicts
the contrast; higher orders, bound-mode shifts, and nonlinear
changes in the entanglement spectrum are consequential.

Our benchmark also tests L=24 and 64, and a distinct reference
potential w=.25. The parameter study is exploratory, not a
pre-registered general statistical test.

We separately report the time/value of the largest *smoothed absolute
contrast* in the interior time window [.65L,1.45L], and mark it
undefined when the maximum falls on the window boundary. This is
NOT labeled an individual first capacity-revival peak.

## 7. What would advance the actual research question?

We now have a known, exact-in-its-order response benchmark, alongside
a matched-transmission null and exact Gaussian evolution.

Possible next nontrivial step:

- Express chi_C through *scattering states and bound poles* using
  a low-dimensional T-matrix and coherent boundary-return kernel.
- Establish whether a reduced set of phase/pole data recovers the
  susceptibility for broad finite-L classes more efficiently.
- Derive and validate the higher-order odd response for moderate delta,
  or a controlled nonperturbative resummation.
- Most importantly, independently predict actual first **capacity**
  peak time and height on held-out defects. Our current response is
  a contrast predictor, not yet that achievement.
- Repeat literature originality audits before claiming more than
  a numerical benchmarking contribution.

## Reproduction

    git pull origin main
    python -m pytest -q
    python scripts/benchmark_capacity_asymmetry_response.py

Relevant files:
- src/capacity_asymmetry_response.py
- tests/test_capacity_asymmetry_response.py
- scripts/benchmark_capacity_asymmetry_response.py
- results/capacity_asymmetry_linear_response.csv
- results/capacity_asymmetry_example.csv
- figures/capacity_asymmetry_response.png
- figures/capacity_asymmetry_error_scaling.png

## Literature anchors

- Aspects of Capacity of Entanglement, Phys Rev D 99 066012 (2019),
  https://doi.org/10.1103/PhysRevD.99.066012 : first-order
  capacity perturbation and capacity under quantum quenches.
- Arias et al., Probing RG flows, symmetry resolution and quench
  dynamics through the capacity of entanglement, JHEP 03 (2023) 175,
  https://doi.org/10.1007/JHEP03(2023)175 .
- Capizzi and Eisler, Entanglement evolution after a global quench
  across a conformal defect, SciPost Physics 14, 070 (2023),
  https://arxiv.org/abs/2209.03297 .
