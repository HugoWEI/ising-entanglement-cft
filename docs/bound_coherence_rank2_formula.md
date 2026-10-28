# Bound–continuum coherence and a rank-two modular-capacity peak condition

**Research milestone (29 October 2026).** Model: an open, even,
number-conserving tight-binding free-fermion chain with two central
onsite potentials, initial staggered half-filled pure Gaussian state,
and a *single* final Hamiltonian eigenstate outside the propagating
band [-2,2]. This is **not** the transverse-field Ising model.

**Scientific status:** We obtained and numerically validated a
physically interpretable, exact **rank-two determinant formula** for
the influence of bound–continuum coherence on the entanglement
spectrum, plus a finite-dimensional Krylov approximation. This
derives from standard Gaussian correlation methods, spectral
projectors and the matrix determinant lemma; novelty of the
application has NOT been established. A simple explicit
universal strong-defect first-peak timing law is still missing.

## 1. Exact decomposition of the occupied orbital propagator

Let the final one-particle Hamiltonian h have eigenvectors
{b,phi_j} where b is the normalized unique out-of-band eigenmode,
h b=E_b b, and phi_j are all other extended eigenmodes.
Define P_b=|b><b|, P_c=I-P_b.

Let F0 be the L x (L/2) matrix of orthonormal occupied orbitals
for the initial pure Slater determinant. Its time evolution is

    F(t)=exp(-i h t)F0 = F_c(t)+F_b(t),

where

    F_c(t)=exp(-i h t) P_c F0,
    F_b(t)=exp(-i E_b t) b q,
    q=b† F0, a row vector of length L/2.

Let A be the left half of the chain and define

    B(t)=F_c(t)[A,:],                [L/2 x L/2]
    x(t)=exp(-i E_b t) b[A],         [L/2 x 1]
    y(t)=B(t) q†,                    [L/2 x 1]
    beta=q q†>=0.

Then, exactly,

    G_A(t)=F(t)[A,:] F(t)[A,:]†
          =B B† + beta x x† + x y† + y x†.

Define a mixed Gaussian reference covariance

    G_deph(t)=B B† + beta x x†.

It *removes only bound–extended covariance interference*.
The corresponding full-minus-reference covariance is

    Delta G(t)=x y† + y x† = X J X†,
    X=[x,y],  J=[[0,1],[1,0]].

This perturbation is rank <=2 for any chain size, even though
it can alter many entanglement eigenvalues.

The result is **phase-gauge invariant** under b -> exp(i gamma)b,
as q -> exp(-i gamma)q and x,y transform consistently.

### Why the reference is physically valid, and what it is not

Since 0<=G0=F0 F0†<=I, we have

    G_deph = P_A U(t) [P_c G0 P_c + P_b G0 P_b] U(t)† P_A,
    0<= G_deph <= I.

It is therefore a valid one-body covariance of a gauge-invariant
mixed Gaussian state, so capacity and entropy *as Gaussian functionals*
are mathematically meaningful. It is also obtained by phase-averaging
the one-body covariance over relative bound/band phases.

**Caveat:** A literal many-body phase-twirled Slater state can be
non-Gaussian. Its actual many-body modular capacity is not generally
the Gaussian functional evaluated on its averaged covariance.
We therefore explicitly label our comparison as a **Gaussian
covariance dephasing reference**, not the measured capacity of an
experimentally dephased many-body state. It is also not the evolution
of the same pure initial state under a physically modified local
defect.

The "band-only" covariance B B† is another valid mixed Gaussian
covariance but changes total particle occupation. It is used
only as an auxiliary diagnostic.

## 2. Exact two-by-two determinant formula

For z off both spectra, the matrix determinant lemma gives

    det(zI-G_A)/det(zI-G_deph)
       = det_2[I_2 - J X† (zI-G_deph)^(-1) X]
       =: D_2(z,t).

**This is a nonperturbative formula**: it resums all orders in
bound-band coherence and exactly encodes the characteristic
polynomial ratio of the FULL and DEPHASED entanglement spectra.

Because modular capacity is a nonlinear spectral functional,

    f(p)=p(1-p)[ln((1-p)/p)]^2,
    C_A=Tr f(G_A),

we do NOT claim C_full equals a sum of "bound capacity" and
"band capacity". Instead, when both covariance spectra are
strictly in (0,1), take a contour Gamma entirely within the
domain of analyticity of f and enclosing both spectra:

    C_full(t)-C_deph(t)
      = (1/(2*pi*i)) ∮_Gamma
          f(z) (partial_z ln D_2(z,t)) dz.

This follows from the contour trace formula for analytic
functions of Hermitian matrices. If eigenvalues sit exactly at
0 or 1, f is continuous but not analytic at the endpoints:
the contour identity then requires suitable limiting
regularization. The determinant identity itself does not have
this restriction.

This immediately yields an exact *implicit* condition for a
capacity revival maximum t_*:

    0 = dC_deph/dt(t_*)
        + (1/(2*pi*i)) ∮_Gamma
            f(z) partial_t partial_z ln D_2(z,t_*) dz,

with the additional condition C_full''(t_*)<0 and a specified
peak selection rule.

This is a **mathematically precise strong-defect peak condition
in terms of a 2x2 determinant**, but still requires the continuum
covariance B(t) and an integral/root search. It is NOT yet a
simple explicit physical-law formula for t_*.

## 3. Numerical validation of the determinant identity

The code directly computes the ratio of two subsystem determinants
for complex z, *independently* of the 2x2 expression.
It verifies agreement around 1e-15 for representative 48- and
64-site examples, across real-time dynamics.

It also verifies that G_deph, G_full, and B B† have all their
occupation eigenvalues within [0,1], and that averaging four
relative bound-band phases at the level of covariance reproduces
G_deph exactly.

## 4. First revival: the original reversed defect

Initial staggered mass m=2.5, two onsite impurity configurations
(u,v)=(1.2,0.35) and (0.35,1.2), Gaussian smoothing width 2,
fixed first broad-peak protocol from src/revival_benchmark.py.

| chain length | physical pure-state peak pair | dephased Gaussian reference pair |
|---:|---:|---:|
| 48 | 26.4 versus 27.4 | 27.8 versus 27.8 |
| 64 | 36.4 versus 37.0 | 37.0 versus 37.0 |

The reversal-dependent first broad peak shift disappears in
the dephased Gaussian covariance reference at both sizes.

For L=48, pair (1.2,0.35), the mean absolute difference
|C_full-C_deph| over the original revival window is about
0.2444 capacity units (nats^2). The maximum difference is
around 0.5785. These are not additive "contributions";
capacity is nonlinear and other quantities should not be
obtained by subtracting independent band and bound capacities.

### Critical negative controls

For L=48, the (0.7,1.4) pair still has a first-peak shift of
+0.2 under dephasing (as well as +0.2 in the pure-state
reference). For the (0.5,0.2) pair, the pure-state shift is
approximately zero whereas its dephased reference shows
a +0.2 shift. Thus the role of bound-band interference
is **defect-dependent**.

The dephased references correspond to different states, so
disappearance of the shift is **diagnostic evidence**, not
proof that bound-band coherence is the sole causal mechanism
in the original pure-state quench.

## 5. Why two entanglement eigenmodes are not sufficient

Although Delta G has rank <=2, its two vectors overlap with
many eigenmodes of G_deph. It can shift the whole nonlinear
entanglement spectrum. The 2x2 determinant itself contains the
full resolvent (zI-G_deph)^(-1), whose poles involve ALL
entanglement eigenvalues.

Our exploratory *static mode selection* retained the K eigenmodes
with the highest leverage ||<u_j|X>||^2. It often gave poor
capacity predictions and failed to reproduce the first peak.

This is a negative result: a low-rank perturbation is not
equivalent to a perturbation of just two entanglement levels.

## 6. A nonperturbative block-Krylov hierarchy

We can approximate the rank-two spectral correction without
discarding its nonlinearity by building

    K_m = span{X, G_deph X, ..., G_deph^(m-1) X}.

Let Q_m span this space orthonormally. Its dimension is at most
2m. The compressed matrices are

    H_m=Q_m† G_deph Q_m,
    Z_m=Q_m† X.

Define an all-orders *compressed* capacity correction

    Delta C_m(t) =
      Tr f(H_m + Z_m J Z_m†) - Tr f(H_m),

and predict

    C_pred,m(t)=C_deph(t)+Delta C_m(t).

The formula does NOT expand in the coherence amplitude,
and it guarantees a physical compressed covariance since it
is the principal compression of the valid full covariance.

It is exactly equal to C_full if the Krylov span fills the
subsystem or is invariant under G_deph. For finite m,
there is no rigorous capacity error bound in this experiment:
f has endpoint derivatives that complicate naive
operator-Lipschitz estimates.

### Numerical pilot

L=48, original orientation (1.2,0.35):

| block-Krylov steps m | retained dimension at most | revival-window capacity RMSE |
|---:|---:|---:|
| 0 (dephased baseline) | 0 | 0.2765 |
| 1 | 2 | 0.6315 |
| 4 | 8 | 0.1299 |
| 6 | 12 | 0.0662 |
| 8 | 16 | 0.0381 |
| 10 | 20 | 0.0190 |

For the reversed orientation (0.35,1.2), m=8 gives RMSE
~0.0256. Both m=8 curves reproduce the physical first
broad peak times 26.4 and 27.4 on our 0.2-time-unit grid.

For L=64, m=8 (max dimension 16 of 32 subsystem modes)
gives RMSE ~0.0423 for the original orientation and
~0.0294 for the reverse. Predicted peak times are
36.2 and 36.8, both within 0.2 of the full reference
36.4 and 37.0, so their *difference* is accurately
reproduced in this exploratory case.

**The m=1 approximation is WORSE than ignoring coherence
altogether.** Eight blocks are far from a universal compact
two-mode formula, and success on this case does not prove
transfer to other defects or preparations.

Most importantly, the tested Krylov approximation still
uses G_deph computed with the EXACT defective Hamiltonian.
This is an error-isolation experiment for the entanglement
correction, not yet a stand-alone low-cost theoretical
peak predictor.

## 7. Roadmap toward a genuinely predictive strong-defect formula

Two ingredients remain:

1. Approximate B(t), the continuum/band occupied orbital
   propagator, directly from the phase-resolved defect
   scattering data and finite-boundary paths, without
   diagonalizing the asymmetric Hamiltonian or evaluating
   a full exact evolution.
2. Bound the error of the combined continuum plus
   low-rank spectrum approximation, and predict peak
   time and height on **held-out** defects/sizes/initial
   preparations without adjusting the approximation.

Our previous parity-contour code can produce the continuum
orbital propagator without asymmetric diagonalization, by
subtracting the exact bound pole from its full scattering
contour, but this is still an expensive exact numerical
representation. A physically interpretable short-path
asymptotic approximation is still an open target.

## 8. Reproduction

    git pull origin main
    python -m pytest -q
    python scripts/benchmark_bound_coherence.py

Files:
- src/bound_coherence_determinant.py: exact spectral decomposition,
  dephased reference and 2x2 determinant identity.
- src/bound_coherence_krylov.py: coherent rank-two block-Krylov
  approximation.
- tests/test_bound_coherence_determinant.py: exact and
  symmetry/physicality tests.
- tests/test_bound_coherence_krylov.py: full-subspace recovery and
  deliberately failing short-subspace control.
- scripts/benchmark_bound_coherence.py: deterministic CSVs and figures.
- results/bound_coherence_reversal.csv,
  results/bound_coherence_krylov.csv.
- figures/bound_coherence_reversal.png,
  figures/bound_coherence_krylov_convergence.png.

## Prior art

- Gouraud, Le Doussal and Schehr, *Stationary time correlations for
  fermions after a quench in the presence of an impurity*,
  Europhys. Lett. 142, 47001 (2023), arXiv:2211.15447:
  bound-state oscillations and memory effects are known.
- Capizzi and Eisler, *Entanglement evolution after a global quench
  across a conformal defect*, SciPost Phys. 14, 070 (2023),
  arXiv:2209.03297: known coherent-boundary quasiparticle physics.
- Matrix determinant lemma, matrix-valued Cauchy transforms,
  block Krylov/Gauss quadrature, and Gaussian fermionic
  correlation-matrix entropy are standard methods.
- Our finite-chain, capacity-specific numerical comparison
  requires a separate originality audit; no peer-reviewed or
  independently established novel physical result is claimed.
