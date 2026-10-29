# Second-order phase control of modular-capacity revival peaks

**29 October 2026 — theory derivation, independently scored numerical test.**

## Scope

We study an open, even, number-conserving free-fermion hopping chain
with a final two-site positive onsite impurity and exactly one
localized eigenmode outside the propagating band. The initial state
is a half-filled pure Slater determinant. A mathematical phase
gate on the bound projector changes the relative initial bound-band
phase theta WITHOUT changing the final Hamiltonian, its scattering
amplitudes, energy eigenstates or energy-mode populations.

This project already has:
- An exact two-vector coherent covariance formula, and
- An EXACT first phase susceptibility for capacity, implemented
  without full defect diagonalization in the pole-exact finite-walk
  solver.

The purpose of this milestone is to predict individual first
revival peak *positions* more accurately for finite phase changes
without fitting any shifted-phase trajectories.

## 1. Exact phase curvature of modular capacity

For a half-chain reduced covariance, write

    G(theta)=D+exp(i theta)x y†+exp(-i theta)y x†,

with D the valid, bound-band-dephased Gaussian covariance.
Define

    A_theta=exp(i theta)x y†,
    H=partial_theta G=i(A_theta-A_theta†),
    K=partial_theta² G=-(A_theta+A_theta†).

For scalar modular-capacity function

    f(p)=p(1-p) log²((1-p)/p)

and eigenvalues/eigenvectors G|i>=p_i|i>, the exact
second phase derivative is

    BOXED:
    kappa_theta := partial_theta² Tr f(G)
      =sum_i f'(p_i) K_ii
       +sum_ij f'^[1](p_i,p_j) |H_ij|²,

where H_ij=<i|H|j>, K_ii=<i|K|i>, and

    f'^[1](p,q) = (f'(p)-f'(q))/(p-q), p!=q,
    f'^[1](p,p) = f''(p).

Explicit derivatives are

    L(p)=log((1-p)/p),
    f'(p)=(1-2p)L(p)^2 -2L(p),
    f''(p)=-2L(p)^2-2(1-2p)L(p)/[p(1-p)]
           +2/[p(1-p)].

This comes directly from the spectral second Fréchet derivative
of a matrix-function trace. It is NONPERTURBATIVE in defect
strength and bound–continuum coherence amplitude.

No eigenvectors of the final full-chain Hamiltonian appear in
this derivative formula. We need only the subsystem correlation
eigenvalues and two local interference vectors x,y.

**Endpoint warning:** f' and f'' are singular at 0 and 1. The code
regularizes these derivatives numerically and reports the number of
near-edge eigenvalues; it does not claim a uniform second-derivative
error theorem.

## 2. Second-order capacity and peak-time formulas

At reference phase theta0, let

    C0(t)=C(t,theta0),
    chi(t)=partial_theta C(t,theta0),
    kappa(t)=partial_theta² C(t,theta0).

Then for eta sufficiently small,

    C(t,theta0+eta)=C0(t)+eta chi(t)
                    +eta² kappa(t)/2+O(eta³).

For a nondegenerate smoothed local peak t0 satisfying
C0'(t0)=0, C0''(t0)<0, implicit differentiation gives

    t_peak(theta0+eta)
        =t0+eta a+eta² b+O(eta³),

    BOXED:
    a=-chi'(t0)/C0''(t0),

    BOXED:
    b=-[ C0'''(t0)a²/2
          +chi''(t0)a+kappa'(t0)/2 ]/C0''(t0).

These are formulas for INDIVIDUAL peak positions. For symmetric
phase contrasts t_peak(+eta)-t_peak(-eta), the even-order
b term cancels, so quadratic control does NOT universally
improve the contrast beyond linear response. This is why our
benchmark scores the individual positive and negative peaks.

All C,chi,kappa are smoothed with the same predetermined Gaussian
temporal filter (sigma=2) before identifying peak positions.

For numerical robustness we also predict peak positions by
directly locating the local smoothed maximum of the quadratic
capacity model, rather than relying solely on noisy third
time derivatives in the explicit b coefficient.

## 3. First independent numerical test: exact continuum reference

Without fitting either phase-shifted curve, test 4 fixed setups:
- L=48 and 64 with onsite pair (1.2,0.35)
- L=48 with pair (0.7,1.4)
- L=56 with pair (1.1,0.1)

Use staggered initial field 2.5, phase offsets
+/-0.08, +/-0.30 and +/-0.60 radians,
timestep .2, first broad capacity-peak protocol
from src/capacity_peak_response.py.

The test obtains the EXACT continuum covariance once
at base phase zero, then predicts each rotated branch
using C0,chi,kappa only. Exact shifted-phase trajectories
are held out and used exclusively for evaluation.

| |phase offset| | mean absolute peak-time error, first order | second order | improvement |
|---:|---:|---:|---:|---:|
| .08 | 0.00119826 | 0.000031165 | 97.4% |
| .30 | 0.0172698 | 0.00180027 | 89.6% |
| .60 | 0.082243 | 0.0396992 | 51.7% |

The reference-mode numerical benchmark passed 34 tests
(GitHub Actions run 40837609139).
These are exploratory finite-size comparisons with a
small prespecified group of defect cases, NOT a theorem
of uniformly increased accuracy or a statistical test.

A meaningful negative result: for L=48, pair (1.2,.35),
the phase -0.60 exact tracked peak is 27.45859,
while the quadratic approximation predicts 27.69369.
The first-order value is 27.12934. The quadratic result
improves the error but is still far from the exact peak.
At +0.60 the quadratic prediction is much closer.

## 4. No-full-chain-diagonalization version

The formula is also inserted in src/walk_phase_response.py,
using src/bound_projected_chebyshev.py to compute the
continuum orbitals via a priori truncated Bessel/Chebyshev
walks and the bound energy/projector using a clean-chain
2x2 Birman–Schwinger equation.

No full-chain defective OR symmetric reference Hamiltonian
diagonalization and no complex-energy contour are necessary
to construct the theoretical C0,chi,kappa curves.

Independent validation is carried out by
- tests/test_walk_phase_curvature.py and
- scripts/benchmark_walk_phase_curvature.py.

The Chebyshev amplitude truncation bound does NOT
automatically bound the phase curvature or peak-time
derivatives, whose endpoint/curvature sensitivities
must be controlled separately.

## 5. Novelty and remaining physics problem

This exact spectral curvature and implicit peak expansion
are standard matrix-calculus/perturbation techniques.
They quantify a particular finite-size defect-revival
response but do not establish a new theorem.

Relevant prior work:
- Donelli, De Chiara, Scazza and Gherardini,
  *Impact of quantum coherence on the dynamics and
  thermodynamics of quenched free fermions coupled to a
  localized defect*, Phys Rev A 113, 013311 (2026),
  DOI 10.1103/5kt1-5m83.
- Capizzi and Eisler, *Entanglement evolution after a
  global quench across a conformal defect*,
  SciPost Physics 14, 070 (2023),
  arXiv:2209.03297.

Potentially publishable next result requires deriving a
SHORT physical scattering-return-path expression for
the phase response or a new asymptotic scaling/identifiability
law with a genuinely new universality class. A more
accurate Taylor polynomial is not, by itself, that result.

## 6. Reproduce

    git pull origin main
    python -m pytest -q tests/test_bound_phase_curvature.py tests/test_walk_phase_curvature.py
    python scripts/benchmark_bound_phase_curvature.py
    python scripts/benchmark_walk_phase_curvature.py

All numerical outputs and plotted errors are recorded
without editing the fixed parameters to match test curves.


## 7. Final no-diagonalization CI verification

**Successful focused workflow run:**
https://github.com/HugoWEI/ising-entanglement-cft/actions/runs/40837992169

40 tests passed, including tests of:
- scalar second derivative and full Gaussian spectral curvature,
- near-degenerate eigenspectrum divided differences,
- local quadratic peak prediction,
- exact-reference versus scattering-walk phase curvature,
- no full-chain Hamiltonian eigendecomposition during prediction.

The strong-defect walk implementation obtains exactly the same
reported predicted peak times as the exact-continuum reference,
to the precision shown here. Both first-order and second-order
models use ONLY reference phase theta0=0 to construct
the +/-shifted prediction.

| L | defect | phase eta | held-out exact peak | first order | second order |
|---:|---|---:|---:|---:|---:|
| 48 | (1.2,.35) | -.30 | 26.68405 | 26.63551 | 26.68788 |
| 48 | (1.2,.35) | +.30 | 26.32566 | 26.29022 | 26.32459 |
| 64 | (1.2,.35) | -.30 | 36.23709 | 36.24383 | 36.23399 |
| 64 | (1.2,.35) | +.30 | 36.73324 | 36.74187 | 36.73720 |
| 56 | (1.1,.1) | -.60 | 30.75553 | 30.69861 | 30.75406 |
| 56 | (1.1,.1) | +.60 | 30.88985 | 30.84196 | 30.89272 |

Across FOUR configurations and both +/- phase orientations:
- +/- .08 radians: first-order mean absolute peak-time error
  0.00119826, quadratic 0.000031165 (97.4% reduction).
- +/- .30 radians: first-order 0.0172698, quadratic
  0.00180027 (89.6% reduction).
- +/- .60 radians: first-order 0.082243, quadratic
  0.0396992 (51.7% reduction).

For the original L=48 case at phase -.60, the quadratic
prediction 27.69369 remains substantially too late
compared with exact 27.45859. This is not a universal
large-phase theory, and no retuning to fix this case
was performed.

The exact-continuum and Chebyshev-walk-based second-order
predictions agree, so remaining deviations are from
the PHASE Taylor truncation / peak-branch sensitivity,
not from noticeable numerical errors in the walk propagator.

Detailed numerical data and figures:
- results/bound_phase_second_order_peak.csv
- figures/bound_phase_quadratic_peak_errors.png
- results/walk_phase_quadratic_peak.csv
- figures/walk_phase_quadratic_peak.png

The workflow publishes all data/figures as an artifact.
