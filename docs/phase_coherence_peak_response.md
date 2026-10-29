# Phase-coherence response of a finite-size modular-capacity revival

**29 October 2026 — analytic and numerical research note.**

## Scope and novelty

We study a number-conserving, even, open free-fermion tight-binding
chain after a global quench, with two central onsite impurities and
EXACTLY ONE localized eigenmode outside the propagating band.

The Hamiltonian, scattering amplitudes, bound eigenenergy, and
final-Hamiltonian eigenmode populations are FIXED. We vary only the
initial phase between the bound mode and the propagating continuum
by the *mathematically exact pure-state gate*

    F_theta(0)=[I+(exp(i theta)-1)P_b]F0.

The family is an admissible pure Slater determinant, and every
final-Hamiltonian spectral occupation and energy moment stays fixed.

**This is not yet a newly discovered physics law.** Ordinary quantum
mechanics already predicts sensitivity to initial offdiagonal
coherence. The exact response formulas below are applications of
Gaussian matrix functional calculus and implicit differentiation.
Relevant prior art includes Donelli et al., Phys. Rev. A 113,
013311 (2026), DOI 10.1103/5kt1-5m83, studying coherence in
nonequilibrium defect-coupled free fermions, albeit with
different observables and geometry.

The value of this experiment is a very specific, falsifiable
**capacity-peak phase response** and a surprisingly simple
characteristic-polynomial phase dependence.

## 1. Exact fixed-Hamiltonian covariance geometry

Let b be the normalized unique final-Hamiltonian bound eigenvector
with energy E_b and P_b=|b><b|. Let

    F_c(t)=exp(-iht)(I-P_b)F0
    q=b†F0
    x(t)=exp(-i E_b t)b[A]
    y(t)=F_c(t)[A,:] q†

with A the left-half subsystem and beta=q q†.

Define the dephased *Gaussian covariance reference*

    D(t)=F_c(t)[A,:]F_c(t)[A,:]† + beta x x†,

which satisfies 0<=D<=I. Then for any preparation phase theta,

    BOXED:
    G_A(t,theta)=D(t)
         +exp(i theta) x(t)y(t)†
         +exp(-i theta) y(t)x(t)†.

This is an EXACT rank-two update for arbitrary theta. It is
NONPERTURBATIVE in the impurity strength and in the coherence
amplitude.

The dephased covariance D is a valid gauge-invariant Gaussian
state, but the physical many-body phase-averaged ensemble
need not itself be Gaussian. Do not conflate its Gaussian
capacity with the measured modular capacity of a phase-twirled
many-body state.

## 2. Exact FIRST-HARMONIC characteristic-polynomial identity

Let X=[x,y], J_theta=[[0,e^(i theta)],[e^(-i theta),0]],
and for spectral z outside the eigenvalues of D,

    M(z,t)=X†(zI-D)^(-1)X.

Using the matrix determinant lemma,

    det(zI-G_A(theta))/det(zI-D)
        =det_2[I_2-J_theta M].

Expand this 2x2 determinant EXACTLY:

    BOXED:
    R(z,t,theta)
       =1-det M(z,t)
        -exp(i theta)M_21(z,t)
        -exp(-i theta)M_12(z,t).

Therefore, at every spectral z, the characteristic-polynomial
ratio is a pure FIRST Fourier harmonic in theta, irrespective
of chain length or defect strength.

But the nonlinear modular capacity

    C_A(theta)=Tr f(G_A(theta)),
    f(p)=p(1-p)ln²[(1-p)/p]

does NOT inherit this exact first-harmonic restriction.
The roots of the polynomial depend nonlinearly on theta,
and the resulting capacity can have higher angular harmonics.

The distinction is testable: our phase-Fourier benchmark fits
a first harmonic with just four cardinal phases, then predicts
12 held-out phase values. It checks the second Fourier
harmonic instead of assuming it is zero.

## 3. Exact phase susceptibility at ANY theta

The time-dependent capacity's exact first phase derivative is

    chi_theta(t,theta)=partial_theta C_A(t,theta)
      =Tr[f'(G_A(t,theta))partial_theta G_A(t,theta)].

The derivative is

    partial_theta G_A
       =i exp(i theta)x y†
         -i exp(-i theta)y x†.

Consequently,

    BOXED:
    chi_theta(t,theta)
        =-2 Im[ exp(i theta)
                y† f'(G_A(t,theta)) x ],

with

    f'(p)=(1-2p)ln²[(1-p)/p]
            -2ln[(1-p)/p].

This susceptibility is nonperturbative in the defect
and the coherence; it is a derivative with respect
to the *preparation phase* alone. It is evaluated
using ONE reference phase, not independently evolved
theta+ and theta- states.

At entanglement eigenvalues p=0 or p=1 the derivative
of f has logarithmic singularities. We clip eigenvalues
only when numerically evaluating f', report the
near-endpoint diagnostic, and validate this
calculation with independently centered finite
differences. This is NOT a global derivative-error theorem.

## 4. Local PHASE-DEPENDENT FIRST-REVIVAL formula

Let \bar C(t,theta) denote the Gaussian-smoothed capacity
with fixed temporal width sigma=2. At an isolated
nondegenerate local maximum t0(theta0),

    partial_t \bar C(t0,theta0)=0,
    partial_t² \bar C(t0,theta0)<0.

Implicit differentiation gives

    BOXED:
    dt_peak/dtheta|theta0
      =-[partial_t \bar chi_theta(t0,theta0)]
         /[partial_t² \bar C(t0,theta0)].

Thus for small phase perturbations ±eta:

    BOXED:
    t_peak(theta0+eta)-t_peak(theta0-eta)
      =-2eta (partial_t \bar chi_theta)/
              (partial_t² \bar C) + O(eta³).

The peak-height contrast similarly satisfies

    C_peak(theta0+eta)-C_peak(theta0-eta)
       =2eta \bar chi_theta(t0,theta0)+O(eta³).

These formulas *predict actual local-maximum peak shifts*, rather
than merely the capacity trajectory, from a single initial
preparation. They are standard implicit-function/linear-response
mathematics; their usefulness is conditional on a smooth
isolated peak branch and small eta. They do not give a
universal first-revival law at large phase rotations.

To make the test reproducible, we reuse the existing repository
peak definition: Gaussian smoothing width 2, first
prominent local maximum in [0.4L,1.65L], prominence
at least 0.08 in +/-0.15L. Sub-grid parabolic
interpolation is used for *local* phase-contrast tests.

## 5. Preliminary local numerical pilot (before CI)

The phase susceptibility was checked against independent
finite differences, with maximum differences of order
1e-6 for 0.001-radian centered steps in four cases.

For small offsets ±0.02 radians, pilot predicted-vs-measured
SMOOTHED peak-time contrasts, using 0.2 observation spacing:

| L | two-site potential pair | analytic prediction | measured |
|---:|---|---:|---:|
| 48 | (1.2,.35) | -0.0206 | -0.0211 |
| 64 | (1.2,.35) | +0.0326 | +0.0333 |
| 48 | (.7,1.4) | +0.0089 | +0.0088 |
| 56 | (1.1,.1) | +0.0047 | +0.0048 |

The SIGN REVERSAL between L=48 and L=64
for the same defect is a warning against a
universal fixed-sign phase-delay story.

The same initial gate changes no energies
or populations; the sensitivity is due to
time-dependent interference and the nonlinear
entanglement spectrum.

An exploratory 16-phase Fourier pilot found
the mean second-harmonic amplitude was
approximately 8–13% of the first harmonic
across the tested cases. A four-cardinal-phase
first-harmonic interpolant gave capacity
RMSE ~0.006–0.018 on the revival window,
but the higher harmonics are real and some
held-out peak indices differ.

The tracked-phase response and Fourier tests
are completely distinct from the original
strong-defect asymmetry-response problem.

## 6. Controls and limitations

- Validate exact determinant identity at complex spectral z,
  including phase factors and off-diagonal complex entries.
- Validate susceptibility against independent differences of
  the *full Gaussian capacity* at phase theta±eps.
- Never derive the ±eta phase prediction by fitting
  either shifted trajectory.
- Hold out 12 of the 16 phase angles to measure the
  quality of the four-phase harmonic approximation.
- A missing or branch-switching local peak must be
  marked undefined, not replaced by a window boundary.
- The exact continuum dynamics is still supplied by
  the full defective Hamiltonian eigensystem for
  this mechanistic experiment. It is NOT yet a
  scattering-only, asymptotically cheap peak law.
- The pure bound-mode gate is not strictly local
  and no experimental hardware implementation is
  demonstrated.

## 7. What could advance novelty?

The current response coefficient can be measured for any fixed
Hamiltonian, so the next real physical target is to derive
its asymptotic scaling with chain length and impurity
localization length from JUST a few coherent scattering
return paths and test a universal collapse or a controlled
obstruction to such collapse. That is not solved by
this note.

## 8. Reproduction

    git pull origin main
    python -m pytest -q tests/test_bound_phase_peak_response.py
    python scripts/benchmark_bound_phase_peak_response.py

Code:
- src/bound_phase_peak_response.py
- tests/test_bound_phase_peak_response.py
- scripts/benchmark_bound_phase_peak_response.py

Outputs:
- results/bound_phase_peak_susceptibility.csv
- results/bound_phase_fourier_heldout.csv
- figures/bound_phase_peak_shift_prediction.png
- figures/bound_phase_harmonic_compression.png


## 9. Independent CI validation and no-diagonalization coupling

**Focused GitHub Actions run (success; 24 tests passed):**
https://github.com/HugoWEI/ising-entanglement-cft/actions/runs/40791880468

We now also combine this exact phase-response identity with the
previous *bound-pole-exact Bessel/Chebyshev walk propagator* in
\`src/walk_phase_response.py\`. For each desired fixed Hamiltonian,
the new hybrid predictor:

1. Obtains the impurity-bound eigenenergy and projector by the
   2x2 **clean-chain Birman–Schwinger equation**, using analytic
   sine eigenfunctions and no Hamiltonian diagonalization.
2. Computes the band occupied orbitals by projected Chebyshev
   walks of degree M fixed by an a-priori Bessel tail tolerance.
3. Constructs D(t), x(t), y(t), exact-in-coherence G(t,theta0),
   and the phase susceptibility chi_theta(t,theta0).
4. Predicts the +/-phase LOCAL peak time/height contrasts
   using only the theta0 preparation.
5. Subsequently uses the exact full Hamiltonian eigensystem
   *only to score held-out phase-rotated preparations*.

This uses **NO full-chain Hamiltonian diagonalization in the
prediction**, including the reference dynamics.

### Validated held-out local peak-time contrasts

For initial theta0=0, phase offsets +/-0.10 radians,
Gaussian-smoothed capacity width 2, time sample spacing .2:

| L | (u,v) | walk prediction | independent exact | |difference| |
|---:|---|---:|---:|---:|
| 48 | (1.2,.35) | -0.105628 | -0.106108 | .000480 |
| 64 | (1.2,.35) | +0.166652 | +0.166270 | .000382 |
| 48 | (.7,1.4) | +0.043785 | +0.043745 | .000040 |
| 56 | (1.1,.1) | +0.023869 | +0.023827 | .000042 |

For +/-0.02 radians the differences agree to about
the sixth decimal or better in these tests.
For +/-0.30 radians, the 48-site original pair has
prediction -0.316884 versus exact -0.358395:
the cubic nonlinear remainder becomes appreciable.
The same model is more accurate at 64 sites for
+/-0.30 (+0.499957 versus +0.496149), so there
is **no uniform finite phase radius of validity**
demonstrated by this pilot.

The a-priori CONTINUUM ORBITAL truncation bounds are
about 9.7e-13 (L=48, polynomial degree 227),
7.7e-13 (L=56, degree 259) and
5.2e-13 (L=64, degree 291). These do NOT
automatically bound the phase susceptibility or
peak-time errors, which require additional spectral
regularity and curvature estimates.

### Validated angular harmonics

The exact spectral determinant is FIRST harmonic in
theta; capacity is not. Across 12 phase angles
held out from the four cardinal fit phases
(0, pi/2, pi, 3pi/2), the measured mean
second-to-first harmonic ratios were:

| L | (u,v) | A2/A1 | average held-out capacity RMSE |
|---:|---|---:|---:|
| 48 | (1.2,.35) | .080 | .00888 |
| 64 | (1.2,.35) | .074 | .00730 |
| 48 | (.7,1.4) | .144 | .01813 |
| 56 | (1.1,.1) | .074 | .00619 |

A first-harmonic capacity approximation is often useful,
but not an exact identity or a universal peak predictor.
No retuning was performed using the held-out angles.

Files:
- src/walk_phase_response.py
- tests/test_walk_phase_response.py
- scripts/benchmark_walk_phase_response.py
- results/walk_phase_peak_response.csv
- figures/walk_phase_peak_response.png

The new result is a quantitatively validated **local
phase-control response at strong defect strength**, built
from standard coherent quantum evolution and matrix
functional calculus. Its mathematical components are
established, and originality still needs a focused
literature comparison before publication claims.
