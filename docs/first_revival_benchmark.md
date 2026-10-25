# First broad entanglement-capacity revival: a falsification benchmark

**Status:** exploratory numerical research diary (26 October 2026).
**No theoretical novelty claim. No phase-aware coherent many-body predictor has been derived.**

## Question

Can the *time of the first broad capacity revival* in a finite,
number-conserving tight-binding chain be predicted from the Wigner reflection
delay of a two-site impurity?

This is deliberately a **minimal model check** before attempting a coherent
multiple-reflection theory.

## Physical setup

Final Hamiltonian: free hopping -1 on L sites, with central onsite potentials
(u,v) in one experiment and spatially reversed (v,u) in the other.

Initial Gaussian state: half-filled gapped ground state of a *uniform* hopping
chain with staggered field m=2.5.

Both defects have the same complex transmission amplitude t(k) for every k,
but different left-reflection phases. We use a fixed coordinate convention.
For reflection-symmetric initial states, mirrored defects must give identical
half-chain entanglement dynamics (already checked by separate tests).

Observable:
C_A(tau)=sum_j p_j(tau)[1-p_j(tau)] log^2[(1-p_j(tau))/p_j(tau)].

## Transparent, reproducible peak definition

The phrase "first revival" is ambiguous: finite chains display rapid
oscillations and sometimes several nearby local maxima. We therefore use a
reproducible *broad envelope peak* operational definition:

- evolve out to 1.8 L in uniform time steps of 0.2
- smooth C_A(tau) with a Gaussian of width sigma_tau=2
- consider candidate maxima in 0.4 L <= tau < 1.65 L
- accept the first local envelope maximum with minimum prominence of 0.08,
  comparing both sides within +/- 0.15 L
- if no such peak exists, return NaN rather than reporting an arbitrary
  search-window boundary as a peak

These rules are practical choices, **not fundamental definitions**.
A robustness analysis over sigma, prominence, and quench protocol is required.

## Models compared

A. **Matched-transmission null baseline:** predict equal first-peak times
   for (u,v) and (v,u), i.e. Delta tau_peak=0. This null control is *not*
   a general implementation of the Capizzi-Eisler quasiparticle theory.

B. **Single-delay phase heuristic:** predict
   Delta tau_peak ~ Delta[d arg r_L(k_star)/dE(k_star)].

k_star is chosen *without fitting the capacity trajectories*, using the
maximum of a physically motivated but **unproved** weight

w(k) = v(k) c_initial(k) T(k)[1-T(k)],

where v(k)=2 sin(k), initial occupation n(k) obeys
n(k)=[1+2 cos(k)/sqrt(m^2+4 cos^2(k))]/2, and

c_initial(k)=n(k)[1-n(k)] log^2[(1-n(k))/n(k)].

The T(1-T) weighting is a heuristic for reflected/transmitted path
interference, NOT a derived or generally valid formula for revival peaks.

## Preliminary result and critical falsification

For the original pair (u,v)=(1.2,0.35), a representative k_star is about
0.792. The predicted first peak displacement is about -0.834 hopping-time
units. Direct finite-chain trajectories yield peak displacements of roughly
-1.0 at L=48 and -0.6 at L=64 under the stated envelope definition.

**Crucially, the same heuristic does NOT generalize to other impurity pairs.**
For example the reversed pair (0.7,1.4) yields a delay prediction of about
+2.15, whereas the observed first broad capacity-peak shifts at L=48 and
L=64 are only about +0.2. For (0.5,0.2), the delay predicts about -0.93
but the first broad peak displacement is approximately zero.

Thus **a single representative Wigner delay is not sufficient to predict
many-body capacity-revival timing.** The benchmark disfavors that simple
model across the selected impurity cases. The test does not disprove a
properly derived phase-resolved *coherent-path* approximation.

These comparisons were explored iteratively; they are not preregistered
held-out statistical tests. They also don't yet test phase-aware peak
*amplitudes*, which need coherent interference information.

## What we learned

1. Reflection phases affect transients with asymmetric initial states.
2. Single-particle Wigner delays predict individual wavepacket arrival
   differences (see docs/phase_scattering_literature_audit.md).
3. A many-body modular-capacity revival integrates a wide spectrum of
   momenta, occupations, boundary return paths and interference. Using a
   single k value can fail even when all scattering inputs are correct.
4. Our next candidate model must integrate *complex amplitudes* over k
   and sum direct/reflected/multiple-reflection paths before computing
   an approximate **correlation matrix** and hence capacity.

## Next derivation (not yet done)

Derive a phase-sensitive one-particle propagator in a finite chain using
multiple-scattering path amplitudes. Construct an approximate reduced
correlation matrix from the initial covariance C(0), and compute capacity
from its eigenvalues. Specify approximation order and verify that adding
paths converges. Compare this model to a phase-erased version using
identical initial state and geometry. Test quantitative timing AND
amplitude without fitting to exact entanglement curves.

Existing exact Gaussian evolution is the numerical reference, not a
proposed new theory.

## Reproduce

Run:
\`python -m pytest -q\`
\`python scripts/benchmark_first_revival.py\`

Outputs:
\`results/first_revival_benchmark.csv\`
\`figures/first_revival_benchmark.png\`

Related literature: Capizzi & Eisler (2023), arXiv:2209.03297, Secs. 3.4-3.5.
