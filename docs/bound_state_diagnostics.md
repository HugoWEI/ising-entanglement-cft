# Bound-state contribution to modular-capacity revivals

**Status (October 2026): numerical mechanism audit; no new general theorem or formula.**

This experiment tests whether the localized eigenstates of our two-site
onsite-potential defects influence the time-dependent capacity spectrum.

## Model and exact decomposition

This is the number-conserving tight-binding free-fermion model used in
\`src/scattering_phase.py\`, *not* the transverse-field Ising Hamiltonian.

For an even open chain, \`h_{j,j+1}=h_{j+1,j}=-1\`, except optional central
hopping \`-g\`; central onsite potentials \`(u,v)\`. The uniform bulk
single-particle energy band is \`[-2,2]\`.

Write
\`h=V diag(e_j) V†\`, with orthogonal projectors \`P_b\` onto the
out-of-band eigenstates \`|e_j|>2+1e-8\` and \`P_c=I-P_b\` onto the
complement (finite-size band states).

Let \`F(0)\` be the orthonormal occupied single-particle orbitals of the
staggered initial Hamiltonian, with mass m=2.5. Define

\`F_c(t)=V_c exp[-i diag(e_c)t] V_c† F(0)\`,
\`F_b(t)=V_b exp[-i diag(e_b)t] V_b† F(0)\`.

Then \`F(t)=F_c(t)+F_b(t)\`, and for the left-half projector \`P_A\`,

\`G_A(t) = G_cc(t)+G_bb(t)+G_cb(t)+G_bc(t)\`,

where \`G_xy=P_A F_x F_y† P_A\`. This is an **exact linear decomposition of
the correlation matrix**. Cross terms \`G_cb+G_bc\` encode interference.

Importantly, capacity is a **nonlinear** functional of eigenvalues of
\`G_A\`; there is no justified additive partition
\`C_A=C_cc+C_bb+C_cross\`. Frobenius norms of those matrix terms are
diagnostics, not physical fractions of capacity.

## A clean but artificial counterfactual

To isolate sensitivity to the *bound-mode dynamical phase* while keeping
the initial state fixed and evolution unitary, replace bound eigenenergies
by the nearest band edge \`e_b -> sign(e_b)*2\`, while leaving all
eigenvectors and band energies unchanged.

Then the counterfactual propagator is
\`U_*(t)=V_c exp(-i e_c t)V_c† + V_b exp[-i sign(e_b)2t]V_b†\`.
Since the projectors are orthogonal, \`U_*\` is unitary and the
counterfactual is a valid pure Gaussian state for every time.

**But the generated Hamiltonian is generically nonlocal, not a
physically realizable local impurity or a "bound-state removal".**
The result quantifies sensitivity to bound-state energy phases; it
does not isolate a causal bound-state contribution in an experimental
Hamiltonian. A true local-defect control is described below.

## Spectral pilot

For the original pairs with L=48, one eigenvalue lies above the bulk
band:

| onsite pair | bound-state energy |
|---|---:|
| (1.2,0.35) | +2.39874792 |
| (0.7,1.4)  | +2.58353126 |
| (0.5,0.2)  | +2.09584000 |

The bound eigenmodes have most of their probability near the central
defect. Their presence is identical under reversing (u,v) to (v,u).

The first broad capacity peak is operationally defined in
\`src/revival_benchmark.py\`: Gaussian smoothing width two time units,
time window [0.4L,1.65L], prominence 0.08 against both sides over
+/-0.15L.

For (u,v)=(1.2,0.35), the preliminary simulations gave:

| length | physical first peak | bound-edge-counterfactual peak |
|---:|---:|---:|
| 48 | 26.4 | 28.6 |
| 64 | 36.4 | 35.6 |

These data establish **sensitivity**, not a universal positive or
negative correction. The shift changes sign under this peak-definition
procedure, and fine oscillations can complicate the notion of
"first revival". Larger-system and window-robustness checks are needed.

In the same setup, the maximum instantaneous capacity difference
\`|C_physical(t)-C_counterfactual(t)|\` can reach order 0.4 during
our explored time interval. This is not a bound-state "fraction" or
"correction" to be added to entropy.

## Physical negative control with no out-of-band states

A weak *central hopping bond* \`g=0.5\`, with zero onsite potentials,
does not produce out-of-band eigenmodes in this finite-chain model.
The bound-phase counterfactual is then **exactly identical** to
the original trajectory, as required. The weak bond still scatters
propagating quasiparticles, and still produces finite-size revivals.

**Caution:** the weak-bond control does not have transmission
\`T(k)\` matched to an onsite-potential scatterer; differences between
their trajectories must not be attributed to the bound mode alone.

## What remains to test

1. Robustness to peak identification, smoothing, system size, initial mass.
2. Whether bound-state phases also predict the *amplitude* of revival
   differences in a simple, independent theory.
3. A coherent continuum-path propagator incorporating both scattering
   cuts and bound-state poles of the Green's function.
4. Model comparisons using held-out defects and an actual
   transmission-only quasiparticle baseline (a null zero peak shift is
   not the full published Capizzi-Eisler approximation).
5. Relationship to existing bound-state oscillation results, including
   Gouraud, Le Doussal & Schehr, arXiv:2211.15447, and known finite-size
   scattering effects.

## Reproduce

Run \`python -m pytest -q\` followed by
\`python scripts/run_bound_state_diagnostics.py\`.

Files:
- \`src/bound_state_diagnostics.py\`: spectral partition and exact matrix
  decomposition with unitary phase counterfactual.
- \`tests/test_bound_state_diagnostics.py\`: independent checks and
  no-bound-state control.
- \`results/bound_state_spectral_summary.csv\`: comparison across defects
  and L=48,64.
- \`results/bound_state_trajectories.csv\`: full time-resolved quantities.
- \`figures/bound_state_phase_ablation.png\`: physical vs counterfactual
  capacity for the first defect.

All numerical analyses are exploratory and should not be represented as
a published or peer-reviewed original finding.
