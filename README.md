# Entanglement across a defect in the critical Ising chain

**Status: work in progress / research reproduction.** This repository studies bipartite ground-state entanglement in an open transverse-field Ising chain with a modified central bond. It is an independent learning-and-research project, **not** a peer-reviewed publication or a claim to have discovered defect-dependent entanglement scaling.

## Research question

How reliably can we extract the defect-dependent logarithmic entanglement coefficient from finite-size lattice data, and what finite-size corrections bias the estimate?

We study
```math
H=-\sum_{j=1}^{L-1}J_j\sigma_j^x\sigma_{j+1}^x-h\sum_{j=1}^L\sigma_j^z,
\quad J_{L/2}=tJ,
```
with open boundaries, even `L`, `J=h=1`, and `t\in[0,1]`.

## Repository contents

- `paper/main.tex` — LaTeX manuscript (initial research-proposal draft).
- `paper/references.bib` — foundational research citations.
- `src/ising.py` — free-Majorana covariance computation and entropy.
- `scripts/run_experiment.py` — generate finite-size data and figures.
- `tests/test_ising.py` — mathematical sanity checks and independent small-chain exact-diagonalization validation.
- `poster/` — poster source will be created after reviewing the results.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_experiment.py
```

The script writes `results/half_chain_entropy.csv` and `figures/entropy_vs_logL.png`. The commands are written for Python 3.10+.

## Physical expectations (established results)

For a *uniform* critical open chain, the central bipartition scales as
```math
S(L/2)=\frac{c}{6}\log L + b + o(1),\qquad c=\frac12.
```
For a central interface defect, the coefficient varies with bond strength. This is **known** from Eisler and Peschel (2010). Finite-size fits are not automatically the thermodynamic asymptote.

## Research roadmap

1. Verify covariance approach against small-system exact diagonalization and the clean-chain CFT result.
2. Reproduce the known defect trend and investigate convergence with system size and fit windows.
3. Read and reproduce the analytic Eisler–Peschel expression before making a quantitative comparison.
4. Develop a clearly stated finite-size research question, with uncertainty and limitations.
5. Compile a research note and poster from the validated results.

## References

- P. Calabrese and J. Cardy, *Entanglement Entropy and Quantum Field Theory*, J. Stat. Mech. P06002 (2004). https://arxiv.org/abs/hep-th/0405152
- V. Eisler and I. Peschel, *Solution of the fermionic entanglement problem with interface defects* (2010). https://arxiv.org/abs/1005.2144

## Authorship and transparency

This portfolio is being developed as a learning/research project. Numerical output, literature comparisons, and scientific claims should be independently checked before use in a PhD application. Computational and writing assistance should be acknowledged appropriately where required.


## Automatically compile the PDF with GitHub Actions

The workflow is in [`.github/workflows/build-paper.yml`](.github/workflows/build-paper.yml).
Every push to `main` that changes the LaTeX manuscript or its Python code
will start a cloud build: check out the code, install dependencies, run
five tests, generate the figure, compile LaTeX, and upload the PDF.

**Try it:**

1. Open the repository's [Actions tab](https://github.com/HugoWEI/ising-entanglement-cft/actions).
2. Select **Build Ising research paper** in the left sidebar.
3. Select the newest run. A green check means the build passed; a red cross means you should expand the failing step to read its log.
4. In the run's **Artifacts** section, open **main.pdf** to view or download the generated PDF (GitHub supports unarchived single-file artifacts).
5. To build again manually, select **Run workflow** in the workflow page (main branch).

To see the automation respond to a change, click the pencil icon on
[`paper/main.tex`](paper/main.tex), make a small edit, then **Commit changes**
to `main`. The workflow should run again automatically.

**Note:** The generated PDF is currently a build artifact, not a committed
`paper.pdf` in the source tree. Artifacts are retained temporarily under
GitHub's retention settings. The manuscript remains a preliminary research
note and its physical claims still need scientific review.

## New direction: nonequilibrium entropy and capacity

We are now exploring whether a global field quench through a fixed central
bond defect produces different *dynamical* responses in entanglement entropy
and capacity (variance of the modular Hamiltonian). This is a literature-
screened **pilot**, not a validated original finding.

- [Research plan and literature caveat](docs/dynamics_research_plan.md)
- [Quench simulation](src/dynamics.py)
- [Independent validation tests](tests/test_dynamics.py)
- [Reproducible quench experiment](scripts/run_quench.py)

Run `python scripts/run_quench.py` to create
`results/quench_trajectory.csv` and `figures/quench_dynamics.png`.
The GitHub Actions workflow also generates these as build artifacts.

Crucial caveat: the 2018 conformal-interface replica-partition-function
approach may already determine the *leading continuum capacity behavior*.
Our possible extension is about realistic lattice deviations and quench-
protocol dependence, subject to further literature checks.


## Phase-sensitive defect pilot (separate tight-binding model)

An exploratory *number-conserving tight-binding* project tests two reflected
two-site defects with identical transmission probabilities at every
momentum but different left-reflection phases. With an asymmetric gapped
staggered initial state their finite-time capacity curves differ; with a
reflection-symmetric initial state they agree exactly, as they must.

- [Literature audit, results, limitations, and next hypothesis](docs/phase_scattering_literature_audit.md)
- [Phase-sensitive scattering solver](src/scattering_phase.py)
- [Automated numerical and symmetry tests](tests/test_scattering_phase.py)
- [Pilot script (CSV + plots)](scripts/run_phase_revivals.py)

Run `python scripts/run_phase_revivals.py` after installing dependencies.

**Do not confuse this hopping model with the Ising Hamiltonian**, and do not
describe phase sensitivity alone as a new discovery. The next challenge is
to use reflection-phase/time-delay information to predict a specific
capacity-revival feature and verify it independently.


## First-capacity-revival benchmark: an important negative control

We tested whether a **single** capacity-weighted Wigner reflection delay
predicts the displacement of the first broad modular-capacity revival.
It approximately describes one original impurity pair, but fails for
other impurity pairs. That falsifies the **simple one-delay heuristic**
as a general predictive model; it does **not** rule out coherent sums
of momentum-resolved reflected/transmitted paths.

- [Research diary, precise peak definition and negative results](docs/first_revival_benchmark.md)
- [Numerical benchmark script](scripts/benchmark_first_revival.py)
- [Peak and phase-delay helper functions](src/revival_benchmark.py)
- [Validation tests](tests/test_revival_benchmark.py)

Run `python scripts/benchmark_first_revival.py` to generate
`results/first_revival_benchmark.csv` and
`figures/first_revival_benchmark.png`.

This is exploratory computational physics; no original publication
result or successful coherent many-body theory is claimed.

## Bound-state spectral audit (new)

The phase-sensitive hopping defects used in our revival pilot have
eigenmodes **outside the uniform propagation band**. A new controlled
numerical experiment partitions the Gaussian propagator into band
states and impurity-bound modes, and measures the interference terms
in the correlation matrix.

We also compare the physical evolution with an **artificial unitary
counterfactual** in which bound-state energies are replaced by band-edge
energies. This is only a sensitivity test: the altered Hamiltonian is
generally nonlocal, and capacity cannot be additively decomposed into
bound-state and band-state contributions.

The results show that bound-state phases influence the capacity
revival trajectory, but do not imply a simple universal time shift.

- [Derivation, numerical results and interpretation](docs/bound_state_diagnostics.md)
- [Spectral and covariance diagnostics](src/bound_state_diagnostics.py)
- [Physical weak-bond/no-out-of-band control tests](tests/test_bound_state_diagnostics.py)
- [Reproduction script](scripts/run_bound_state_diagnostics.py)

After `git pull`, run `python scripts/run_bound_state_diagnostics.py`
for CSV files and plots. GitHub Actions also uploads these outputs.


## Coherent scattering from a 2×2 defect T-matrix

We can now derive the exact finite-chain scattering correction from a
**two-by-two** T-matrix rather than diagonalizing the defective chain.
Its secular determinant predicts bound-state energies and localized
eigenvectors. This is a standard, validated scattering-theory
reformulation—not yet a theory of the capacity revival peaks.

- [Mathematical derivation and scientific limitations](docs/coherent_scattering_tmatrix.md)
- [Analytic clean-chain Green function and 2×2 defect T-matrix](src/defect_tmatrix.py)
- [Independent dense-matrix, bound-pole and multiple-scattering tests](tests/test_defect_tmatrix.py)
- [Pole and series-convergence experiment](scripts/run_tmatrix_experiment.py)

Run `python scripts/run_tmatrix_experiment.py` to generate
`results/tmatrix_bound_poles.csv`,
`results/tmatrix_scattering_convergence.csv`, and
`figures/tmatrix_scattering_poles.png`.

A short multiple-scattering series works away from the bound pole but
converges poorly near it. The full 2×2 resummation includes both
coherent reflection and bound-state physics, preparing the next
time-domain approximation.


## New time-domain milestone: unitary reduced dynamics

Two **unitary, second-order numerical time integrators** now evolve the
finite hopping-chain Gaussian state without diagonalizing the defective
Hamiltonian: a two-site Woodbury/Cayley step and a clean-sine/onsite-kick
Strang splitting step. The latter treats clean propagation exactly and
was substantially more accurate for capacity revivals at matched steps.

- [Full derivations, accuracy results and limitations](docs/unitary_time_domain_milestone.md)
- [Rank-two unitary Cayley solver](src/rank2_cayley.py)
- [Exact-clean/defect-kick Strang solver](src/clean_defect_strang.py)
- [Independent validation tests](tests/test_reduced_time_propagators.py)
- [Capacity-revival numerical benchmark](scripts/benchmark_reduced_time_propagators.py)

Run `python scripts/benchmark_reduced_time_propagators.py`.

**Important:** these are numerical solvers of the original Hamiltonian.
Their accuracy is NOT evidence for an original analytical prediction
of capacity revival timing or amplitude. Our open task remains deriving
and validating a genuinely reduced coherent scattering model.


## Coherent boundary-reflection paths: two-site Volterra model

The clean open-chain propagator is now represented as a sum of complex
infinite-chain amplitudes and their **reflected images**. The defect
dynamics obey a closed **two-site Dyson/Volterra convolution equation**,
which resums multiple impurity encounters without diagonalizing the
full defective Hamiltonian.

The finite-image/time-grid approximation is NOT automatically unitary.
The benchmark explicitly reports isometry violations and optionally
applies a nearest-column-isometry (polar) correction. Keeping two image
orders greatly improves the 48-site capacity prediction over keeping
zero or one, but this is still a numerical scattering-path method,
not a new compact analytical revival formula.

- [Derivation, empirical errors, caveats](docs/boundary_return_volterra.md)
- [Infinite-chain Fourier kernel and finite hard-wall images](src/boundary_images.py)
- [Two-site Dyson/Volterra solver and explicit physicality checks](src/boundary_image_volterra.py)
- [Independent validation tests](tests/test_boundary_image_volterra.py)
- [Reproducible exact-vs-path benchmark](scripts/benchmark_boundary_image_volterra.py)

Run `python scripts/benchmark_boundary_image_volterra.py` for the
CSV and two figures. GitHub Actions publishes these as artifacts.


## Analytic capacity response to reversed weak defects

A controlled predictive calculation now evaluates the first-order
difference in modular capacity between two reversed onsite defects,
**using only the symmetric reference Hamiltonian**, and compares it
with the exact asymmetric evolutions afterward. This is a standard
Duhamel/Kubo linear-response application, not new theoretical physics.

For final potentials (w+delta/2,w-delta/2) and their reverse, the
antisymmetric capacity difference is predicted as
`Delta C_A(t) = 2 delta chi_C(t) + O(delta^3)`, away from
entanglement-spectrum endpoint singularities.

Weak-asymmetry cases show strong improvements over the no-response
baseline, while moderate/large asymmetries demonstrably fail.
This predicts a **capacity contrast**, not necessarily the first
capacity peak timing or amplitude.

- [Complete mathematical derivation and honest scope](docs/capacity_asymmetry_linear_response.md)
- [Analytic spectral susceptibility](src/capacity_asymmetry_response.py)
- [Independent finite-difference and reflection controls](tests/test_capacity_asymmetry_response.py)
- [Capacity-response numerical benchmark](scripts/benchmark_capacity_asymmetry_response.py)

Run `python scripts/benchmark_capacity_asymmetry_response.py` after
`git pull origin main` to generate CSVs and figures.


## Exact all-orders scattering response under defect reversal

For reversed central onsite potentials
`(w+delta/2,w-delta/2)`, a symmetric reference chain splits into
even and odd parity scattering channels. The antisymmetric defect
couples only these two channels.

An **exact 2x2 Dyson resummation** predicts the full single-particle
resolvent difference between the reversed defects for arbitrary
asymmetry `delta`, using *only the symmetric reference Hamiltonian*:

`R_delta - R_-delta = delta (R0|+><-|R0 + R0|-><+|R0) / (1-delta^2*g_+*g_-/4)`.

The result matches full asymmetric-matrix inversions to numerical
precision even when first-order response has 7–32% relative error.

**Scope caveat:** This is established symmetry/Dyson algebra, not a
new result on entanglement capacity. The nonlinear capacity functional
does not inherit this simple resummation automatically. A physically
interpretable prediction for actual capacity-revival peak times and
heights still requires a time-domain approximation and validation.

- [Complete mathematical derivation, numbers, originality audit](docs/parity_resummation.md)
- [Exact parity-channel resummation](src/parity_resummed_scattering.py)
- [Independent tests against direct dense inverses](tests/test_parity_resummed_scattering.py)
- [Strong-defect spectral benchmark and figures](scripts/benchmark_parity_resummation.py)

Run `python scripts/benchmark_parity_resummation.py` to recreate the
CSV and figures. GitHub Actions publishes these as artifacts.


## First explicit first-capacity-peak prediction formula (weak asymmetry)

We now have a controlled predictive formula for the shift of an
isolated smoothed capacity maximum under defect reversal. With
asymmetry delta about a symmetric reference and its first isolated
maximum at t0, the analytically predicted time and height
differences are

`Delta t_peak=-2*delta*chi_C'(t0)/C0''(t0)+O(delta^3)`
and
`Delta C_peak=2*delta*chi_C(t0)+O(delta^3)`.

The capacity susceptibility `chi_C` is computed from the symmetric
reference alone via Duhamel/Kubo response, before evaluating either
asymmetric chain. The peak time and height predictions are
validated for delta=.02 across six length/reference choices.
The formula **fails** for the original strong asymmetry delta=.85;
it is NOT an original general strong-defect revival theory.

For all delta, the exact parity-resummed scattering resolvent
and Gaussian Rényi determinant provide an **implicit exact
spectral formula** for capacity, but this is established theory,
not yet an economical explicit first-revival prediction.

- [Derivation, numerical results, validity assumptions and exact all-delta implicit formula](docs/capacity_peak_formula.md)
- [Predictive smoothed-peak formulas with branch tracking](src/capacity_peak_response.py)
- [Independent weak- and strong-asymmetry validation tests](tests/test_capacity_peak_response.py)
- [Reproducible peak-time and height benchmark](scripts/benchmark_capacity_peak_response.py)

Run `python scripts/benchmark_capacity_peak_response.py`
after `git pull origin main`. The Actions pipeline also
uploads the CSV and figures.


## Nonperturbative, exact-form capacity via parity scattering contour

Beyond the **weak-defect peak shift formula**, we can also compute
the full capacity for arbitrarily strong two-site defect asymmetry
using an exact complex-energy contour integral of the resummed
parity-channel T-matrix. This only diagonalizes the symmetric
reference Hamiltonian, while the asymmetric eigen-decomposition
is used independently for validation.

The contour method reproduces the strong-asymmetry capacity
curves and first-broad-maximum locations to numerical accuracy.
It checks state orthonormality without silently repairing
large contour quadrature errors.

- [Full contour, Gaussian determinant and peak derivation](docs/capacity_peak_formula.md)
- [All-asymmetry contour capacity code](src/parity_contour_capacity.py)
- [Independent strong-defect and symmetry tests](tests/test_parity_contour_capacity.py)
- [Reproducible strong-defect benchmark](scripts/benchmark_parity_contour_capacity.py)

**This is a known exact spectral/Gaussian representation, not a
simple novel physical peak-timing law.** It generally needs contour
quadrature and may be computationally heavier than direct
diagonalization. A short, accurate strong-defect revival rule
is still an open target.


## Bound–continuum coherence: exact two-by-two entanglement-spectrum formula

For any defect with **one** localized single-particle bound mode,
the full reduced correlation matrix differs from the bound–band
**dephased Gaussian covariance** by a rank-two operator. The
matrix determinant lemma expresses the full change in the
entanglement-spectrum characteristic polynomial through a
**2x2 determinant**. This is nonperturbative in coherence and
gives an exact implicit capacity peak condition, once the
continuum-band covariance is supplied.

In the original reversed defect pair, the smoothed first
capacity-peak time shifts from 26.4 to 27.4 (L=48), and
36.4 to 37.0 (L=64). The corresponding peaks of the
**dephased Gaussian reference** coincide at 27.8 and 37.0
respectively. This implicates bound-band coherence in that
specific reversal shift; other defect pairs are negative
controls and do not all behave this way.

A nonperturbative **block-Krylov hierarchy** approximates the
rank-two capacity correction. For L=48 a 16-dimensional
subspace (8 blocks) reproduces the two original peak times
on our 0.2 time grid; a 2-dimensional truncation is
UNRELIABLE. The benchmark still gets its continuum
covariance from exact evolution, so this is a conditional
entanglement correction rather than a stand-alone predictive
scattering law.

- [Exact determinant, mathematical peak condition, numerical data,
  dephasing interpretation, limitations](docs/bound_coherence_rank2_formula.md)
- [Single-bound-mode covariance and determinant formula](src/bound_coherence_determinant.py)
- [Nonperturbative compressed rank-two capacity correction](src/bound_coherence_krylov.py)
- [Independent exact and physicality tests](tests/test_bound_coherence_determinant.py)
- [Krylov convergence/failure controls](tests/test_bound_coherence_krylov.py)
- [Reproduce time-series, CSVs and figures](scripts/benchmark_bound_coherence.py)

Run `python scripts/benchmark_bound_coherence.py`.

**Scientific status:** This is a tested application of known
Gaussian covariance and determinant-lemma mathematics, not
yet a proven novel simple strong-defect peak law.


## Composite strong-defect formula from symmetric scattering and bound poles

We now combine three existing ingredients to produce a
**nonperturbative capacity prediction without diagonalizing the
asymmetric Hamiltonian**:

1. The exact parity-channel two-site scattering T-matrix supplies
   the propagated occupied orbitals by an energy contour.
2. The dressed bound-state energy and projector are obtained
   from the same scalar scattering pole; subtracting the pole
   gives the continuum orbitals.
3. A pre-specified **eight-block Krylov approximation** evaluates
   the fully nonlinear rank-two bound–continuum coherence
   correction to the Gaussian modular capacity.

Predicted broad peak times/heights are validated *afterward*
against separately diagonalized asymmetric Hamiltonians for
our original strong defect and an additional configuration.

- [Full mathematical formula, assumptions and scientific limits](docs/scattering_pole_krylov_formula.md)
- [Combined scattering-only predictor](src/scattering_pole_krylov_capacity.py)
- [Independent full-Gaussian and symmetry tests](tests/test_scattering_pole_krylov_capacity.py)
- [Held-out capacity-peak benchmark](scripts/benchmark_scattering_pole_krylov.py)

Run `python scripts/benchmark_scattering_pole_krylov.py`.

**Scientific limitation:** The continuum propagator is still
computed by an expensive near-exact scattering contour. This
is a rigorous nonperturbative *computational formula*, not yet
the compact, original analytical first-revival law we seek.


## Pure-state phase control: same defect, different revivals

We additionally tested whether the first capacity revival depends
on the initial coherence between the localized bound mode and
propagating modes.

A pure-state unitary `U_theta=I+(exp(i theta)-1)P_bound`
changes ONLY their relative phase. It preserves the Hamiltonian,
all scattering amplitudes, bound-state energy, and all modal
occupations and energy moments. Nevertheless, for L=48 and
potentials (1.2,.35), the first broad capacity peak shifts
among 26.4, 26.6, 27.6, and 28.6 as theta goes through
0, pi/2, pi, and 3pi/2.

The exact two-by-two entanglement-spectrum determinant
generalizes by replacing the coherence matrix
`J=[[0,1],[1,0]]` with
`J_theta=[[0,e^(i theta)],[e^(-i theta),0]]`.
This is a *mathematical pure-state preparation control*;
it does not imply a real experimental phase gate exists.

- [Mathematical derivation, 48/64-site results and limitations](docs/bound_coherence_rank2_formula.md)
- [Pure phase-kick revival benchmark](scripts/benchmark_bound_phase_kicks.py)
- [Independent Gaussian/energy-preservation tests](tests/test_bound_coherence_determinant.py)

Run `python scripts/benchmark_bound_phase_kicks.py` after pulling.
