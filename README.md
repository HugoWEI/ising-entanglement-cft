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
