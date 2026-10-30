# Clean-propagator witness for the Airy-generated impurity drive

The Airy solver generates a two-site Hermite drive `Y(t)`. This module then **reconstructs the state using the exact clean-chain sine-mode propagator** and derives a source-only, full-cell fourth-order residual upper bound. No eigendecomposition of the defective Hamiltonian is required for the predictor or the mathematical error inequality.

**Main files:** `clean_witness.py` implements the reconstruction and analytic between-node Hermite bound; `conditional_peak.py` applies the physical Gaussian-state capacity continuity inequality to a sampled Gaussian peak; `peak_study.py` performs a strictly post-hoc exact-diagonalization benchmark; `validate_against_exact.py` compares measured errors with the conditional upper bounds; `test_clean_witness.py` contains six tests.

**Scope:** analytically derived inequalities are rigorous *given exact nodal values, norms, and integrals*. This implementation uses double precision and non-validated Gauss quadrature, so its numerical outputs are **conditional diagnostics**, not machine-certified enclosures. Peak-root enclosures also require continuous interval extrema (the current script uses a diagnostic mesh).

See [RESEARCH_NOTE.md](RESEARCH_NOTE.md) for the derivations, exact assumptions, benchmarks, and limitations.
