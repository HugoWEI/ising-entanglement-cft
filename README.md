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
