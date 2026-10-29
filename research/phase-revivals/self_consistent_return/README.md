# Self-consistent Airy first-return scattering — numerical validation

**30 October 2026.** A reproducible update to `quantum-entanglement-revivals`, following the sampled Airy peak-transfer analysis. This is a computational diagnostic and improvement in bookkeeping, **not** a new published theorem, not a formal interval certificate, and not yet a self-contained fourth-order Airy predictor.

## What was wrong with the old post-hoc model?

For the clean open chain, separate the exact image propagator as `U_0(t) = K_0(t) + W(t) + R(t)` where `K_0` includes direct propagation and the first reflection at each end, `W` consists of the two leading wrap images (the first long reflected return), and `R` includes higher images. The old approximation solved impurity scattering with `K_0` and *then* added `W(t)Q` to the occupied amplitudes. That omitted the additional impurity response to the arriving return wave.

For an embedding `B` of the two defect sites, with onsite matrix `V=diag(u,v)`, the **self-consistent** two-site equation for a clean propagation kernel `K = K_0 + W` is

\[
Y(t)=[K(t)Q]_D-i\int_0^t K_{DD}(t-s)VY(s)\,ds,
\qquad
F(t)=K(t)Q-i\int_0^t K_{:D}(t-s)VY(s)\,ds.
\]

The first-return images are

\[
W_{ij}(t)=i^{N+i-j}J_{N+i-j}(2t)+i^{N-i+j}J_{N-i+j}(2t),\quad N=2(L+1).
\]

`airy` replaces only these two Bessel factors by the leading uniform Olver-Airy approximation. All other retained images are Bessel-evaluated. The `full_bessel` method also includes the next left-image `-i^{N+i+j+2}J_{N+i+j+2}(2t)`.

Expanding `Y=Y_0+\delta Y` makes the formerly missing terms explicit:

\[
\delta Y = W_DQ-iK_{0,DD}*(V\delta Y)-iW_{DD}*(VY_0)-iW_{DD}*(V\delta Y).
\]

Similarly,

\[
\delta F=WQ-iK_{0,:D}*(V\delta Y)-iW_{:D}*(VY_0+V\delta Y).
\]

The old post-hoc approximation kept **only** `WQ`. In the new implementation all these terms are included once, through the same Volterra solve, rather than appended repeatedly.

## Numerical results

Open fermion chain `L=48`, half filling, initial staggered mass `m=2.5`, hopping `J=1`, Gaussian smoothing of capacity over samples `t=0,...,36` with width `sigma=2`. Fourth-order clean-chain sine-mode Hermite Volterra is the independent **source-only** comparator, not diagonalization of the defective Hamiltonian.

| Defect (u,v) | h | post-hoc Airy peak | self-consistent Airy peak | Hermite reference peak | Airy–Hermite error |
|---|---:|---:|---:|---:|---:|
| (1.20,.35) | .1000 | 26.403163 | 26.403147 | 26.422635 | -0.019488 |
| (1.20,.35) | .0500 | 26.417596 | 26.417580 | 26.422677 | -0.005097 |
| (1.20,.35) | .0250 | 26.421402 | 26.421386 | 26.422680 | -0.001295 |
| (1.20,.35) | .0125 | 26.422367 | 26.422350 | 26.422681 | -0.000331 |
| (.35,1.20) | .1000 | 27.216161 | 27.216459 | 27.311727 | -0.095268 |
| (.35,1.20) | .0500 | 27.286092 | 27.286453 | 27.311932 | -0.025479 |
| (.35,1.20) | .0250 | 27.304993 | 27.305373 | 27.311954 | -0.006581 |
| (.35,1.20) | .0125 | 27.309817 | 27.310201 | 27.311960 | -0.001758 |

A two-grid h² Richardson extrapolation \((4t_{h/2}-t_h)/3\) at `h=0.0125` yields **26.422672** and **27.311811**; corresponding differences to the same-grid Hermite references are **-9.1e-6** and **-1.49e-4**. This is an *extrapolated numerical estimate*, not a rigorously controlled fourth-order error bound.

Held-out weaker defect `(.50,.20)` yields `26.479825` self-consistent Airy at `h=.025`, while Hermite predicts `26.479763`, difference `6.3e-5`; Richardson using `.05/.025` gives `26.479758`, difference `-4.3e-6`.

**Interpretation:** The explicit first-return feedback modifies the peak by approximately `-1.7e-5` for `(1.2,.35)` and `+3.8e-4` for its reverse at fine spacing. The dominant baseline error of the old method is the second-order product-integration error, rather than a failure of the Olver-Airy return waveform. The Airy/Bessel difference itself remains about `-8.0e-6` and `-1.4e-4` at `h=.0125`; increasing numerical order exposes that smaller physical-approximation error.

The next higher Bessel image changes the observed first-peak time by less than `5e-13` across these conditions. This is a numerical diagnostic, not a global truncation certificate.

## Implementation and tests

`self_consistent_return.py` tabulates all relevant clean-chain Bessel images once, assembles the two-site Volterra matrices and solves by trapezoidal product integration. It outputs capacity curves from polar-orthonormalized occupied amplitudes. The original defective Hamiltonian is never diagonalized. The fourth-order reference uses the clean-chain sine basis and a 4×4 Hermite collocation solve.

```bash
python -m pip install numpy scipy matplotlib pytest
OPENBLAS_NUM_THREADS=1 python -m pytest -q test_self_consistent_return.py
OPENBLAS_NUM_THREADS=1 python self_consistent_return.py --dt 0.025 --T 36
OPENBLAS_NUM_THREADS=1 python summarize_results.py
```

**Four tests passed**, covering a direct one-shell image identity, agreement with an independent existing Volterra solver, feedback into the two-site impurity amplitudes, and basic wavefront/initial-state consistency. A `.0125` calculation is more expensive; the repository includes machine-readable convergence results from the runs.

## Limits and research next step

The present step clarifies and implements first-return impurity feedback, but the main gain in peak accuracy comes from numerical refinement/extrapolation. The final reduced model still requires the original numerical baseline and polar projection, and the extrapolated peaks have no formal interval enclosure. We should **replace the second-order baseline with fourth-order collocation directly in the Airy-modified kernel**, or derive a provably stable correction that avoids a second solution. Then propagate the *continuous* Airy remainder and low-rank truncation into the capacity-current stability bound.

Existing coherent defect-induced entanglement revivals: Capizzi & Eisler, SciPost Phys. **14**, 070 (2023). The current work does not claim novelty over that theory without an explicit asymptotic comparison.
