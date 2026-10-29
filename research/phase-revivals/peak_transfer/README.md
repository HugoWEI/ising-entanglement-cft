# Sampled-to-continuous first-peak stability for phase-resolved modular capacity

**Research continuation: 30 October 2026.** Model-specific error-transfer theorem and reproducible numerical tests for the first-return Airy and Bessel scattering *surrogates* for the open free-fermion chain. No final defective-Hamiltonian diagonalization is used in these calculations. **The numerical enclosures are conditional diagnostics, not formally verified interval-arithmetic certificates, and not a certificate relative to the true defective-chain dynamics.**

## What the new result controls

Let the discrete Gaussian smoother be **defined**, for all continuous argument times `t`, from a fixed sample grid `t_j` and sample capacities `c_j` by

\[
\bar C(t)=\sum_jw_j(t)c_j,\quad
w_j(t)=\frac{\exp[-(t-t_j)^2/(2\sigma^2)]}{\sum_k\exp[-(t-t_k)^2/(2\sigma^2)]}.
\]

This is **not** the same as a continuously convolved capacity curve or SciPy's `gaussian_filter1d` with a chosen boundary condition.

If pointwise sample differences obey `|c_j - c'_j| <= eps_j`, then for *every continuous* `t`:

\[
|\bar C^{(k)}(t)-\bar C'^{(k)}(t)|\le \sum_j |w_j^{(k)}(t)|\,\epsilon_j, \quad k=0,1,2.
\]

Let `a_j(t)=(t_j-t)/sigma^2`, `mu=sum w_j*a_j`, and `v=sum w_j*(a_j-mu)^2`. Then

\[
w_j'=w_j(a_j-\mu),\qquad
w_j''=w_j[(a_j-\mu)^2-v].
\]

We use explicit Gaussian numerator and denominator interval majorants to upper-bound these weighted sums on an entire peak-search interval. We also bound the third derivative of the Airy smoothed curve, which converts a dense grid of its second derivative into a conservative continuous-interval curvature lower estimate.

### Local peak-location theorem

Suppose \(\bar C_A'(t_A)=0\), and on a neighborhood \([t_A-R,t_A+R]\) the Airy curve has \(\bar C_A''\le-\kappa<0\). If our capacity-sample uncertainty implies \(|\bar C_B' - \bar C_A'|\le\eta_1\) and \(|\bar C_B''-\bar C_A''|\le\eta_2<\kappa\) throughout the neighborhood, and the Bessel current has positive/negative signs at the two endpoints (as ensured by the corresponding strict Airy-sign margins), the Bessel smoother has exactly one maximum there with

\[
|t_B-t_A|\le \eta_1/(\kappa-\eta_2).
\]

No capacity-eigenvalue derivative is needed in this step; the nodewise \(\epsilon_j\) themselves come from the previously established endpoint-regular Gaussian-state geometric bound.

### Building sample envelopes from the scattering calculation

For each sample, define a common numerical baseline `Z0(t_j)` (direct and singly reflected clean paths, with onsite potentials resummed by a **second-order trapezoidal two-site Volterra solver**) and add either the Bessel first-return wavefront or its leading Olver–Airy approximation: \(Z_B=Z_0+W_BQ\), \(Z_A=Z_0+W_AQ\). Apply a polar orthonormalization to each.

In exact arithmetic, if \(s_j\le \inf_{\lambda\in[0,1]}\sigma_{\min}(Z_A+\lambda(Z_B-Z_A))\) is a verified positive lower bound and `M<1.22` is the universal geometric constant,

\[
\epsilon_j=\frac{2M\sqrt m}{s_j}\,\|(W_B(t_j)-W_A(t_j))Q\|_F
\]

bounds the capacity difference at node `j`. The implementation takes a conservative *algebraic* singular-value lower estimate \(s_j=\sigma_{\min}(Z_A)-\|Z_B-Z_A\|_2\).

However, `scipy.special.jv`, `scipy.special.airy`, and `numpy.linalg.svd` here provide floating-point values **without outward-rounded inclusion guarantees**. The `1+1e-10` multiplicative slack and `5e-12` absolute capacity slack in the numerical script are **not a proof of rounding-error coverage**. Indeed, before adding the small absolute slack, 14–31 early-time samples showed discrepancies of \(\approx10^{-12}\) where the theoretical error was below floating-point resolution. This is an important cautionary test, not evidence the theorem fails.

## Reproducible result for L=48

Sample grid `t_j=0,0.025,...,36`, 1441 nodes, `sigma=2`, staggered initial mass `m0=2.5`, hopping `J=1`. `t_A`/`t_B` are peaks of the explicitly defined Gaussian smoother.

| Defect (u,v) | Airy t_A | Bessel t_B | observed | conditional **continuous-neighborhood** bound |
|---|---:|---:|---:|---:|
| (1.20, 0.35) | 26.42140246 | 26.42141046 | 7.99e-6 | 0.006556 |
| (0.35, 1.20) | 27.30499321 | 27.30513360 | 1.40e-4 | 0.025208 |
| (0.50, 0.20) | 26.47982101 | 26.47982578 | 4.76e-6 | 0.002432 |

The quoted bounds use a radius `R=0.05` and the conservative continuous Gaussian and curvature majorants, evaluated in floating point. The endpoint-sign/curvature inequalities pass for all three cases. An additional adaptive positivity check also passes on `[19.2,t_A-0.05]`, conditional on the same numerical envelopes; it therefore excludes an earlier local maximum of the surrogate smoother in that search interval. A sampled prominence comparison at peak vs. points 7.2 units on either side yields lower margins >0.18 for the original two defects, exceeding the 0.08 benchmark threshold. **This does not exclude a qualifying peak before t=19.2 or establish a first maximum of the exact defective Hamiltonian.**

The 1,441-node sample arrays, numerical witness/analytic bounds, and full diagnostic JSON are stored alongside the source. The low-level baseline solver and Airy approximation are imported from the preceding `certificate/` research module; the Gaussian geometry bound comes from `capacity_geometry/`.

## Important decomposition of remaining error

These results isolate the leading **Bessel-to-Airy replacement** while holding the approximate baseline fixed. The relative peak shifts are much smaller than earlier **Airy-versus-full-defective-chain** numerical discrepancies (of order 0.001 and 0.007 in the original two orientations). This is numerical evidence that the Airy replacement itself is not the principal source of total predictive error. To certify the actual quantum dynamics we still need to bound the baseline, neglected impurity feedback, image paths, and Volterra quadrature, all within a common consistent smoothing definition.

**No originality claim:** Gaussian kernel differentiation, curvature-based root stability, Olver asymptotics, and polar Gaussian-state geometry are established techniques. Coherent defect entanglement revivals were studied in Capizzi & Eisler, SciPost Phys. 14, 070 (2023), https://doi.org/10.21468/SciPostPhys.14.4.070. This is a tested quantitative application, not yet new physics.

## Run

From the `peak_transfer/` directory, with `numpy`, `scipy`, `mpmath` available:

```bash
OPENBLAS_NUM_THREADS=1 python sampled_peak_transfer.py
OPENBLAS_NUM_THREADS=1 python continuous_peak_envelope.py
OPENBLAS_NUM_THREADS=1 python -m unittest -v test_sampled_peak_transfer
```

Reproduction evaluates approximately 1,441 time slices for each of three defect pairs and may take a few minutes. Numerical files are deterministic with NumPy/SciPy within ordinary floating-point tolerances; this is **not** a formal machine certificate.
