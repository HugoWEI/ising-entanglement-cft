# Endpoint-regular modular-capacity error bound

**Research continuation, 30 October 2026.** Mathematical and numerical verification of a sharper, physical-state-dependent capacity-continuity estimate for the first-return Airy/Bessel correction. This does **not** prove an independently certified first-revival peak-time interval or establish novelty.

## Theorem (pure number-conserving fermionic Gaussian states)

Let \(X\in\mathbb C^{L\times m}\) satisfy \(X^\dagger X=I\), and let \(A\) be the orthogonal projector onto the left subsystem. Set \(B=X^\dagger A X\) with eigenvalues \(p_j\in[0,1]\). The Gaussian modular capacity equals \(\mathrm{Tr}\,f(B)\), where \(f(p)=p(1-p)\log^2[(1-p)/p]\), extended by zero at 0 and 1.

For any differentiable isometric path \(X(s)\), put \(H=(I-XX^\dagger)\dot X\). Then

\[
|\dot C_A|\le 2\left[\sum_j p_j(1-p_j) f'(p_j)^2\right]^{1/2}\|H\|_F
\le 2M\sqrt m\,\|H\|_F,
\quad M=\max_{p\in[0,1]}\sqrt{p(1-p)}|f'(p)|<1.22.
\]

**Proof:** At interior eigenvalues, differentiating the trace and dropping the vertical right-unitary gauge gives \(\dot C=2\operatorname{Re}\operatorname{Tr}[H^\dagger (I-XX^\dagger)AX f'(B)]\). Since
\[
\|(I-XX^\dagger)AX f'(B)\|_F^2
=\mathrm{Tr}[f'(B)^2(B-B^2)]
=\sum_j p_j(1-p_j)f'(p_j)^2,
\]
Cauchy–Schwarz proves the first estimate. Despite \(f'\) diverging at 0 and 1, the horizontal gradient weight \(\sqrt{p(1-p)}f'(p)\) vanishes there. The bound extends to those endpoints along smooth isometric paths.

For \(Z_\lambda=Z_0+\lambda\Delta Z\) with full column rank, \(X_\lambda=\operatorname{polar}(Z_\lambda)\), its horizontal derivative is \(H_\lambda=(I-X_\lambda X_\lambda^\dagger)\Delta Z(Z_\lambda^\dagger Z_\lambda)^{-1/2}\). If \(\sigma_{\min}(Z_\lambda)\ge s>0\), it follows that
\[
|\Delta C_A|\le \frac{2M\sqrt m}{s}\|\Delta Z\|_F.
\]
For an initial occupied-mode isometry \(Q\), the first-return Bessel-to-Airy substitution has \(\|\Delta Z\|_F=\|(W_B-W_A)Q\|_F\le\|W_B-W_A\|_F\).

## Constant audit

Write \(u=|\log[(1-p)/p]|\). Then \(M=\sup_{u\ge0}|u^2\tanh(u/2)-2u|/[2\cosh(u/2)]\). Numerical maximization yields \(M\approx1.21080189\), at \(u\approx5.39862\). The companion elementary-exponential interval check covers \([0,30]\) using 6,000 subintervals, with worst upper envelope <1.21451; for \(u\ge30\), \((u^2+2u)e^{-u/2}\le960e^{-15}<0.000294\). **This is conditional on mpmath.iv primitive inclusion and is not an independent Arb/MPFI certification.**

## L=48 first-return source-only benchmark

| onsite defect | sampled t | previous endpoint-modulus upper | new geometric upper | observed Bessel–Airy difference |
|---|---:|---:|---:|---:|
| (1.20, 0.35) | 26.4 | 0.0190524 | **0.00052704** | 0.000005288 |
| (0.35, 1.20) | 27.3 | 0.0234828 | **0.00062144** | 0.000001935 |

The reduction is **36.1× and 37.8×**, respectively. The upper bounds use pointwise wavefront enclosures from the earlier finite-order audit and floating-point estimates of minimum singular values. They are **conditional**, not a completed formally interval-verified peak-time certificate. The observed differences are retrospective numerical tests, not inputs to the upper bounds.

## Reproduction

- \`python -m unittest -v test_capacity_geometry\`
- \`python verify_constant.py\`
- Python dependencies: numpy, scipy, mpmath.

## Context / novelty

Established CS decomposition, Grassmann horizontal calculus, polar perturbations and Gaussian correlation-matrix methods underlie this estimate. This is a model-specific derivation and useful quantitative improvement, **not a claim of new mathematics or new coherent-revival physics**. For prior defect-revival theory see Capizzi & Eisler, SciPost Phys. 14, 070 (2023), https://doi.org/10.21468/SciPostPhys.14.4.070. For Gaussian modular capacity see Bhattacharjee, Nandy & Pathak, Phys. Rev. B 104, 214306 (2021).

The **next open step** is a time-uniform Airy and interpolation error bound, plus continuous peak-current sign/curvature verification.