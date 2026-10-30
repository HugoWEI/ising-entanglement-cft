# Source-only Airy first-return residual audit

**Research continuation, 30 October 2026.** This derives a time-uniform clean-propagator error inequality from the Schrödinger-equation residual of the leading Airy approximation. It does not establish a formal interval-arithmetic certificate, a complete defective-chain peak-time enclosure, or scientific novelty.

## Theorem

Let \(h_0\) be the open-chain hopping Hamiltonian, \(U_0(t)=e^{-ih_0t}\), and \(K(t)\) the explicitly constructed first-return Airy image propagator. Define \(R_K=\dot K+ih_0K\). Since \(D=K-U_0\) solves \(\dot D=-ih_0D+R_K\), Duhamel's formula and unitarity give

\[
\boxed{\|K(t)-U_0(t)\|_2\le \|K(0)-I\|_2+\int_0^t\|R_K(s)\|_2\,ds.}
\]

For occupied initial modes \(Q\), the same proof gives the sharper bound \(\|(K-U_0)Q\|_2\le\|(K(0)-I)Q\|_2+\int_0^t\|R_K(s)Q\|_2\,ds\). These inequalities hold for every \(t\le T\) once the continuous residual integral is bounded.

The code constructs \(K\) from the same six open-chain Bessel-image orders as the preceding Hermite–Airy solver, replacing two wraparound images with the leading uniform Olver–Airy term. It differentiates the Airy approximation analytically instead of using finite differences. Near the turning point, \(\zeta=2^{1/3}(1-z)[1+\tfrac3{10}(1-z)+\tfrac{32}{175}(1-z)^2+\tfrac{1037}{7875}(1-z)^3+\tfrac{103727}{1010625}(1-z)^4+\cdots]\). The current polynomial truncation and floating-point integration are **not interval-certified**.

## Independently measured clean-kernel errors

| L | t | Measured \(\|K-U_0\|_2\) | Sampled integral \(\int\|R_K\|\) |
|---|---:|---:|---:|
| 48 | 26.4 | 3.0566e-5 | 4.7453e-5 |
| 48 | 30.0 | 6.4944e-5 | 1.4729e-4 |
| 64 | 36.5 | 2.3517e-5 | 6.0974e-5 |
| 64 | 40.0 | 4.9660e-5 | 1.4902e-4 |
| 80 | 45.0 | 1.8549e-5 | 5.1184e-5 |
| 80 | 50.0 | 4.7763e-5 | 1.5203e-4 |

The right column uses ordinary trapezoid quadrature at \(\Delta t=0.1\): it is a diagnostic of the exact inequality, **not a rigorously upper-rounded number**. The left column uses the independent clean sine-eigenbasis only for post-hoc validation. The Bessel image mode has integrated residual near \(10^{-14}\) for \(L=48,T=30\).

## Transfer to the defective dynamics: remaining term

Let \(B\) embed the two impurity sites, \(V=\operatorname{diag}(u,v)\), and let an arbitrary proposed drive \(Y(t)\) define \(\widetilde F=KQ-iK*(BVY)\), \(q=B^\dagger\widetilde F-Y\). Exact differentiation gives

\[
\dot{\widetilde F}+ih_f\widetilde F
=R_KQ-iR_K*(BVY)+i(I-K(0))BVY+iBVq.
\]

Unitarity of \(e^{-ih_ft}\) then bounds the full occupied-mode error by the integral of the right-hand residual plus the initial error. **The open problem is a uniform, independently certified bound on the inter-node interpolation defect \(q(t)\)** and on all special-function and quadrature rounding. Nodal self-consistency alone is insufficient.

## Reproduction

Requires NumPy, SciPy and pytest:

\`OPENBLAS_NUM_THREADS=1 python -m pytest -q test_airy_kernel_residual.py\`

The four targeted mathematical/numerical checks passed locally. We do not claim new Bessel asymptotics or a novel form of Duhamel's formula. See [NIST DLMF §10.20](https://dlmf.nist.gov/10.20) and Capizzi–Eisler, SciPost Phys. 14, 070 (2023), for relevant established mathematics and coherent-defect revival physics.
