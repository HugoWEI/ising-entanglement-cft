# Finite-order Airy remainder and trace-sensitive modular capacity

**Research continuation, 29 October 2026.** This is a self-contained, reproducible *diagnostic* advancing the previous conditional a posteriori first-peak analysis. It does not establish a new publishable physics theorem or complete interval-certified peak enclosure.

## Goal and what is genuinely new in this run

1. Construct finite-order *pointwise* enclosures on the difference between the exact Bessel first-return images and the leading uniform Olver Airy approximation. The Bessel and Airy values are independently calculated by series with explicit geometric tail bounds; no defective Hamiltonian solver is used.
2. Propagate these scalar remainder enclosures into a norm bound for the entire first-return propagation matrix and, conditionally on physical polar factors and their minimum singular values, into a projector bound.
3. Demonstrate why generic capacity-modulus inequalities are far too conservative and verify the sharper **exact trace-response identity** along a physically valid polar path.

The underlying ideas (Bessel/Airy asymptotics, polar perturbation, matrix calculus) are known; the results are a model-specific quantitative audit.

## A. Finite-order special-function enclosures

For integer `n>=1` and rational `t>=0`, the Bessel series is

\[
 J_n(2t)=\sum_{k=0}^{K}\frac{(-1)^k t^{n+2k}}{k!(n+k)!}+R_{n,K}(t).
\]

Let `a_{K+1}=t^(n+2K+2)/((K+1)!(n+K+1)!)`. If

\[
 q=\frac{t^2}{(K+2)(n+K+2)}<1,
 \qquad |R_{n,K}|\le \frac{a_{K+1}}{1-q}.
\]

The bound follows by taking absolute values and comparing successive omitted terms with a geometric series. We use `K=100` and high-precision interval arithmetic.

For the Airy function, its ODE \(\mathrm{Ai}''(x)=x\mathrm{Ai}(x)\) gives the Taylor recursion

\[
 a_{k+2}=\frac{a_{k-1}}{(k+2)(k+1)},\qquad
 a_0=\frac{1}{3^{2/3}\Gamma(2/3)},\quad
 a_1=-\frac{1}{3^{1/3}\Gamma(1/3)},\quad a_2=0.
\]

Only coefficients with indices `0 mod 3` and `1 mod 3` survive. For either branch the ratio of successive terms in absolute value is at most \(|x|^3 /[(3j+p+3)(3j+p+2)]\), which yields an explicit geometric tail after sufficiently many terms. We use `J=110`. In the positive-Airy decay region `x>15`, we avoid cancellation-prone Taylor evaluation and use the *inequality* (DLMF 9.7.15)

\[
0<\mathrm{Ai}(x)\le \frac{e^{-2x^{3/2}/3}}{2\sqrt\pi x^{1/4}}.
\]

**Scope of interval claim.** The code uses `mpmath.iv` at 110 decimal digits for arithmetic and for `Gamma`, `log`, `sqrt`, `atan2`, and `exp`. The library's interval Bessel and interval Airy routines fail in this environment, hence the independently implemented power-series enclosures. All six automated tests pass, including independent 160-digit checks. However, we have **not audited the numerical interval library itself** or performed an externally validated Arb/MPFI check. Therefore this is *interval-assisted evidence conditional on the inclusion correctness of its primitives*, not a formally machine-verified proof.

## B. From first-return scalar to matrix error

Let \(N=2(L+1)\). The wavefront retained in the model is

\[
W_{ij}(t)=i^{N+i-j}J_{N+i-j}(2t)+i^{N-i+j}J_{N-i+j}(2t).
\]

Define \(e_n(t)=J_n(2t)-J_n^{\rm Olver,0}(2t)\). Interval enclosures for each `n` yield enclosures for each complex matrix entry in \(D=W-W_{\rm Ai}\). Since the phases are integer powers of `i`, their relative sign is known exactly (equal for even `i-j`, opposite for odd `i-j`). Thus 

\[
\|D\|_2\le \|D\|_F\le \left(\sum_{ij} d_{ij}^2\right)^{1/2},
\]

where each \(d_{ij}\) bounds the modulus of the true entry using the interval endpoints (including phase cancellations). For any orthonormal initial occupied modes `Q`, \(\|(W-W_{\rm Ai})Q\|_2 \le \|D\|_F\).

| L | t | interval-assisted pointwise upper bound on `||D||_2` | independent measured `||D||_2` |
|---|---:|---:|---:|
| 48 | 26.4214 | 4.42925e-5 | 3.07007e-5 |
| 48 | 27.3050 | 5.20066e-5 | 3.38566e-5 |
| 64 | 36.4766 | 4.44313e-5 | 2.35387e-5 |
| 64 | 37.0491 | 5.24646e-5 | 2.89291e-5 |
| 80 | 45.0000 | 3.64135e-5 | 1.85492e-5 |

All statements are **pointwise at the rational times shown**. They are not time-uniform enclosures, do not include omitted paths or feedback, and do not by themselves certify the peak locations.

## C. Physical projector and capacity transfer

Let \(Z_B=Z_0+W_BQ\) and \(Z_A=Z_0+W_AQ\), with `Z0` the common direct+single-end reflection baseline from the source-only two-site Volterra solver. Define physical occupied subspaces via \(X_B=\operatorname{polar}(Z_B)\) and \(X_A=\operatorname{polar}(Z_A)\).

For full-column-rank matrices, the standard polar perturbation inequality and \(P=XX^\dagger\) imply

\[
\boxed{\|P_B-P_A\|_2\le\frac{4\|Z_B-Z_A\|_2}{\sigma_{\min}(Z_B)+\sigma_{\min}(Z_A)}.}
\]

The numerical small singular values are near `0.9995` for both original orientations, so the pointwise wavefront estimates imply a bound of `8.82e-5` (forward, `t=26.4`) and `1.04e-4` (reverse, `t=27.3`) *conditional on those unverified floating-point lower singular-value estimates*.

Using Weyl's inequality for the left-half correlation eigenvalues and a state-dependent capacity modulus gives conservative capacity difference bounds `0.0191` (forward) and `0.0235` (reverse). The directly measured Bessel-vs-Airy capacity differences are only `5.29e-6` and `1.94e-6`, respectively. Thus worst-case eigenvalue perturbation loses over three orders of magnitude: the limiting difficulty is no longer the intrinsic Airy wavefront replacement, but propagation of that small error through an inherently structured spectral observable.

## D. An exact trace response that preserves cancellations

Interpolate the **full-amplitude** wavefronts,

\[
Z_\lambda=Z_A+\lambda(Z_B-Z_A),\quad X_\lambda=Z_\lambda(Z_\lambda^\dagger Z_\lambda)^{-1/2},\quad 0\le\lambda\le1.
\]

The physically relevant horizontal derivative of the occupied space is

\[
H_\lambda=(I-X_\lambda X_\lambda^\dagger)
 (Z_B-Z_A)(Z_\lambda^\dagger Z_\lambda)^{-1/2}.
\]

For a smooth full-rank path whose entanglement spectrum stays in `(0,1)`, with \(f(p)=p(1-p)\log^2((1-p)/p)\),

\[
\boxed{\frac{dC_A}{d\lambda}=2\operatorname{Re}\operatorname{Tr}\left[X_{A,\lambda}^\dagger f'(G_{A,\lambda}) H_{A,\lambda}\right].}
\]

The potential gauge component of \(\dot X\) cancels because the spectrum of `G_A` is invariant under right-unitary rotations of occupied orbitals. Integration over \([0,1]\) is an exact identity. Test calculations use eigenvalue clipping only to evaluate the derivative numerically at machine precision; the derivation itself requires no clipping and must handle endpoint crossings separately.

| defect at L=48 | measured `C_Bessel - C_Airy` | trace integral | worst-case capacity modulus |
|---|---:|---:|---:|
| (1.2,0.35), t=26.4 | -5.28789115e-6 | -5.28789123e-6 | 0.01905 |
| (0.35,1.2), t=27.3 | -1.93500415e-6 | -1.93500376e-6 | 0.02348 |

This identity is **not a new theorem**; it is a standard matrix trace-derivative applied to our physical polar interpolation. Its usefulness is that it suggests an error certificate that retains cancellation rather than applying independent eigenvalue worst cases.

## What is still unfinished

1. **Interval-special-function auditing:** substitute a rigorously documented ball implementation (Arb/FLINT) or verify all required primitive interval routines independently. The current library's `besselj`/`airyai` interval calls are broken, which motivated the series construction.
2. **Time-uniform bound:** extend the pointwise \(e_n(t)\) enclosures across the whole peak window, including the `z=1` crossing where the present leading-Olver implementation changes formula.
3. **Full peak certificate:** certify singular-value lower bounds and the trace-response integral uniformly across wavefront interpolation and the full peak-time interval; combine with omitted-image, impurity feedback and discretization errors. We have not completed this.
4. **Literature novelty:** compare with Capizzi–Eisler's conformal-defect revival formula and established Olver asymptotics. The new result here is primarily a tested, model-specific certification workflow, not a novel universal physics law.

## Run

`OPENBLAS_NUM_THREADS=1 python -m unittest -v test_finite_order test_response`

`OPENBLAS_NUM_THREADS=1 python run_finite_order_audit.py`

`OPENBLAS_NUM_THREADS=1 python run_projector_transfer.py`

For the projector-transfer script, `code/` contains the few modules taken unchanged from the preceding research bundle; those modules also import dependencies from the same folder. Numerical evaluation uses NumPy/SciPy; high precision interval work uses `mpmath.iv`.

## References

- NIST DLMF §10.20, uniform Bessel–Airy asymptotic expansions: https://dlmf.nist.gov/10.20
- NIST DLMF §9.7.15, positive-Airy exponential upper bound: https://dlmf.nist.gov/9.7.E15
- Capizzi and Eisler (2023), *Entanglement evolution after a global quench across a conformal defect*: https://arxiv.org/abs/2209.03297
