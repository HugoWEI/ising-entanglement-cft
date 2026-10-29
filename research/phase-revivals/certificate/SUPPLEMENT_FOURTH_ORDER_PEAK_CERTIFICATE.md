# Fourth-order impurity collocation and conditional first-revival peak enclosure

**Research supplement — 29 October 2026**  
**Model:** finite half-filled free-fermion chain with two adjacent onsite impurity potentials; initial state is the ground state of the staggered-mass chain.  
**Scope:** self-contained mathematical argument, reproducible computational construction and independent benchmark. **No new-physics or publication novelty claim.**  
**Result:** a two-site, fourth-order, source-only numerical integrator with a nonvacuous *conditional exact-arithmetic* local capacity-peak enclosure. The evaluation is in double precision, not directed interval arithmetic. The result is for the full two-site scattering solver, **not** yet for the reduced Airy-only model.

## 1. Set-up and conventions

Let `L` be even and `M=L/2`. Define the one-particle Hamiltonians

\[
 h_0=-\sum_{j=0}^{L-2}(|j\rangle\langle j+1|+|j+1\rangle\langle j|),
 \qquad h=h_0+BVB^\dagger,
\]

with `B` the `L×2` coordinate-embedding matrix at sites `D={L/2-1,L/2}` and `V=diag(u,v)`. Set `hbar=1`, nearest-neighbor hopping equal to one, and choose an `L×M` initial matrix `Q` of orthonormal occupied eigenmodes of the staggered-mass initial Hamiltonian. The exact occupied-mode matrix is `F(t)=exp(-iht)Q`.

For *any* continuously differentiable two-site candidate `Y(t)` with `Y(0)=B†Q`, define the **clean-chain driven reconstruction**

\[
F_Y(t)=e^{-ih_0t}Q-i\int_0^t e^{-ih_0(t-s)}BVY(s)\,ds,
\qquad r(t)=Y(t)-B^\dagger F_Y(t).
\]

The exact `F` and approximate `F_Y` obey the residual-forced equation

\[
\dot{(F-F_Y)}=-ih(F-F_Y)-iBVr,
\]

and therefore, using `||exp(-iht)||_2=1`,

\[
\boxed{\|F(t)-F_Y(t)\|_2\le b\int_0^t\|r(s)\|_2\,ds},
\quad b=\|V\|_2=\max(|u|,|v|).
\]

This requires **no diagonalization or time evolution of the defective Hamiltonian for the bound**. The norm refers to the operator norm of `L×M` or `2×M` matrices, as appropriate. All inequalities here refer to exact arithmetic unless explicitly qualified.

## 2. Fourth-order Hermite collocation using only a 4×4 impurity solve

The clean open-chain single-particle eigenvalues and normalized sine-wave eigenvectors are known analytically:

\[
 e_k=-2\cos\frac{\pi k}{L+1},\qquad S_{jk}=\sqrt{\frac2{L+1}}\sin\frac{\pi(j+1)k}{L+1},\quad 1\le k\le L.
\]

This diagonalizes **the clean Hamiltonian only**. On each time cell `[t_k,t_{k+1}]`, let `h=t_{k+1}-t_k` and approximate the impurity amplitude by cubic Hermite interpolation constrained by `Y_k,Y_{k+1}` and their slopes `D_k,D_{k+1}`:

\[
Y(t_k+hx)=H_{00}(x)Y_k+hH_{10}(x)D_k+H_{01}(x)Y_{k+1}+hH_{11}(x)D_{k+1}, \quad 0\le x\le1,
\]

where `H00=2x³−3x²+1`, `H10=x³−2x²+x`, `H01=−2x³+3x²`, `H11=x³−x²`.

The forced convolution over one cell can be evaluated with entire *phi functions*

\[
\varphi_j(z)=\sum_{n=0}^\infty\frac{z^n}{(n+j)!}
\]

and weights, for `z=-ieh`,

\[
\begin{aligned}
w_{00}&=\varphi_1-6\varphi_3+12\varphi_4,\\
w_{10}&=\varphi_2-4\varphi_3+6\varphi_4,\\
w_{01}&=6\varphi_3-12\varphi_4,\\
w_{11}&=-2\varphi_3+6\varphi_4.
\end{aligned}
\]

The unknown endpoint amplitudes and slopes are determined by simultaneously imposing

\[
 Y_{k+1}=B^\dagger F_Y(t_{k+1}),\quad
 D_{k+1}=B^\dagger\big(-ih_0F_Y(t_{k+1})-iBVY_{k+1}\big).
\]

They form a `4×4` complex linear system, shared across the `M` occupied-mode right-hand sides. Implemented in `high_order_volterra.py`. There is no stepwise nonlinear capacity fitting or final-defect diagonalization.

## 3. A continuous-time residual bound for the cubic interpolant

At every node, the construction imposes `r(t_k)=r'(t_k)=0` in exact arithmetic. Since the two-site drive is cubic inside each cell, `r^{(4)}(t)=-B†F_Y^{(4)}(t)`. Repeated differentiation of the clean forced equation gives

\[
 F_Y^{(4)}=h_0^4 F_Y+h_0^3 BVY+i h_0^2 BVY'-h_0BVY''-iBVY'''.
\]

Here `||h0||≤2`. If `M4,k` bounds `||r^{(4)}||` on the whole cell, the endpoint-Hermite interpolation remainder yields

\[
 \|r(t)\|_2\le\frac{M_{4,k}}{4!}(t-t_k)^2(t_{k+1}-t)^2,
 \qquad b\int_{t_k}^{t_{k+1}}\|r(s)\|_2ds\le\frac{b M_{4,k}h^5}{720}.
\]

The implementation deliberately uses the **looser** coefficient `1/384` in place of `1/720`, conservatively. Supremum norms of `Y,Y',Y'',Y'''` follow from its explicit cubic coefficients, while `sup_cell||F_Y||≤||F_Y(t_k)||+hb sup_cell||Y||` follows from clean propagation and Duhamel. The computable continuous-time majorant is

\[
M_{4,k}\le 16F_{\max,k}+8bY_{\max,k}+8bY'_{\max,k}+2bY''_{\max,k}+bY'''_{\max,k}.
\]

The exact coefficient of `Y'` could be `4b`; `8b` is again a safe overestimate. These bounds refer to any complex matrix entries, including all initially occupied modes.

**Numerical subtlety:** the theorem assumes exact collocation and exact evaluation of the clean propagator. The present NumPy/SciPy calculation has ordinary floating-point residuals at the nodes, at about machine precision, and does **not** rigorously enclose roundoff in the phi functions, linear solves, eigenvalue computations or accumulated quadrature. It is *not* an interval-arithmetic certificate.

## 4. Convergence benchmark: exact Hamiltonian used only after prediction

For `L=48, T=30`, initial mass `m=2.5`:

| Step `h` | `(1.20,0.35)` source-only bound | `(0.35,1.20)` source-only bound |
|---:|---:|---:|
| 0.2 | 9.49×10⁻³ | 7.76×10⁻³ |
| 0.1 | 5.25×10⁻⁴ | 4.34×10⁻⁴ |
| 0.05 | 3.09×10⁻⁵ | 2.57×10⁻⁵ |
| 0.025 | 1.87×10⁻⁶ | 1.56×10⁻⁶ |
| 0.0125 | 1.15×10⁻⁷ | 9.64×10⁻⁸ |
| 0.00625 | 7.14×10⁻⁹ | 5.98×10⁻⁹ |

For forward `h=0.025`, the independent full defective Hamiltonian check gives `sup_nodes||F-F_Y||≈2.02×10⁻⁷`; the bound is about `1.87×10⁻⁶`. At `h=0.00625`, those figures fall to `7.91×10⁻¹⁰` and `7.14×10⁻⁹`. The corresponding reversed-defect comparisons are also below their source-only bounds.

See `fourth_order_convergence.json`, `figures/fourth_order_residual_convergence.png` and `high_order_volterra.diagnostic`. This numerical validation is **not** an ingredient of the source-only estimate.

## 5. Endpoint-safe capacity control

Let `P=FF†` be the exact rank-`M` projector. For sufficiently small `E=||F-F_Y||`, polar-normalize `F_Y` to `X=F_Y(F_Y†F_Y)^−1/2` and set `P_Y=XX†`. Standard nearby-isometry arguments give the conservative bound `||P-P_Y||_2≤4E` for our small-error regime. Consequently the eigenvalues of the restricted correlation blocks move by at most `4E`.

Capacity is the scalar spectral functional

\[
 C_A=\sum_{\ell=1}^M f(p_\ell),\qquad f(p)=p(1-p)\log^2\frac{1-p}{p}.
\]

Define the global scalar modulus `ω_f(δ)=sup_{x,y∈[0,1],|x−y|≤δ}|f(x)-f(y)|`. Despite the unbounded derivative at the endpoints, `ω_f(δ)→0` with `δ`. For `0<δ<0.06`, a conservative explicit bound is

\[
 \omega_f(\delta)\le \max\left\{
  \delta[\log^2(1/\delta)+2\log(1/\delta)],\quad
  2\delta\log^2[1/(2\delta)]\right\}.
\]

Then, independently for each exact time sample,

\[
\boxed{|C_A(t_j)-\widetilde C_A(t_j)|\le M\,\omega_f(4E)}.
\]

Here the computed capacity uses the **polar-normalized** `F_Y`; one must not apply this formula to arbitrary unnormalized approximate states.

## 6. A deliberately explicit operational definition of the smoothed peak

To eliminate ambiguity about padding and sampling, in this supplement the smoother is

\[
 \bar C_h(t)=\frac{\sum_{j=0}^N e^{-(t-t_j)^2/(2\sigma^2)}C_A(t_j)}
 {\sum_{j=0}^N e^{-(t-t_j)^2/(2\sigma^2)}},\qquad t_j=jh,\quad \sigma=2.
\]

This normalized Gaussian **sampled smoother** is not identical to continuous convolution, nor to the reflect-padded filter in the earlier research note. The peak certified here is *local* within a stated window; the first *global prominent* peak selection remains an additional condition to verify.

Let `w_j(t)` be normalized Gaussian weights and `a_j=(t_j−t)/σ²`, `u_j=a_j−Σw_k a_k`, `v_2=Σw_j u_j²`. Then

\[
 w'_j=w_j u_j,\qquad w''_j=w_j(u_j^2-v_2).
\]

An error `|C_j−\tilde C_j|≤ε_C` therefore implies

\[
 |\bar C'_h(t)-\tilde{\bar C}'_h(t)|\le\epsilon_C\sum_j|w'_j(t)|,
 \quad
 |\bar C''_h(t)-\tilde{\bar C}''_h(t)|\le\epsilon_C\sum_j|w''_j(t)|.
\]

The program `peak_certificate.py` bounds these weight-derivative sums **uniformly over a continuous interval** using explicit Gaussian numerator and denominator envelopes. It also bounds `|\tilde{\bar C}'''_h|` throughout that interval, allowing a finite sampled check of the candidate curvature to yield a continuous lower bound. Under (i) opposite-sign endpoint derivatives with margins exceeding the derivative-error majorant, and (ii) the candidate negative curvature lower bound exceeding the second-derivative-error majorant, the exact smoothed capacity has a **unique local maximum** in that interval. If `η1` bounds current error, `η2` curvature error and `κ` bounds the magnitude of the candidate negative curvature throughout the window, then

\[
 \boxed{|t_*−\tilde t_*|\le \frac{\eta_1}{\kappa-\eta_2}},
 \quad \text{provided }\kappa>\eta_2\text{ and robust endpoint signs hold.}
\]

The code's `improved_strict_interval_diagnostic` evaluates these exact-arithmetic majorants in double precision; it does **not** use a verified interval package.

## 7. Results: conditional LOCAL first-revival enclosure

Parameter choices: `L=48`, mass `m=2.5`, horizon `T=40`, time step `h=0.00625`, smoother `σ=2`, local half-window `R=0.03` around each computed peak. The two orientations are assessed independently, with no use of their exact capacities in computing error bounds.

| Quantity | `(1.20,0.35)` | `(0.35,1.20)` |
|:--|--:|--:|
| Predicted local peak | 26.4226541973 | 27.3115339862 |
| Source-only amplitude bound through T=40 | 9.54×10⁻⁹ | 7.98×10⁻⁹ |
| Endpoint-safe capacity sample error bound | 0.00049178 | 0.00042065 |
| Uniform current error `η1` | 0.00021011 | 0.00017972 |
| Uniform curvature error `η2` | 0.00026764 | 0.00022892 |
| Uniform negative-curvature lower bound `κ` | 0.02857068 | 0.00961905 |
| Satisfies robust endpoint/curvature conditions | Yes | Yes |
| **Conditional local peak radius** | **0.007424** | **0.019140** |

Independent defective-Hamiltonian calculation of the *same defined smoother* gives `26.42265419797` and `27.31153398921`, respectively. The sub-`10^-8` observed differences serve **only** as post-hoc verification; they are not what proves the analytical bound. The certificate radius is intentionally far larger than the observed differences.

**Exact interpretation:** assuming exact evaluation of the input and majorant quantities and exact arithmetic in solving the 4×4 systems, the inequalities certify a unique *local* peak in the stated window. Because the actual numerical results are ordinary floating-point output, this is currently a **conditional, source-only numerical certificate**, not a formal computer-assisted proof or validated interval-arithmetic enclosure.

For the peak displacement between reversed orientations, the crude combined local radius would be the **sum** of the two individual radii, roughly `0.02656` hopping-time units (in the same conditional sense). Such a bound may be too loose for applications that seek shifts below that scale.

## 8. Airy error budget and honest novelty boundary

Separately, our prior Airy-versus-Bessel first-return test found sampled `||(W_Ai−W_Bessel)Q||_2≈4.97×10^-5` through `t=30` for `L=48`. This quantity is small, but it is **not a rigorous uniform bound** on all intermediate times, and does not include every approximation and polar projection in the reduced Airy generator. We therefore **cannot transfer** the new small source-only peak intervals to the Airy-only low-rank surrogate yet.

The historical one-shell Bessel image-tail theorem does bound a *different* omitted-image contribution, of size `5.16×10^-12` for the clean kernel through `t=30` at `L=48`; the corresponding continuous-time impurity response estimate is `7.06×10^-9`. We must not add these to an uncertified Airy grid sample and call it rigorous.

Most importantly, **Capizzi & Eisler (2023), Eq. (64)** already describe coherent finite-size defect revivals in the conformal-hopping setting, and §3.5 analyzes a momentum-dependent hopping defect at short times. Their published Rényi formulas also imply a capacity prediction by differentiating at `n=1`. The present two-site **onsite-potential**, finite-size, phase-sensitive setup is distinct, but distinct does not mean novel. A proper novelty claim requires a more exhaustive theorem and literature comparison, and a genuinely reduced predictor rather than a new numerical representation of existing free-fermion dynamics.

## 9. Reproduction and next work

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python -m unittest -v test_certificate
OPENBLAS_NUM_THREADS=1 python run_peak_certificate.py
OPENBLAS_NUM_THREADS=1 python plot_convergence.py
```

`run_peak_certificate.py` outputs `peak_certificate_results.json` and `figures/peak_current_local_windows.png`. The accompanying JSON contains all first-order and curvature margins, to prevent suppressing failed conditions. `fourth_order_convergence.json` stores benchmark data; `plot_convergence.py` visualizes both source-only bounds and independently measured errors. The earlier `RESEARCH_NOTE.md` remains the **Phase-I** analysis, and its last section predates the present fourth-order result.

**Remaining tasks before a publishable numerical theorem:** outward-rounded special functions, interval linear algebra and global accumulated-roundoff bounds; *uniform* Airy remainder including the reduced two-sided scattering generator and omitted paths; and a proof that the certified local maximum is the **first relevant** capacity revival under a preregistered prominence criterion. For a claim of new physics, additionally validate across held-out chain sizes, stagger parameters, defect pairs and a properly applicable published theory baseline.

### References

- L. Capizzi and V. Eisler, *Entanglement evolution after a global quench across a conformal defect*, SciPost Phys. 14, 070 (2023), arXiv:2209.03297, §§3.4–3.5, Eq. (64): https://arxiv.org/abs/2209.03297
- NIST DLMF, §10.20, large-order uniform Bessel–Airy asymptotics: https://dlmf.nist.gov/10.20
- NIST DLMF, §10.12, Bessel generating function for the image tail: https://dlmf.nist.gov/10.12
