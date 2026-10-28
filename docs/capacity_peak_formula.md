# A quantitative formula for weak-asymmetry capacity-revival peak shifts

**28 October 2026, preliminary analytical/numerical research result.**

**Scientific status:** We now have a *validated predictive formula* for the
timing and height of a nondegenerate, isolated, smoothed capacity maximum
under weak defect asymmetry. It is not a novel theorem: it follows
from standard Kubo/Duhamel linear response and implicit differentiation.
It is not reliable for the original strongly asymmetric pair, and no
general strong-defect revival law has yet been established.

## 1. Physical system

This is the **number-conserving tight-binding free-fermion model**, not
the transverse-field Ising Hamiltonian from the first project.

Set hopping and hbar to one. Even open chain L. Post-quench onsite
defect potentials at the two central sites are

    u=w+delta/2, v=w-delta/2.

Reversing the defect changes delta -> -delta. The initial state,
kept identical for both comparisons, is the half-filled Gaussian
ground state of a clean chain with onsite stagger m=2.5.

Capacity is the nonlinear Gaussian function

    C_A(t;delta)=Tr f(G_A(t;delta)),
    f(p)=p(1-p) ln^2[(1-p)/p],
    G_A=P_A U_delta(t) G(0) U_delta(t)† P_A.

## 2. Linear-response coefficient calculated without asymmetric runs

For h(delta)=h_sym+delta W,
W=(|a><a|-|b><b|)/2, the propagator derivative is

    dU_delta/d(delta)|0
      = -i int_0^t U_sym(t-s) W U_sym(s) ds.

This is explicitly calculated from one symmetric eigensystem.
The capacity susceptibility is

    chi_C(t)=d C_A(t;delta)/d(delta)|0
       =Tr[f'(G_A,sym) dG_A/d(delta)|0],

with

    f'(p)=(1-2p) ln^2[(1-p)/p]
            -2 ln[(1-p)/p].

For smooth positive entanglement eigenvalues away from 0 and 1,

    C_A(t;delta) = C_0(t)+delta chi_C(t)+O(delta^2).

The derivative of the capacity itself requires a limit at
occupation eigenvalues near 0 and 1. Our numerical implementation
reports and handles near-endpoint sensitivity explicitly.

## 3. THE PREDICTIVE PEAK FORMULA

Let t0 be an isolated nondegenerate local maximum of C0, so

    C0'(t0)=0, C0''(t0)<0.

By the implicit-function theorem there is a nearby smooth peak branch
t_peak(delta) satisfying
d C_A(t_peak(delta);delta)/dt=0.
Differentiate at delta=0:

    0=C0''(t0) t_peak'(0)+chi_C'(t0),

therefore

    BOXED: t_peak(delta)
       =t0-delta*chi_C'(t0)/C0''(t0)+O(delta^2).

For two reversed defects, the even powers of delta cancel:

    BOXED: Delta t_peak = t_peak(+delta)-t_peak(-delta)
       =-2delta*chi_C'(t0)/C0''(t0)+O(delta^3).

Expanding the capacity at its moving maximum:

    BOXED: Delta C_peak = C_peak(+delta)-C_peak(-delta)
       =2delta*chi_C(t0)+O(delta^3).

The cubic-remainder claims require enough regularity and a stable
nondegenerate branch. The individual peak times/heights only have
O(delta^2) accuracy from first order.

## 4. Operational first-maximum identification and smoothing

Our finite open chain can have multiple peaks. To obtain a
reproducible *first broad post-ballistic maximum* we define:

- Uniform times dt=0.2, t in [0,1.7L].
- Gaussian smoothing of C and chi with width sigma=2.
- First local smoothed maximum in [0.4L,1.65L] with prominence >=0.08
  relative to nearby +/-0.15L windows.
- Require negative curvature magnitude >=1e-4. Use three-sample
  parabolic interpolation for sub-grid time and height.
- After predicting the peak from symmetric C0 and chi only, track
  the nearest local maximum within +/-3 time units on the two
  *independent* exact asymmetric curves.
- If no such local maximum exists, mark the comparison undefined.
  Do not invent a peak at a search-window endpoint.

Because Gaussian smoothing is a *fixed linear operator*, smoothing
C0 and chi first preserves the formal linear-response expansion and
peak formulas for the smoothed observable. It does NOT claim the
same numerical peak location for unsmoothed capacity.

A change of peak branch at large delta violates the perturbative
hypotheses; such cases must be flagged, not presented as successes.

## 5. Independent numerical validation

For L=48, reference mean potential w=.775, 0.2 sample spacing,
smoothed width sigma=2:

| delta | predicted peak-time contrast | measured contrast | predicted peak-height contrast | measured contrast |
|---:|---:|---:|---:|---:|
| 0.02 | +0.01620 | +0.01612 | -0.00373 | -0.00372 |
| 0.10 | +0.08101 | +0.06883 | -0.01865 | -0.01734 |
| 0.30 | +0.24304 | +0.03239 | -0.05595 | -0.03521 |
| 0.85 | +0.68862 | -0.88616 | -0.15852 | -0.09571 |

For L=64 and w=.775, the time contrasts are respectively

| delta | predicted | measured |
|---:|---:|---:|
| .02 | -.00901 | -.00899 |
| .10 | -.04504 | -.04217 |
| .30 | -.13511 | -.10586 |

At delta=.85 the local tracked maximum is not well defined
within the preselected branch window; the benchmark outputs NaN.

For all six (L,w) combinations with L=24,48,64 and
w=.25,.775, the time-contrast error at delta=.02 was <.001.
A standard result, but a real predictive test requiring only the
symmetric eigensystem for the theoretical side.

**Important negative result:** The first-order peak formula fails for
some larger deltas, sometimes reversing the predicted shift sign.
Solving for maxima of C0(t)+delta chi(t) nonlinearly does not
systematically repair the failure. Higher-order changes of the
spectrum and peak branches matter.

## 6. An exact (but implicit) formula valid at arbitrary asymmetry

In the parity-channel basis, let |+> and |-> be the central
symmetric/antisymmetric states, and

    R0(z)=(zI-h_sym)^-1,
    g_plus=<+|R0|+>, g_minus=<-|R0|->,
    X(z)=R0(z)[|+>,|->].

The exact defect resolvent follows from the two-channel resummation:

    R_delta(z)=R0(z)+X(z) T_delta(z) X(z)^T,

    T_delta(z)= 1/D_delta(z)
      [[delta^2*g_minus/4,delta/2],
       [delta/2,delta^2*g_plus/4]],

    D_delta(z)=1-delta^2*g_plus*g_minus/4.

The exact time-domain propagator is *formally* the contour integral

    U_delta(t) = (1/(2*pi*i)) * contour_C
                 exp(-i z t) R_delta(z) dz,

where C encloses the full discrete spectrum anticlockwise.
This automatically contains scattering phases, boundaries, and
bound-state poles. The Gaussian Rényi determinant is

    Z_alpha(t)=Tr[rho_A(t)^alpha]
      =det[G_A(t)^alpha+(I-G_A(t))^alpha],

    capacity C_A(t)=d^2/dalpha^2 log Z_alpha(t)|_{alpha=1}.

This is an EXACT IMPLICIT spectral formula for every finite L and
arbitrary delta, but it is based entirely on established Gaussian
fermion and resolvent theory. Evaluating the contour and the
matrix determinant remains computational work. It is therefore
**not** the sought simple, efficient or original strong-defect
first-revival law.

The meaningful open problem is whether asymptotic evaluation or a
controlled handful of coherent scattering paths/defect poles reduces
this exact formula to an accurate explicit *peak* prediction in a
specified strong-defect regime.

## 7. Reproduction and independent controls

    git pull origin main
    python -m pytest -q
    python scripts/benchmark_capacity_peak_response.py

Files:
- src/capacity_asymmetry_response.py: symmetric Kubo susceptibility
- src/capacity_peak_response.py: smoothed local-maximum formula
- tests/test_capacity_peak_response.py: six independent weak-delta
  timing/height comparisons; strong-delta negative control; no-peak
  and symmetry checks
- scripts/benchmark_capacity_peak_response.py: 24 case benchmarks
- results/capacity_peak_response_benchmark.csv
- figures/capacity_peak_shift_prediction.png
- figures/capacity_peak_height_prediction.png

## Prior art / novelty boundaries

- *Aspects of Capacity of Entanglement*, Phys. Rev. D 99, 066012
  (2019), DOI:10.1103/PhysRevD.99.066012, develops capacity from
  the Rényi generating function and examines perturbations.
- Capizzi and Eisler, *Entanglement evolution after a global
  quench across a conformal defect*, SciPost Phys. 14 070 (2023),
  arXiv:2209.03297, studies quasiparticle revivals at defects.
- Jin/Korepin and subsequent free-fermion works establish the
  Gaussian correlation/Rényi determinant method.
- Implicit-function derivatives of nondegenerate maxima are
  ordinary calculus; the expressions above are not claimed as
  independently original theory.

**Summary:** An actual, quantitatively validated *weak-defect peak
prediction* now exists. The all-delta resolvent/capacity formula is
formally exact but does not yet solve the sought physical
strong-defect revival-prediction problem in a compact form.
