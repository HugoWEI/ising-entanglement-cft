# Strong-defect capacity from an exact bound pole and finite coherent walks

**Status: reproducible computational formula with analytic truncation bounds,
29 October 2026.** This is a technically stronger and more transparent
alternative to the contour calculation, but its underlying techniques
are **established quantum numerical methods**. It is NOT a new
few-reflection, explicit universal first-revival timing law.

## 1. The physical question

Given a fixed strong two-site potential defect in a finite open
number-conserving free-fermion chain, can we predict the first
finite-size modular-capacity revival *without* diagonalizing the
defective (or symmetric-reference) Hamiltonian, without complex-energy
contour integration, and with a rigorously controlled propagation
approximation?

For this restricted model class, **yes**.

Set hopping amplitude and hbar to one, even L, central onsite
potentials u,v>=0. We require exactly ONE localized eigenstate
E_b>2 above the bulk continuum band [-2,2] and no other bound
states. The initial half-filled Slater orbitals F0 are orthonormal.

For this model h has nearest-neighbor hopping -1, the
continuum eigenvalues lie in [-2,2], and the bound eigenvalue
lies above 2. The finite clean chain eigenfunctions are known
analytically:

    q_m=m*pi/(L+1),  E_m^0=-2cos(q_m),
    phi_m(j)=sqrt[2/(L+1)] sin(j q_m).

## 2. Exact bound state from an analytic two-site scattering matrix

Let h0 be the CLEAN hopping Hamiltonian and V be the
two-site, nonnegative onsite potential.

The clean Green function is

    G0(z)=(zI-h0)^-1,
    g_D(z)=G0(z)[central two sites,central two sites].

We compute g_D(z) from the known sine eigenfunctions,
with NO matrix diagonalization. The two-site
Birman-Schwinger matrix is

    K(E)=sqrt(V_D) g_D(E) sqrt(V_D),  E>=2.

Since g_D(E) is positive definite and monotonically decreases
for E>max spectrum(h0), eigenvalues of K(E) decrease.
An eigenenergy E>2 of h=h0+V is equivalent to
an eigenvalue of K(E) crossing 1.

We establish the one-bound-state eligibility condition
by evaluating the two eigenvalues of K(2). Exactly one must
exceed 1 (the other must remain below 1). This requires only
an analytically calculated 2x2 Green function.

Then the ONE positive bound energy E_b is obtained
by scalar bisection of

    lambda_max(K(E_b))=1.

The full bound eigenvector b is reconstructed using
the clean Green function and a TWO-DIMENSIONAL null
vector, as implemented in src/defect_tmatrix.py.
No symmetric or asymmetric defective Hamiltonian
eigenvector is required.

The code explicitly rejects negative onsite potentials
and multiple out-of-band bound modes, which would need
the corresponding generalized bound-projector sum.

## 3. EXACT coherent-walk formula

Define P_b=|b><b| and P_c=I-P_b. Since P_b commutes
with h, the propagator separates exactly:

    U(t) = exp(-iE_b t)P_b + exp(-iht)P_c.

The spectral theorem gives spectrum(h P_c) within [-2,2],
so apply the Jacobi-Anger/Chebyshev identity to its
continuum subspace:

    exp(-i2t x)= J_0(2t)
                 +2 sum_{m=1}^infty (-i)^m J_m(2t) T_m(x),
    -1<=x<=1.

The resulting EXACT infinite series for occupied orbitals is

    F(t)= exp(-iE_b t)P_bF0
          +J_0(2t)P_cF0
          +2 sum_{m=1}^infty (-i)^m J_m(2t)
                                 T_m(h/2)P_cF0.

The important point is that the bound-mode contribution is
treated exactly, and every continuum term is a POLYNOMIAL
in the hopping Hamiltonian.

The m-th Chebyshev polynomial T_m(h/2) contains coherent
walks built from up to m nearest-neighbor hops and onsite
interactions. Finite boundaries and defects enter exactly
through the Hamiltonian action, so every retained path
carries its correct complex phase and interference.
This is a **walk-length expansion**, not a list of a few
continuum scattering reflection trajectories.

### Practical recurrence

Compute W_0=P_c F0, W_1=P_c(h/2)W_0, and

    W_{m+1}=2 P_c(h/2)P_c W_m-W_{m-1}.

The matrix action uses just nearest-neighbor hopping,
two onsite multiplications, and a rank-one projector,
taking O(L*N_occ) work per recursion step.

**Projection at EVERY step is mandatory in floating-point.**
If one only projects the initial vector and then applies
the unprojected polynomial recurrence, an O(roundoff)
component along the bound eigenvector is amplified
approximately like exp[m acosh(E_b/2)] because E_b/2>1.
The projected recurrence is bounded on the continuum
and avoids this explosive leakage.

The polynomial moments W_m are time-independent and can
be reused for every requested t. The time coefficients
are calculated from a sampled Fourier integral using an FFT.

## 4. Rigorous finite-walk truncation error

Truncate after m=M and denote the result F_M(t).
Because ||T_m(h/2)P_c||_2<=1,

    ||F_M(t)-F(t)||_2
       <= 2 sum_{m=M+1}^infty |J_m(2t)|.

For m>2t, the Bessel contour shift gives

    |J_m(2t)| <= exp[
       -m acosh(m/(2t)) + sqrt(m^2-(2t)^2)
    ].

The logarithm of this bound has derivative
-acosh(m/(2t)), so a geometric tail majorant yields
for n=M+1>2t:

    epsilon_walk(t,M)
       = 2 exp[-n acosh(n/(2t))+sqrt(n^2-4t^2)]
             / [1-exp(-acosh(n/(2t)))].

Thus for every t in [0,T], selecting M using
epsilon_walk(T,M)<=tolerance gives a deterministic
a-priori continuum propagation bound. When n<=2T
this simple bound is unavailable and the code
reports +infinity, not a fabricated accuracy certificate.

The numerical FFT Bessel coefficient approximation has
an additional alias upper bound. For FFT length N,

    epsilon_alias <= 2 epsilon_walk(T,N-M-1),

provided N-M>2T. We choose N sufficiently large.
This Fourier-alias bound is distinct from floating-point
roundoff and from numerical errors in the scalar bound pole.

## 5. Physicality and a non-Lipschitz capacity bound

The finite Chebyshev sum is not exactly unitary. Let F_M
be its occupied orbital matrix. We explicitly apply the
nearest-column-isometry (polar factor)

    Fhat_M=F_M(F_M†F_M)^(-1/2).

This is a declared numerical repair, NOT an extra
physical interaction. It guarantees that Fhat_M is a
pure Slater state and that its half-chain occupation
eigenvalues lie in [0,1] up to roundoff.

In exact arithmetic, if epsilon bounds
||F_M-F||_2<1 with F†F=I, the singular values of F_M
are within epsilon of 1. Hence

    ||Fhat_M-F_M||_2<=epsilon,
    ||Fhat_M-F||_2<=2epsilon,
    ||Ghat_A-G_A||_2<=4epsilon.

Set s=4epsilon. For

    f(p)=p(1-p)ln²[(1-p)/p],  0<=p<=1,

the derivative diverges logarithmically near the
endpoints, so there is NO global uniform Lipschitz
constant. Nevertheless

    |f'(p)|<= ln²(1/q)+2ln(1/q),
    q=min(p,1-p).

Integrating this endpoint bound and allowing for both
ends of the spectrum gives a conservative uniform
modulus, for 0<s<1/2:

    omega_f(s) <= 2 s [ln(1/s)+2]^2.

Weyl's inequality for covariance eigenvalues then yields

    BOXED:
    |C_A[Fhat_M(t)]-C_A[F(t)]|
       <= 2 d_A s [ln(1/s)+2]^2,
       s=4(epsilon_walk+epsilon_alias),
       d_A=L/2.

This is a formal DETERMINISTIC error estimate in
exact arithmetic and with an exact bound projector.
The benchmark checks the observed error independently.
It is NOT a rigorous bound on finite-precision effects
or pole-root rounding, which are separately monitored.

The estimates can be conservative, especially for
moderate errors. At small truncation tolerances they
become useful enough to test individual capacity peaks.

## 6. First broad peak GRID-INDEX certificate

The project's operational first broad capacity peak
uses a Gaussian-smoothed sequence, width 2 time units,
first local max in [0.4L,1.65L] that is prominent by
at least 0.08 on both sides over +/-0.15L.

Since the Gaussian smoothing matrix has nonnegative
weights summing to one (including reflected padding),
a uniform samplewise capacity bound eta remains eta
AFTER smoothing.

A sufficient condition for a predicted index i to be
the SAME first qualifying peak index of every
capacity sequence within eta is:

  (i) S_i beats both neighbors by >2eta,
  (ii) its prominence beats 0.08 by >2eta,
  (iii) every earlier sample j in the search window
        is either definitely not a local max
        (some neighbor exceeds it by >2eta)
        or definitely not prominent
        (estimated prominence +2eta <0.08).

This is implemented in src/first_peak_certificate.py.
A successful certificate is stronger than visually
matching the exact simulation: the peak index is
protected against ANY curve within the formal
uniform truncation error.

**Limits:** This is a discrete sampled peak certificate,
NOT a continuous-time asymptotic revival law or a
proof that the first peak is the dominant global maximum.
It is conditional on the stated exact-arithmetic and
exact-pole assumptions. The peak certificate explicitly
returns "uncertified" when sufficient inequalities fail,
instead of asserting a result.

## 7. Empirical design and falsification controls

To avoid fitting a convenient polynomial order to the
exact simulation, the benchmark fixes:
- initial staggered ground-state mass 2.5,
- times at resolution 0.4 through 1.8L,
- two orientations of (1.2,.35), (1.1,.1) and (.7,1.4),
- a priori WALK tolerances 1e-3, 1e-7 and 1e-12,
- the same peak-identification method in every case.

Prediction data are generated without diagonalizing a
single-particle defective Hamiltonian or evaluating
a complex-energy contour. Independent dense diagonalization
is used *afterwards* only for scoring.

The benchmark reports:
  full capacity RMSE, maximum absolute error,
  peak times and heights, the Bessel error bounds,
  raw orthonormality deviations, explicit polar repairs,
  and conditional local/first-peak grid certifications.

The low-accuracy tolerances provide meaningful negative
controls: not every finite polynomial order is claimed
sufficient for the first revival. If a peak fails
or its certificate is unavailable, that is reported.

## 8. Literature and originality boundaries

The Chebyshev/Bessel time-propagation method is well established:
- H. Tal-Ezer and R. Kosloff (1984), *An Accurate and Efficient
  Scheme for Propagating the Time Dependent Schrödinger
  Equation*, J. Chem. Phys. 81, 3967–3971.
  DOI 10.1063/1.448136.
- F. Alexander Wolf, J. A. Justiniano,
  I. P. McCulloch and U. Schollwöck (2015),
  *Spectral functions and time evolution from the
  Chebyshev recursion*, Phys. Rev. B 91, 115144.
  DOI 10.1103/PhysRevB.91.115144.

The Birman–Schwinger principle, Dyson/T-matrix
bound poles, finite-rank spectral projectors and
Gaussian capacity formula are also known.

Our specific *pole-exact, error-controlled capacity peak
benchmark* may be useful for demonstrating a rigorous
regime of validity, but its novelty is **not established**.
A journal-level claim needs a new physical scaling
prediction, error theorem beyond standard techniques,
or substantially novel application together with a
deep originality audit.

The original aspiration of a short, closed-form
phase-aware reflection-path rule that predicts all
strong-defect first revivals remains unfulfilled.
This finite-walk expansion is a mathematically
valid controlled alternative, not a substitute claim.

## 9. Reproduction

    git pull origin main
    python -m pytest -q
    python scripts/benchmark_bound_chebyshev.py

Source:
- src/bound_projected_chebyshev.py
- src/defect_tmatrix.py
- src/first_peak_certificate.py

Tests:
- tests/test_bound_projected_chebyshev.py
- tests/test_first_peak_certificate.py

Results:
- results/bound_chebyshev_revivals.csv
- figures/bound_chebyshev_convergence.png
- figures/bound_chebyshev_capacity.png

GitHub Actions uploads the data/figures after the
normal paper and previous research benchmarks.
