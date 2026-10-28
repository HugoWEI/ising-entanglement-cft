# Nonperturbative defect reversal via exact parity-channel resummation

**Status: validated spectral Green-function identity (28 October 2026).**
This is a significant mathematical simplification for our existing
tight-binding model, but a known application of symmetry and
finite-rank Dyson identities, **not a new discovery or a closed-form
prediction of entanglement-capacity revival times**.

## 1. Precise question

The earlier weak-asymmetry result

    C_A(t;delta)-C_A(t;-delta)
        =2 delta chi_C(t)+O(delta^3)

accurately predicted small defect differences but failed at
delta=.85. Can we resum **all powers of delta** in the
scattering response without diagonalizing either asymmetric
Hamiltonian?

We can do so *exactly for the single-particle resolvent*.

## 2. Finite-chain Hamiltonian and reflection basis

Work with the number-conserving free-fermion hopping model

    h(delta) = h_sym + delta W,
    h_sym = -sum_j (|j><j+1| + h.c.)
            +w(|a><a|+|b><b|),
    W=(|a><a|-|b><b|)/2,

where a,b are the two central sites of an even open chain.
The physical onsite pair is
(u,v)=(w+delta/2,w-delta/2); the reversed pair has -delta.

Let P reflect sites j -> L+1-j. Since P h_sym P=h_sym,
the normalized defect states

    |+>=(|a>+|b>)/sqrt(2),   |->=(|a>-|b>)/sqrt(2)

have even and odd reflection parity. The asymmetry operator is

    W=(|+><-| + |-><+|)/2.

This means asymmetry only couples **opposite parity** channels.

Define the reference resolvent

    R0(z)=(zI-h_sym)^(-1),

and parity-projected local Green functions

    g_plus(z)=<+|R0(z)|+>,
    g_minus(z)=<-|R0(z)|->.

The mixed overlap <+|R0|-> vanishes *exactly* by reflection symmetry.
The reference resolvent is obtained from a SINGLE symmetric-chain
eigendecomposition; no asymmetric matrix is diagonalized for
the prediction.

## 3. Exact all-orders formula

In the {|+>,|->} subspace, the defect perturbation is

    delta W_D=(delta/2) [[0,1],[1,0]].

The reference local resolvent is diagonal, diag(g_plus,g_minus).
The resummed T-matrix is

    T_D(z,delta)
      = [I - delta W_D diag(g_plus,g_minus)]^(-1) delta W_D
      = 1/D(z,delta)
          [[delta^2*g_minus/4, delta/2],
           [delta/2,delta^2*g_plus/4]],

with the exact even-in-delta denominator

    D(z,delta)=1 - delta^2*g_plus(z)*g_minus(z)/4.

Therefore

    R_delta(z)
      = R0(z)+R0(z) P_D T_D(z,delta) P_D^T R0(z),

where P_D has columns |+>,|->.

Subtract the reversal (delta -> -delta). The diagonal even terms
cancel, giving a particularly compact exact identity:

    BOXED:
    R_delta(z)-R_-delta(z)
      = [delta/D(z,delta)]
         [R0(z)|+><-|R0(z)
          +R0(z)|-><+|R0(z)].

This holds for arbitrary delta, finite L, and complex z outside the
spectra (interpreted by analytic continuation away from poles).

For |delta^2 g_plus*g_minus/4|<1, one can formally expand

    delta/D = delta * sum_{n>=0}
                          (delta^2*g_plus*g_minus/4)^n.

The first term is the original linear Green-function response, and
the next terms resum repeated even/odd parity conversion at the
defect. The geometric series **must not** be truncated near
resonances where it converges slowly or not at all.

The exact denominator accounts for bound-state and other spectral
pole shifts; poles at reference eigenvalues may cancel between
numerators and denominators. Avoid treating every apparent pole
of an intermediate factor as a physical eigenstate.

## 4. Independent numerical verification

For L=24, w=.775, two representative points off the spectrum,

    z1=0.2+0.4i and z2=2.8+0.35i,

calculate the resolvent difference directly by inverting both
defective Hamiltonians (reference ONLY), and independently through
the parity formula using the symmetric Hamiltonian (prediction).

| delta | linear relative error at z1 | linear relative error at z2 | resummed relative error |
|---:|---:|---:|---:|
| 0.02 | 0.00390% | 0.00503% | below 1e-11 |
| 0.30 | 0.876% | 1.131% | below 1e-11 |
| 0.85 | 7.036% | 9.081% | below 1e-11 |
| 1.60 | 24.929% | 32.176% | below 1e-11 |

"Below 1e-11" denotes the relative Frobenius numerical error, not
a mathematically new result. The parity expression is an **identity**,
so an accurate numerical implementation should match direct inversion.

Numerical tests also cover different chain lengths, negative mean
potentials, complex energies, reflection-odd contrast symmetry,
zero asymmetry, and the cubic remainder of the first-order
resolvent contrast.

## 5. What DOES this tell us about entanglement capacity?

An exact spectral formula can in principle be converted to a
time-domain propagator:

    U_delta(t) = (1/(2*pi*i)) contour_int
                        exp(-i z t) R_delta(z) dz,

where the contour encloses the appropriate finite-chain eigenvalues
with the conventional orientation.

Then
    G_A(t;delta)
      = P_A U_delta(t) G(0) U_delta(t)^dag P_A,
    C_A(t;delta) = Tr f(G_A(t;delta)),
    f(p)=p(1-p) log^2[(1-p)/p].

**However, the nonlinear trace functional does not inherit the
simple rational delta/D factor from the resolvent difference.**

Even knowing R_delta - R_-delta exactly does not make
C_A(delta)-C_A(-delta) a simple scalar multiple of chi_C.
The even-in-delta changes of both evolved states and the
entanglement eigenvectors still matter.

Consequently, this experiment does NOT show that linear capacity
response has been repaired at delta=.85. It rigorously repairs the
underlying *spectral-scattering* response, an indispensable ingredient.

## 6. Scientific novelty / limitations

- The parity decomposition, Woodbury/Dyson resolvent identity, and
  geometric resummation are standard tools.
- Exact two-impurity Green-function denominators have already
  appeared in the literature, e.g. the two-impurity note in
  Frontiers in Physics (2019),
  https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2019.00007/full
- Existing nonequilibrium free-fermion and defect-entanglement work
  already contains related scattering mathematics.
- No original physics claim is justified by re-deriving this identity.

The nontrivial remaining problem is to use the parity-dressed
propagator to derive a *controlled, interpretable approximation*
for capacity **peak timing and height** and test it on held-out
defects, system sizes, and initial states.

A next test could isolate a small number of dominant spectral poles
or coherent boundary paths and quantify the error before proposing
a first-revival law. A direct numerical contour inversion that
recovers the exact full propagator is a useful control, but is not
by itself a simpler predictive theory.

## 7. Reproduction

    git pull origin main
    python -m pytest -q
    python scripts/benchmark_parity_resummation.py

Files:
- src/parity_resummed_scattering.py
- tests/test_parity_resummed_scattering.py
- scripts/benchmark_parity_resummation.py
- results/parity_resummation_benchmark.csv
- figures/parity_resummation_error.png
- figures/parity_channel_dressing.png

Earlier response reference:
docs/capacity_asymmetry_linear_response.md
