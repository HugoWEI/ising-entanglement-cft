# Coherent boundary-return paths and a two-site Dyson/Volterra equation

**Numerical and theoretical milestone, October 2026.**

**Research honesty:** This is standard free-fermion propagator theory
implemented as a controlled scattering-path numerical approximation.
It is NOT yet an original analytical formula for capacity-revival
timings or amplitudes. The numerical reference is independent full
defective-chain Gaussian evolution.

## A. The model and clean propagator

We use the *number-conserving tight-binding hopping chain*, not the
transverse-field Ising model:

    H = sum_{ij} h_ij c_i^dag c_j,
    h=h0+V,
    h0=-sum_{j=1}^{L-1} (|j><j+1|+|j+1><j|),
    V=u|a><a|+v|b><b|,

where defect sites a=L/2 and b=L/2+1 are 1-indexed.

In the infinite chain, the clean amplitude between sites distance m
apart at evolution time tau (hbar=1) is

    U_inf(m,tau)=integral_{-pi}^{pi} (dk/2pi)
                 exp[i*k*m + 2i*tau*cos(k)]
                = i^m J_m(2tau).

This is an exact standard lattice Fourier/Bessel representation.

With hard-wall (Dirichlet) conditions at sites 0 and L+1, the *clean
finite-chain* propagator follows from the odd image extension:

    U0_open(i,j;tau)
      = sum_{n in Z} [
          U_inf(i-j+2n(L+1),tau)
          - U_inf(i+j+2n(L+1),tau)
        ],

for 1<=i,j<=L. The second term enforces zero amplitude at the
boundaries. Each image carries its own complex phase, so it represents
coherent boundary returns, not classical reflected probabilities.

The infinite-chain integral is evaluated numerically by a high-resolution
Fourier sum (NumPy inverse FFT) with an explicit aliasing guard.

The truncation |n|<=R defines a controlled hierarchy of boundary-image
approximations *when omitted image amplitudes are small*.

For a 48-site chain, the relative Frobenius errors in the CLEAN
propagator at time tau=67.2 were:

| maximum image order R | relative clean-propagator error |
|---:|---:|
| 0 | about 0.875 |
| 1 | about 0.339 |
| 2 | about 2.4e-14 |

Thus the second image order is crucial by that late time.

## B. Exact two-site Volterra reduction of defect dynamics

For occupied single-particle orbitals F(0) (an L x (L/2) matrix),
the exact Duhamel/Dyson identity is

    F(t)=U0(t)F(0)
         -i integral_0^t U0(t-s)[:,D] V_D Y(s) ds,

where D={a,b}, V_D=diag(u,v), and Y(s)=F(s)[D,:] is only
2 x (L/2).

Restricting both sides to the defect sites gives the **closed**
two-site Volterra integral equation

    Y(t)=Y0(t)
         -i integral_0^t K(t-s) V_D Y(s) ds,

where Y0(t)=U0(t)[D,:]F(0), and K(t)=U0(t)[D,D].

This equation automatically resums repeated defect encounters and
retains phases of all paths. Unlike a generic LxL diagonalization,
the time-dependent unknown localized at the scatterer is only 2 x
(L/2), although reconstructing the full evolved orbital state still
requires additional work.

### Numerical quadrature

We discretize time at t_n=n delta. The trapezoidal-rule system is

    [I2 + (i delta/2) K(0)V_D]Y_n
       = Y0_n
         - i delta [
             (1/2)K(t_n)V_D Y_0
             + sum_{m=1}^{n-1} K(t_n-t_m)V_D Y_m
           ].

The only implicit inversion is 2x2. We reconstruct F(t_n) using
the analogous discretization of the full Duhamel identity, and
compute the reduced half-chain correlation matrix from its
occupied orbitals.

## C. An essential physicality limitation

A finite image truncation and the trapezoidal time grid are NOT
exactly unitary. They can produce approximate occupied orbitals
F with F^dag F != I and reduced occupation eigenvalues outside
the allowed physical range 0<=p<=1. This must not be hidden.

The code reports the raw isometry error

    epsilon_iso = ||F^dag F - I||_F.

For an optional, explicitly declared *numerical correction*, apply
the closest column-isometry/polar factor

    F_polar = F (F^dag F)^(-1/2),

provided F has full column rank. This projection enforces a valid
pure Slater determinant but changes the approximate numerical state.
It is not a physical interaction and is NOT the solution of the
original Volterra equation.

A no-correction mode sets entropy/capacity outputs to NaN if
occupation eigenvalues violate [0,1] by more than 1e-7, rather
than clipping them silently.

## D. Pilot results

At initial staggered field m=2.5, after quenching into the final
two-site onsite defect (u,v)=(1.2,0.35), with delta=0.2,
we compare the polar-corrected Volterra approximation with direct
full defective-chain Gaussian dynamics.

The capacity RMSE over 0.4 L <= tau <= 1.6 L is:

| L | R=0 | R=1 | R=2 |
|---:|---:|---:|---:|
| 24 | ~1.676 | ~0.0937 | ~0.0186 |
| 48 | ~3.784 | ~0.1212 | ~0.0214 |

The **raw isometry deviation** at L=48 before correction is
approximately 3.89, 3.01, and 0.0139 for R=0,1,2 respectively.

Before correction, the R=1 calculation produces physicality violations
as large as a subsystem occupation eigenvalue around 1.44. These values
are mathematically inadmissible; reporting the corresponding
"capacity" by clipping them would be misleading.

For R=2 the raw occupation overshoot is much smaller (~0.00005 at
delta=.2) and tends to shrink as the time step is refined.
The remaining R=2 capacity error comes largely from time quadrature.

These are numerical pilot results, not proven asymptotic error bounds.

## E. Scientific interpretation

1. Boundary-reflected quantum amplitudes are indispensable in the
   time range where capacity revivals occur.
2. A naive truncation may fail both numerically and physically;
   accuracy and valid fermionic occupations must be checked separately.
3. The 2-site Volterra equation is a transparent resummation of
   impurity scattering and retains the complex propagation phases.
4. The unitary clean/defect Strang method in
   docs/unitary_time_domain_milestone.md is a *more stable* numerical
   solver at comparable time step for these examples. The Volterra
   method remains useful for exposing which paths and kernels matter.
5. The exact bound-state poles are naturally contained in the
   resummed scattering kernel, although this implementation has not
   split out their residues analytically.

## F. What still needs to be demonstrated for genuine originality

We still do NOT have a compact, predictive formula for the location
and amplitude of the *first capacity revival*.

Next derive a controlled phase- and pole-resolved asymptotic expression
for selected boundary-return paths, where scattering phases and
Wigner delays appear explicitly. Its prediction must be computed
BEFORE looking at the exact revival peak, and validated on held-out
defect configurations, lengths, and initial preparations.

Full correlation-matrix evolution through a low-dimensional integral
equation is scientifically rigorous but does not by itself qualify
as a novel many-body theory.

## Reproduction

    git pull origin main
    python -m pytest -q
    python scripts/benchmark_boundary_image_volterra.py

Files:
- src/boundary_images.py: clean infinite-chain Fourier kernel and
  finite hard-wall reflection image formula.
- src/boundary_image_volterra.py: implicit two-site convolution and
  full-orbital reconstruction; explicit physicality correction.
- tests/test_boundary_image_volterra.py: independent clean-sine and
  exact-defect Gaussian checks, plus invalid-state detection.
- scripts/benchmark_boundary_image_volterra.py: CSV+figures comparing
  image orders and the exact capacity trajectory.

Background: Duhamel/Dyson equation, method of images, standard
Gaussian free-fermion entanglement. Capizzi & Eisler (2023),
arXiv:2209.03297, discuss coherent reflection and complex
finite-size revival behavior.
