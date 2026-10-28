# From coherent defect scattering to exact bound-state poles

**Research milestone, October 2026.** This is a rigorous foundation for a
future approximate theory of modular-capacity revivals, **not** a successful
first-revival prediction or a claim of newly discovered physics.

## 1. Which model?

Our phase-sensitive revival experiments use an **open number-conserving
tight-binding chain**, distinct from the transverse-field Ising model that
the repository originally studied:

    H = sum_{ij} h_ij c_i^dagger c_j,

with h0 nearest-neighbor hopping -1 and a two-site onsite impurity

    V = u |a><a| + v |b><b|,

where a=L/2-1 and b=L/2 in zero-index notation. The finite clean
single-particle band has eigenvalues E_m=-2*cos(m*pi/(L+1)), all strictly
inside the infinite-chain band (-2,2).

There is a mathematically exact Gaussian time-evolution reference:
C(t)=U(t) C(0) U(t)^dagger with U(t)=exp(-i h t).
For the half-chain C_A=P_A C(t)P_A, its eigenvalues p_j give

    S_A = sum_j [-p_j log p_j-(1-p_j) log(1-p_j)]
    C_A = sum_j p_j(1-p_j) [log((1-p_j)/p_j)]^2.

The challenge is to explain/predict the revival structure without using
a dense full-chain diagonalization for each proposed defect.

## 2. Exact finite-rank Dyson identity

Set G0(z)=(zI-h0)^-1, G(z)=(zI-h)^-1.
The resolvent identity G=G0+G0 V G can be algebraically resummed as

    G(z) = G0(z) + G0(z)[:,D] T_D(z) G0(z)[D,:],
    T_D(z)=[I - V_D g_D(z)]^-1 V_D,
    g_D(z)=G0(z)[D,D], D={a,b}, V_D=diag(u,v).

**The only nontrivial inverse in the defect correction is 2x2.**

The clean open-chain resolvent is analytic from its sine modes:

    q_m=m*pi/(L+1)
    E_m=-2*cos(q_m)
    phi_m(j)=sqrt(2/(L+1)) sin(j*q_m), j=1,...,L
    G0_ij(z)=sum_m phi_m(i)*phi_m(j)/(z-E_m).

This identity is standard low-rank scattering theory. It is exact for
z outside the spectrum, and includes coherent propagation, boundary
reflection, and repeated impurity encounters.

## 3. Coherent repeated scattering

Formally, if spectral_radius(V_D*g_D(z)) < 1,

    T_D(z) = V_D + V_D g_D V_D + V_D g_D V_D g_D V_D + ...

Each term represents an additional encounter with the defect. Since g_D
is a complex Green function, the expansion retains quantum phase
interference, not merely transmission probabilities.

**Important limitation:** the series is not uniformly convergent near
a bound-state pole. The fully resummed 2x2 inverse is therefore the
proper starting point before making approximations.

An independent numerical pilot at L=48, onsite (1.2,.35):

| Complex energy z | radius rho(V g_D) | relative T error after 16 terms |
|---|---:|---:|
| 2.8 + 0.5 i | ~0.594 | ~0.00023 |
| 2.398748 + 0.05 i | ~0.996 | ~0.932 |

The truncated expansion works well away from the pole, while the same
number of scattering encounters fails badly near the bound-state pole.
This supports *resumming* important repeated-scattering processes and
treating the bound-state contribution separately.

## 4. Defect-bound poles from a 2x2 secular equation

Defect eigenvalues not in the clean spectrum satisfy

    D(z)=det[I - g_D(z) V_D]=0.

The OUT-OF-BAND zeros |E_b|>2 are localized defect modes, which are
not described by real propagating momenta k in (0,pi).

To obtain the bound eigenvector, choose a null vector q_b of

    [I - g_D(E_b) V_D] q_b=0,

then reconstruct the full L-component state:

    |psi_b> = normalize( G0(E_b)[:,D] V_D q_b ).

This provides bound-state energies and wavefunctions using only clean
open-chain eigenmodes and a 2x2 secular determinant. It does NOT
diagonalize the defective LxL Hamiltonian.

### Independent verification

For L=48, direct diagonalization gives the identical positive
out-of-band eigenvalues:

| (u,v) | E_b via 2x2 secular root | E_b via full diagonalization |
|---|---:|---:|
| (1.2,0.35) | 2.3987479158 | 2.3987479158 |
| (0.7,1.4)  | 2.5835312561 | 2.5835312561 |
| (0.5,0.2)  | 2.0958400030 | 2.0958400030 |

The bound wavefunctions reconstructed from the two-site null space
also satisfy ||h psi_b - E_b psi_b|| at numerical precision.

The code checks several system lengths, positive and negative
impurities, reflection-related pairs, and a clean no-bound-state case.
The root finder uses finite brackets and is not a certified
all-parameters root finder for poles arbitrarily close to a band edge.

## 5. Why this is relevant for capacity revivals

For a finite Hermitian matrix, the resolvent has exact spectral form

    G(z) = sum_m |psi_m><psi_m|/(z-E_m),

and the propagator can be recovered by the residue formula

    U(t) = (1/(2*pi*i)) contour integral
           exp(-i*z*t) G(z) dz
         = sum_m exp(-i*E_m*t) |psi_m><psi_m|.

Separating band and bound poles:

    U(t) = U_band(t) + U_bound(t),
    U_bound(t)=sum_bound exp(-i*E_b*t)|psi_b><psi_b|.

Since C_A is nonlinear in the eigenvalues of the *restricted
correlation matrix*, we must first combine amplitudes coherently,
then form C_A(t), then compute capacity. We cannot separately compute
"band capacity" and "bound capacity" and add them.

The 2x2 resolvent now makes it possible to express both contributions
from scattering data and pole information. That is a route to a
controlled approximation—not yet an actual prediction of a peak.

## 6. Next predictive step

1. Construct a *time-domain* propagator by evaluating the resummed
   defect resolvent and separating residue contributions from real-band
   propagation. Avoid simply recovering every exact eigenpair and
   calling it a theory.
2. Develop a controlled approximation to the *band* propagator,
   retaining complex scattering amplitudes and finite-end returns.
3. Add bound poles analytically from the secular roots, preserving
   coherence between continuum and bound contributions.
4. Enforce valid Gaussian-state occupations (0 <= p_j <= 1),
   or clearly quantify and correct any unitarity violations.
5. Predict first broad capacity-peak time AND height, out of sample,
   against the same existing exact-lattice benchmark.
6. Compare runtime/error/interpretability with both
   transmission-only theory and full dense diagonalization.

A two-by-two resolvent identity alone is standard mathematical physics.
A genuine project contribution would require a NEW, validated,
useful reduced predictive approximation or a well-defined regime
boundary where such an approximation succeeds/fails.

## 7. Reproduce

    git pull origin main
    python -m pytest -q
    python scripts/run_tmatrix_experiment.py

Outputs:
- results/tmatrix_bound_poles.csv
- results/tmatrix_scattering_convergence.csv
- figures/tmatrix_scattering_poles.png

Source: src/defect_tmatrix.py
Tests: tests/test_defect_tmatrix.py

### Background literature

- Capizzi & Eisler (2023), arXiv:2209.03297, coherent-reflection
  quasiparticles and momentum-dependent defects.
- Gouraud, Le Doussal & Schehr (2023), arXiv:2211.15447, defect-bound
  states and nonequilibrium correlations.
- The finite-rank Dyson/T-matrix and spectral-residue identities are
  standard linear algebra and scattering theory.
