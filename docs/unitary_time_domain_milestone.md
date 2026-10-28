# From a two-site T-matrix to unitary time-domain calculations

**October 2026 — validated numerical-method milestone, not a new theory
of entanglement-capacity revival peaks.**

## Scientific goal and its limitations

We aim to predict finite-time modular-capacity revivals from reduced
defect/scattering information, ideally without diagonalizing the full
defective lattice Hamiltonian. In the previous milestone we derived the
exact two-site T-matrix and its bound-state poles. The present experiment
tests how to evolve the relevant quantum amplitudes in the time domain
while preserving coherent interference and valid fermionic occupations.

**Crucial distinction:** The algorithms below approximately evolve the
*original Hamiltonian*. Successful numerical convergence is NOT itself
a new reduced physical explanation or a predictive quasiparticle theory.
They provide an independent numerical benchmark for one.

## 1. Hamiltonian in clean normal modes

For an open chain of length L with nearest-neighbor hopping -1, the
clean one-particle Hamiltonian h0 is diagonalized by

    q_m = m*pi/(L+1),
    E_m = -2 cos(q_m),
    Q_jm = sqrt(2/(L+1)) sin(j*q_m).

A two-site onsite defect acts at a=L/2-1 and b=L/2 (zero-indexed):

    V = P_D^T V_D P_D,   V_D=diag(u,v).

In the sine-mode basis define B=Q^T P_D^T (shape Lx2, B^T B=I2).
Then

    h_clean-basis = E + B V_D B^T,

where E=diag(E_m). The entire nontrivial defect contribution is rank two.

## 2. Exactly unitary Cayley update using a 2x2 inverse

For delta t>0, let alpha=i delta t/2 and Z(t)=Q^T F(t),
where columns of F are initially occupied single-particle orbitals.

The implicit-midpoint/Cayley step is

    Z_{n+1} = [I+alpha(E+B V_D B^T)]^-1
              [I-alpha(E+B V_D B^T)] Z_n.

Let D=I+alpha E (diagonal), K=alpha V_D.
Using Woodbury,

    [D+B K B^T]^-1 Y
    = D^-1 Y
       - D^-1 B [I2 + K B^T D^-1 B]^-1 K B^T D^-1 Y.

Only the diagonal D and a **2x2 inverse** are needed.

Since h is Hermitian, the Cayley transform is exactly unitary:
its eigenvalues are (1-i delta t lambda/2)/(1+i delta t lambda/2),
with modulus one for every real lambda.

This preserves pure Gaussian-state occupation constraints. It is
globally second-order: phase errors accumulate O(delta t^2).

There is a physical interpretive warning: a Cayley step replaces
the exact single-particle energy-dependent phase exp(-i lambda delta t)
with exp[-2i arctan(lambda delta t/2)], changing even the CLEAN
dispersion relation. In long finite-size revivals, accumulated clean
propagation phase error can be important.

## 3. Higher-accuracy clean-bulk / rank-two defect Strang splitting

A second unitary method treats the clean propagation exactly and
isolates the defect to a local two-site kick:

    U_split(delta t)
      = exp(-i E delta t/2)
        exp(-i B V_D B^T delta t)
        exp(-i E delta t/2).

Since B^T B=I2, the *defect kick* has the exact identity

    exp(-i B V_D B^T delta t)
      = I + B [exp(-i V_D delta t)-I2] B^T.

Thus it can be evaluated without any dense defective-chain exponential:
multiply diagonal clean sine phases, apply a 2x2 onsite phase kick, then
multiply the remaining clean phases.

Every factor is unitary, so the algorithm preserves occupied-orbital
orthonormality. The error is globally O(delta t^2) from noncommuting
clean and defect Hamiltonians. When the defect is absent, the scheme
reproduces EXACT clean dynamics at any step size (up to roundoff).

It includes impurity-bound eigenstate physics **implicitly** through
repeated time steps but does NOT explicitly isolate the bound pole or
provide a closed-form capacity-peak prediction.

## 4. Capacity from occupied orbitals

Given occupied modes F(t), restrict to the left half A:

    C_A(t) = [F(t)]_A [F(t)]_A^dagger.

If p_j are eigenvalues of C_A,

    Capacity(t) = sum_j p_j (1-p_j) log^2((1-p_j)/p_j).

Both numerical methods maintain orthonormal orbitals, so all p_j
remain in [0,1] up to machine precision, and the Gaussian
correlation matrix remains physically admissible.

Our exact reference separately diagonalizes the full defective
Hamiltonian once and evaluates exp(-i h t) at each observation time.

## 5. Independent validation and numerical convergence

Tests independently verify:
- One Cayley step equals a dense implicit-midpoint solve.
- One Strang step equals a direct dense clean-half/defect-kick/clean-half
  operator multiplication.
- Both methods preserve orthonormal occupied orbitals.
- Without a defect, Strang matches exact time evolution for any dt.
- Strang converges approximately quadratically with shrinking dt.
- At t=0, both reproduce the same initial capacity as direct Gaussian
  evolution.
- Invalid time grids and parameters are rejected.

For L=48 and (u,v)=(1.2,0.35), the capacity RMSE over the
revival-time window 0.4L <= t < 1.5L is:

| time step | rank-two Cayley RMSE | exact-clean Strang RMSE |
|---:|---:|---:|
| 0.20 | ~0.12113 | ~0.01091 |
| 0.10 | ~0.03851 | ~0.00271 |
| 0.05 | ~0.00982 | ~0.000677 |

The error reductions are consistent with second-order convergence as
dt decreases. Similar improvements hold at L=64 and for (0.7,1.4)
and (0.5,0.2).

**Interpretation:** treating the clean dispersion and its accumulated
propagation phases exactly is advantageous for these tested finite-size
revival trajectories. This is a numerical-scheme comparison and does
not prove that a Wigner-delay model predicts the revival physics.

## 6. What is still missing for genuine research originality?

A physically explanatory theory must start from the complex scattering
data and bound-state poles and **predict** first-revival time and
amplitude without running near-exact numerical evolution of the full
chain.

Our next objective is to:
1. Construct an explicitly phase-resolved, finite-boundary propagator
   using scattering paths or a low-rank Dyson kernel.
2. Obtain controlled *truncations* or asymptotics with transparent error
   estimates and retained Gaussian physicality.
3. Separate continuum-band and bound-pole terms in those predictions.
4. Predict held-out first-revival times and heights before examining
   exact capacity curves.
5. Compare against an actual transmission-only quasiparticle baseline,
   not merely a zero-shift matched-transmission null control.

A model that runs the full quantum dynamics with another time-step
integrator is useful for validation, but is **not** that final theory.

## Reproduce

    git pull origin main
    python -m pytest -q
    python scripts/benchmark_reduced_time_propagators.py

Outputs:
- results/reduced_time_propagator_benchmark.csv
- figures/reduced_time_propagator_comparison.png
- figures/reduced_time_step_convergence.png

Related source:
- src/defect_tmatrix.py: exact stationary two-site T-matrix and poles.
- src/rank2_cayley.py: unitary Woodbury/Cayley evolution.
- src/clean_defect_strang.py: exact clean evolution and local defect kick.
- tests/test_reduced_time_propagators.py: independent solver checks.

Background: ordinary Woodbury matrix identity, Cayley transform,
Strang splitting, and standard Gaussian free-fermion correlation
methods. This document does not claim these methods were invented here.
