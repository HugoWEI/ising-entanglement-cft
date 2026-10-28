# Composite strong-defect capacity prediction from scattering poles and a Krylov spectrum

**October 2026.** This is a reproducible, explicitly stated
nonperturbative **computational prediction formula**, but NOT a
compact new universal physical law for first revival time and height.
All individual identities are established matrix resolvent,
Gaussian-state, matrix determinant, and Krylov mathematics.

## 1. Exactly what we compute without diagonalizing a strong defect

Given symmetric reference h_sym with central onsite potentials
(w,w), define the two central even/odd parity channels |+>, |->.
For desired asymmetry delta, the dressed local scattering matrix is

    D(z)=1-delta^2*g_plus(z)*g_minus(z)/4,
    T_delta(z)=1/D(z) *
        [[delta^2*g_minus(z)/4, delta/2],
         [delta/2,delta^2*g_plus(z)/4]].

The full occupied orbital propagator can be obtained from the
symmetric reference and a one-dimensional energy contour:

    F_delta(t)=U_sym(t)F0
       + (1/(2*pi*i)) ∮_Gamma e^(-izt)
          R_sym(z) B T_delta(z) B† R_sym(z) F0 dz,

where B=[|+>,|->]. No eigenvectors of h(delta) are needed.

The bound energy E_b and rank-one projector P_b can also be derived
from the same scalar denominator D(E_b)=0 and its residue:

    P_b=Res_(z=E_b) R_delta(z).

For the tested positive central potentials there is exactly ONE
out-of-band mode E_b>2. We explicitly reject other situations
rather than silently ignoring extra modes.

Define the continuum occupied orbitals

    F_c(t) = F_delta(t)- e^(-i E_b t) P_b F0.

From them calculate B_A=F_c(t)[left,:], the vectors

    x=e^(-i E_b t) b[left],
    q=b†F0,
    y=B_A q†,

and the mixed Gaussian reference covariance

    G_deph=B_A B_A†+||q||^2 x x†.

The phase-coherent correction is rank <=2,

    G_full=G_deph+X J X†,
    X=[x,y], J=[[0,1],[1,0]].

## 2. Explicit NONPERTURBATIVE compressed capacity formula

For f(p)=p(1-p)log²[(1-p)/p], define a block-Krylov basis Q_m
of dimension <=2m that spans

    span{X,G_deph X,...,G_deph^(m-1)X}.

Then our *approximation* is

    BOXED:
    C_pred,m(t;delta) =
        Tr f(G_deph)
       +Tr f(Q_m†G_deph Q_m +
             Q_m†X J X†Q_m)
       -Tr f(Q_m†G_deph Q_m).

This formula does not linearize in delta or bound-coherence
strength, and therefore remains mathematically defined for
strong defect potentials. For m large enough to span the
subsystem it is exact up to contour numerical integration
errors. For finite m it is an ordinary projected matrix-function
approximation whose accuracy must be *empirically checked*.

Finally the predicted first broad capacity peak is defined by
the same pre-specified smoothing/peak selection applied to
t -> C_pred,m(t;delta), without looking at the exact
asymmetric-defect capacity curve.

The prediction is generated only from h_sym, delta, F0,
the contour scattering integral, E_b and P_b, and fixed
Krylov block count m. No exact asymmetric eigenbasis
is used for the prediction.

## 3. Why this is not the desired final asymptotic theory

The continuum orbital propagator F_c still requires evaluating
the exact energy contour integral. The method can be SLOWER
than straightforward diagonalization of h(delta) for small L.
Also m=8 retains 16 entanglement directions; this is not a
two-mode phenomenological explanation.

Thus this is a **fully specified conditional strong-defect
predictor and error-isolation tool**, not a claimed new
analytical law, nor evidence of computational superiority.

Its genuine utility is to expose separate approximation errors:
- Contour/scattering propagation error.
- Exact bound-pole reconstruction error.
- Finite-dimensional entanglement-spectrum compression error.
- Peak-branch definition and smoothing effects.

## 4. Held-out validation

The script compares against an independently diagonalized
asymmetric Hamiltonian at
(L,w,delta)=(48,.775,±.85), (64,.775,±.85), and
(48,1.05,±.7), corresponding to our original defect
(1.2,.35) and a separate pair (.7,1.4).

Use m=8 Krylov blocks (at most 16 retained entanglement
directions), a fixed numerical time step .4 for reporting,
and N=1536 contour nodes. No fit to any of the exact curves.

Both forward and reversed defects are checked independently.
The output reports the first broad peak-time error, peak height
error, revival-window capacity RMSE, and a strict orbital
isometry check for contour quadrature.

**No claim of experimental confirmation or original theory**;
this remains controlled numerical theoretical research.

## 5. Reproduce

    git pull origin main
    python -m pytest -q
    python scripts/benchmark_scattering_pole_krylov.py

Files:
- src/parity_resummed_scattering.py
- src/parity_contour_capacity.py
- src/bound_coherence_determinant.py
- src/bound_coherence_krylov.py
- src/scattering_pole_krylov_capacity.py
- tests/test_scattering_pole_krylov_capacity.py
- scripts/benchmark_scattering_pole_krylov.py
- results/scattering_pole_krylov_peaks.csv
- results/scattering_pole_krylov_timeseries.csv
- figures/scattering_pole_krylov_first_rev.png

## 6. Next theory challenge

Obtain F_c(t) from a *small, controlled number of continuum
scattering phases/boundary-reflected paths* instead of a
near-exact contour reconstruction, and validate held-out
capacity peaks without any exact capacity input.

This step is essential before claiming the originally
requested compact physically predictive strong-defect
revival-time law has been found.


### Additional prospectively fixed test

After selecting the eight-block model from the exploratory 48/64
benchmarks, we added a distinct 56-site chain with mean
potential w=.6 and asymmetry delta=1.0, i.e. onsite potentials
(1.1,.1) and their reversal, **before computing their capacity
prediction errors**. The post-quench Hamiltonian has one
out-of-band state at about +2.30009465, verified as a
model-class eligibility condition only.

The fixed approximation settings (m=8, contour nodes 1536,
observation dt=.4) are unchanged. Its accuracy or failure must
be reported, not used for retuning m.
