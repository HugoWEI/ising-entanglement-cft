# A conserved phase-memory budget for entanglement-capacity revivals

**29 October 2026. Status: rigorous elementary bound, numerical tests,
not an independently novel theorem or a universal revival-time law.**

This note isolates a physical limitation on phase-controlled
finite-size modular-capacity dynamics. Unlike simply fitting another
peak-shift coefficient, it identifies how much bound–continuum
coherence can exist in an initial pure Gaussian state and how much
of it can enter the chosen subsystem.

Related project developments:
- docs/bound_coherence_rank2_formula.md (rank-two entanglement spectrum)
- docs/phase_coherence_peak_response.md (exact phase susceptibility)
- docs/second_order_phase_peak.md (second-order local peak prediction)
- docs/error_controlled_revival_walks.md (no-diagonalization propagation)

## 1. Assumptions

Let h be ANY finite Hermitian single-particle Hamiltonian with a
normalized isolated eigenvector b and eigenenergy E_b.
Let P_b=|b><b|, P_c=I-P_b.
Our application uses the even, open, number-conserving hopping chain
with one impurity mode outside the band, but the identity below
requires only a rank-one eigenprojector.

The initial pure Gaussian Slater determinant has correlation
projector P0=F0 F0†, where F0† F0=I and P0²=P0.

Let

    beta=<b|P0|b>,  0<=beta<=1.

This is the initial occupation of the localized eigenmode.

The preparation phase gate

    V_theta=P_c+exp(i theta)P_b

is UNITARY and commutes with h. It changes neither the
post-quench Hamiltonian nor its energy eigenvalues, scattering
coefficients, final-mode occupations, or energy moments.
It changes only the initial phase relation between bound and
extended sectors.

## 2. Exact CONSERVED coherence budget

Define the orthogonal bound-to-continuum coherence vector

    v=P_c P0 b=P0 b-beta b.

Because P0 is a rank-r orthogonal projector,

    ||v||² = <b|P0²|b>-beta² = beta(1-beta).

In particular, bound–continuum coherence vanishes EXACTLY
when beta=0 or beta=1 and is maximized at beta=1/2.

The phase-rotated global covariance at any time t is

    P_theta(t)=exp(-iht) V_theta P0 V_theta† exp(iht).

For two preparation phases theta and phi, its difference is

    P_theta(t)-P_phi(t)
      = d |b_t><v_t| + d* |v_t><b_t|,

where d=exp(i theta)-exp(i phi),
b_t=exp(-iht)b,
v_t=exp(-iht)v,
and <b_t|v_t>=0.

This matrix is Hermitian and rank <=2, with nonzero eigenvalues

    lambda_±= +/- |d| sqrt[beta(1-beta)]
            = +/-2|sin((theta-phi)/2)|sqrt[beta(1-beta)].

The resulting norms are EXACTLY

    BOXED:
    ||P_theta(t)-P_phi(t)||_1
      =4|sin((theta-phi)/2)|sqrt[beta(1-beta)].

    BOXED:
    ||P_theta(t)-P_phi(t)||_F
      =2sqrt(2)|sin((theta-phi)/2)|sqrt[beta(1-beta)].

These are independent of t. The phase coherence is not
generated or destroyed by the fixed Hamiltonian; it is
redistributed spatially and can interfere differently with
returning quasiparticle paths.

**This is standard projector/unitary-invariance algebra, not a
new universal theorem.** It gives our capacity problem a
quantitative conserved *preparation coherence budget*.

## 3. A spatial subsystem cannot access arbitrary phase contrast

Let A be any spatial subsystem, P_A its one-particle projector,
and alpha_A=||P_A b||², the spatial weight of the localized mode.

The reduced covariance difference is exactly

    Delta G_A
      =d |P_A b_t><P_A v_t|
         +d* |P_A v_t><P_A b_t|.

Because ||P_A b_t||=sqrt(alpha_A) for stationary bound b
and ||P_A v_t||<=||v_t||=sqrt[beta(1-beta)], the
nuclear/trace norm obeys

    BOXED:
    ||Delta G_A||_1
      <=4|sin((theta-phi)/2)|sqrt[alpha_A beta(1-beta)].

The upper bound is UNIFORM IN TIME and does not scale
explicitly with chain length. It can be loose if the
extended coherence wavefunction has small weight in A
or if interference cancels.

The bound vanishes if beta is 0 or 1, if b has no spatial
support in A, or if theta=phi modulo 2pi. It does not say
that beta=1/2 necessarily leads to a large capacity change:
the spatial and spectral structure can suppress it.

## 4. A rigorous uniform bound on modular-capacity change

For the Gaussian modular capacity function

    f(p)=p(1-p) log²((1-p)/p),  p in [0,1],
    C_A=Tr f(G_A),

set f(0)=f(1)=0 by continuity.

The derivative diverges logarithmically near the endpoints,
so NO global Lipschitz constant exists. However,

    |f'(p)|
      <= log²(1/q)+2 log(1/q), q=min(p,1-p).

For 0<q<=1/2, optimizing sqrt(q) log²(1/q) and
sqrt(q) log(1/q) separately gives

    sqrt(q)|f'(p)|
       <= 16/e² +4/e <4.

Thus |f'(p)|<=4/sqrt(min(p,1-p)).
Integrating on either side of p=1/2 and using
sqrt(a)+sqrt(b)<=sqrt[2(a+b)] proves

    |f(p)-f(q)| <=8sqrt(2)|p-q|^(1/2)
                 <12|p-q|^(1/2).

For d_A eigenvalues, the Hermitian trace-norm
eigenvalue perturbation inequality and
Cauchy–Schwarz imply

    |C_A(theta)-C_A(phi)|
       <= 12 sqrt[d_A * ||Delta G_A||_1].

Combining with the phase-memory budget yields

    BOXED (UNCONDITIONAL, ENDPOINT-SAFE):
    |Delta C_A|
       <=12 sqrt[
           4 d_A |sin((theta-phi)/2)|
           sqrt(alpha_A beta(1-beta))
         ].

It is safe when the entanglement eigenvalues approach 0
or 1, but can be VERY conservative and grows as sqrt(d_A).

### Stronger result with an entanglement-spectrum gap

If BOTH Gaussian reduced covariances have spectra entirely
inside [eps,1-eps] for some 0<eps<1/2, define

    L_eps=log((1-eps)/eps).

The scalar mean-value theorem gives the sharper bound

    |Delta C_A|
       <= [L_eps²+2L_eps] ||Delta G_A||_1
       <=4[L_eps²+2L_eps]
           |sin((theta-phi)/2)|
           sqrt[alpha_A beta(1-beta)].

This bound is dimension-independent, conditional on an
entanglement-spectrum gap eps. In many free-fermion
finite-size systems some occupations approach 0 or 1,
so the gap may shrink quickly with system size.

**Neither bound directly constrains peak-time shifts.**
Peak locations can be arbitrarily sensitive to an
arbitrarily small capacity perturbation if an isolated
maximum becomes flat, nearly degenerate or changes branch.
A peak-time theorem must also control temporal derivatives
and curvature, as in docs/second_order_phase_peak.md.

## 5. Falsification / empirical value

The implementation checks, independently:
- the exact conserved global trace and Frobenius norms at
  multiple times, phases, chain lengths and defect strengths;
- the rigorous reduced-subsystem trace norm bound;
- the endpoint-safe modular-capacity bound;
- the stronger gap bound when both spectra are separated
  from 0 and 1;
- explicit pure Slater preparations with beta=0, .5 and 1;
- exact phase invisibility at beta=0 and beta=1;
- adversarial scalar p,q including the endpoints.

The benchmark varies the INITIAL staggered gap at fixed
post-quench defect and the bound-phase gate. It measures
bound occupation beta, the spatial bound weight alpha_A,
and the spread of the first broad capacity peaks over
four phases. **We do not assume a monotone relationship
between beta and timing**, nor that the upper bounds
will be tight. Their role is to distinguish forbidden
responses from possible ones.

Source:
- src/phase_memory_budget.py
- tests/test_phase_memory_budget.py
- scripts/benchmark_phase_memory_budget.py
- .github/workflows/phase-memory-budget-check.yml

Expected outputs:
- results/phase_memory_budget.csv
- results/phase_memory_time_samples.csv
- figures/phase_memory_occupation_and_peaks.png
- figures/phase_memory_capacity_bounds.png

## 6. Originality limitations and next step

The conservation of singular values under unitary
time evolution and the projector identity
||P_c P0 b||²=beta(1-beta) are elementary.
The subsystem trace bound is an elementary rank-two
perturbation estimate; the modular-capacity Hölder
modulus uses standard matrix eigenvalue inequalities.
No novelty is established by repackaging them.

The current literature already connects initial
coherences with defect-coupled free fermion dynamics:
- Donelli, De Chiara, Scazza & Gherardini (2026),
  Phys Rev A 113 013311, DOI 10.1103/5kt1-5m83.
  The paper studies orthogonalization and work
  quasiprobabilities rather than our finite-chain
  modular-capacity peak response.
- Capizzi & Eisler (2023), SciPost Phys 14 070,
  arXiv:2209.03297, treats coherent defect revivals.
- The capacity functional and Gaussian entanglement
  determinant are established theory.

A potentially new physics claim would require a
controlled continuum/boundary asymptotic law predicting
the *distribution* of the conserved coherence budget
among entanglement modes at t~L, including the
initial gap and defect localization length.
This work only supplies rigorous necessary bounds.

Reproduce:

    git pull origin main
    python -m pytest -q tests/test_phase_memory_budget.py
    python scripts/benchmark_phase_memory_budget.py


## 7. Independent validation: first numerical results

The focused GitHub Actions workflow PASSED, including 18
independent proof/numerical tests and all preselected preparation
sweeps:
https://github.com/HugoWEI/ising-entanglement-cft/actions/runs/40839476308

The tests cover system sizes L=24,48,64, onsite defects
(1.2,.35) and (.7,1.4), initial staggered masses
m=.5,2.5,8, and four phase preparations
theta=0, pi/2, pi, 3pi/2.

### A numerical sign of a preparation-controlled coherence budget

At L=48, for the first defect (u,v)=(1.2,.35):

| initial staggered mass | beta | sqrt[beta(1-beta)] | range of first-peak times over phases |
|---:|---:|---:|---:|
| .5 | .08363 | .27683 | 0.0 on dt=.4 time grid |
| 2.5 | .31442 | .46428 | 2.4 |
| 8 | .50065 | .50000 | 2.4 |

For the second defect (u,v)=(.7,1.4):

| initial mass | beta | coherence budget | phase-peak time range |
|---:|---:|---:|---:|
| .5 | .00956 | .09728 | .4 |
| 2.5 | .13969 | .34666 | .4 |
| 8 | .29507 | .45607 | .8 |

These observations highlight a necessary-but-not-sufficient
role for initial coherence:
1. The bound occupation and its coherence budget are controlled
   by the INITIAL state, even while the post-quench defect is fixed.
2. Greater coherence *can* enable larger revival timing responses.
3. The observed response is NOT determined by beta alone: the
   second defect has weaker phase sensitivity at comparable
   coherence, and finite-sampling peak quantization matters.

**Do not infer that an observed 0.0 peak-time span on dt=.4 grid
means exactly zero continuous-time phase sensitivity.** This
experiment uses the original operational broad-peak finder and
does not resolve smaller sub-grid shifts. For that use the
analytic phase susceptibility and parabolic interpolation from
docs/phase_coherence_peak_response.md.

The independent output also checked the stronger trace-distance
inequality at three times per configuration. The global
one-body covariance distinguishability remains constant
at all tested times, even though its subsystem manifestation
and modular-capacity response vary.

**Assessment:** A rigorous *conserved phase-memory resource*
has now been isolated and falsification-tested. It provides
a quantitative constraint and a useful physical control.
Its algebra is established and its peak-timing explanatory
power remains incomplete: predicting the *spatial/time
distribution* of this coherence is the harder physics task.


## 8. A many-body interpretation: exact fidelity and quantum Fisher information

Let n_b=c_b†c_b be the occupation operator of the
localized single-particle eigenmode. The one-particle
phase rotation V_theta lifts to the many-body unitary
exp(i theta n_b) up to the usual fermion representation.

Because n_b has only eigenvalues 0 and 1 and
<Ψ0|n_b|Ψ0>=beta, ANY initial pure state (not just
a Slater determinant) obeys

    <Psi_theta|Psi_phi>
       =1-beta+beta exp[i(phi-theta)].

Consequently the exact many-body fidelity is

    BOXED:
    F(theta,phi)
       =|<Psi_theta|Psi_phi>|²
       =1-4beta(1-beta) sin²[(theta-phi)/2].

This fidelity remains constant under common unitary
time evolution. Because [h,n_b]=0 in the fixed
post-quench Hamiltonian, all final spectral mode
populations are also unchanged by the phase rotation.

For pure-state encoding of theta by exp(i theta n_b),
standard quantum Fisher information is

    BOXED:
    F_Q =4 Var(n_b)=4beta(1-beta).

This equals four times the squared coherence budget
defined in section 2. It reaches the **maximum F_Q=1**
at beta=1/2 and vanishes at beta=0 and 1.

The fidelity is independently validated in code using
the Slater determinant overlap

    <Psi_theta|Psi_phi>
       =det(F_theta† F_phi),

which reduces by the rank-one matrix determinant
lemma to the same scalar 1-beta+beta exp(iΔtheta).
The tests also check time invariance, the
beta endpoints, and the small-angle expansion

    F(theta,theta+delta)
       =1-F_Q delta²/4+O(delta⁴).

**This is standard pure-state quantum metrology,
not a novel theorem.** It gives an operational
interpretation of the initial phase-memory budget
and implies that any reduced-subsystem information
cannot exceed the global phase information
under data processing. It does NOT tell us the
time of the first modular-capacity revival;
that requires propagation and entanglement
spectral dynamics.
