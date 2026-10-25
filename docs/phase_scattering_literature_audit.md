# Scattering phases and modular-capacity revivals: literature audit + controlled pilot
**Updated 25 October 2026. Status: preliminary computational exploration; no original research claim.**

## Questions under investigation

1. Can a defect's scattering **phase** help predict transient entanglement-capacity dynamics beyond a quasiparticle approximation using only transmission probabilities?
2. Can interference and scattering phases explain/predict *capacity revivals* in finite free-fermion chains?

### Strongest prior art (must cite)

- L. Capizzi & V. Eisler, *Entanglement evolution after a global quench across a conformal defect*, SciPost Phys. 14, 070 (2023), https://arxiv.org/abs/2209.03297 .
  - Sec. 3.4, Eqs. (62)-(64), constructs an interference-aware quasiparticle explanation for **entropy** revivals with a conformal defect.
  - Sec. 3.5, Eqs. (65)-(66), treats a **momentum-dependent hopping defect**.
  - Discussion explicitly reports highly irregular entropy revivals for hopping defects, attributed to interference and warranting further investigation.
  - Their general Rényi formulas may already imply corresponding capacity predictions in cases where their approximations apply. Therefore the idea that "reflection and transmission interference affects revivals" is not itself new.
- Kusuki, Suzuyama & Tang (2026), https://arxiv.org/abs/2610.02309, studies **equilibrium** capacity and conformal interfaces and suggests nonequilibrium capacity filtering as future work.
- G. Parez, R. Bonsignori & P. Calabrese (2021), https://arxiv.org/abs/2106.13115, derives time-dependent Rényi/symmetry-resolved observables for free fermions, including delay phenomena; not our precise two-site phase pilot.
- S. Fraenkel & M. Goldstein (2021), https://arxiv.org/abs/2105.00740, develops nonequilibrium entanglement across generic scattering regions and full-counting-statistics relationships in steady states.

**Novelty status:** no full predictive *capacity revival* theory for exactly the proposed phase-sensitive setup was identified in this literature check. That is not an exhaustive proof that it is unpublished; review citations, preprints and derivations before representing this as an original result.

## Why we changed the test model

The original repo studies the **transverse-field Ising chain**, but this phase test uses a distinct **number-conserving tight-binding free-fermion chain**, which matches the hopping-defect setting more naturally. Do NOT treat them as the same Hamiltonian, even though both are Gaussian/free-fermion problems.

We take a finite open chain with nearest-neighbor hopping -1 and two neighboring onsite potentials straddling the central cut. Compare:

- Configuration A: (u,v)=(1.2,0.35), ordered left-to-right
- Configuration B: (v,u)=(0.35,1.2), the spatial reverse.

Scattering on the **infinite** chain obeys E(k)=-2 cos k. For this pair, the transmission amplitude t_A(k)=t_B(k) at *all* momenta, while the left-reflection amplitudes r_A(k) and r_B(k) have different phases. This is verified numerically by solving the two-site transfer matrix.

At k=1.1 (radians/site), the two transmission probabilities are both about 0.63386, while left-reflection phases are approximately -1.955 and -0.768 rad, respectively, using the same reference convention.

## Our pilot quench: same input, same T(k), different phase

Prepare the **same** gapped Gaussian ground state of a uniform hopping chain with staggered onsite field (+/- 2.5). At time 0, quench to either (u,v) or its reverse with no staggered field. Both are half-filled, open, length L=48.

Evolve the occupied single-particle modes, restrict the one-particle correlation matrix to half the chain, and compute

S = sum[-p ln(p) - (1-p) ln(1-p)]
C = sum p(1-p)*[ln((1-p)/p)]^2.

These p are eigenvalues of the restricted correlation matrix.

At evolution time tau=2 in units hbar/hopping, preliminary values are:

| Configuration | Entropy S | Capacity C |
|---|---:|---:|
| (1.2,0.35) | 1.573403 | 0.815212 |
| (0.35,1.2) | 1.366163 | 1.131601 |

The initial S and C are equal because the initial state is identical. Over tau in [0,52], the maximum observed absolute differences are approximately 0.314 in entropy and 0.343 in capacity (different units; comparing the raw differences DOES NOT establish greater sensitivity of capacity). This confirms that **transmission probability alone does not determine the exact transient bipartite entanglement observables for this asymmetric preparation**.

## Essential null control

The two final Hamiltonians are related by spatial reflection. If the initial state is reflection symmetric, a half-chain bipartition has identical entropy and capacity under both. We verified this numerically for an exactly reflection-symmetric half-filled Fock state, and our tests enforce it.

**Without this negative control, differences could be mistaken for universal phase physics.** Reflection amplitudes, preparation asymmetry, and finite geometry must all be specified. Scattering phase shifts themselves depend on choice of reference planes: only gauge-/reference-invariant combinations or carefully fixed reference conventions can underlie a physical prediction.

## What has NOT yet been demonstrated

- No *quantitative prediction* from the reflection-phase derivative / Wigner delay for the shapes or timings of capacity revivals has been checked.
- The two configurations are mirror images, not two unrelated defects; with a parity-symmetric preparation they cannot be distinguished by this bipartite observable.
- A toy model's phase sensitivity is a test of principle, not an original discovery.
- We have not shown that capacity reveals more physics than entropy after controlling for scale, variance and initial state.

## Next genuine research milestone

1. Derive scattering amplitudes r_L(k),r_R(k),t(k) with a fixed coordinate convention.
2. Compute directional time delays tau_W(k)=d arg[r_L(k)]/dE or the appropriate physically invariant scattering-channel delay.
3. Predict a *specific* shift, amplitude or revival feature from the phase information **before** comparing to numerical capacity trajectories.
4. Test against large L, different gapped initial preparations and matched-transmission controls; evaluate predictions out of sample.
5. Check whether existing Rényi-entanglement formulas (especially Capizzi-Eisler 2023 Eq. 64) already imply that predicted feature.

Until step 3, we should describe our work as **numerical phase-sensitivity benchmarking**, not an explanatory theory of capacity revivals.

## Reproduction

Run `python -m pytest -q` and `python scripts/run_phase_revivals.py`.

Outputs: `results/two_site_scattering.csv`, `results/phase_revival_pilot.csv`, and `figures/phase_revival_pilot.png`. CI saves these as a named artifact on GitHub Actions.


## Independent phase-delay prediction (known scattering theory)

The Wigner reflection delay, for a fixed lead/reference-plane convention,
is tau_W(k)=d[arg r_L(k)]/dE, with E=-2*cos(k). Our exact two-site
scattering amplitudes give at k0=1.1:

- Defect (1.2,0.35): tau_W ≈ 0.30737
- Defect (0.35,1.2): tau_W ≈ 1.21195
- **Predicted difference** (first minus second) ≈ -0.90458 time units.

A separately simulated narrow single-particle Gaussian wavepacket
(L=240, initial center at site 70, width 12, momentum k0=1.1;
measure reflected probability at site 55) peaks at times **63.8**
and **64.7**, giving observed difference **-0.90** time units.
The agreement supports the stationary-phase explanation, and is a
useful benchmark for our numerical scattering conventions.

**This prediction is an established Wigner-delay result, not a new
scientific discovery.** The important *unperformed* research test is
whether combining such delays with a coherent multiple-reflection
model correctly predicts the locations and amplitudes of **modular
capacity revivals**. Wavepacket delay agreement alone cannot answer that.

Reproduce with `python scripts/validate_phase_delay.py` (outputs a CSV
and figure in `results/` and `figures/`).
