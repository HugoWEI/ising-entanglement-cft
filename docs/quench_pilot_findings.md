# Initial quench pilot (exploratory, NOT a research discovery)

## Model and method

- Open Ising chain, L=24, 48, 64, J=1, central defect bond d.
- Ground state at h_initial=2; at tau=0 quench to h_final=1.
- Keep defect d fixed throughout the quench; measure entanglement between
  the two half chains.
- Evaluate S_A and modular capacity C_A using evolved Majorana covariances.
- Fit ordinary least-squares straight lines to the time window
  tau=2,2.25,...,5 (units hbar/J). This is an arbitrary practical window,
  **not** a proven universal linear-growth regime.

## L=48 illustrative fitted slopes

| d | fitted dS/dtau | fitted dC/dtau | entropy / clean | capacity / clean |
|---:|---:|---:|---:|---:|
| 0.25 | 0.10727 | 0.28164 | 0.283 | 0.810 |
| 0.50 | 0.25625 | 0.36914 | 0.676 | 1.062 |
| 0.75 | 0.35103 | 0.35324 | 0.926 | 1.016 |
| 1.00 | 0.37914 | 0.34772 | 1.000 | 1.000 |

The same slopes at these early times agree to 5 decimal places for
L=24, 48 and 64, consistent with boundary reflections not yet affecting
the central cut. This is a *consistency check*, not convergence to a
universal continuum coefficient.

**Observation:** for d=0.5 in this window, fitted entropy growth is
about 32% smaller than the clean value, while capacity growth is about
6% larger. This is not, by itself, a new effect: the 2026 equilibrium
interface paper already shows that capacity can exceed its clean-chain
logarithmic coefficient for partially transmitting defects.

## Main cautions

1. The CFT replica geometry analysis in Wen--Wang--Ryu (2018,
   arXiv:1711.02126) suggests that the leading *continuum* dynamical
   capacity behavior can be inferred from already-known replica
   functions. Do not present those leading relations as original.
2. Our initial h=2 state is not exactly a regulated conformal boundary
   state. Band curvature, nonthermal mode occupations and finite-size
   effects may all matter. Comparison must respect these assumptions.
3. This isn't a universal statement about C_A(tau)>S_A(tau). The
   equilibrium bound concerns asymptotic coefficients only.
4. Apparent fit slopes depend on the chosen time interval and protocol.
   Before any scientific conclusion we need complete-window plots,
   quench-depth variations, and theory comparison.

## Concrete follow-up

Analyze whether the lattice-to-CFT deviation in the **capacity** growth
rate can be quantitatively predicted from post-quench mode occupations
and interface transmission probabilities. This is a narrower potential
novelty than the mere existence of different entropy/capacity rates,
but the literature still needs a deeper derivation-level comparison.

## Reproducibility

Run `python scripts/run_quench.py` for a saved L=24 pilot. For the L=48
slopes here, reproduce locally with
`quench_trajectory(48, d, 2., 1., np.arange(0, 5.01, .25))` and fit
on time in [2,5] using `np.polyfit`. The automated tests compare the
dynamics to independent spin-basis exact diagonalization at L=4.
