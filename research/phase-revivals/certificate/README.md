# Airy first-return revival: conditional a posteriori first-peak enclosure

**Date:** 29 October 2026. **Model:** L=48, staggered mass m=2.5, central onsite potentials (1.2,0.35) and (0.35,1.2). Natural hopping units, Gaussian **normalized sampled** smoother with width sigma=2, grid 0:0.025:36. Not necessarily the same smoother as the older reflect-padding plot.

## What this package establishes

The Airy first-return prediction is assembled from the clean Bessel-image direct/single-boundary baseline and a leading Olver Airy wraparound wave, then polar normalized. The `airy` columns in `data/` are independent **predictions**, not fits to the exact defective-Hamiltonian trajectory.

Separately, the error-controlled, fourth-order **two-site** Hermite Volterra solver, on a finer grid h=0.00625, computes a source-only validation witness. Its calculated continuous-time Duhamel residual bound controls the unknown exact occupied-mode amplitudes in exact arithmetic. The predictor and witness are compared at each node. A state-dependent Weyl-eigenvalue bound, which is valid at 0 and 1, bounds sample capacity error; interval envelopes of the Gaussian weights bound the current and its curvature. This proves, **conditionally on exact evaluation of the numerical inputs**, that the exact smoothed capacity has a unique local maximum in a specified neighborhood of each Airy peak.

The same interval-envelope technique verifies positive exact current on [19.2, Airy peak−0.06]. Values 7.15 time units on either side of the Airy peak certify that each first maximum satisfies prominence >0.08 within the stipulated ±0.15L neighborhood. These together establish a **conditional first qualifying peak** under this smoother.

## Results (see FINAL_RESULTS.json)

| Configuration | Airy peak | Conditional radius | Prominence lower bound |
|---|---:|---:|---:|
| (1.2,0.35) | 26.4214024623 | 0.0042463063 | 0.22156590 |
| (0.35,1.2) | 27.3049932124 | 0.0105651682 | 0.18248801 |

Difference Airy forward minus reverse: −0.88359075. Combined error radius: 0.01481147, yielding approximately **[−0.898403, −0.868779]** conditional on exact arithmetic.

Post-hoc exact-diagonalization benchmark (not part of the certificate): 26.42268042 and 27.31195479, respectively, both within the conditional enclosure.

## State-dependent spectral capacity bound

For f(p)=p(1−p)log²((1−p)/p), set f(0)=f(1)=0. If the projector error is ≤delta, Weyl's inequality places the exact j-th entanglement eigenvalue in I_j=[max(0,phat_j−delta),min(1,phat_j+delta)]. Let fmin_j and fmax_j be extrema of f on I_j. These can be computed from interval endpoints and the only interior stationary points: p*=0.0832217201995, 1/2, 1−p*. Then, without requiring a spectral gap,

    |C_exact-C_witness| <= Σ_j max(f(phat_j)−fmin_j, fmax_j−f(phat_j)).

The right-hand side uses only the computed physical witness spectrum and the independently derived norm bound. The Airy-sample uncertainty is this quantity **plus** |C_Airy−C_witness|, calculated without accessing exact defective-Hamiltonian data.

For the normalized Gaussian weights w_j(t), direct differentiation gives w'_j and w''_j. If |C_j−C^Airy_j|≤eps_j, then the current and curvature errors obey

    |Cbar'−Cbar_Airy'| <= Σ_j |w'_j| eps_j,
    |Cbar''−Cbar_Airy''| <= Σ_j |w''_j| eps_j.

Interval upper envelopes for weights and centered moments turn these into bounds valid over entire time cells, not only sample points. A sampled curvature bound with an explicit third-derivative majorant controls between-grid variation.

## What this does **not** prove

1. **Not a validated interval-arithmetic proof.** Special functions, eigenvalues, phi functions, numerical sums and source residual majorants use ordinary double precision, not outward rounding. The numerical intervals are **conditional on exact evaluation** of their inputs. The mathematical inequalities are valid in exact arithmetic.
2. **Not a standalone Airy remainder theorem.** This is an *a posteriori* certificate: it uses the full two-site, source-only Volterra witness after making the Airy prediction. It does not certify the Airy model using only asymptotic coefficients. Thus it does not establish computational superiority of the reduced Airy model.
3. **Not a claim of novel physics.** Capizzi & Eisler already derive coherent free-fermion defect revivals; Olver's Airy asymptotics, Duhamel bounds, and Weyl perturbation inequalities are known. At most, this is a tested model-specific computational application pending literature comparison.
4. **Not a result for arbitrary initial states/defects/sizes.** Only the two stated orientations and particular smoother are verified here. Tests need expansion.

## Reproduction

Install numpy, scipy, matplotlib (`pip install -r requirements.txt`), then:

    OPENBLAS_NUM_THREADS=1 python -m unittest -v test_aposteriori
    OPENBLAS_NUM_THREADS=1 python run_report.py
    OPENBLAS_NUM_THREADS=1 python plot_certificates.py

The supplied arrays `airy_witness_*.npz` contain the independently calculated source-only witness and per-node error envelopes. To recompute them using the two-site Hermite solver (heavier computation):

    OPENBLAS_NUM_THREADS=1 python airy_aposteriori_certificate.py

To regenerate the Airy-only predictions from clean path physics (no defective Hamiltonian diagonalization; slower):

    OPENBLAS_NUM_THREADS=1 python generate_airy_curves.py

## The scientifically meaningful next step

Apply outward-rounded interval arithmetic to the source-only fourth-order solver and the spectral bound; then derive an intrinsic, uniform finite-L Bessel-to-Airy remainder that does not require this independent full two-site witness. Compare a scalable generalization against the Capizzi–Eisler conformal limit and held-out nonconformal defects.

References: Capizzi & Eisler, SciPost Phys. 14, 070 (2023), https://arxiv.org/abs/2209.03297; NIST DLMF §10.20, https://dlmf.nist.gov/10.20.
