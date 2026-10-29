# Error-budget decomposition: Airy first return versus source-only Hermite witness

This note supplements the conditional sampled-to-continuous Airy/Bessel peak certificate. The previously archived **fourth-order source-only Hermite–Volterra witness** offers an independent approximation of the physical full two-site scattering response; it was **not** used to generate the Airy or Bessel curves and does not diagonalize the final defective Hamiltonian. The witness itself was checked independently against the exact solution in earlier work, but none of these calculations use formally verified interval arithmetic.

All three peak times are obtained from the **same** discrete Gaussian smoother with `sigma=2` and `t_j=0,0.025,...,36`.

| defect (u,v) | leading Airy peak | Bessel-image peak | source-only Hermite witness peak | fraction of Airy–witness discrepancy remaining after Bessel replacement |
|---|---:|---:|---:|---:|
| (1.20, 0.35) | 26.421402462 | 26.421410456 | 26.422680423 | **99.37%** |
| (0.35, 1.20) | 27.304993212 | 27.305133598 | 27.311954782 | **97.98%** |

This is **not** an additive proof of cause, as interacting truncations can cancel and the witness still has its own errors. Nevertheless, it strongly suggests that the remaining observed peak error is mainly from the *common direct/single-reflection trapezoidal baseline and missing image/impurity feedback*, rather than the Bessel-to-leading-Olver-Airy substitution itself.

## Priority for the next theory stage

1. Replace the second-order direct/single-reflection baseline with the validated fourth-order two-site Hermite solver (or derive its reduced equivalent) **without double-counting reflected paths**.
2. Include the wraparound reflected-wave forcing **self-consistently** in the Volterra equation. Simply adding `W(t)Q` to the baseline bypasses the impurity feedback term from that return path.
3. Repeat the same conditional peak-enclosure calculation on a reduced source-only model with these corrections, then assess if the leading Airy replacement remains small.
4. Only after the dynamical discrepancy is acceptably small should fully verified interval arithmetic and a novelty audit become the main bottlenecks.

The main technical lesson is about prioritization: intrinsic wavefront asymptotic error is now smaller than the error of assembling the quantum scattering dynamics. No novelty claim is made.
