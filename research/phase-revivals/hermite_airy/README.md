# Fourth-order self-consistent Airy–Volterra revival solver

Research continuation, October 30, 2026. This is a numerical result, not a formally interval-certified first-peak theorem or a claim of new quantum physics.

The initial condition is a half-filled open fermionic hopping chain, with staggered initial mass 2.5; the final Hamiltonian has central adjacent onsite defects (u,v). The observable is Gaussian modular capacity, computed after polar-normalizing occupied amplitudes, smoothed by a discrete Gaussian of sigma=2.

## Numerical method

For the same approximate clean propagator K(t) in both source and kernel, solve the two-site self-consistent Volterra equation:
Y(t)=G(t)-i(K_DD * VY)(t), G(t)=[K(t)Q]_D.
Integration by parts gives:
Y'(t)=G'(t)-i K_DD(t) V Y(0)-i(K_DD * VY')(t).
We interpolate the impurity drive Y using a cubic Hermite polynomial per time cell, integrating the four Hermite shapes and their four derivatives against K with 8-point Gauss–Legendre quadrature. The unknown Y_k,Y'_k satisfy one complex 4-by-4 block system per time step. After solving, reconstruct the full occupied modes using the same cubic convolution. The returning Airy wave enters source and impurity response exactly once.

The leading Olver–Airy expression replaces only the two first-return Bessel image functions in mode airy. Mode bessel retains them exactly and is checked against the independent clean-sine-basis fourth-order Volterra solver. The final defective Hamiltonian is never diagonalized for prediction.

The source derivative uses a five-point finite difference with spacing 0.001. The leading Airy evaluation switches representations close to the turning point; a uniform mathematical fourth-order proof including that switch, derivative evaluation, and quadrature rounding has not been established.

## Benchmarks

| L | step | defect | Airy first peak | Bessel / independent reference | difference |
|---:|---:|---|---:|---:|---:|
| 48 | 0.1 | (1.2,0.35) | 26.4226271300 | 26.4226351832 | -8.05e-6 |
| 48 | 0.1 | (0.35,1.2) | 27.3115851764 | 27.3117272008 | -1.42e-4 |
| 48 | 0.1 | (0.5,0.2) | 26.4797582152 | 26.4797629780 | -4.76e-6 |
| 48 | 0.025 | (1.2,0.35) | 26.4226721991 | 26.4226802545 | -8.06e-6 |
| 48 | 0.025 | (0.35,1.2) | 27.3118119232 | 27.3119540093 | -1.42e-4 |
| 48 | 0.025 | (0.5,0.2) | 26.4797577865 | 26.4797625493 | -4.76e-6 |
| 64 | 0.1 | (1.2,0.35) | 36.4771871223 | 36.4772578994 | -7.08e-5 |
| 64 | 0.1 | (0.35,1.2) | 37.0505861145 | 37.0506657758 | -7.97e-5 |

The old second-order Airy h=0.1 errors were approximately -0.01949 and -0.09527, respectively, for the two original orientations. The new direct fourth-order model agrees with the independently written Bessel solver to about 1e-12 in peak time when Airy replacement is switched off. The Airy branch is not an isometry before polar normalization.

## Reproduction

Install numpy, scipy, pytest and matplotlib. From this directory, run:

OPENBLAS_NUM_THREADS=1 python -m pytest -q test_hermite_airy_solver.py

OPENBLAS_NUM_THREADS=1 python hermite_airy_solver.py

Depends on the existing sibling research modules: certificate, self_consistent_return and capacity_geometry. Five new local regression tests passed. The full downloadable research archive contains sampled trajectories, convergence plot and more detailed benchmarks.

## Prior art and open tasks

The Bessel–Airy asymptotics are established (NIST DLMF 10.20). Capizzi and Eisler, SciPost Physics 14, 070 (2023), already derive coherent defect-driven entanglement revivals. The new work is a rigorously documented numerical improvement in this model, not a confirmed new physical effect.

Next steps: a verified fourth-order quadrature remainder, time-uniform Airy error and first-peak interval certificate, and an analytical distinction from published coherent-revival results.
