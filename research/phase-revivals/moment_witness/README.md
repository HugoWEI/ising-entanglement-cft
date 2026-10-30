# Analytical cubic-Hermite moments for the clean witness

**30 October 2026; independently tested research prototype.** This speeds up reconstruction of the first-return Airy-generated impurity drive and removes the need for a second numerical convolution. It is NOT a formally certified first-peak theorem, and no novelty claim is made.

Let h0=S diag(lambda) S† be the clean open-chain Hamiltonian, with a two-site potential B V B†. On each time cell [t_j,t_j+h], write Y(t_j+uh)=sum_{k=0}^3 c[j,k]u^k for 0<=u<=1. The exactly reconstructed clean response to that *specified drive* satisfies

\[
\widehat F_{j+1}=e^{-i\Lambda h}\left(\widehat F_j-ih\sum_{k=0}^3\mu_k(i\Lambda h)S^\dagger B Vc_{j,k}\right),\quad
\mu_k(z)=\int_0^1 e^{zu}u^k du=\sum_{q=0}^{\infty}\frac{z^q}{q!(q+k+1)}.
\]

For a Taylor approximation through degree N, the exact-arithmetic tail obeys

\[
|R_{k,N}(z)| \le \frac{|z|^{N+1}}{(N+1)!(N+k+2)[1-|z|/(N+2)]},\quad |z|<N+2.
\]

Since the clean evolution is unitary, a bound on the global clean-response truncation error is h||V|| sum_{j,k} ||c[j,k]|| R_{k,N}(2h). This is not a bound on the error of the Airy drive itself or on floating-point rounding.

The mathematical n-point Gauss–Legendre convolution error on one Hermite cell has the bound

\[
E_{j,n}\leq \|V\| h^{2n+1}\frac{(n!)^4}{(2n+1)[(2n)!]^3}\sum_{k=0}^3 {2n\choose k}2^{2n-k} M_{k,j},
\]

where M_{k,j} is a valid upper bound on ||Y^{(k)}|| on that cell; see standard Gauss–Legendre remainder formulas.

## Numerical validation

L=48, staggered initial mass 2.5, h=0.05, T=30, Taylor N=14, and 12-point Gauss:

| Defect | Taylor tail bound | Gauss truncation bound | Observed max element difference | Speed-up |
|---|---:|---:|---:|---:|
| (1.20,0.35) | 1.43e-27 | 5.96e-58 | 9.16e-15 | 5.24x |
| (0.35,1.20) | 1.39e-27 | 3.93e-58 | 8.76e-15 | 4.46x |
| (0.50,0.20) | 6.15e-28 | 1.34e-58 | 9.07e-15 | 5.09x |

At T=36 the moment and Gauss first peaks agree to ~1.3e-12: 26.422677008246, 27.311932364024, and 26.479761445705. The new method is an O(n_time) recurrence instead of the previous O(n_time^2) history sum. Optional full real-space reconstruction still requires O(L^2 m) per step.

The remainder bounds assume exact inputs and arithmetic. Floating-point rounding dominates the measured discrepancies and is *not certified*. The previous continuous-cell physical-state bound remains conditional (~1.49e-5 forward and ~1.28e-5 reverse through t=30). The next task is outward-rounded sine-mode/eigenspace evaluation, residual norms and peak-curvature verification.

## Reproduction

Install numpy, scipy, pytest. From repo root set PYTHONPATH to the seven phase-revivals module directories (moment_witness, clean_witness, hermite_airy, self_consistent_return, certificate, certificate/src, capacity_geometry), then run pytest against test_moment_witness.py. The companion research ZIP contains more benchmarks and the full derivation.
