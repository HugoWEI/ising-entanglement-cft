"""Exact finite-rank defect T-matrix for a finite open tight-binding chain.

THEORETICAL SCOPE
-----------------
This is a scattering-theory reformulation of an exactly solvable FREE
FERMION Hamiltonian, not a new many-body revival theory and not the Ising
Hamiltonian in src/ising.py.

The clean chain h0 has nearest-neighbor hopping -1, open boundaries.
A local defect V = u|a><a| + v|b><b| acts on the two middle sites.
For z off the spectrum, the exact resolvent is

  G(z) = (zI-h0-V)^(-1)
       = G0(z) + G0(z)[:,D] T_D(z) G0(z)[D,:],

  T_D(z) = [I - V_D G0(z)[D,D]]^(-1) V_D.

The 2x2 denominator determines the defect-induced bound-state poles.
The Neumann expansion T=V+V g V+V g V g V+... corresponds to repeated
defect scattering, but converges only if spectral_radius(V g)<1.

For bound energy E_b outside [-2,2], a normalized eigenvector is obtained
without diagonalizing the defective chain, via the null space of
[I-g_D(E_b) V_D] and G0(E_b)[:,D] V_D.
"""
from __future__ import annotations
import numpy as np


class TwoSiteDefectResolvent:
    def __init__(self, length: int, onsite=(1.2, .35)):
        if length < 4 or length % 2:
            raise ValueError("length must be even and at least four")
        self.length = int(length)
        self.onsite = tuple(float(x) for x in onsite)
        if len(self.onsite) != 2:
            raise ValueError("onsite must have exactly two entries")
        self.defect_sites = np.array([length // 2 - 1, length // 2],
                                     dtype=int)
        self.v_defect = np.diag(self.onsite)
        wave_n = np.arange(1, length + 1)
        q = np.pi * wave_n / (length + 1)
        j = np.arange(1, length + 1)[:, None]
        self.clean_energies = -2. * np.cos(q)
        self.clean_vectors = np.sqrt(2. / (length + 1)) * np.sin(j * q)

    def clean_green(self, z: complex, rows=None, cols=None) -> np.ndarray:
        """Clean resolvent elements from analytic sine modes."""
        n = self.length
        rows = np.arange(n, dtype=int) if rows is None else np.atleast_1d(rows)
        cols = np.arange(n, dtype=int) if cols is None else np.atleast_1d(cols)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        if np.any(rows < 0) or np.any(rows >= n):
            raise ValueError("rows outside chain")
        if np.any(cols < 0) or np.any(cols >= n):
            raise ValueError("cols outside chain")
        inv = 1.0 / (complex(z) - self.clean_energies)
        return (self.clean_vectors[rows, :] * inv) @ self.clean_vectors[cols, :].T

    def defect_green(self, z: complex):
        return self.clean_green(z, self.defect_sites, self.defect_sites)

    def secular_determinant(self, z: complex) -> complex:
        """det[I - g_D(z) V_D]. Roots outside [-2,2] are bound poles."""
        return complex(np.linalg.det(
            np.eye(2) - self.defect_green(z) @ self.v_defect))

    def t_matrix(self, z: complex, scattering_order: int | None = None):
        """Exact or truncated 2x2 multiple-scattering T matrix.

        scattering_order=M retains M terms starting with V (M>=1).
        Truncation is not reliable when rho(V*g_D)>=1 or close to 1.
        """
        g = self.defect_green(z)
        v = self.v_defect.astype(complex)
        if scattering_order is None:
            return np.linalg.solve(np.eye(2) - v @ g, v)
        if not isinstance(scattering_order, int) or scattering_order < 1:
            raise ValueError("scattering_order must be a positive integer")
        total = np.zeros((2,2), dtype=complex)
        term = v
        for _ in range(scattering_order):
            total += term
            term = term @ g @ v
        return total

    def scattering_radius(self, z: complex) -> float:
        return float(max(abs(np.linalg.eigvals(
            self.v_defect @ self.defect_green(z)))))

    def green(self, z: complex, rows=None, cols=None,
              scattering_order: int | None = None):
        """Full/restricted resolvent via the two-site defect T-matrix."""
        n = self.length
        rows = np.arange(n) if rows is None else np.atleast_1d(rows)
        cols = np.arange(n) if cols is None else np.atleast_1d(cols)
        d = self.defect_sites
        g0 = self.clean_green(z, rows, cols)
        l = self.clean_green(z, rows, d)
        r = self.clean_green(z, d, cols)
        return g0 + l @ self.t_matrix(z, scattering_order) @ r

    def bound_energies(self, n_grid: int = 500, tol: float = 1e-12):
        """Find all OUT-OF-BAND real secular roots by sign-bracket bisection.

        Intended for finite-support onsite impurities. Roots extremely close
        to the band edge can evade the finite scan; do not use as a certified
        general root finder without additional interval analysis.
        """
        if n_grid < 20:
            raise ValueError("n_grid must be >=20")
        maximum = 2. + max(8., sum(abs(x) for x in self.onsite)+4.)
        positive = np.geomspace(1e-9, maximum - 2., n_grid) + 2.
        segments = [-positive[::-1], positive]
        roots = []
        for xs in segments:
            f_prev = float(self.secular_determinant(float(xs[0])).real)
            x_prev = float(xs[0])
            for x in xs[1:]:
                x = float(x)
                fx = float(self.secular_determinant(x).real)
                if np.sign(f_prev) != np.sign(fx):
                    a,b = x_prev,x
                    fa = f_prev
                    for _ in range(90):
                        mid=(a+b)/2
                        fm=float(self.secular_determinant(mid).real)
                        if (b-a) < tol:
                            break
                        if np.sign(fm) == np.sign(fa):
                            a,fa=mid,fm
                        else:
                            b=mid
                    root=(a+b)/2
                    if not any(abs(root-r)<1e-7 for r in roots):
                        roots.append(float(root))
                x_prev,f_prev=x,fx
        return np.array(sorted(roots),dtype=float)

    def bound_wavefunction(self, energy: float):
        """Unit-normalized bound eigenstate inferred from the 2x2 pole.

        Only for real out-of-band eigenvalues solving the secular equation.
        """
        energy = float(energy)
        if abs(energy) <= 2.:
            raise ValueError("energy must be outside [-2,2]")
        gdd=self.defect_green(energy)
        a=np.eye(2)-gdd @ self.v_defect
        _,singular,vh=np.linalg.svd(a)
        if singular[-1] > 1e-5:
            raise ValueError("energy is not sufficiently close to a pole")
        q=vh[-1,:].conj()
        psi=self.clean_green(energy, cols=self.defect_sites) @ (self.v_defect @ q)
        norm=np.linalg.norm(psi)
        if norm < 1e-13:
            raise RuntimeError("bound wavefunction has near-zero norm")
        return psi/norm
