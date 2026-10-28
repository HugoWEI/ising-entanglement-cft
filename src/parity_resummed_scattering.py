"""Exact parity-channel resummation of reversed two-site impurity resolvents.

Hamiltonian h(delta)=h_sym+delta W, with central potentials
(w+delta/2,w-delta/2) and W=(|a><a|-|b><b|)/2.
The symmetric reference h_sym commutes with reflection.

Define normalized central even/odd states
  |+>=(|a>+|b>)/sqrt(2), |->=(|a>-|b>)/sqrt(2).
Then W=(|+><-|+|-><+|)/2. At z off the reference spectrum,
g_+(z)=<+|(zI-h_sym)^-1|+>, similarly g_-(z), with
<+|R0|->=0 by reflection symmetry.

The full 2x2 T-matrix in parity basis is exactly
    1 / D * [[delta^2*g_-/4, delta/2],
             [delta/2, delta^2*g_+/4]],
  D=1 - delta^2*g_+*g_-/4.

The *exact* odd-under-reversal resolvent contrast is
    R_delta - R_-delta =
      delta/D * (R0|+><-|R0 + R0|-><+|R0).

The numerator is first order, the denominator contains ALL
even powers of delta. No asymmetric Hamiltonian is diagonalized
or inverted to predict these Green functions. The exactness holds
for finite open chains and arbitrary delta when z avoids poles.

This is a symmetry-resolved finite-rank Dyson identity (known algebra),
not a new analytical prediction for entanglement-capacity peaks.
"""
from __future__ import annotations
import numpy as np

from scattering_phase import tight_binding_hamiltonian


class ParityResummedDefect:
    def __init__(self, length: int, mean_potential: float):
        if length < 4 or length % 2:
            raise ValueError("length must be even and >=4")
        self.length=int(length)
        self.mean_potential=float(mean_potential)
        reference=tight_binding_hamiltonian(
            length,(self.mean_potential,self.mean_potential))
        self.e,self.q=np.linalg.eigh(reference)
        a,b=length//2-1,length//2
        self.channels=np.zeros((length,2),dtype=float)
        self.channels[a,:]=1/np.sqrt(2)
        self.channels[b,0]=1/np.sqrt(2)
        self.channels[b,1]=-1/np.sqrt(2)
        self.channel_eigenbasis=self.q.T@self.channels

    def _reference_amplitudes(self,z: complex):
        z=complex(z)
        if np.min(abs(z-self.e))<1e-11:
            raise ValueError("z at or too close to reference spectrum")
        inv=1/(z-self.e)
        r0=(self.q*inv)@self.q.T
        x=(self.q*inv)@self.channel_eigenbasis
        gp=np.vdot(self.channel_eigenbasis[:,0],
                   inv*self.channel_eigenbasis[:,0])
        gm=np.vdot(self.channel_eigenbasis[:,1],
                   inv*self.channel_eigenbasis[:,1])
        off=np.dot(self.channel_eigenbasis[:,0],
                   inv*self.channel_eigenbasis[:,1])
        # In a reflection-symmetric finite chain this vanishes.
        if abs(off)>1e-9:
            raise RuntimeError("reference parity channels are not decoupled")
        return r0,x,complex(gp),complex(gm)

    def channel_data(self,z: complex,delta: float):
        """Return g+, g-, resummation denominator D, and scalar factor."""
        _,_,gp,gm=self._reference_amplitudes(z)
        delta=float(delta)
        denominator=1-(delta**2/4)*gp*gm
        if abs(denominator)<1e-12:
            raise ValueError("z at or too close to a dressed eigenvalue")
        return gp,gm,denominator,delta/denominator

    def contrast(self,z: complex,delta: float,linear=False):
        """R_delta(z)-R_-delta(z) from *one* symmetric reference only.

        linear=True drops the even-order denominator and gives the
        first-order Green-function response; this is NOT the linear
        susceptibility of nonlinear modular capacity.
        """
        _,x,gp,gm=self._reference_amplitudes(z)
        delta=float(delta)
        factor=delta if linear else delta/(1-(delta**2/4)*gp*gm)
        if not linear and abs(1-(delta**2/4)*gp*gm)<1e-12:
            raise ValueError("z near a dressed eigenvalue")
        return factor*(np.outer(x[:,0],x[:,1])+
                       np.outer(x[:,1],x[:,0]))

    def green(self,z: complex,delta: float,linear=False):
        """Full R_delta(z) predicted without inverting h(delta).

        linear=True returns R0+delta*R0 W R0, NOT a unitary
        time-domain propagator.
        """
        r0,x,gp,gm=self._reference_amplitudes(z)
        d=float(delta)
        t=np.array([[d*d*gm/4,d/2],[d/2,d*d*gp/4]],
                   dtype=complex)
        denom=1-d*d*gp*gm/4
        if linear:
            t=np.array([[0,d/2],[d/2,0]],dtype=complex)
            denom=1
        if abs(denom)<1e-12:
            raise ValueError("z near a dressed eigenvalue")
        return r0+(x@t@x.T)/denom


    def scalar_local_channels(self,z: float):
        """g_plus(z),g_minus(z) using only symmetric spectral weights."""
        z=float(z)
        if np.min(abs(z-self.e))<1e-12:
            raise ValueError("energy too close to reference eigenvalue")
        weights=self.channel_eigenbasis**2
        inverse=1/(z-self.e)
        return (float(weights[:,0]@inverse),
                float(weights[:,1]@inverse))

    def largest_eigenenergy(self,delta: float,tol: float=5e-13) -> float:
        """Highest dressed eigenvalue from scalar parity-channel secular root.

        Unique root of D(E)=1-delta^2*g_plus*g_minus/4 to the right
        of the largest symmetric eigenvalue, for nonzero delta.
        This is a *bound* state only if this eigenenergy exceeds 2.
        No asymmetric Hamiltonian diagonalization is used.
        """
        delta=abs(float(delta))
        if not np.isfinite(delta) or tol<=0:
            raise ValueError("invalid delta or tolerance")
        emax=float(self.e[-1])
        if delta==0:
            return emax
        low=emax+1e-10
        gp,gm=self.scalar_local_channels(low)
        if 1-.25*delta**2*gp*gm>=0:
            raise ValueError("asymmetry shift below scalar root resolution")
        high=emax+max(4.,2*delta+2)
        for _ in range(70):
            gp,gm=self.scalar_local_channels(high)
            if 1-.25*delta**2*gp*gm>0:
                break
            high=emax+2*(high-emax)
        else:
            raise RuntimeError("failed to bracket largest dressed eigenvalue")
        for _ in range(100):
            mid=(low+high)/2
            if high-low<tol:
                break
            gp,gm=self.scalar_local_channels(mid)
            if 1-.25*delta**2*gp*gm<0:
                low=mid
            else:
                high=mid
        return float((low+high)/2)

    def top_eigenenergy_quadratic(self,delta: float) -> float:
        """Leading even-in-delta Rayleigh-Schrodinger energy correction."""
        delta=float(delta)
        weights=self.channel_eigenbasis**2
        top=int(np.argmax(weights[-1]))
        opposite=1-top
        coefficient=.25*float(weights[-1,top])*float(
            np.sum(weights[:-1,opposite]/(self.e[-1]-self.e[:-1])))
        return float(self.e[-1]+coefficient*delta**2)

    def parity_diagnostic(self):
        """Symmetric reference must split into exact reflection sectors."""
        h=tight_binding_hamiltonian(
            self.length,(self.mean_potential,self.mean_potential))
        return float(np.linalg.norm(h-h[::-1,::-1]))
