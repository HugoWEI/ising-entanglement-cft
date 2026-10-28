"""Exact all-asymmetry capacity from a parity-resummed complex contour.

An EXACT formal scattering/Gaussian formula, numerically quadrature.
NOT a new closed-form or efficient physical law for first-revival peaks.

R_delta(z)=R_sym(z)+X(z) T_delta(z) X(z)^T,
X(z)=R_sym(z)[|+>,|->],
T_delta=1/D * [[delta^2*g_minus/4, delta/2],
               [delta/2,delta^2*g_plus/4]],
D(z)=1-delta^2*g_plus*g_minus/4.

The exact time-domain propagator is
U_delta(t)=U_sym(t)+(1/(2*pi*i)) contour_C
           exp(-i*z*t) X(z) T_delta(z) X(z)^T dz.

We apply the contour correction directly to occupied Slater orbitals,
rather than constructing the full propagator, and compute modular
capacity from eigenvalues of the restricted correlation matrix.

Complex-energy quadrature has cancellation issues at long times.
No unitarity repair or occupation-eigenvalue clipping is used to hide
large numerical errors; both are checked explicitly.
"""
from __future__ import annotations
import numpy as np
from scattering_phase import entropy_capacity
from parity_resummed_scattering import ParityResummedDefect


class ContourCapacity:
    def __init__(self,length,mean_potential,occupied_modes):
        self.reference=ParityResummedDefect(length,mean_potential)
        self.length=int(length)
        f0=np.asarray(occupied_modes,dtype=complex)
        if f0.shape!=(length,length//2):
            raise ValueError("initial orbitals must have shape L x (L/2)")
        if not np.allclose(f0.conj().T@f0,np.eye(length//2),atol=1e-9):
            raise ValueError("initial Slater orbitals must be orthonormal")
        self.f0=f0
        self.mode_occupations=self.reference.q.T@f0

    def propagate(self,delta,times,quadrature_nodes=768,
                  real_semiaxis=None,imag_semiaxis=None):
        """Predict U_delta(t)F0 without asymmetric h(delta) eigenvectors."""
        r=self.reference
        d=float(delta)
        t=np.asarray(times,dtype=float)
        if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.min(t)<0:
            raise ValueError("times must be a nonempty finite 1d array >=0")
        if not np.isfinite(d):
            raise ValueError("delta must be finite")
        if not isinstance(quadrature_nodes,int) or quadrature_nodes<64:
            raise ValueError("quadrature_nodes must be >=64")
        radius_min=float(np.max(abs(r.e))+abs(d)/2+.30)
        a=(max(3.05,radius_min) if real_semiaxis is None
           else float(real_semiaxis))
        if a<radius_min:
            raise ValueError("real semiaxis must enclose dressed spectrum")
        b=(min(.18,8/max(float(np.max(t)),1.))
           if imag_semiaxis is None else float(imag_semiaxis))
        if b<=0 or b>=a:
            raise ValueError("invalid imaginary semiaxis")

        theta=2*np.pi*np.arange(quadrature_nodes)/quadrature_nodes
        z=a*np.cos(theta)+1j*b*np.sin(theta)
        derivative=-a*np.sin(theta)+1j*b*np.cos(theta)
        inv=1/(z[:,None]-r.e[None,:])
        projected=r.channel_eigenbasis
        x=np.einsum("ik,zk,kp->zip",r.q,inv,projected,optimize=True)
        gp=np.sum(inv*projected[:,0]**2,axis=1)
        gm=np.sum(inv*projected[:,1]**2,axis=1)
        denominator=1-d*d*gp*gm/4
        if np.min(abs(denominator))<1e-10:
            raise ValueError("contour too close to dressed pole")
        tmatrix=np.empty((quadrature_nodes,2,2),dtype=complex)
        tmatrix[:,0,0]=(d*d/4)*gm/denominator
        tmatrix[:,0,1]=tmatrix[:,1,0]=(d/2)/denominator
        tmatrix[:,1,1]=(d*d/4)*gp/denominator

        # x^T F0: [nodes, parity_channel, occupied_orbital].
        overlap=np.einsum("zip,im->zpm",x,self.f0,optimize=True)
        left=np.einsum("zip,zpq->ziq",x,tmatrix,optimize=True)
        correction=np.einsum("ziq,zqm->zim",left,overlap,optimize=True)
        weights=(np.exp(-1j*t[:,None]*z[None,:])*
                 (derivative[None,:]/(1j*quadrature_nodes)))
        delta_f=np.einsum("tz,zim->tim",weights,correction,optimize=True)

        phase=np.exp(-1j*t[:,None]*r.e[None,:])
        clean=np.einsum("ik,tk,km->tim",r.q,phase,self.mode_occupations,
                        optimize=True)
        return clean+delta_f

    def capacity_trajectory(self,delta,times,quadrature_nodes=768,
                            real_semiaxis=None,imag_semiaxis=None,
                            physicality_tolerance=2e-6):
        """Return exact-formula quadrature capacity plus physicality tests.

        Values are [entropy,capacity]. If finite quadrature violates
        isometry or fermionic occupations by more than tolerance,
        raise rather than projecting the state or silently clipping.
        """
        f=self.propagate(delta,times,quadrature_nodes,
                         real_semiaxis,imag_semiaxis)
        n=self.length
        result=np.empty((len(f),2))
        errors=np.empty(len(f))
        occup=np.empty((len(f),2))
        for i,modes in enumerate(f):
            errors[i]=np.linalg.norm(modes.conj().T@modes-np.eye(n//2))
            block=modes[:n//2]
            p=np.linalg.eigvalsh(block@block.conj().T)
            occup[i]=[np.min(p),np.max(p)]
            if errors[i]>physicality_tolerance:
                raise RuntimeError("contour quadrature unitarity tolerance exceeded")
            if p.min() < -physicality_tolerance or p.max()>1+physicality_tolerance:
                raise RuntimeError("contour produced nonphysical occupations")
            result[i]=entropy_capacity(p)
        return {
            "times":np.asarray(times,float),
            "values":result,
            "max_isometry_error":float(np.max(errors)),
            "max_occupation":float(np.max(occup[:,1])),
            "min_occupation":float(np.min(occup[:,0])),
        }
