"""Reproducible sanity checks for exact-arithmetic identities and numerical code."""
import os, sys, unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'src'))
from defect_residual import clean_modes,phi12,defect_drive_response,hermite_residual_integral_majorant
from published_baseline import capizzi_eisler_capacity,mode_capacity
from image_tail_certificate import image_tail_op_bound
import volterra_scattering_revival as vol


class CertificationChecks(unittest.TestCase):
    def test_clean_eigensystem(self):
        L=12
        e,S=clean_modes(L)
        H=np.diag(-np.ones(L-1),-1)+np.diag(-np.ones(L-1),1)
        for t in [0,.5,3]:
            self.assertLess(np.linalg.norm(S@(np.exp(-1j*e*t)[:,None]*S.T)-expm(-1j*H*t)),1e-12)

    def test_phi_moments(self):
        for e in [-1.8,-.1,0,1.2]:
            h=.2
            p1,p2=phi12(np.array([-1j*e*h]))
            c1=complex(quad(lambda t:np.cos(-e*(h-t)),0,h)[0],quad(lambda t:np.sin(-e*(h-t)),0,h)[0])
            c2=complex(quad(lambda t:t*np.cos(-e*(h-t)),0,h)[0],quad(lambda t:t*np.sin(-e*(h-t)),0,h)[0])
            self.assertLess(abs(p1[0]*h-c1),1e-13)
            self.assertLess(abs(p2[0]*h*h-c2),1e-13)

    def test_piecewise_linear_forcing_zero(self):
        L=8;t=np.linspace(0,2,11)
        Q=np.eye(L)[:,:4]
        F,r=defect_drive_response(L,t,Q,(.3,-.2),np.zeros((11,2,4)))
        e,S=clean_modes(L)
        for ti,f in zip(t,F):
            self.assertLess(np.linalg.norm(f-S@(np.exp(-1j*e*ti)[:,None]*(S.T@Q))),1e-12)

    def test_residual_majorizes_exact_error(self):
        L=12;dt=.15;T=3.;t=np.arange(0,T+dt/4,dt)
        h0=np.diag(-np.ones(L-1),-1)+np.diag(-np.ones(L-1),1)
        _,q=np.linalg.eigh(h0+np.diag(2.5*(-1)**np.arange(L)));Q=q[:,:L//2]
        pair=(.7,.3)
        Z,Y=vol.defect_response(L,t,*pair,Q,shells=1,return_full=True,return_defect=True)
        F,r=defect_drive_response(L,t,Q,pair,Y)
        maj=hermite_residual_integral_majorant(L,t,Q,pair,Y,F)
        H=h0.copy();H[L//2-1,L//2-1]+=pair[0];H[L//2,L//2]+=pair[1]
        e,U=np.linalg.eigh(H);M=U.T@Q
        actual=max(np.linalg.norm(f-U@(np.exp(-1j*e*ti)[:,None]*M),2) for f,ti in zip(F,t))
        self.assertLess(actual,maj+1e-9)
        self.assertGreater(maj,0)

    def test_conformal_baseline_limits(self):
        t=np.arange(0,10.1,.5)
        self.assertLess(np.max(abs(capizzi_eisler_capacity(48,2.5,t,lam=0))),1e-10)
        self.assertGreater(capizzi_eisler_capacity(48,2.5,t,lam=1)[5],0)
        # C=-2 d S_n/dn at n=1, verifying the differentiation used in the baseline
        p=.18
        def s(n):return np.log(p**n+(1-p)**n)/(1-n)
        eps=1e-4
        derivative=(s(1+eps)-s(1-eps))/(2*eps)
        self.assertAlmostEqual(-2*derivative,mode_capacity(p),places=6)

    def test_image_tail_source(self):
        self.assertLess(image_tail_op_bound(48,1,30,3),1e-10)

if __name__=='__main__':unittest.main()

# Additional fourth-order checks intentionally remain in a separate test class,
# so the original six tests can be audited independently.
class HighOrderChecks(unittest.TestCase):
    def test_cubic_hermite_weights_and_node_consistency(self):
        from high_order_volterra import hermite_weights,solve_hermite_volterra,cubic_drive_error_majorant
        z=np.array([0j])
        w=hermite_weights(z)
        np.testing.assert_allclose([float(x[0].real) for x in w],[.5,1/12,.5,-1/12],atol=1e-13)
        L=12;t=np.arange(0,3.01,.15)
        h0=np.diag(-np.ones(L-1),-1)+np.diag(-np.ones(L-1),1)
        _,q=np.linalg.eigh(h0+np.diag(2.5*(-1.)**np.arange(L)));Q=q[:,:6]
        pair=(.7,.3)
        F,Y,DY,check=solve_hermite_volterra(L,t,Q,pair,True)
        self.assertLess(check['node_r'],1e-12)
        self.assertLess(check['node_dr'],1e-12)
        bound=cubic_drive_error_majorant(L,t,Q,pair,F,Y,DY)
        self.assertGreater(bound,0)
        H=h0.copy();H[5,5]+=.7;H[6,6]+=.3
        e,U=np.linalg.eigh(H);M=U.T@Q
        actual=max(np.linalg.norm(f-U@(np.exp(-1j*e*tt)[:,None]*M),2) for f,tt in zip(F,t))
        self.assertLess(actual,bound)

    def test_interval_gaussian_bounds(self):
        from peak_certificate import gaussian_interval_weight_majorants,weights_derivatives,improved_gaussian_interval_bounds,scalar_modulus_safe
        nodes=np.linspace(0,40,401)
        m1,m2,m3=gaussian_interval_weight_majorants(nodes,2,26.3,26.5)
        caps=6+np.cos(nodes/2)
        M1,M2,M3,B3,meanbound=improved_gaussian_interval_bounds(nodes,caps,2,26.3,26.5)
        self.assertGreater(M1,0)
        self.assertGreater(M2,0)
        self.assertGreater(B3,0)
        for t in [26.3,26.4,26.5]:
            w,dw,ddw=weights_derivatives(nodes,2,t)
            self.assertLessEqual(np.sum(abs(dw)),M1+1e-12)
            self.assertLessEqual(np.sum(abs(ddw)),2*M2+1e-12)
        self.assertGreater(scalar_modulus_safe(1e-7),0)
