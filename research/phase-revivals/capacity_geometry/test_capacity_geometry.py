import unittest
import numpy as np
from capacity_geometry import polar,capacity,gradient_norm,derivative,capacity_bound,horizontal
from verify_constant import verify

class GeometryTests(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(capacity(np.array([[0.],[1.]])),0.,places=14)
        x=np.array([[1e-8],[np.sqrt(1.-1e-16)]])
        h=np.array([[np.sqrt(1.-1e-16)],[-1e-8]])
        self.assertLess(abs(derivative(x,h)),1e-4)

    def test_analytic_derivative(self):
        rng=np.random.default_rng(37)
        for L in (8,16,32):
            m=L//2
            X=polar(rng.normal(size=(L,m))+1j*rng.normal(size=(L,m)))
            D=rng.normal(size=X.shape)+1j*rng.normal(size=X.shape)
            H=D-X@(X.conj().T@D)
            eps=1e-5
            fd=(capacity(polar(X+eps*H))-capacity(polar(X-eps*H)))/(2*eps)
            self.assertAlmostEqual(fd,derivative(X,H),delta=1e-7)
            self.assertLessEqual(abs(fd),gradient_norm(X)*np.linalg.norm(H,'fro')+1e-8)

    def test_path_bound(self):
        rng=np.random.default_rng(14); L=32; m=L//2
        Z=polar(rng.normal(size=(L,m))); D=0.01*rng.normal(size=Z.shape)
        s=1-np.linalg.norm(D,2)
        self.assertLess(abs(capacity(polar(Z+D))-capacity(Z)),
                        capacity_bound(np.linalg.norm(D,'fro'),m,s))
        X,H=horizontal(Z+.4*D,D)
        eps=1e-5
        fd=(capacity(polar(Z+(.4+eps)*D))-capacity(polar(Z+(.4-eps)*D)))/(2*eps)
        self.assertAlmostEqual(fd,derivative(X,H),delta=1e-7)

    def test_interval_constant(self):
        self.assertLess(verify()['max_interval_upper'],1.22)

if __name__=='__main__': unittest.main()
