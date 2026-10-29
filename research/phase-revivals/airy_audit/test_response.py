"""Small synthetic checks of the physical trace derivative."""
import unittest,numpy as np
from run_projector_transfer import capacity_response, capacity_of_X, polar

class TraceTests(unittest.TestCase):
    def test_polar_path_response(self):
        rng=np.random.default_rng(23)
        z0=rng.normal(size=(12,6))+1j*rng.normal(size=(12,6))
        q=polar(z0)
        perturb=(rng.normal(size=(12,6))+1j*rng.normal(size=(12,6)))*1e-3
        res=capacity_response(q,q+perturb,81)
        expected=capacity_of_X(polar(q+perturb))-capacity_of_X(q)
        self.assertLess(abs(expected-res['integral']),1e-9)
    def test_polar_preserves_occupied_orthogonality(self):
        rng=np.random.default_rng(4)
        z=rng.normal(size=(16,8))+1j*rng.normal(size=(16,8))
        x=polar(z)
        self.assertLess(np.linalg.norm(x.conj().T@x-np.eye(8)),1e-12)
if __name__=='__main__':unittest.main()
