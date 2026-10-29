import unittest
import numpy as np
from adaptive_modulus import f,local_capacity_modulus,pstar
from run_report import evaluate

class TestAposteriori(unittest.TestCase):
    def test_stationary_points(self):
        self.assertLess(abs((1-2*pstar)*np.log((1-pstar)/pstar)-2),1e-10)
        self.assertLess(abs(f(pstar)-f(1-pstar)),1e-12)
    def test_local_modulus_is_upper_bound_for_random_perturbations(self):
        rng=np.random.default_rng(20261029)
        for delta in (1e-8,1e-5,0.01,0.1):
            for _ in range(100):
                p=rng.uniform(size=24)
                q=np.clip(p+rng.uniform(-delta,delta,size=24),0,1)
                bound=local_capacity_modulus(p,delta)
                actual=abs(float(np.sum(f(q)-f(p))))
                self.assertLessEqual(actual,bound+1e-12)
    def test_endpoint_inclusion(self):
        self.assertGreater(local_capacity_modulus(np.array([0.,1.,.5]),1e-4),0)
    def test_forward_first_peak(self):
        r=evaluate(1.2,.35)
        self.assertTrue(r['conditional_first_prominent'])
        self.assertLess(r['chosen_radius']['radius'],.005)
        self.assertGreater(r['prominence_lower'],.08)
    def test_reverse_first_peak(self):
        r=evaluate(.35,1.2)
        self.assertTrue(r['conditional_first_prominent'])
        self.assertLess(r['chosen_radius']['radius'],.011)
        self.assertGreater(r['prominence_lower'],.08)
if __name__=='__main__':unittest.main()
