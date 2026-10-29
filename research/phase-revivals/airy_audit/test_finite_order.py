import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from mpmath import mp
from interval_series import bessel_interval,olver_interval
from run_finite_order_audit import comparison_one

mp.dps=160

def enclosed(ivval,truevalue):
    lower=mp.make_mpf(ivval._mpi_[0]);upper=mp.make_mpf(ivval._mpi_[1])
    return lower<=truevalue<=upper

class RemainderChecks(unittest.TestCase):
    def test_bessel_series(self):
        for n,t in [(50,'26.4'),(98,'26.4'),(145,'26.4'),(98,'49')]:
            self.assertTrue(enclosed(bessel_interval(n,t),mp.besselj(n,2*mp.mpf(t))))
    def test_airy_olver_at_turning(self):
        for n,t in [(50,'25'),(50,'26.4'),(98,'49')]:
            z=2*mp.mpf(t)/n
            if z==1: val=(mp.mpf(2)/n)**(mp.mpf(1)/3)*mp.airyai(0)
            elif z<1:
                q=mp.sqrt(1-z*z);e=mp.log((1+q)/z)-q
                zz=(mp.mpf('1.5')*e)**(mp.mpf(2)/3)
                val=(4*zz/(1-z*z))**mp.mpf('.25')/n**(mp.mpf(1)/3)*mp.airyai(n**(mp.mpf(2)/3)*zz)
            else:
                q=mp.sqrt(z*z-1);e=q-mp.atan(q)
                zz=-(mp.mpf('1.5')*e)**(mp.mpf(2)/3)
                val=(4*zz/(1-z*z))**mp.mpf('.25')/n**(mp.mpf(1)/3)*mp.airyai(n**(mp.mpf(2)/3)*zz)
            self.assertTrue(enclosed(olver_interval(n,t),val))
    def test_positive_airy_enclosure(self):
        for n,t in [(110,'26.42'),(120,'26.42'),(145,'26.42')]:
            z=2*mp.mpf(t)/n;q=mp.sqrt(1-z*z);e=mp.log((1+q)/z)-q
            zz=(mp.mpf('1.5')*e)**(mp.mpf(2)/3)
            val=(4*zz/(1-z*z))**mp.mpf('.25')/n**(mp.mpf(1)/3)*mp.airyai(n**(mp.mpf(2)/3)*zz)
            self.assertTrue(enclosed(olver_interval(n,t),val))
    def test_matrix_bound(self):
        for L,t in [(48,'26.4214'),(64,'36.4766')]:
            res=comparison_one(L,t)
            self.assertLess(res['measured_matrix_operator_norm'],res['matrix_frobenius_upper'])

if __name__=='__main__':unittest.main()
