"""Tests for exact input-envelope inequalities, excluding numerical-rounding claims."""
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from continuous_peak_envelope import verify_window
from airy_aposteriori_certificate import local_analytic_weight_error,certify_no_earlier_peak
from peak_certificate import weights_derivatives

class EnvelopeTests(unittest.TestCase):
    def test_exact_weights(self):
        t=np.linspace(0,12,601); cap=np.cos(t/2)+.2*np.sin(t)
        for x in (1.,4.,6.,11.):
            w,d,dd=weights_derivatives(t,2,x)
            h=1e-4
            wp=weights_derivatives(t,2,x+h)[0]
            wm=weights_derivatives(t,2,x-h)[0]
            self.assertLess(np.max(abs(d-(wp-wm)/(2*h))),2e-9)
            self.assertLess(np.max(abs(dd-(wp-2*w+wm)/h**2)),3e-8)

    def test_perturbation_envelopes(self):
        ts=np.linspace(0,12,401);c=np.sin(ts/3)+.2*np.cos(ts/4)
        eps=.001+.0001*(1+np.cos(ts))
        mid=5.4; R=.09
        e1,e2=local_analytic_weight_error(ts,eps,2,mid-R,mid+R)
        rng=np.random.default_rng(14)
        pert=eps*rng.uniform(-1,1,len(ts))
        for x in np.linspace(mid-R,mid+R,101):
            _,d,dd=weights_derivatives(ts,2,x)
            self.assertLess(abs(d@pert),e1)
            self.assertLess(abs(dd@pert),e2)

    def test_peak_root_perturbation(self):
        ts=np.linspace(0,15,601)
        c=np.exp(-(ts-7)**2/5)
        eps=np.ones_like(ts)*1e-4
        root=brentq(lambda x:weights_derivatives(ts,2,x)[1]@c,6,8)
        result=verify_window(ts,c,eps,root,.1)
        self.assertTrue(result['conditional_isolated_peak'])
        rng=np.random.default_rng(9)
        for j in range(8):
            D=eps*rng.uniform(-1,1,len(ts))
            r=brentq(lambda x:weights_derivatives(ts,2,x)[1]@(c+D),6,8)
            self.assertLess(abs(r-root),result['conditional_peak_shift_upper'])

    def test_first_peak_current_check(self):
        ts=np.linspace(0,20,801);c=np.exp(-(ts-12)**2/5)
        eps=np.ones_like(ts)*1e-5
        check=certify_no_earlier_peak(ts,c,eps,7,11.95,min_width=.001)
        self.assertTrue(check['proved_positive_conditional'])

if __name__=='__main__':unittest.main()
