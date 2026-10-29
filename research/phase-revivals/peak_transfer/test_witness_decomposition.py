import unittest
import numpy as np
from witness_decomposition import run,ROOT

class WitnessDecompositionTests(unittest.TestCase):
    def test_sum_of_peak_shifts(self):
        rows=run()
        self.assertEqual(len(rows),2)
        for r in rows:
            self.assertAlmostEqual(r['Airy_to_Bessel_shift']+r['Bessel_to_witness_shift'],
                                   r['Airy_to_witness_total_shift'],places=12)
            self.assertLess(r['Airy_to_Bessel_shift'],r['Airy_to_witness_total_shift']/10)
        self.assertGreater(rows[0]['fraction_abs_peak_gap_remaining_after_Bessel'],.99)
        self.assertGreater(rows[1]['fraction_abs_peak_gap_remaining_after_Bessel'],.97)

if __name__=='__main__':unittest.main()
