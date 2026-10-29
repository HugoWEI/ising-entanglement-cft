"""Diagnostic decomposition of Airy/ Bessel / Hermite-source-only peak shifts.

The witness NPZ arrays were computed in the previous certification project; their
peak times are not exact-Hamiltonian values or formal interval certificates.
"""
import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent
CERT=ROOT.parent/'certificate' if (ROOT.parent/'certificate').exists() else Path('/mnt/data/airy_certificate_work')
sys.path.insert(0,str(CERT))
from peak_certificate import weights_derivatives

def peak(t,c):
    a,b=24.,29.
    return float(brentq(lambda x:weights_derivatives(t,2.,x)[1]@c,a,b,xtol=1e-12))

def run():
    rows=[]
    for u,v in ((1.2,.35),(.35,1.2)):
        d=np.load(ROOT/f'sampled_{u}_{v}.npz');w=np.load(CERT/f'airy_witness_{u}_{v}.npz')
        assert np.max(abs(w['t']-d['t']))<1e-12
        assert np.max(abs(w['airy']-d['airy']))<5e-9
        pA=peak(d['t'],d['airy']);pB=peak(d['t'],d['bessel']);pW=peak(d['t'],w['witness'])
        rows.append({'defect':[u,v],'Airy_peak':pA,'Bessel_peak':pB,'source_only_Hermite_witness_peak':pW,
                     'Airy_to_Bessel_shift':pB-pA,'Bessel_to_witness_shift':pW-pB,
                     'Airy_to_witness_total_shift':pW-pA,
                     'fraction_abs_peak_gap_remaining_after_Bessel':abs(pW-pB)/abs(pW-pA),
                     'interpretation':'Benchmark decomposition only; Hermite witness supplies residual control but ordinary floating point here. Does not certify either actual peak.'})
    (ROOT/'WITNESS_DECOMPOSITION.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
    return rows
if __name__=='__main__':run()
