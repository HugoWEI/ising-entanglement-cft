"""Verify peak neighborhoods and first-prominent status using stored a posteriori witnesses.
No diagonalization of the defective Hamiltonian; no exact-capacity data.
Numerical majorants are conditional exact-arithmetic, NOT outward-rounded.
"""
import os,json
import numpy as np
from peak_certificate import weights_derivatives
from airy_aposteriori_certificate import certify_peak,certify_no_earlier_peak

HERE=os.path.dirname(__file__)

def evaluate(a,b):
    data=np.load(os.path.join(HERE,f'airy_witness_{a}_{b}.npz'))
    t,C,eps=(data[k] for k in ['t','airy','eps'])
    pk=certify_peak(t,C,eps,win=(25,28.8))
    checks=[r for r in pk['checks'] if r['good']]
    r=min(checks,key=lambda q:q['radius'])
    tp=pk['airy_peak']; R=r['R']; error=r['radius']
    early=certify_no_earlier_peak(t,C,eps,.4*48,tp-R)
    # Left and right witness points are within 0.15*L of actual peak,
    # because certified |tpeak-tp| <0.02 and chosen distance=7.15<7.2.
    xpts=[tp,tp-7.15,tp+7.15]
    vals=[]
    for x in xpts:
        w,_,_=weights_derivatives(t,2.,x)
        vals.append({'t':float(x),'airy_smooth':float(w@C),'error_upper':float(w@eps)})
    lower=min(vals[0]['airy_smooth']-vals[0]['error_upper']-vals[k]['airy_smooth']-vals[k]['error_upper'] for k in [1,2])
    return {'pair':[a,b], 'airy_peak':tp,'chosen_radius':r,'earlier_derivative_positive':early,
            'prominence_witnesses':vals,'prominence_lower':float(lower),
            'conditional_first_prominent':bool(r['good'] and early['proved_positive_conditional'] and lower>0.08 and error<0.05)}

if __name__=='__main__':
    results=[evaluate(1.2,.35),evaluate(.35,1.2)]
    out={'method':'conditional a posteriori validation of archived Airy first-return predictor',
        'model':'L48,m2.5, gaussian normalized finite-grid smoother sigma2, t grid .025 to T36',
        'rounding':'double precision; NOT formal interval arithmetic',
        'witness':'source-only fourth-order two-site Hermite Volterra h=.00625',
        'uses_exact_defective_H':False,'results':results}
    with open(os.path.join(HERE,'FINAL_RESULTS.json'),'w') as f:json.dump(out,f,indent=2)
    for d in results: print(json.dumps(d,indent=2))
