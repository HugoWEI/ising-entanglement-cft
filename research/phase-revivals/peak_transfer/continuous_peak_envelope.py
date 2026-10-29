"""Conditional continuous-parameter peak enclosure of DISCRETE Gaussian smoother.

No defective final-Hamiltonian data used; compares full Bessel first return and
leading Olver Airy first return on a common numerical Volterra baseline.
Nodewise eps are double-precision diagnostics plus a 5e-12 numerical floor,
not rigorous outward-rounded eigenvalue and special-function enclosures.
"""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
CERT=(PROJECT/'certificate') if (PROJECT/'certificate').exists() else Path('/mnt/data/airy_certificate_work')
sys.path.insert(0,str(CERT))
from airy_aposteriori_certificate import local_analytic_weight_error,certify_no_earlier_peak
from peak_certificate import weights_derivatives,improved_gaussian_interval_bounds

SIGMA=2.0

def verify_window(nodes,airy,eps,peak,radius,n=1001):
    a=peak-radius;b=peak+radius
    eta1,eta2=local_analytic_weight_error(nodes,eps,SIGMA,a,b)
    m1,m2,m3,B3,mu=improved_gaussian_interval_bounds(nodes,airy,SIGMA,a,b)
    xx=np.linspace(a,b,n)
    curv=np.array([weights_derivatives(nodes,SIGMA,x)[2]@airy for x in xx])
    step=(b-a)/(n-1)
    kappa=float(np.min(-curv))-(step/2)*B3
    left=float(weights_derivatives(nodes,SIGMA,a)[1]@airy)
    right=float(weights_derivatives(nodes,SIGMA,b)[1]@airy)
    good=kappa>eta2 and left>eta1 and -right>eta1
    return dict(radius=radius,eta1_continuous_majorant=float(eta1),eta2_continuous_majorant=float(eta2),
                kappa_continuous_lower=float(kappa),third_derivative_continuous_majorant=float(B3),
                left_current_sign_margin=float(left-eta1),right_current_sign_margin=float(-right-eta1),
                conditional_isolated_peak=bool(good),
                conditional_peak_shift_upper=float(eta1/(kappa-eta2)) if good else None)

def run():
    rows=json.loads((ROOT/'RESULTS.json').read_text())
    for row in rows:
        pair=row['defect']; data=np.load(ROOT/f'sampled_{pair[0]}_{pair[1]}.npz')
        # Technical floating stability floor based on observed early-time eigensolver noise.
        # Not a mathematically validated rounding/error estimate.
        eps=data['eps']+5e-12
        outputs=[verify_window(data['t'],data['airy'],eps,row['peak_airy'],r) for r in (.05,.1,.2,.4,.8)]
        row['continuous_enclosures']=outputs
        row['first_peak_check_from_t19_2']=certify_no_earlier_peak(data['t'],data['airy'],eps,19.2,row['peak_airy']-.05,min_width=.001)
        observed=np.max(np.abs(data['airy']-data['bessel'])-eps)
        row['max_observed_violation_with_numerical_floor']=float(observed)
        row['floating_slack_added_to_all_capacity_nodes']=5e-12
        print(pair, 'observed peak shift',row['peak_shift_bessel_minus_airy'],flush=True)
        for o in outputs:
            print('   ',o['radius'], 'passes',o['conditional_isolated_peak'],'bound',o['conditional_peak_shift_upper'],
                  'kappa',o['kappa_continuous_lower'],'eta1',o['eta1_continuous_majorant'],flush=True)
    (ROOT/'CONTINUOUS_RESULTS.json').write_text(json.dumps(rows,indent=2))
    return rows
if __name__=='__main__':run()
