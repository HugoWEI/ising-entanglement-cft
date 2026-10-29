"""A posteriori conditional peak certificate for archived AIRY surrogate.

The Airy curve is the *prediction*. An independently computed, source-only
(two-impurity-site) Hermite-Volterra solution serves as a validation witness,
not as an input to the Airy prediction. The final defective Hamiltonian is
never diagonalized. All numerical bounds are double-precision diagnostics,
NOT outward-rounded/interval certificates.
"""
import os, io, json, zipfile
import numpy as np
from scipy.optimize import brentq
from high_order_volterra import solve_hermite_volterra,cubic_drive_error_majorant
from phase_revival_helper import initial_occupied
from peak_certificate import capacity_polar,weights_derivatives,improved_gaussian_interval_bounds
from adaptive_modulus import local_capacity_modulus,f

ART='/mnt/data/airy_first_return_generator_bundle_2026-10-29.zip'
LOCAL_DATA=os.path.join(os.path.dirname(__file__),'data')
SIGMA=2.

def read_airypred(pair):
    entry=f'release/data/L48_u{pair[0]}_v{pair[1]}_dt0.025.npz'
    local=os.path.join(LOCAL_DATA,f'L48_u{pair[0]}_v{pair[1]}_dt0.025.npz')
    if os.path.exists(local):
        d=np.load(local)
        return d['t'],d['airy']
    with zipfile.ZipFile(ART) as z:
        d=np.load(io.BytesIO(z.read(entry)))
        return d['t'],d['uniform_airy_rank_None']

def nodewise_airy_uncertainty(t,surrogate,full_F,E):
    delta=4*E
    vals=[];err=[];local=[]
    for airy,Fr in zip(surrogate,full_F):
        ev,v=np.linalg.eigh(Fr.conj().T@Fr)
        X=Fr@((v*ev**(-.5))@v.conj().T)
        m=Fr.shape[1]
        p=np.linalg.eigvalsh(X[:m]@X[:m].conj().T)
        C=float(np.sum(f(p)))
        local_err=local_capacity_modulus(p,delta)
        vals.append(C);local.append(local_err);err.append(abs(C-airy)+local_err)
    return np.asarray(err),np.asarray(vals),np.asarray(local)

def local_analytic_weight_error(nodes,eps,sigma,a,b):
    mid=(a+b)/2;R=(b-a)/2
    d=np.abs(nodes-mid);dmin=np.maximum(d-R,0);dmax=d+R
    upper=np.exp(-dmin**2/(2*sigma**2));lower=np.exp(-dmax**2/(2*sigma**2))
    wmax=upper/lower.sum();aa=dmax/sigma**2
    wmid,_,_=weights_derivatives(nodes,sigma,mid)
    mu_mid=float(wmid@((nodes-mid)/sigma**2))
    A2=float(np.sum(wmax*aa**2))
    mu=abs(mu_mid)+R*max(A2,1/sigma**2)
    u=aa+mu
    mu2=float(np.sum(wmax*u*u))
    eta1=float(np.sum(eps*wmax*u))
    eta2=float(np.sum(eps*wmax*(u*u+mu2)))
    return eta1,eta2

def certify_peak(t,C,eps,win=(24,29),radii=(.06,.1,.2,.3,.5,1.0)):
    def deriv(x):
        w,dw,ddw=weights_derivatives(t,SIGMA,x)
        return float(dw@C),float(ddw@C)
    xs=np.linspace(*win,201)
    roots=[brentq(lambda x:deriv(x)[0],a,b) for a,b in zip(xs[:-1],xs[1:]) if deriv(a)[0]>0 and deriv(b)[0]<0]
    if not roots: return {'error':'no local max'}
    peak=roots[0]; results=[]
    for R in radii:
        a,b=peak-R,peak+R
        eta1,eta2=local_analytic_weight_error(t,eps,SIGMA,a,b)
        *_,B3,_=improved_gaussian_interval_bounds(t,C,SIGMA,a,b)
        grid=np.linspace(a,b,1001);values=np.array([deriv(x)[1] for x in grid])
        kappa=float(np.min(-values))-(grid[1]-grid[0])/2*B3
        left,right=deriv(a)[0],deriv(b)[0]
        good=bool(kappa>eta2 and left>eta1 and -right>eta1)
        results.append(dict(R=float(R),eta1=eta1,eta2=eta2,kappa=kappa,
            leftmargin=float(left-eta1),rightmargin=float(-right-eta1),
            good=good,radius=float(eta1/(kappa-eta2)) if good else None))
    return {'airy_peak':float(peak),'checks':results}

def run(pair=(1.2,.35),step=.00625):
    t_air,C_air=read_airypred(pair)
    t_full=np.arange(0,36+step/4,step)
    assert np.max(abs(t_full[::4]-t_air))<1e-10 if abs(step-.00625)<1e-10 else True
    Q=initial_occupied(48)
    F,Y,DY=solve_hermite_volterra(48,t_full,Q,pair)
    E=cubic_drive_error_majorant(48,t_full,Q,pair,F,Y,DY)
    stride=int(round(.025/step));
    if abs(stride*step-.025)>1e-10:raise ValueError('step must divide .025')
    samples=F[::stride]
    assert len(samples)==len(t_air)
    eps,cfull,local=nodewise_airy_uncertainty(t_air,C_air,samples,E)
    # Conditional a posteriori certificate uses the source-only full solver as witness.
    c=certify_peak(t_air,C_air,eps,win=(25,28.8))
    return {'pair':pair,'full_solver_dt':step,'source_error_bound':float(E),
        'source_only_airyprediction':True,'conditional_not_interval':True,
        'max_airy_vs_source_capacity_discrepancy':float(np.max(abs(C_air-cfull))),
        'peak_window_max_airy_vs_source_capacity_discrepancy':float(np.max(abs(C_air[(t_air>24)&(t_air<29)]-cfull[(t_air>24)&(t_air<29)]))),
        'max_source_spectral_capacity_error':float(np.max(local)),
        'peak_window_max_total_capacity_error':float(np.max(eps[(t_air>24)&(t_air<29)])),
        'peak':c,'t':t_air,'airy':C_air,'witness':cfull,'eps':eps}

if __name__=='__main__':
    out=[]
    for pair in [(1.2,.35),(.35,1.2)]:
        print('running',pair,flush=True)
        d=run(pair)
        np.savez_compressed(f'/mnt/data/airy_certificate_work/airy_witness_{pair[0]}_{pair[1]}.npz',t=d.pop('t'),airy=d.pop('airy'),witness=d.pop('witness'),eps=d.pop('eps'))
        print(json.dumps(d,indent=2),flush=True)
        out.append(d)
    with open('/mnt/data/airy_certificate_work/airy_aposteriori_results.json','w') as h:json.dump(out,h,indent=2)

def early_positive_cell(nodes,C,eps,sigma,a,b):
    mid=(a+b)/2;R=(b-a)/2
    d=np.abs(nodes-mid);dmin=np.maximum(d-R,0);dmax=d+R
    upper=np.exp(-dmin*dmin/(2*sigma*sigma));lower=np.exp(-dmax*dmax/(2*sigma*sigma))
    wm=upper/lower.sum();aa=dmax/sigma**2
    w,dw,ddw=weights_derivatives(nodes,sigma,mid)
    mu0=float(w@((nodes-mid)/sigma**2))
    mu=abs(mu0)+R*max(float(np.sum(wm*aa**2)),1/sigma**2)
    u=aa+mu; M2=float(np.sum(wm*u**2))
    center=float(dw@C)
    cref=float(w@C)
    curvature_bound=float(np.sum(wm*abs(C-cref)*(u**2+M2)))
    err_bound=float(np.sum(wm*eps*u))
    return center-R*curvature_bound-err_bound

def certify_no_earlier_peak(nodes,C,eps,start,stop,sigma=2.,min_width=.0005):
    """Analytic interval envelope for positivity of exact smoothed derivative.
    Conditions are evaluated with ordinary floating point, not verified rounding.
    """
    queue=[(start,stop)];checked=0;min_margin=float('inf');fails=[];max_depth=0
    while queue:
        a,b=queue.pop()
        margin=early_positive_cell(nodes,C,eps,sigma,a,b)
        if margin>0:
            min_margin=min(min_margin,margin);checked+=1
            continue
        if b-a<min_width:
            fails.append((a,b,margin));continue
        mid=(a+b)/2
        queue.extend([(a,mid),(mid,b)])
    return {'proved_positive_conditional':not fails,'accepted_intervals':checked,'worst_accepted_margin':float(min_margin),'failed_intervals':fails[:10]}
