"""Conditional continuous-in-t peak enclosures for a *discrete Gaussian smoother*.

The source-only amplitude majorant and endpoint-safe scalar modulus give exact-
arithmetic error bounds. Numerical evaluation of inputs is still double precision,
NOT interval-arithmetic verification. This defines an operational smoothed peak:
   Cbar(t)=sum_j exp(-(t-t_j)^2/(2 sigma^2)) C(t_j) / sum_j weights.
All comparisons refer to this explicit Gaussian-kernel smoother, not directly
to the continuous Gaussian convolution of C(t).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from high_order_volterra import solve_hermite_volterra,cubic_drive_error_majorant
from phase_revival_helper import initial_occupied


def scalar_modulus_safe(delta):
    """Upper bound on omega_f(delta) for 0<delta<0.06, exact arithmetic.
    Either x,y in [delta,1-delta], derivative bound; or both close to
    one endpoint, in [0,2delta] or [1-2delta,1]. For small 2delta<e^-2,
    f(x)<=x log^2(1/x) increasing to 2delta.
    """
    assert 0<delta<.06
    M=np.log(1/delta)
    interior=delta*(M*M+2*M)
    endpoint=2*delta*np.log(1/(2*delta))**2
    return max(interior,endpoint)


def capacity_polar(F):
    # F is L x M, with full column rank, at a single time.
    ev,v=np.linalg.eigh(F.conj().T@F)
    X=F@((v*ev**(-.5))@v.conj().T)
    p=np.linalg.eigvalsh(X[:X.shape[1]]@X[:X.shape[1]].conj().T)
    pc=p[(p>1e-12)&(p<1-1e-12)]
    return float(np.sum(pc*(1-pc)*np.log((1-pc)/pc)**2))


def weights_derivatives(nodes,sigma,t):
    a=-(nodes-t)**2/(2*sigma**2)
    w=np.exp(a-np.max(a));w=w/w.sum()
    vel=(nodes-t)/sigma**2
    mv=np.sum(w*vel)
    v=vel-mv
    var=np.sum(w*v*v)
    return w, w*v, w*(v*v-var)


def candidate_certificate(L=48,pair=(1.2,.35),T=40,dt=.00625,sigma=2.,window=(24.5,29.)):
    t=np.arange(0,T+dt/4,dt)
    Q=initial_occupied(L)
    F,Y,DY,_=solve_hermite_volterra(L,t,Q,pair,True)
    E=cubic_drive_error_majorant(L,t,Q,pair,F,Y,DY)
    delta=4*E
    cap_err=(L//2)*scalar_modulus_safe(delta)
    capacities=np.array([capacity_polar(x) for x in F])
    # root via fine scan: first positive to negative current in requested window.
    def derivs(x):
        w,dw,ddw=weights_derivatives(t,sigma,x)
        return float(dw@capacities),float(ddw@capacities),float(np.sum(abs(dw))*cap_err),float(np.sum(abs(ddw))*cap_err)
    scan=np.linspace(*window,100)
    roots=[]
    for a,b in zip(scan[:-1],scan[1:]):
        if derivs(a)[0]>0 and derivs(b)[0]<0:
            roots.append(brentq(lambda x:derivs(x)[0],a,b,xtol=1e-11))
    if not roots:return dict(no_root=True,pair=pair,dt=dt,source_E=E,delta_P=delta,capacity_bound=cap_err)
    peak=roots[0]
    out={'pair':pair,'dt':dt,'T':T,'peak':peak,'source_E':float(E),'projector_bound':float(delta),
         'capacity_bound':float(cap_err),'root_curvature':derivs(peak)[1]}
    for radius in [.05,.10,.2,.4,.8,1.]:
        grid=np.linspace(peak-radius,peak+radius,61)
        curvature=np.array([derivs(x)[1] for x in grid])
        eta1=max(derivs(x)[2] for x in grid)
        eta2=max(derivs(x)[3] for x in grid)
        kap=-max(curvature)
        # Bound is valid *conditional* on certified sup curvature and inputs;
        # sampled grid extrema are not an interval enclosure.
        root_limit=eta1/(kap-eta2) if kap>eta2 else float('inf')
        left,right=derivs(peak-radius)[0],derivs(peak+radius)[0]
        bracket=left>derivs(peak-radius)[2] and -right>derivs(peak+radius)[2]
        out[f'window_{radius}']={'kappa_sampled':float(kap),'eta1_sampled':float(eta1),'eta2_sampled':float(eta2),
                                  'root_radius_diagnostic':float(root_limit),'robust_bracket_sampled':bool(bracket),
                                  'curvature_sign_robust_sampled':bool(kap>eta2)}
    return out

if __name__=='__main__':
    import json
    for pair in [(1.2,.35),(.35,1.2)]:
        x=candidate_certificate(pair=pair)
        print(json.dumps(x,indent=2),flush=True)


def gaussian_interval_weight_majorants(nodes,sigma,a,b):
    """Continuous-in-t analytic upper bounds for normalized Gaussian weights.

    For every t in [a,b], use Gaussian numerator envelope and denominator
    lower envelope to bound moments of a_j=(t_j-t)/sigma**2 about its weighted mean.
    Numerical evaluation uses floats, NOT outward-rounded intervals.
    """
    mid=(a+b)/2; R=(b-a)/2
    d=np.abs(nodes-mid)
    dmin=np.maximum(0,d-R)
    dmax=d+R
    top=np.exp(-dmin**2/(2*sigma**2))
    bottom=np.exp(-dmax**2/(2*sigma**2))
    zmin=bottom.sum()
    wm=top/zmin
    aa=dmax/sigma**2
    A1=np.sum(wm*aa)
    mom=[float(np.sum(wm*(aa+A1)**k)) for k in (1,2,3)]
    return mom


def strict_interval_diagnostic(nodes,cap,sigma,peak,radius,epsC,ngrids=1001):
    """Continuous *analytic* majorants except floating evaluation/roundoff.

    Does not certify the global 'first prominent' peak, only a local maximum
    in [peak-radius,peak+radius] of the discrete Gaussian-kernel smoother.
    """
    a,b=peak-radius,peak+radius
    m1,m2,m3=gaussian_interval_weight_majorants(nodes,sigma,a,b)
    eta1=epsC*m1;eta2=2*epsC*m2
    M3=float(np.max(abs(cap)))*(2*m3+3*m2*m1)
    xx=np.linspace(a,b,ngrids)
    curv=np.array([weights_derivatives(nodes,sigma,x)[2]@cap for x in xx])
    step=xx[1]-xx[0]
    kappa=float(np.min(-curv))-0.5*step*M3
    d_left=float(weights_derivatives(nodes,sigma,a)[1]@cap)
    d_right=float(weights_derivatives(nodes,sigma,b)[1]@cap)
    good=kappa>eta2 and d_left>eta1 and -d_right>eta1
    return dict(radius=radius,eta1_uniform=float(eta1),eta2_uniform=float(eta2),
                third_derivative_uniform=float(M3),kappa_uniform_lower=float(kappa),
                left_margin=float(d_left-eta1),right_margin=float(-d_right-eta1),
                robust_unique_root_by_analytic_majorants=bool(good),
                radius_upper=float(eta1/(kappa-eta2)) if kappa>eta2 else float('inf'))


def improved_gaussian_interval_bounds(nodes,cap,sigma,a,b):
    """Continuous analytic Gaussian-weight envelopes using near-zero weighted
    mean offset (estimated at midpoint with an *analytic derivative bound*).
    Returns source-only, non-interval-float majorants.
    """
    mid=(a+b)/2;R=(b-a)/2;d=np.abs(nodes-mid)
    dmin=np.maximum(d-R,0);dmax=d+R
    upper=np.exp(-dmin**2/(2*sigma**2)); lower=np.exp(-dmax**2/(2*sigma**2))
    wmax=upper/lower.sum();amax=dmax/sigma**2
    A2=float(np.sum(wmax*amax**2))
    wmid,_,_=weights_derivatives(nodes,sigma,mid)
    mu_mid=float(wmid@((nodes-mid)/sigma**2))
    mubound=abs(mu_mid)+R*max(A2,1/sigma**2)
    uu=amax+mubound
    M1=float(np.sum(wmax*uu));M2=float(np.sum(wmax*uu**2));M3=float(np.sum(wmax*uu**3))
    # Exact formula w'''=w(u^3 -3Var*u -third-central-moment), hence
    # bound derivative of weighted sum after subtracting a constant.
    cref=float(weights_derivatives(nodes,sigma,mid)[0]@cap)
    absdiff=np.abs(cap-cref)
    b3=float(np.sum(wmax*absdiff*(uu**3+3*M2*uu+M3)))
    return M1,M2,M3,b3,float(mubound)


def improved_strict_interval_diagnostic(nodes,cap,sigma,peak,radius,epsC,ngrids=1001):
    a,b=peak-radius,peak+radius
    m1,m2,m3,B3,mu=improved_gaussian_interval_bounds(nodes,cap,sigma,a,b)
    eta1=epsC*m1;eta2=2*epsC*m2
    xx=np.linspace(a,b,ngrids)
    curv=np.array([weights_derivatives(nodes,sigma,x)[2]@cap for x in xx])
    step=xx[1]-xx[0]
    kappa=float(np.min(-curv))-0.5*step*B3
    d_left=float(weights_derivatives(nodes,sigma,a)[1]@cap)
    d_right=float(weights_derivatives(nodes,sigma,b)[1]@cap)
    good=kappa>eta2 and d_left>eta1 and -d_right>eta1
    return dict(radius=radius,eta1_uniform=float(eta1),eta2_uniform=float(eta2),
                third_derivative_uniform=float(B3),mean_offset_uniform=float(mu),
                kappa_uniform_lower=float(kappa),left_margin=float(d_left-eta1),
                right_margin=float(-d_right-eta1), robust_unique_root_by_analytic_majorants=bool(good),
                radius_upper=float(eta1/(kappa-eta2)) if kappa>eta2 else float('inf'))
