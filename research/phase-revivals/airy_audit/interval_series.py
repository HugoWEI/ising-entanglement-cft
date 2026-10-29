"""Finite-L, finite-time interval diagnostic for Olver Airy vs Bessel.

This uses mpmath.iv (gamma, logarithm, atan2 and powers) plus mathematical
majorants for the Bessel and Airy power-series tails. It does NOT prove a
uniform-in-time asymptotic bound; all reported enclosures are pointwise.
It assumes the library's special-function interval implementations are
inclusion-preserving (not externally machine-verified here).
"""
from __future__ import annotations
from mpmath import iv
from fractions import Fraction
import math

iv.dps=110

def rv(x):
    if isinstance(x, Fraction): return iv.mpf(x.numerator)/x.denominator
    if isinstance(x,str): return iv.mpf(x)
    return iv.mpf(x)

def pow_frac(x,num,den):
    return iv.exp(rv(Fraction(num,den))*iv.log(x))

def aup(x):
    return math.nextafter(max(abs(float(x.a)), abs(float(x.b))),math.inf)

def add_err(x,R):
    # float bound R rounded upward, represented by interval enclosure
    R=math.nextafter(float(R),math.inf)
    return x+iv.mpf([-R,R])

def airy_series(x, J=110):
    """Ai(x) via exact ODE coefficient recurrences and two geometric tail bounds.

    Ai(0)=1/(3^(2/3)Gamma(2/3)), Ai'(0)=-1/(3^(1/3)Gamma(1/3)).
    a_{k+2}=a_{k-1}/((k+2)(k+1)).
    """
    a0=1/(pow_frac(rv(3),2,3)*iv.gamma(rv(Fraction(2,3))))
    a1=-1/(pow_frac(rv(3),1,3)*iv.gamma(rv(Fraction(1,3))))
    total=iv.mpf(0)
    R=0.
    for p,base in [(0,a0),(1,a1*x)]:
        term=base
        for j in range(J+1):
            total+=term
            term=term*x**3/((3*j+p+3)*(3*j+p+2))
        q=aup(x)**3/((3*(J+1)+p+3)*(3*(J+1)+p+2))
        if not (0<=q<1):raise ValueError('Increase series length')
        R+=aup(term)/(1-q)
    return add_err(total,R)

def bessel_interval(n,t,K=100):
    """Exact rational power series J_n(2t), positive integer n, t >=0."""
    t=rv(t)
    term=t**n/math.factorial(n)
    total=iv.mpf(0)
    for k in range(K+1):
        total+= term if k%2==0 else -term
        term*=t*t/((k+1)*(n+k+1))
    q=aup(t)**2/((K+2)*(n+K+2))
    if not 0<=q<1:raise ValueError('Increase K')
    R=aup(term)/(1-q)
    return add_err(total,R)

def olver_interval(n,t):
    """Leading Olver approximation used in the preceding study, at x=2t."""
    t=rv(t)
    if float(t.b)==0: return iv.mpf(0)
    x=2*t
    z=x/n
    if float(z.a)==float(z.b)==1.0:
        return pow_frac(rv(Fraction(2,n)),1,3)*airy_series(iv.mpf(0))
    if float(z.b)<1:
        q=iv.sqrt(1-z*z)
        eta=iv.log((1+q)/z)-q
        zeta=pow_frac((rv(Fraction(3,2))*eta),2,3)
    elif float(z.a)>1:
        q=iv.sqrt(z*z-1)
        eta=q-iv.atan2(q,iv.mpf(1))
        zeta=-pow_frac(rv(Fraction(3,2))*eta,2,3)
    else:
        raise ValueError('z interval straddles 1, choose exact rational point or subdivide')
    amp=pow_frac(4*zeta/(1-z*z),1,4)/pow_frac(rv(n),1,3)
    arg=pow_frac(rv(n),2,3)*zeta
    if float(arg.a)>15:
        # DLMF 9.7.15 (positive-real Airy bound), with outward interval operations.
        xi=rv(Fraction(2,3))*pow_frac(arg,3,2)
        top=iv.exp(-xi)
        bound=amp*top/(2*iv.sqrt(iv.pi)*pow_frac(arg,1,4))
        return iv.mpf([0,bound.b])
    return amp*airy_series(arg)

def compare(n, ts):
    ans=[]
    for t in ts:
        b=bessel_interval(n,t)
        a=olver_interval(n,t)
        diff=b-a
        ans.append({'n':n,'t':str(t),'B_mid':(float(b.a)+float(b.b))/2,
                    'A_mid':(float(a.a)+float(a.b))/2,
                    'difference_interval':[float(diff.a),float(diff.b)],
                    'enclosure_width':float(diff.b)-float(diff.a)})
    return ans

if __name__=='__main__':
    import json
    vals=compare(50,['25.5','26','26.4'])+compare(98,['26.4','49','51'])
    print(json.dumps(vals,indent=2))
