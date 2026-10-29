"""Intrinsic finite-order Bessel/Olver Airy remainder audit (no defective solver).

Per-order differences at rational time points are enclosed by interval power
series for J_n and Ai, a rigorous geometric tail, and the DLMF positive Airy
bound 9.7.15. The interval engine's special functions need separate auditing.
Bounds below are POINTWISE in t; they do not by themselves enclose a peak time.
"""
import json, math, sys
import numpy as np
from scipy.special import jv,airy
from interval_series import bessel_interval,olver_interval,aup,rv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent / "code"))
from airy_hybrid import leading_olver_j
from mpmath import iv
from pathlib import Path
from zipfile import ZipFile

OUT=Path(__file__).resolve().parent

def comparison_one(L,tstr,save=False):
    t=float(tstr);N=2*(L+1)
    ns=np.arange(N-(L-1),N+(L-1)+1,dtype=int)
    bounds={}; mids={}
    for n in ns:
        b=bessel_interval(int(n),tstr)
        a=olver_interval(int(n),tstr)
        d=b-a
        bounds[n]=[math.nextafter(float(d.a),-math.inf),math.nextafter(float(d.b),math.inf)]
        mids[n]=(float(d.a)+float(d.b))/2
    mags={n:math.nextafter(max(abs(lo),abs(hi)),math.inf) for n,(lo,hi) in bounds.items()}
    # Valid enclosures on each complex component of W_B-W_A.
    # Since N is even, two phases have the same/different sign determined by i-j parity.
    # W_ij = i^(N+i-j)e_(N+i-j) + i^(N-i+j)e_(N-i+j).
    dM=np.zeros((L,L),dtype=float)
    for i in range(L):
        for j in range(L):
            x=i-j
            n1=N+x;n2=N-x
            lo1,hi1=bounds[n1];lo2,hi2=bounds[n2]
            sign=1 if x%2==0 else -1
            lo=lo1+(lo2 if sign==1 else -hi2)
            hi=hi1+(hi2 if sign==1 else -lo2)
            dM[i,j]=math.nextafter(max(abs(lo),abs(hi)),math.inf)
    accumulator=iv.mpf(0)
    for val in dM.ravel(): accumulator+=rv(float(val))**2
    frob=math.nextafter(float(iv.sqrt(accumulator).b),math.inf)
    # Measured benchmark with standard numerical libraries: not used in the bound.
    Bval=np.array([jv(int(n),2*t) for n in ns])
    from_airy=np.array([leading_olver_j(int(n),np.array([2*t]))[0] for n in ns])
    i=np.arange(L)[:,None];j=np.arange(L)[None,:]
    ns_idx1=N+i-j-ns[0];ns_idx2=N-i+j-ns[0]
    Wdiff= (1j**(N+i-j))*((Bval-from_airy)[ns_idx1])+(1j**(N-i+j))*((Bval-from_airy)[ns_idx2])
    op=float(np.linalg.svd(Wdiff,compute_uv=False)[0])
    # With Q†Q=I, ||(W_B-W_A)Q||_2 <= Frobenius bound (even without knowing Q).
    result={'L':L,'time_rational':tstr,'orders':[int(ns[0]),int(ns[-1])],
            'max_orderwise_diff_bound':max(mags.values()),'matrix_frobenius_upper':frob,
            'measured_matrix_operator_norm':op,
            'pointwise_only':True,'interval_engine':'mpmath.iv, 110 decimal digits',
            'note':'Assumes inclusion-preserving gamma, atan2, log and exp interval operations. Not a proof of time-uniform error.'}
    if save:
        with open(OUT/f'order_error_L{L}_t{tstr}.csv','w') as f:
            f.write('n,lower,upper\n')
            for n in ns: f.write(f'{n},{bounds[n][0]:.18e},{bounds[n][1]:.18e}\n')
        return result,ns,np.array([mags[n] for n in ns])
    return result


def main():
    cases=[(48,'26.4214'),(48,'27.3050'),(64,'36.4766'),(64,'37.0491'),(80,'45.0')]
    results=[];curves=[]
    for L,t in cases:
        r,ns,err=comparison_one(L,t,save=True)
        print(json.dumps(r),flush=True)
        results.append(r)
        curves.append((L,t,ns,err))
    (OUT/'finite_order_results.json').write_text(json.dumps(results,indent=2))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for L,t,ns,err in curves[:4]:
        ax.semilogy(ns, np.maximum(err,1e-90),label=f'L={L}, t={t}')
    ax.set_xlabel('Bessel order n')
    ax.set_ylabel('Pointwise enclosing upper bound on |J_n(2t) − Olver$_0$|')
    ax.set_title('Intrinsic first-return Bessel–Airy discrepancy (finite order)')
    ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(OUT/'finite_order_airy_remainder.png',dpi=170)

if __name__=='__main__': main()
