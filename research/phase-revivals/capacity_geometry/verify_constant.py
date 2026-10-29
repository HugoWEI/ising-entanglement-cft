"""Interval-assisted, elementary-exponential audit of M < 1.22.

Not independently Arb/MPFI verified; mpmath.iv primitives assumed correct.
"""
from mpmath import iv

def verify(step_milli=5):
    iv.dps=35
    threshold=iv.mpf('1.22')
    mx=0.
    for i in range(0,30000,step_milli):
        u=iv.mpf([f'{i/1000:.3f}',f'{(i+step_milli)/1000:.3f}'])
        r=iv.exp(-u)
        val=abs(u*u*(1-r)/(1+r)-2*u)*iv.exp(-u/2)/(1+r)
        if not bool(val.b < threshold):
            raise AssertionError(f'failure at {i/1000}: {val}')
        mx=max(mx,float(val.b))
    tail=iv.mpf(960)*iv.exp(-iv.mpf(15))
    assert bool(tail.b < threshold)
    return {'M_upper':1.22,'interval_count':30000//step_milli,
            'max_interval_upper':mx,'tail_upper_approx':float(tail.b),
            'scope':'conditional on inclusion of mpmath.iv primitives'}

if __name__=='__main__': print(verify())
