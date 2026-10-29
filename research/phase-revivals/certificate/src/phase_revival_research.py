"""Independent reproducible research follow-up: exact dualities and perturbative first-peak shifts.

Number conserving spinless fermions: even L; H_i = hopping + m*(-1)^j;
H_f = hopping + (w+delta) n_{L/2-1} + (w-delta) n_{L/2}.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def hopping(L):
    h = np.zeros((L,L), dtype=float)
    ii=np.arange(L-1);h[ii,ii+1]=h[ii+1,ii]=-1
    return h

def hfinal(L,u,v):
    h=hopping(L); h[L//2-1,L//2-1]=u;h[L//2,L//2]=v
    return h

def initial_projector(L,m):
    e,q=np.linalg.eigh(hopping(L)+np.diag(m*(-1.)**np.arange(L)))
    Q=q[:,:L//2]
    return Q@Q.T

def entropy_capacity_and_derivative(G,dG=None,eps=1e-12):
    p,Q=np.linalg.eigh((G+G.conj().T)/2)
    pc=np.clip(p,eps,1-eps)
    c=pc*(1-pc)*np.log((1-pc)/pc)**2
    C=float(c.sum())
    if dG is None: return C
    # Endpoint clipping here is strictly numerical; the analytic formula applies in (0,1).
    logratio=np.log((1-pc)/pc)
    fp=(1-2*pc)*logratio**2-2*logratio
    # For exponentially small occupations, check derivative stability separately.
    val=np.real(np.sum(fp*np.diag(Q.conj().T@dG@Q)))
    return C,float(val)

def exact_capacity_series(L,m,u,v,times):
    G0=initial_projector(L,m)
    E,V=np.linalg.eigh(hfinal(L,u,v))
    B=V.T@G0@V
    out=[]
    for t in times:
        U=V@np.diag(np.exp(-1j*E*t))@V.T
        G=(U@G0@U.conj().T)[:L//2,:L//2]
        out.append(entropy_capacity_and_derivative(G))
    return np.array(out)

def reference_linear_response(L,m,w,times):
    """C(t;w+delta,w-delta) = C0(t)+delta*C1(t)+O(delta^2)
    with C1 evaluated by exact Fréchet derivative of the matrix exponential.
    Does not solve any asymmetric defect Hamiltonian.
    """
    H=hfinal(L,w,w)
    E,V=np.linalg.eigh(H)
    G0=initial_projector(L,m)
    W=np.zeros((L,L));W[L//2-1,L//2-1]=1;W[L//2,L//2]=-1
    A=V.T@W@V
    deltaE=E[:,None]-E[None,:]
    out=[]
    for t in times:
        phases=np.exp(-1j*E*t)
        dd=np.empty((L,L),dtype=complex)
        np.divide(phases[:,None]-phases[None,:],deltaE,out=dd,where=np.abs(deltaE)>1e-9)
        dd[np.abs(deltaE)<=1e-9]=(-1j*t*phases[:,None].repeat(L,axis=1))[np.abs(deltaE)<=1e-9]
        U=(V*phases)@V.T
        dU=V@(dd*A)@V.T
        G=U@G0@U.conj().T
        dG=dU@G0@U.conj().T + U@G0@dU.conj().T
        out.append(entropy_capacity_and_derivative(G[:L//2,:L//2], dG[:L//2,:L//2]))
    return np.array(out)

def smooth(y,dt):
    return gaussian_filter1d(y,2/dt,mode='reflect',truncate=4)

def first_peak(t,c,L):
    from scipy.signal import find_peaks
    z=smooth(c,t[1]-t[0]); pp,_=find_peaks(z)
    for ix in pp:
        if t[ix] < .4*L or t[ix]>=1.65*L: continue
        dt=t[1]-t[0]; rad=max(2,round(.15*L/dt))
        lo=max(ix-rad,0);hi=min(ix+rad+1,len(t))
        prom=min(z[ix]-min(z[lo:ix+1]),z[ix]-min(z[ix:hi]))
        if prom>=.08:return ix,t[ix],z[ix]
    return None,np.nan,np.nan

def first_order_peak_shift(t,C0,C1,L,delta):
    dt=t[1]-t[0]; idx,tpeak,h=first_peak(t,C0,L)
    if idx is None or idx<1 or idx>=len(t)-1:return np.nan,tpeak
    C0s=smooth(C0,dt);C1s=smooth(C1,dt)
    dC1=(C1s[idx+1]-C1s[idx-1])/(2*dt)
    ddC0=(C0s[idx+1]-2*C0s[idx]+C0s[idx-1])/dt**2
    shift=-2*delta*dC1/ddC0
    return shift,tpeak

def quantization_polynomial_eigvalues(L,u,v):
    """From 2-arm Schur complement Chebyshev formula, use polynomial roots."""
    from numpy.polynomial import Polynomial as P
    n=L//2
    E=P([0,1]);p0=P([1]);p1=E
    if n-1==1:pmn1,pmn2=p1,p0
    else:
        for j in range(2,n):
            p2=E*p1-p0;p0,p1=p1,p2
        pmn1,pmn2=p1,p0
    q=((E-u)*pmn1-pmn2)*((E-v)*pmn1-pmn2)-pmn1**2
    return np.sort(q.roots())

if __name__=='__main__':
    print('MIRROR+CHIRAL DUALITY CHECK (numerical precision):')
    for L in [12,24,48]:
        tt=np.array([0.7,2.0,8.0,20.0])
        a=exact_capacity_series(L,2.5,.8,.2,tt)
        b=exact_capacity_series(L,2.5,.2,.8,tt)
        c=exact_capacity_series(L,2.5,-.8,-.2,tt)
        d=exact_capacity_series(L,-2.5,.8,.2,tt)
        print('L=',L, 'max errors b-c:',f'{max(abs(b-c)):.3e}', 'b-d:',f'{max(abs(b-d)):.3e}', 'asymmetry max:',f'{max(abs(a-b)):.5f}')
    print('QUANTIZATION ROOTS CHECK:')
    for L in [8,12,20]:
        for u,v in [(1.2,.35),(.5,.2)]:
            roots=quantization_polynomial_eigvalues(L,u,v)
            exact=np.linalg.eigvalsh(hfinal(L,u,v))
            print('L=',L,(u,v),'max root error:',f'{max(abs(np.real(roots)-exact)):.3e}', 'max imaginary:',f'{max(abs(np.imag(roots))):.2e}')
    print('LINEAR RESPONSE TEST (no fit to asymmetric curves):')
    for L,w in [(48,.2),(48,.8),(64,.2),(64,.8)]:
        times=np.arange(0,1.8*L+.001,.2)
        ref=reference_linear_response(L,2.5,w,times)
        print('L,w',L,w)
        for delta in [.05,.1,.2]:
            predicted,tp=first_order_peak_shift(times,ref[:,0],ref[:,1],L,delta)
            a=exact_capacity_series(L,2.5,w+delta,w-delta,times)
            b=exact_capacity_series(L,2.5,w-delta,w+delta,times)
            _,ta,ha=first_peak(times,a,L)
            _,tb,hb=first_peak(times,b,L)
            print(' delta',delta,'baseline peak',round(tp,3),'pred shift',round(predicted,4),'obs shift',round(ta-tb,4),'times',ta,tb)
