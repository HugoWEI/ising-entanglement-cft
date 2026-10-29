"""Remove individual physical boundary path families from Bessel-image kernel.
The physical path decomposition is exact before truncation: direct infinite-lead,
single reflections at left and right end, and additional multi-bounce wraps.
"""
import sys,json,time
import numpy as np
from scipy.special import jv
import volterra_scattering_revival as vol
from phase_revival_research import hopping,exact_capacity_series


def ablated_kernel(L,times,shells,rows=None,cols=None,families=('direct','left','right','wrap')):
    if rows is None: rows=np.arange(L)
    if cols is None: cols=np.arange(L)
    i=np.asarray(rows,dtype=int)[:,None];j=np.asarray(cols,dtype=int)[None,:]
    t=np.asarray(times,float);out=np.zeros((len(t),len(i),len(cols)),dtype=complex)
    N=2*(L+1)
    n_all=np.arange(2*(shells+1)*(L+1)+1)
    basis=(1j**n_all)[:,None]*jv(n_all[:,None],2*t[None,:])
    def add(n,sign):
        nonlocal out
        out+=sign*basis[np.abs(n)].transpose(2,0,1)
    if 'direct' in families: add(i-j,+1)
    if 'left' in families:add(i+j+2,-1)
    if 'right' in families:add(i+j+2-N,-1)
    if 'wrap' in families or 'wrap_plus' in families:
        add(i-j+N,+1)
    if 'wrap' in families or 'wrap_minus' in families:
        add(i-j-N,+1)
    if 'higher' in families:
        add(i+j+2+N,-1)
    return out

def run(L=48,dt=.1,tmaxfactor=.85,pairs=((1.2,.35),(.35,1.2))):
    t=np.arange(0,L*tmaxfactor+dt/4,dt)
    _,q=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)))
    Q=q[:,:L//2]
    refs=[exact_capacity_series(L,2.5,*pair,t) for pair in pairs]
    families_list={
       'direct':('direct',),
       'left':('direct','left'),
       'right':('direct','right'),
       'both_ends':('direct','left','right'),
       'plus_wrap_plus':('direct','left','right','wrap_plus'),
       'plus_wrap_minus':('direct','left','right','wrap_minus'),
       'plus_wrap':('direct','left','right','wrap'),
       'full_shell1':('direct','left','right','wrap','higher')
    }
    all_results=[]
    for label,fam in families_list.items():
        def fun(L,times,shells,rows=None,cols=None):
            return ablated_kernel(L,times,shells,rows,cols,fam)
        vol.image_bessel_kernel=fun
        for pair,ref in zip(pairs,refs):
            tic=time.perf_counter()
            ca,mx,mn=vol.capacity_from_amplitudes(vol.defect_response(L,t,*pair,Q,1),diagnostic=True)
            ep=vol.peak(t,ref,L);ap=vol.peak(t,ca,L)
            row=dict(L=L,dt=dt,label=label,potentials=pair,mae=float(np.mean(abs(ca-ref))),maxerror=float(max(abs(ca-ref))),eigleak=float(max(0,max(mx)-1,-min(mn))),exact_peak_t=ep[0],pred_t=ap[0],time_error=ap[0]-ep[0],height_error=ap[1]-ep[1],runtime=time.perf_counter()-tic)
            print(json.dumps(row),flush=True)
            all_results.append(row)
    with open('/mnt/data/phase_revival_new/path_ablation_results.json','w') as f:json.dump(all_results,f,indent=2)

if __name__=='__main__':
  dt=float(sys.argv[1]) if len(sys.argv)>1 else .1
  run(dt=dt)
