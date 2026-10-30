"""Peak comparison (post-hoc exact benchmark); not an interval certificate."""
import json
from pathlib import Path
import numpy as np
from clean_witness import witness_nodes,majorant
from hermite_airy_solver import solve_hermite_image
from phase_revival_helper import initial_occupied
from self_consistent_return import cap_curve,smoothed_peak

def go(L=48,T=36.,dt=.05,pairs=((1.2,.35),(.35,1.2),(.5,.2))):
 t=np.arange(0,T+dt/4,dt);Q=initial_occupied(L)
 results=[]
 for pair in pairs:
  Air,Y,DY=solve_hermite_image(L,t,Q,pair,'airy',n_gauss=8)
  Clean=witness_nodes(L,t,Q,pair,Y,DY,12)
  E,U=np.linalg.eigh(np.diag(-np.ones(L-1),1)+np.diag(-np.ones(L-1),-1)+np.diag(np.r_[np.zeros(L//2-1),pair[0],pair[1],np.zeros(L//2-1)]))
  P=U.T@Q
  Exact=np.einsum('ik,tk,kq->tiq',U,np.exp(-1j*t[:,None]*E[None,:]),P)
  ap=smoothed_peak(t,cap_curve(Air));cp=smoothed_peak(t,cap_curve(Clean));xp=smoothed_peak(t,cap_curve(Exact))
  B=majorant(L,t,pair,Y,DY,Clean)
  result=dict(L=L,dt=dt,T=T,pair=pair,airy_peak=ap,clean_witness_peak=cp,exact_peak=xp,
              clean_peak_error=cp-xp,airy_peak_error=ap-xp,
              conditional_amplitude_bound=B['state_error_l1_bound'])
  results.append(result);print(json.dumps(result),flush=True)
 (Path(__file__).parent/'PEAK_RESULTS.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':go()
