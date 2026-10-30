"""Post-hoc validation only: exact defective Hamiltonian and midpoint convolution.
No exact defective-chain data enters the certificate construction.
"""
from pathlib import Path
import json,sys
import numpy as np
from clean_witness import sine_modes,witness_nodes,majorant
from hermite_airy_solver import solve_hermite_image
from phase_revival_helper import initial_occupied
from scipy.linalg import eigh_tridiagonal

def benchmark(dt=.1,T=30.,pairs=((1.2,.35),(.35,1.2))):
 L=48;t=np.arange(0,T+dt/4,dt);Q=initial_occupied(L)
 out=[]
 for pair in pairs:
  F_A,Y,DY=solve_hermite_image(L,t,Q,pair,'airy',n_gauss=8)
  F8=witness_nodes(L,t,Q,pair,Y,DY,8)
  F12=witness_nodes(L,t,Q,pair,Y,DY,12)
  cert=majorant(L,t,pair,Y,DY,F12)
  h=np.diag(-np.ones(L-1),1)+np.diag(-np.ones(L-1),-1)
  h[L//2-1,L//2-1]=pair[0];h[L//2,L//2]=pair[1]
  E,U=np.linalg.eigh(h)
  P=U.T@Q
  maxerr=0.;errt=0.
  for j in range(len(t)):
   exact=U@ (np.exp(-1j*t[j]*E)[:,None]*P)
   err=np.linalg.norm(exact-F12[j],2)
   maxerr=max(maxerr,err)
   if j==len(t)-1:errt=err
  record={'L':L,'T':T,'dt':dt,'defect':pair,
   'conditional_bound_final':cert['state_error_l1_bound'],
   'observed_max_node_error':maxerr,'observed_final_error':errt,
   'quadrature_8_vs_12_max_matrix_entry':float(np.max(abs(F8-F12))),
   'max_nodal_q':cert['q_nodes_max'],
   'max_nodal_qprime':cert['dq_nodes_max'],
   'hermite_integral_component':cert['q_hermite_l1_bound'],
   'node_integral_component':cert['q_node_l1_bound']}
  out.append(record);print(json.dumps(record),flush=True)
 return out
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dt',type=float,default=.1);a=p.parse_args()
 r=benchmark(dt=a.dt);(Path(__file__).parent/f'VALIDATION_h{a.dt}.json').write_text(json.dumps(r,indent=2))
