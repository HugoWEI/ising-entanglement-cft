"""Conditional first-peak stability envelopes from the full-cell clean witness.
No defective Hamiltonian used. Extrema evaluated on a mesh, not interval-certified.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.optimize import brentq
from clean_witness import witness_nodes,majorant
from hermite_airy_solver import solve_hermite_image
from phase_revival_helper import initial_occupied
from self_consistent_return import cap_curve,smoothed_peak
from peak_certificate import weights_derivatives


def analyze(L=48,T=36,dt=.05,pairs=((1.2,.35),(.35,1.2),(.5,.2)),sigma=2.):
 t=np.arange(0,T+dt/4,dt);m=L//2;Q=initial_occupied(L)
 report=[]
 for pair in pairs:
  _,Y,DY=solve_hermite_image(L,t,Q,pair,'airy')
  F=witness_nodes(L,t,Q,pair,Y,DY)
  cur=cap_curve(F)
  b=max(map(abs,pair))
  cert=majorant(L,t,pair,Y,DY,F)
  node_bnd=np.r_[0.,b*np.cumsum(cert['cell'][:,0]+cert['cell'][:,1])]
  # End-point-stable capacity Lipschitz constant from physical polar path.
  assert max(node_bnd)<1
  eps=2*1.22*m*node_bnd/(1-node_bnd)
  p=smoothed_peak(t,cur)
  # +/-0.5 around surrogate peak, continuous Gaussian weights are evaluated on grid.
  xs=np.linspace(p-.5,p+.5,401)
  grad=[];curv=[];eta1=[];eta2=[]
  for x in xs:
   w,wd,wdd=weights_derivatives(t,sigma,x)
   grad.append(float(wd@cur))
   curv.append(float(wdd@cur))
   eta1.append(float(np.abs(wd)@eps))
   eta2.append(float(np.abs(wdd)@eps))
  grad=np.array(grad);curv=np.array(curv);eta1=np.array(eta1);eta2=np.array(eta2)
  margin=np.min(-curv-eta2)
  radius=np.max(eta1)/margin if margin>0 else np.inf
  signL=grad[0]-eta1[0];signR=grad[-1]+eta1[-1]
  result={'pair':pair,'L':L,'T':T,'dt':dt,'airy_drive_clean_recon_peak':p,
    'max_node_amplitude_bound':float(node_bnd.max()),
    'max_node_capacity_bound':float(eps.max()),
    'local_radius_estimate':float(radius),
    'min_curvature_margin_estimate':float(margin),
    'left_current_lower_estimate':float(signL),
    'right_current_upper_estimate':float(signR),
    'first_peak_certification':'not certified; meshes are not interval bounds',
    'extrema_mesh_step':float(xs[1]-xs[0])}
  report.append(result);print(json.dumps(result),flush=True)
 (Path(__file__).parent/'PEAK_ENVELOPES.json').write_text(json.dumps(report,indent=2))
 return report
if __name__=='__main__':analyze()
