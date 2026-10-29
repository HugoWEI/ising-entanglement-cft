import numpy as np,sys,json
# Dependencies are packaged in this directory.

from analytic_first_return import base_kernel,wrap_matrix
from scattering_generator import polar_decomp,projector
from phase_revival_research import hopping,hfinal
from physical_echo import f_cap
import volterra_scattering_revival as vol
ORG=vol.image_bessel_kernel

def cap(x):
 p=np.linalg.eigvalsh(x[:len(x)//2]@x[:len(x)//2].conj().T)
 return float(np.sum(f_cap(p)))
def check(L=48,dt=.05,pair=(1.2,.35),T=26.4):
 tt=np.arange(0,T+dt/4,dt)
 _,q=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)));q=q[:,:L//2]
 vol.image_bessel_kernel=base_kernel
 z=vol.defect_response(L,tt,*pair,q,1,return_full=True)[-1]
 vol.image_bessel_kernel=ORG
 e=wrap_matrix(L,np.array([tt[-1]]),'uniform_airy')[0]@q
 u,s,vh=np.linalg.svd(e,full_matrices=False)
 xf,_=polar_decomp(z+e)
 rows=[]
 for r in (2,3,4,5,6,7,8,10):
  er=(u[:,:r]*s[:r])@vh[:r]
  xr,_=polar_decomp(z+er)
  tail=s[r] if r<len(s) else 0
  mins=np.linalg.svd(z+e,compute_uv=False)[-1]
  m2=np.linalg.svd(z+er,compute_uv=False)[-1]
  rows.append(dict(rank=r,tail_sigma=float(tail),tail_fro=float(np.linalg.norm(s[r:])),
   projector_error=float(np.linalg.norm(projector(xr)-projector(xf),2)),
   polar_bound=float(4*tail/(mins+m2)),capacity_error=float(cap(xr)-cap(xf))))
 return dict(L=L,dt=dt,pair=pair,t=float(tt[-1]),singular_values=s.tolist(),ranks=rows)

if __name__=='__main__':
 cases=[(48,.05,(1.2,.35),26.4),(48,.05,(.35,1.2),27.3),(64,.1,(1.2,.35),36.5),(64,.1,(.35,1.2),37.)]
 out=[]
 for l,dt,p,t in cases:
  d=check(l,dt,p,t);out.append(d)
  print('\n',l,p,'t',t)
  for r in d['ranks']:print(r)
 with open('rank_certificates.json','w') as f:json.dump(out,f,indent=2)
