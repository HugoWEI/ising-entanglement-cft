import os,sys,json
import numpy as np
# Dependencies are packaged in this directory.

from analytic_first_return import wrap_matrix, base_kernel, rank_approx
from scattering_generator import polar_decomp,tangent_generator,projector,projected_derivative, transport
from phase_revival_research import hopping,hfinal
import volterra_scattering_revival as vol
from physical_echo import f_cap

ORIGINAL=vol.image_bessel_kernel

def check(L=48, dt=.05, pair=(1.2,.35), at=None):
    times=np.arange(0,.75*L+dt/4,dt)
    e,Q=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)));Q=Q[:,:L//2]
    vol.image_bessel_kernel=base_kernel
    z0=vol.defect_response(L,times,*pair,Q,1,return_full=True)
    vol.image_bessel_kernel=ORIGINAL
    z1=vol.defect_response(L,times,*pair,Q,1,return_full=True)
    target_t=at or (.55*L)
    ix=np.argmin(abs(times-target_t))
    z=z0[ix]; d=z1[ix]-z
    E=wrap_matrix(L,np.array([times[ix]]),'uniform_airy')[0]@Q
    eb=wrap_matrix(L,np.array([times[ix]]),'bessel')[0]@Q
    ea,e_diag=rank_approx(E[None,:,:],rank=6);ea=ea[0]
    x,xH=polar_decomp(z+E)
    x0,_=polar_decomp(z)
    xfull,_=polar_decomp(z1[ix]);xr,_=polar_decomp(z+ea)
    # finite differencing projector w.r.t added Airy strength
    xg,uu,s,vh,a=tangent_generator(z+.37*E,E)
    eps=1e-5
    xp,_=polar_decomp(z+(.37+eps)*E);xm,_=polar_decomp(z+(.37-eps)*E)
    derr=np.linalg.norm((projector(xp)-projector(xm))/(2*eps)-projected_derivative(xg,a))
    ty=transport(z,E,nsteps=8)
    tail=np.linalg.svd(E,compute_uv=False)[6]
    s1=np.linalg.svd(z+E,compute_uv=False)[-1]
    s2=np.linalg.svd(z+ea,compute_uv=False)[-1]
    bound=4*tail/(s1+s2)
    phase_err=np.linalg.norm(projector(x)-projector(xfull),2)
    trunc_proj=np.linalg.norm(projector(x)-projector(xr),2)
    data=dict(L=L,dt=dt,pair=pair,t=float(times[ix]),
        bessel_airy_err_operator=float(np.linalg.norm(E-eb,2)),
        airy_vs_full_path_err_operator=float(np.linalg.norm(E-d,2)),
        airy_vs_full_path_relative_fro=float(np.linalg.norm(E-d)/np.linalg.norm(d)),
        full_path_delta_norm=float(np.linalg.norm(d,2)),
        airy_delta_norm=float(np.linalg.norm(E,2)),
        projector_airy_vs_full_path=float(phase_err),
        projector_baseline_vs_full_path=float(np.linalg.norm(projector(x0)-projector(xfull),2)),
        rank6_tail_sigma=float(tail), rank6_projection_error=float(trunc_proj),rank6_projection_bound=float(bound),
        finite_difference_generator_error=float(derr),
        transport_8_projector_error=float(np.linalg.norm(projector(ty)-projector(x),2)),
        orthonormal_error=float(np.linalg.norm(x.conj().T@x-np.eye(L//2))),
        capacity_airy=float(np.sum(f_cap(np.linalg.eigvalsh(x[:L//2]@x[:L//2].conj().T)))),
        capacity_full_path=float(np.sum(f_cap(np.linalg.eigvalsh(xfull[:L//2]@xfull[:L//2].conj().T))))
    )
    return data

if __name__=='__main__':
    all=[]
    for pair in ((1.2,.35),(.35,1.2)):
        d=check(pair=pair,at=26.4 if pair[0]>pair[1] else 27.3)
        print(json.dumps(d,indent=2),flush=True);all.append(d)
    with open('generator_diagnostics.json','w') as f:json.dump(all,f,indent=2)
