"""Regenerate the Airy first-return predictions without diagonalizing the
final impurity Hamiltonian. This is the prediction step, not a certificate.
Warning: trapezoidal baseline can require substantial run time at dt=.025.
"""
from __future__ import annotations
import os,sys
import numpy as np
HERE=os.path.dirname(__file__)
sys.path.insert(0,os.path.join(HERE,'src'))
import analytic_first_return as ar
import volterra_scattering_revival as vol
from scattering_generator import polar_decomp,capacity
from phase_revival_helper import initial_occupied

def regenerate(L=48,dt=.025,T=36,pair=(1.2,.35)):
    t=np.arange(0,T+dt/4,dt)
    Q=initial_occupied(L)
    old=vol.image_bessel_kernel
    try:
        vol.image_bessel_kernel=ar.base_kernel
        base=vol.defect_response(L,t,*pair,Q,1,return_full=True)
    finally: vol.image_bessel_kernel=old
    echo=ar.wrap_matrix(L,t,'uniform_airy')@Q
    airy=np.array([capacity(polar_decomp(z+e)[0]) for z,e in zip(base,echo)])
    out=os.path.join(HERE,'data',f'L{L}_u{pair[0]}_v{pair[1]}_dt{dt}.npz')
    np.savez_compressed(out,t=t,airy=airy)
    print('Wrote',out)
if __name__=='__main__':
    for pair in [(1.2,.35),(.35,1.2)]:regenerate(pair=pair)
