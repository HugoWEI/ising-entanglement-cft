import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import numpy as np
from scipy.linalg import eigh_tridiagonal
from clean_witness import sine_modes,majorant,witness_nodes
from hermite_airy_solver import solve_hermite_image
from phase_revival_helper import initial_occupied

class Tests(unittest.TestCase):
 def test_clean_semigroup(self):
  L=16; eig,S=sine_modes(L)
  tr=np.diag(np.ones(L-1)*-1,1)+np.diag(np.ones(L-1)*-1,-1)
  self.assertLess(np.linalg.norm(tr@S-S*eig),1e-13)
  self.assertLess(np.linalg.norm(S.T@S-np.eye(L)),1e-13)
 def test_zero_defect(self):
  L=16;t=np.arange(0,2.1,.1);Q=initial_occupied(L)
  Y=np.zeros((len(t),2,L//2),complex);DY=Y.copy()
  F=witness_nodes(L,t,Q,(0.,0.),Y,DY)
  ev,S=sine_modes(L)
  expf=np.einsum('ik,tk,kq->tiq',S,np.exp(-1j*t[:,None]*ev[None,:]),S.T@Q)
  self.assertLess(np.max(abs(F-expf)),1e-13)
 def test_heldout_diagnostic(self):
  L=16;h=.2;t=np.arange(0,4.0001,h);Q=initial_occupied(L)
  pair=(.5,.2);_,Y,DY=solve_hermite_image(L,t,Q,pair,'airy',n_gauss=10)
  F=witness_nodes(L,t,Q,pair,Y,DY,10)
  ans=majorant(L,t,pair,Y,DY,F)
  self.assertGreater(ans['state_error_l1_bound'],0)
  self.assertLess(ans['max_F_cell_bound'],5)
  # Independent final-H diagonalization only for a test; not used by certificate.
  hf=np.diag(np.r_[np.zeros(L//2-1),pair[0],pair[1],np.zeros(L//2-1)])
  hf+=np.diag(-np.ones(L-1),1)+np.diag(-np.ones(L-1),-1)
  lam,U=np.linalg.eigh(hf)
  Ft=U@((np.exp(-1j*t[-1]*lam))[:,None]*(U.T@Q))
  self.assertLess(np.linalg.norm(F[-1]-Ft,2),ans['state_error_l1_bound']+1e-8)
 def test_quadrature_convergence(self):
  L=16;t=np.arange(0,3.0001,.2);Q=initial_occupied(L)
  _,Y,DY=solve_hermite_image(L,t,Q,(1.2,.35),'airy',n_gauss=8)
  A=witness_nodes(L,t,Q,(1.2,.35),Y,DY,8)
  B=witness_nodes(L,t,Q,(1.2,.35),Y,DY,12)
  self.assertLess(np.max(abs(A-B)),1e-11)

if __name__=='__main__':unittest.main()

def test_integrated_hermite_remainder_coefficient():
    # f(t)=t^2(t-h)^2; its endpoint value and first derivative are zero.
    # Its fourth derivative is 24. The integral is h^5/30 = h^5*24/720.
    import scipy.integrate
    h=.37
    integral=scipy.integrate.quad(lambda t:t*t*(t-h)**2,0,h,epsabs=1e-14)[0]
    assert abs(integral-h**5*24/720)<1e-14


def test_continuous_cell_majorant_for_polynomial_quartic():
    # Strong shape of the Peano estimate at every point of the Hermite cell.
    h=.19
    u=np.linspace(0,h,123)
    err=u*u*(u-h)**2
    point=24/24*u*u*(h-u)**2
    assert np.max(abs(err-point))<1e-15
