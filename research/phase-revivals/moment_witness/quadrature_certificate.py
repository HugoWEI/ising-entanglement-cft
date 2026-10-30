"""Analytic exact-arithmetic Gauss--Legendre truncation bound for a cubic drive."""
from math import factorial,comb
import numpy as np

def cubic_cell_bounds(y0,dy0,y1,dy1,h):
    if h<=0:raise ValueError("nonpositive h")
    norm=lambda a:float(np.linalg.norm(a,2))
    ypp0=6*(y1-y0)/h**2-(4*dy0+2*dy1)/h
    ypp1=-6*(y1-y0)/h**2+(2*dy0+4*dy1)/h
    yppp=-12*(y1-y0)/h**3+6*(dy0+dy1)/h**2
    m2=max(norm(ypp0),norm(ypp1))
    return np.array([norm(y0)+h*norm(dy0)+h*h*m2/2,
                     norm(dy0)+h*m2,m2,norm(yppp)])

def gauss_coefficient(n):
    if n<2:raise ValueError("n>=2 required")
    return factorial(n)**4/((2*n+1)*factorial(2*n)**3)

def cell_quadrature_bound(mk,h,n=12,b=1.2,hinorm=2):
    d=sum(comb(2*n,k)*hinorm**(2*n-k)*mk[k] for k in range(4))
    return b*gauss_coefficient(n)*h**(2*n+1)*d

def bound_clean_convolution(Y,DY,h,n=12,b=1.2,hinorm=2):
    if Y.shape!=DY.shape or Y.ndim!=3:raise ValueError("shape mismatch")
    cells=[cell_quadrature_bound(cubic_cell_bounds(
               Y[j],DY[j],Y[j+1],DY[j+1],h),h,n,b,hinorm)
           for j in range(len(Y)-1)]
    return dict(cells=len(cells),n_gauss=n,max_cell=max(cells),
                uniform_all_nodes_bound=float(sum(cells)))
