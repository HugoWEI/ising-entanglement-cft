"""Clean open tight-binding propagator via coherent method of images.

For 1-indexed sites i,j in {1,...,L}, open Dirichlet boundaries:
 U_open(i,j;t) = sum_{n in Z} [
    U_infty(i-j+2n(L+1),t) - U_infty(i+j+2n(L+1),t)],
 U_infty(m,t) = integral_{-pi}^pi dk/(2pi) exp[i*k*m+2i*t*cos(k)].

The formula is EXACT for the clean chain if all images are included.
Finite image truncation approximates the number of boundary returns.
Its complex amplitudes must be summed before forming correlations.
The integral is evaluated by a sufficiently large NumPy FFT.

This is well-known method-of-images physics, not a novel result.
"""
from __future__ import annotations
import numpy as np


def infinite_chain_kernel(time: float, fft_points: int=4096):
    if time < 0 or fft_points < 32:
        raise ValueError("invalid time or fft_points")
    momenta=2*np.pi*np.arange(fft_points)/fft_points
    return np.fft.ifft(np.exp(2j*time*np.cos(momenta)))


def finite_clean_image_propagator(length: int, time: float,
                                  max_image_order: int=2,
                                  fft_points: int=4096):
    if length < 2 or max_image_order < 0 or time < 0:
        raise ValueError("invalid length, image order or time")
    # Conservative FFT alias check for the largest image displacement
    # and a ballistic light-cone width of about 2*time.
    min_points=2*(2*(max_image_order+1)*(length+1)+2*time+4)
    if fft_points<=min_points:
        raise ValueError("fft_points insufficient for alias control")
    kernel=infinite_chain_kernel(time,fft_points)
    i=np.arange(length)[:,None]
    j=np.arange(length)[None,:]
    result=np.zeros((length,length),complex)
    period=2*(length+1)
    for n in range(-max_image_order,max_image_order+1):
        direct=(i-j+n*period)%fft_points
        reflected=(i+j+2+n*period)%fft_points
        result+=kernel[direct]-kernel[reflected]
    return result


def exact_clean_sine_propagator(length: int,time: float):
    """Independent reference from finite open-chain sine eigenfunctions."""
    if length < 2 or time < 0:
        raise ValueError("invalid length or time")
    q=np.pi*np.arange(1,length+1)/(length+1)
    j=np.arange(1,length+1)[:,None]
    sine=np.sqrt(2/(length+1))*np.sin(j*q)
    return (sine*np.exp(2j*time*np.cos(q)))@sine.T
