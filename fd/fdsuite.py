# -*- coding: utf-8 -*-
"""
This module provides functions for solving differential equations on 
bounded and periodic intervals by the finite difference method. The
module includes functions for generating differentiation matrices of
arbitrary order on uniform and non-uniform grids. Auxiliary functions
are included for incorporating boundary conditions and performing 
interpolation 

I.  Generation of finite difference weights

1.  fdweights : finite difference weights of arbitrary order

II. Differentiation Matrices - uniform grids

1.  cdif : Central difference differentiation matrices
2.  fdif : Forward difference differenation matrices
3.  bdif : Backward difference differentation matrices

III. Differentation Matrices - non-uniform grids


IV. Boundary conditions

V. Interpolation

"""

import numpy as np
import scipy as sp
from scipy import sparse

__all__=['fdweights', 'cdif']

def fdweights(m, x0, x, dtype='float_'):
    '''
    Calculate finite difference weights.

    Returns the finite difference weights for derivatives 1, 2, ..m at the
    location x0 for the stencil defined in the array of points x.
    
    Parameters
    ----------
    
    m  : int
         maximum order of the derivative

    x0 : scalar
         location at which the derivative is needed
         
    x  : array_like
         the points of the stencil, len(x) > m

    Returns
    -------
    w : ndarray
        (m+1) x len(x) array of weights where w[p,:] contains the
        weights for the p-th derivative. The first row, w[0,:], contains the
        weights of the interpolating polynomial. 
        
    Notes
    -----
    
    This function provides the weights w_i^p for the finite difference
    approximation of the p-th derivative of the function f, at the point x0,
    using the values of the function at the points x[0], x[1], ... x[n-1]. 
    The p-th derivative is obtained as :

    .. math:: 
    
    f^(p)(x0) = \sum_{i=0}^{n-1} w_i^p f(x_i) 
    
    The weights w_i^p are returned in the array w[p,i]. It is not necessary 
    for the elements x[i] to be monotonically increasing or for x0 to be in 
    the interval (x[0], x[n-1]), though these are the most commonly occuring 
    situations. The x[i] need not be equally spaced but must be distinct.    

    The weights are calculated in a numerically stable manner using the 
    algorithm of Fornberg.      

    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids,Mathematics of Computation 51, no. 184 (1988): 699-706.
        
    Examples
    --------
    
    The weights of the 3-point central difference approximation are obtained
    as 
    
    >>> w =  pyddx.fd.fdweights[[-1, 0, 1], 0, 2]
    
    w[0], w[1] and w[2] contain the weights, respectively, of the 
    interpolating polynomial and the first two derivatives. 
    
    The weights of the 5-point central difference approximation are obtained,
    similarly, as
    
    >>> w = pyddx.fd.fdweights[[-2, -1, 0, 1, 2], 0, 4]
    
    Now, w[0], w[1], w[2], w[3] and w[4] contain the weights, respectively,
    of the interpolating polynomial and the first 4 derivatives. 
    
    '''
    
    n = len(x)
    
    if m >= n:
        raise Exception('length of x must be larger than m')

    x=np.asanyarray(x)
    weights = np.zeros((m+1,n), dtype=dtype)
    weights[0,0] = 1.
    betaold = 1.
    for i in range(1,n):
            beta = np.prod(x[i]-x[0:i])
            for k in range(0,min(i,m)+1):
                    weights[k,i] = betaold*(k*weights[k-1,i-1]-(x[i-1]-x0)*weights[k,i-1])/beta
            betaold=beta

            for j in range(0,i):
                    for k in range(min(i,m),-1,-1):
                            weights[k,j] = ((x[i]-x0)*weights[k,j]-k*weights[k-1,j])/(x[i]-x[j])

    # Clear very small entries:
    weights[np.absolute(weights)<1e-10] = 0
    return weights


def cdif(N, m, stw, xh):
    '''
    Calculate differentiation matrices using central differences.

    Returns the differentiation matrix Dm corresponding to the the m-th 
    derivative of the function f, on an equispaced periodic grid x of N
    points, using variable order central difference approximations. 
    
    Parameters
    ----------
    
    N   : int 
          number of grid points
         
    m   : int
          order of the derivative

    stw : int
          stencil width, i.e, number of points in the stencil
          must be an odd number for central difference stencils
         
    xh  : double
          grid spacing

    Returns
    -------
    Dm : ndarray
         N x N  differentiation matrix  
        
    Notes
    -----
    This function returns the differentiation matrix for the m-th derivative
    on an equispaced, periodic grid of N points. The differentiation
    matrix is constructed using variable order central difference formulae.
    The differentiation matrix is then a band diagonal matrix, with constant
    entries on the diagonals (Toeplitz) and with a bandwidth equal
    to the stencil width. On a periodic grid the matrix is also circulant
    so that the i-th and (N-i)th diagonals are identical. 
    
    The m-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    
    .. math::
    
    f^{(m)}_i = D^{(m)}_{ij}f_j
     
    The weights are calculated in a numerically stable manner using the 
    algorithm of Fornberg. 
    
    Finite difference approximations on equispaced grids rapidly lose
    accuracy at the boundary due to the Runge phenomenon. Thus, high order
    differentiation matrices should be used with care in bounded domains.      

    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids,Mathematics of Computation 51, no. 184 (1988): 699-706.
        
    Examples
    --------
    To construct the 3-point central difference Laplacian on a grid of 32
    points, 

    >>> from pyddx.fd import fdsuite as fds
    >>> L = fds.cdif(32, 2, 3, 1)    
    >>> spy(L)
    '''
    
    if m >= N:
        raise Exception('number of grid points must be greater than m')
         
    if m >= stw:
        raise Exception('stencil width must be greater than m')
       
    if stw % 2 == 0:
        raise Exception('stencil width must be odd')
    
    
    sthw = np.int(stw/2 - 1/2)                  # stencil half width
    stpts = np.arange(-sthw, sthw + 1)          # stencil points
    weights = fdweights(m, 0, stpts)            # finite difference weights 
    wm = weights[m,:]                           # m-th derivative weights

    
    # array holding band diagonals of the differentiation matrix
    data = np.ones((stw, N))
    
    # band diagonals are the finite difference weights
    for i in np.arange(stw):
        data[i,:] = data[i,:]*wm[i]
        
    # indices of the band diagonals are the stencil points
    diags = stpts
    
    # the differentiation matrix    
    D = sp.sparse.spdiags(data, diags, N, N)
     
    # for periodic boundaries the differentiation matrix is circulant
    # the i-th and (i-N)th diagonals are identical, i = 1, 2, ... N-1

    # the lower circulant diagonals
    diags = np.arange(-(N-1), -(N-1) + sthw, 1)
    D = D + sp.sparse.spdiags(data[sthw + 1 : stw, :], diags, N, N) 
    
    # the upper circulant diagonals
    diags = np.arange(N - sthw, N, 1) 
    D = D + sp.sparse.spdiags(data[0 : sthw, :], diags, N, N)
        
    return D
