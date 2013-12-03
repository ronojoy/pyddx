''' Library of finite difference matrices
'''
import numpy as np
import scipy as sp
from scipy import sparse
from . import fdweights

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
    differentiation matrices should be used with great care in bounded domains.      

    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids,Mathematics of Computation 51, no. 184 (1988): 699-706.
        
    Examples
    --------
    
    
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
    data = np.ones(stw, N)
    
    # band diagonals are the finite difference weights
    for i in np.arange(stw):
        data[i,:] = data[i,:]*wm[i]
        
    # indices of the band diagonals are the stencil points
    diags = stpts
    
    # the differentiation matrix    
    D = sp.sparse.spdiags(data, diags, N, N, 'lil')
     
    # for periodic boundaries the differentiation matrix is circulant
    # the i-th and (i-N)th diagonals are identical, i = 1, 2, ... N-1

    # the upper circulant diagonals ; 
    # switch sign for lower circulant diagonals
    diags = np.arange(N-1, N-1-sthw, -1)
    
    D = D + sp.sparse.spdiags(data[0 : sthw - 1, :], -diags, N, N, 'lil') \
          + sp.sparse.spdiags(data[sthw + 1, stw - 1, :], diags, N, N, 'lil')
         
    D = D.tocsr()
        
    return D
