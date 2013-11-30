''' Library of finite difference matrices
'''
import numpy as np
import scipy as sp
from scipy import sparse

def fdweights(x0, x, m, dtype='float_'):
        '''
        Calculate the Finite Difference weights for a FD stencil on a base of points x
        at the locations x0. Calculates m levels of stencils (m=0 corresponding to the identity)
        
        Algorithm taken from:
        B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily Spaced Grids,
        Mathematics of Computation 51, no. 184 (1988): 699-706.
        
        Code as implemented in https://code.google.com/p/polymode/
        '''
        x=np.asanyarray(x)
        n = len(x)

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

def fdmatrix(N, h, ddxorder, stencilwidth):
    '''
    Constructs sparse central difference matrices using stencils 2m+1 points wide on a
    periodic grid of N points with spacing h. 
    
    This implementation is limited to stencils that are no larger than 
    9 points wide. This limits both the order of the derivative 
    (no greater than 8) and the accuracy (no greater than 8th order). 
    
    The weights are obtained from :
    B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily 
    Spaced Grids,  Mathematics of Computation 51, no. 184 (1988): 699-706.
    '''
    
    if ddxorder > 8 :
        print "please choose a lower order of derivative"
        return 0
        
    if stencilwidth not in [3, 5, 7, 9]:
        print "please choose a stencil of width 3, 5, 7 or 9"
        return 0
        
    # the stencil edges
    stencilhalfwidth = (stencilwidth - 1) / 2 
    stencilpoints =  np.arange(-stencilhalfwidth, stencilhalfwidth + 1)
    
    # obtain complete array of finite difference weights
    weights = fdweights(0, stencilpoints, ddxorder)
    
    # obtain weights for the order of derivative requested
    w = weights[ddxorder,:]
    
    # assemble the matrix by defining diagonals of correct bandwidth
    dx = np.ones([stencilwidth, N])    
    
    for i in np.arange(stencilwidth):        
         dx[i, :] = dx[i, :]*w[i]
        
    D = sp.sparse.spdiags(dx, stencilpoints, N, N, 'lil') 
     
    # for pbc construct upper boundary matrix
    diags1 = np.arange(N-stencilhalfwidth, N)
    dx1 = np.ones([stencilhalfwidth, N])    
     
    # odd/even derivates are antisymmetric/symmetric  
    for i in np.arange(stencilhalfwidth): 
        dx1[i, :] = dx1[i, :]*w[i]            
           
    D1 = sp.sparse.spdiags(dx1, diags1, N, N, 'lil') 
    
    # for pbc construct lower boundary matrix
    diags2 = np.arange(-N+stencilhalfwidth, -N,-1)
    dx2 = np.ones([stencilhalfwidth, N]) 
    
    # odd/even derivates are antisymmetric/symmetric 
    for i in np.arange(stencilhalfwidth): 
        if ddxorder not in [2, 4, 6, 8]:
            dx2[i, :] = -dx2[i, :]*w[i]
        else:
            dx2[i, :] = dx2[i, :]*w[i]
    
    D2 = sp.sparse.spdiags(dx2, diags2, N, N, 'lil') 
    
    # assemble all parts
    DD = (D+D1+D2)/pow(h,ddxorder) 
    DD = DD.tocsr()
    
    return DD
    
    dx = np.ones([stencilwidth, N])    
    
    for i in np.arange(stencilwidth):        
        dx[i, :] = dx[i, :]*wdiags[i]
        
    D = sp.sparse.spdiags(dx, diags, N, N, 'lil') 
    
    D = D.tocsr()
        
    return D
