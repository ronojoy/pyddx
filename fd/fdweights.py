import numpy as np

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
    
    >>> w =  pyddx.fd.fdweights[2, 0, [-1, 0, 1]]
    
    w[0,:], w[1,:] and w[2,:] contain the weights, respectively, of the 
    interpolating polynomial and the first two derivatives. 
    
    The weights of the 5-point central difference approximation are obtained,
    similarly, as
    
    >>> w = pyddx.fd.fdweights[4, 0, [-2, -1, 0, 1, 2]]
    
    Now, w[0,:], w[1,:], w[2,:], w[3,:] and w[4,:] contain the weights, respectively,
    of the interpolating polynomial and the first four derivatives. 
    
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
