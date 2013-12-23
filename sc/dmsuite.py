# -*- coding: utf-8 -*-
"""
This module provides native Numpy implementations of the seventeen 
m-files provided in the DMSuite library of Weidemann and Reddy in 
ACM Transactions of Mathematical Software, 4, 465-519 (2000). The
authors describe their library as:

 "functions for solving differential equations on bounded, periodic,
 and infinite intervals by the spectral collocation (pseudospectral)
 method.  The package includes functions for generating differentiation
 matrices of arbitrary order corresponding to Chebyshev, Hermite,
 Laguerre, Fourier, and sinc interpolants. In addition, functions
 are included for computing derivatives via the fast Fourier transform
 for Chebyshev, Fourier, and Sinc interpolants.  Auxiliary functions
 are included for incorporating boundary conditions, performing
 interpolation using barycentric formulas, and computing roots of
 orthogonal polynomials.  In the accompanying paper it is demonstrated
 how to use the package by solving eigenvalue, boundary value, and
 initial value problems arising in the fields of special functions,
 quantum mechanics, nonlinear waves, and hydrodynamic stability."
 
The summary of the Numpy functions, named exactly as the original DMSuite 
functions :
 
I. Differentiation Matrices (Polynomial Based)

1.  poldif  : General differentiation matrices.
2.  chebdif : Chebyshev differentiation matrices. 
3.  herdif  : Hermite differentiation matrices. 
4.  lagdif  : Laguerre differentiation matrices.

II. Differentiation Matrices (Non-Polynomial)

1.  fourdif : Fourier differentiation matrices.  
2.  sincdif : Sinc differentiation matrices. 

III. Boundary Conditions

1.  cheb2bc : Chebyshev 2nd derivative matrix incorporating Robin conditions.
2.  cheb4c  : Chebyshev 4th derivative matrix incorporating clamped conditions.

IV. Interpolation

1.  polint  : Barycentric polynomial interpolation on arbitrary distinct nodes
2.  chebint : Barycentric polynomial interpolation on Chebyshev nodes. 
3.  fourint : Barycentric trigonometric interpolation at equidistant nodes. 

V. Transform-based derivatives  

1.  chebdifft : Chebyshev derivative.
2.  fourdifft : Fourier derivative.
3.  sincdift  : Sinc derivative.

VI. Roots of Orthogonal Polynomials

1.  legroots : Roots of Legendre polynomials.
2.  lagroots : Roots of Laguerre polynomials. 
3.  herroots : Roots of Hermite polynomials.  

VII. Examples

1.     cerfa.m: Function file for computing the complementary  error function.  Boundary condition (a) is used.
2.     cerfb.m: Same as cerfa.m but boundary condition (b) is used.
3.   matplot.m: Script file for plotting the characteristic curves of Mathieu's equation.
4.       ce0.m: Function file for computing the Mathieu cosine elliptic function.
5.     sineg.m: Script file for solving the sine-Gordon equation.
6.     sgrhs.m: Function file for computing the right-hand side of the sine-Gordon system.
7.    schrod.m: Script file for computing the eigenvalues of the Schr\"odinger equation.
8.    orrsom.m: Script file for computing the eigenvalues of the Orr-Sommerfeld equation.
"""
from __future__ import division
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.linalg import eig
from scipy.linalg import toeplitz


__all__ = ['poldif', 'chebdif', 'herdif', 'lagdif', 'fourdif',
           'sincdif', 'cheb2bc', 'cheb4bc', 'polint', 'chebint',
           'fourint', 'chebdifft', 'fourdifft', 'sincdift', 
           'legroots', 'lagroots', 'herroots', 'cerfa', 'cerfb',
           'matplot', 'ce0', 'sineg', 'sgrhs', 'schrod', 'orrsom']


def poldif(*arg):
    """
    Calculate differentiation matrices on arbitrary nodes.
      
    Returns the differentiation matrices D1, D2, .. DM corresponding to the 
    M-th derivative of the function f at arbitrarily specified nodes. The
    differentiation matrices can be computed with unit weights or 
    with specified weights.
    
    Parameters 
    ----------

    x       : ndarray
              vector of N distinct nodes
     
    M       : int 
              maximum order of the derivative, 0 < M <= N - 1    
                
    
    OR (when computing with specified weights)

    x       : ndarray
              vector of N distinct nodes
     
    alpha   : ndarray
              vector of weight values alpha(x), evaluated at x = x_j.
                
    B       : int
              matrix of size M x N, where M is the highest derivative required.  
              It should contain the quantities B[l,j] = beta_{l,j} = 
              l-th derivative of log(alpha(x)), evaluated at x = x_j.   

    Returns
    -------
   
    DM : ndarray
         M x N x N  array of differentiation matrices 
     
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the 
    1st, 2nd, ... M-th derivates on arbitrary nodes specified in the array
    x. The nodes must be distinct but are, otherwise, arbitrary. The 
    matrices are constructed by differentiating N-th order Lagrange 
    interpolating polynomial that passes through the speficied points. 
    
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    
    .. math::
    
    f^{(m)}_i = D^{(m)}_{ij}f_j
     
    This function is based on code by Rex Fuzzle
    https://github.com/RexFuzzle/Python-Library
    
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
 
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix 
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    
    """
    if len(arg) > 3:
        raise Exception('numer of arguments are either two OR three')
        
    if len(arg) == 2:
    # unit weight function : arguments are nodes and derivative order    
        x, M = arg[0], arg[1]          
        N = np.size(x); alpha = np.ones(N); B = np.zeros((M,N))
    
    # specified weight function : arguments are nodes, weights and B  matrix   
    elif len(arg) == 3:
        x, alpha, B =  arg[0], arg[1], arg[2]        
        N = np.size(x); M = B.shape[0]   
        
    I = np.eye(N)                       # identity matrix
    L = np.logical_or(I,np.zeros(N))    # logical identity matrix 
    XX = np.transpose(np.array([x,]*N))
    DX = XX-np.transpose(XX)            # DX contains entries x(k)-x(j)
    DX[L] = np.ones(N)                  # put 1's one the main diagonal
    c = alpha*np.prod(DX,1)             # quantities c(j)
    C = np.transpose(np.array([c,]*N)) 
    C = C/np.transpose(C)               # matrix with entries c(k)/c(j).    
    Z = 1/DX                            # Z contains entries 1/(x(k)-x(j)
    Z[L] = 0 #eye(N)*ZZ;                # with zeros on the diagonal.      
    X = np.transpose(np.copy(Z))        # X is same as Z', but with ...
    Xnew=X                
    
    for i in range(0,N):
        Xnew[i:N-1,i]=X[i+1:N,i]

    X=Xnew[0:N-1,:]                     # ... diagonal entries removed
    Y = np.ones([N-1,N])                # initialize Y and D matrices.
    D = np.eye(N);                      # Y is matrix of cumulative sums
    
    DM=np.empty((M,N,N))                # differentiation matrices
    
    for ell in range(1,M+1):
        Y=np.cumsum(np.vstack((B[ell-1,:], ell*(Y[0:N-1,:])*X)),0) # diags
        D=ell*Z*(C*np.transpose(np.tile(np.diag(D),(N,1))) - D)    # off-diags         
        D[L]=Y[N-1,:]
        DM[ell-1,:,:] = D 

    return DM

def chebdif(N,M):
    '''    
    Calculate differentiation matrices using Chebyshev collocation.
      
    Returns the differentiation matrices D1, D2, .. DM corresponding to the 
    M-th derivative of the function f, at the N Chebyshev nodes in the 
    interval [-1,1].   
    
    Parameters
    ----------
     
    N   : int 
          number of grid points
         
    M   : int
          maximum order of the derivative, 0 < M <= N - 1

    Returns
    -------
    x  : ndarray
         N x 1 array of Chebyshev points 
         
    DM : ndarray
         M x N x N  array of differentiation matrices 
        
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the 
    1st, 2nd, ... M-th derivates on a Chebyshev grid of N points. The 
    matrices are constructed by differentiating N-th order Chebyshev 
    interpolants.  
    
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    
    .. math::
    
    f^{(m)}_i = D^{(m)}_{ij}f_j
     
    The code implements two strategies for enhanced accuracy suggested by 
    W. Don and S. Solomonoff :
    
    (a) the use of trigonometric  identities to avoid the computation of
    differences x(k)-x(j) 
    
    (b) the use of the "flipping trick"  which is necessary since sin t can 
    be computed to high relative precision when t is small whereas sin (pi-t) 
    cannot.
    
    It may, in fact, be slightly better not to implement the strategies 
    (a) and (b). Please consult [3] for details.
    
    This function is based on code by Nikola Mirkov 
    http://code.google.com/p/another-chebpy

    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
 
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix 
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487 
           
    Examples
    --------
    
    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Chebyshev
    approximation of the first two derivatives of y = f(x) can be obtained
    as 
    
    >>> N = 32; M = 2; pi = np.pi
    >>> from pyddx.sc import dmsuite as dms
    >>> x, D = dms.chebdif(N, M)        # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.sin(2*pi*x)              # function at Chebyshev nodes
    >>> plot(x, y, 'r', x, D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper left')
    '''

    if M >= N:
        raise Exception('numer of nodes must be greater than M')
        
    if M <= 0:
         raise Exception('derivative order must be at least 1')

    DM = np.zeros((M,N,N))
    
    n1 = N/2; n2 = int(round(N/2.))     # indices used for flipping trick
    k = np.arange(N)                    # compute theta vector
    th = k*np.pi/(N-1)

    # Compute the Chebyshev points

    #x = np.cos(np.pi*np.linspace(N-1,0,N)/(N-1))                # obvious way   
    x = np.sin(np.pi*((N-1)-2*np.linspace(N-1,0,N))/(2*(N-1)))   # W&R way
    x = x[::-1]
    
    # Assemble the differentiation matrices
    T = np.tile(th/2,(N,1))
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T)               # trigonometric identity
    DX[n1:,:] = -np.flipud(np.fliplr(DX[0:n2,:]))    # flipping trick
    DX[range(N),range(N)]=1.                         # diagonals of D
    DX=DX.T

    C = toeplitz((-1.)**k)           # matrix with entries c(k)/c(j)
    C[0,:]  *= 2
    C[-1,:] *= 2
    C[:,0] *= 0.5
    C[:,-1] *= 0.5

    Z = 1./DX                        # Z contains entries 1/(x(k)-x(j))
    Z[range(N),range(N)] = 0.        # with zeros on the diagonal.          

    D = np.eye(N)                    # D contains differentiation matrices.
                                          
    for ell in range(M):
        D = (ell+1)*Z*(C*np.tile(np.diag(D),(N,1)).T - D)      # off-diagonals    
        D[range(N),range(N)]= -np.sum(D,axis=1)        # negative sum trick
        DM[ell,:,:] = D                                # store current D in DM

    return x,DM

def herdif(N, M, b):
    '''    
    Calculate differentiation matrices using Hermite collocation.
      
    Returns the differentiation matrices D1, D2, .. DM corresponding to the 
    M-th derivative of the function f, at the N Chebyshev nodes in the 
    interval [-1,1].   
    
    Parameters
    ----------
     
    N   : int 
          number of grid points
         
    M   : int
          maximum order of the derivative, 0 < M < N

    b   : float
          scale parameter, real and positive
          
    Returns
    -------
    x  : ndarray
         N x 1 array of Hermite nodes which are zeros of the N-th degree 
         Hermite polynomial, scaled by b
         
    DM : ndarray
         M x N x N  array of differentiation matrices 
        
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the 
    1st, 2nd, ... M-th derivates on a Hermite grid of N points. The 
    matrices are constructed by differentiating N-th order Hermite
    interpolants. 
    
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    
    .. math::
    
    f^{(m)}_i = D^{(m)}_{ij}f_j
     
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
 
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix 
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487 
           
    Examples
    --------
    
    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Chebyshev
    approximation of the first two derivatives of y = f(x) can be obtained
    as 
    
    >>> N = 32; M = 2; pi = np.pi
    >>> from pyddx.sc import dmsuite as dms
    >>> x, D = dms.chebdif(N, M)        # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.sin(2*pi*x)              # function at Chebyshev nodes
    >>> plot(x, y, 'r', x, D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper left')
    '''    
    if M >= N - 1:
        raise Exception('numer of nodes must be greater than M - 1')
        
    if M <= 0:
         raise Exception('derivative order must be at least 1')    
    
    
    x = herroots(N)                   # compute Hermite nodes
    alpha = np.exp(-x*x/2)            # compute Hermite  weights.

    beta = np.zeros([M + 1, N])  
    
    # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
    beta[0,:] = np.ones(N)         
    beta[1,:] = -x   
                                      
    for ell in range(2, M + 1):                        
        beta[ell,:] = -x*beta[ell-1,:]-(ell-2)*beta[ell-2,:]
    
    # remove initialising row from beta    
    beta = np.delete(beta, 0, 0)

    # compute differentiation matrix (b=1)
    DM = poldif(x, alpha, beta)     

    # scale nodes by the factor b
    x = x/b                          
    
    # scale the matrix by the factor b
    for ell in range(M):             
        DM[ell,:,:] = (b^(ell+1))*DM[ell,:,:]
    
    return x, DM

def lagdif(N, M, b):
    '''    
    Calculate differentiation matrices using Laguerre collocation.
      
    Returns the differentiation matrices D1, D2, .. DM corresponding to the 
    M-th derivative of the function f, at the N Laguerre nodes.
        
    Parameters
    ----------
     
    N   : int 
          number of grid points
         
    M   : int
          maximum order of the derivative, 0 < M < N

    b   : float
          scale parameter, real and positive
          
    Returns
    -------
    x  : ndarray
         N x 1 array of Hermite nodes which are zeros of the N-th degree 
         Hermite polynomial, scaled by b
         
    DM : ndarray
         M x N x N  array of differentiation matrices 
        
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the 
    1st, 2nd, ... M-th derivates on a Hermite grid of N points. The 
    matrices are constructed by differentiating N-th order Hermite
    interpolants. 
    
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    
    .. math::
    
    f^{(m)}_i = D^{(m)}_{ij}f_j
     
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
 
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix 
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487 
           
    Examples
    --------
    
    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Laguerre
    approximation of the first two derivatives of y = f(x) can be obtained
    as 
    
    >>> N = 32; M = 2; b = 30
    >>> from pyddx.sc import dmsuite as dms
    >>> x, D = dms.lagdif(N, M, b)      # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.exp(-x)                  # function at Laguerre nodes
    >>> plot(x, y, 'r', x, -D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper right')
    '''    
    if M >= N - 1:
        raise Exception('numer of nodes must be greater than M - 1')
        
    if M <= 0:
         raise Exception('derivative order must be at least 1')    

    # compute Laguerre nodes
    x = 0                               # include origin 
    x = np.append(x, lagroots(N-1))     # Laguerre roots
    alpha = np.exp(-x/2);               # Laguerre weights

        
    # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
    beta = np.zeros([M , N])      
    d = np.ones(N)

    for ell in range(0, M):                           
        beta[ell,:] = pow(-0.5, ell+1)*d

    # compute differentiation matrix (b=1)
    DM = poldif(x, alpha, beta)     

    # scale nodes by the factor b
    x = x/b                 

    for ell in range(M):             
        DM[ell,:,:] = pow(b, ell+1)*DM[ell,:,:]

    return x, DM


def fourdif():
#    function [x, DM] = fourdif(N,m)
#%
#% The function [x, DM] = fourdif(N,m) computes the m'th derivative Fourier 
#% spectral differentiation matrix on grid with N equispaced points in [0,2pi)
#% 
#%  Input:
#%  N:        Size of differentiation matrix.
#%  M:        Derivative required (non-negative integer)
#%
#%  Output:
#%  x:        Equispaced points 0, 2pi/N, 4pi/N, ... , (N-1)2pi/N
#%  DM:       m'th order differentiation matrix
#%
#% 
#%  Explicit formulas are used to compute the matrices for m=1 and 2. 
#%  A discrete Fouier approach is employed for m>2. The program 
#%  computes the first column and first row and then uses the 
#%  toeplitz command to create the matrix.
#
#%  For m=1 and 2 the code implements a "flipping trick" to
#%  improve accuracy suggested by W. Don and A. Solomonoff in 
#%  SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
#%  The flipping trick is necesary since sin t can be computed to high
#%  relative precision when t is small whereas sin (pi-t) cannot.
#%
#%  S.C. Reddy, J.A.C. Weideman 1998.  Corrected for MATLAB R13 
#%  by JACW, April 2003.
# 
#
#    x=2*pi*(0:N-1)'/N;                       % gridpoints
#    h=2*pi/N;                                % grid spacing
#    zi=sqrt(-1);
#    kk=(1:N-1)';
#    n1=floor((N-1)/2); n2=ceil((N-1)/2);
#    if m==0,                                 % compute first column
#      col1=[1; zeros(N-1,1)];                % of zeroth derivative
#      row1=col1;                             % matrix, which is identity
#
#    elseif m==1,                             % compute first column
#      if rem(N,2)==0                         % of 1st derivative matrix
#    topc=cot((1:n2)'*h/2);
#        col1=[0; 0.5*((-1).^kk).*[topc; -flipud(topc(1:n1))]]; 
#      else
#    topc=csc((1:n2)'*h/2);
#        col1=[0; 0.5*((-1).^kk).*[topc; flipud(topc(1:n1))]];
#      end;
#      row1=-col1;                            % first row
#
#    elseif m==2,                             % compute first column  
#      if rem(N,2)==0                         % of 2nd derivative matrix
#    topc=csc((1:n2)'*h/2).^2;
#        col1=[-pi^2/3/h^2-1/6; -0.5*((-1).^kk).*[topc; flipud(topc(1:n1))]];
#      else
#    topc=csc((1:n2)'*h/2).*cot((1:n2)'*h/2);
#        col1=[-pi^2/3/h^2+1/12; -0.5*((-1).^kk).*[topc; -flipud(topc(1:n1))]];
#      end;
#      row1=col1;                             % first row 
#
#    else                                     % employ FFT to compute
#      N1=floor((N-1)/2);                     % 1st column of matrix for m>2
#      N2 = (-N/2)*rem(m+1,2)*ones(rem(N+1,2));  
#      mwave=zi*[(0:N1) N2 (-N1:-1)];
#      col1=real(ifft((mwave.^m).*fft([1 zeros(1,N-1)])));
#      if rem(m,2)==0,
#    row1=col1;                           % first row even derivative
#      else
#    col1=[0 col1(2:N)]'; 
#    row1=-col1;                          % first row odd derivative
#      end;
#    end;
#    DM=toeplitz(col1,row1);            
    
    pass

def sincdif():
    pass

def cheb2bc():
    pass

def cheb4bc():
    pass

def polint():
    pass

def chebint():
    pass

def fourint():
    pass

def chebdifft():
    pass

def fourdifft():
    pass

def sincdift():
    pass

def legroots(N):
    """
    Compute roots of the Legendre polynomial of degree N
    
    Parameters
     ----------
     
    N   : int 
          degree of the Legendre polynomial
        
    Returns
    -------
    x  : ndarray
         N x 1 array of Laguerre roots
         
    """
    
    n = np.arange(1, N)                     # indices
    p = np.sqrt(4*n*n - 1)                  # denominator :)
    d = n/p                                 # subdiagonals
    J = np.diag(d, 1) + np.diag(d, -1)      # Jacobi matrix
    
    mu, v = eig(J)
    
    return np.sort(mu)    

def lagroots(N):
    """
    Compute roots of the Laguerre polynomial of degree N
    
    Parameters
     ----------
     
    N   : int 
          degree of the Hermite polynomial
        
    Returns
    -------
    x  : ndarray
         N x 1 array of Laguerre roots
         
    """
    d0 = np.arange(1, 2*N, 2)
    d = np.arange(1, N)
    J = np.diag(d0) - np.diag(d,1) - np.diag(d,-1)

    # compute eigenvectors and eigenvalues    
    mu, v = eig(J)
    
    # return sorted, normalised eigenvalues
    return np.sort(mu)
    
def herroots(N):
    """
    Compute roots of the Hermite polynomial of degree N
    
    Parameters
     ----------
     
    N   : int 
          degree of the Hermite polynomial
        
    Returns
    -------
    x  : ndarray
         N x 1 array of Hermite roots
         
    """
    
    # Jacobi matrix
    d = np.sqrt(np.arange(1, N))
    J = np.diag(d, 1) + np.diag(d, -1)
        
    # compute eigenvectors and eigenvalues    
    mu, v = eig(J)
    
    # return sorted, normalised eigenvalues
    return np.sort(mu)/np.sqrt(2)
    
def cerfa():
    pass

def cerfb():
    pass

def matplot():
    pass

def ce0():
    pass

def sineg():
    pass

def sgrhs():
    pass

def schrod():
    pass

def orrsom(N, R):
    '''
    Compute the eigenvalues of the Orr-Sommerfeld equation using Chebyshev
    collocation.
    
    Parameters
    ----------
     
    N   : int 
          number of grid points
         
    R   : float
          Reynolds number
          
    Returns
    -------
    x  : ndarray
         N x 1 array of Hermite nodes which are zeros of the N-th degree 
         Hermite polynomial, scaled by b
         
    DM : ndarray
         M x N x N  array of differentiation matrices 
        
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the 
    1st, 2nd, ... M-th derivates on a Hermite grid of N points. The 
    matrices are constructed by differentiating N-th order Hermite
    interpolants. 
    
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    
    .. math::
    
    f^{(m)}_i = D^{(m)}_{ij}f_j
     
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
 
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix 
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487 
           
    Examples
    --------
    
    The derivatives of functions can be obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Chebyshev
    approximation of the first two derivatives of y = f(x) can be obtained
    as 
    
    >>> N = 32; M = 2; pi = np.pi
    >>> from pyddx.sc import dmsuite as dms
    >>> x, D = dms.chebdif(N, M)        # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.sin(2*pi*x)              # function at Chebyshev nodes
    >>> plot(x, y, 'r', x, D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper left')
    '''        
    

#i = np.sqrt(-1);
#
## compute Chebyshev differentiation matrices
#[x,DM] = chebdif(N+2,2); 
#
## extract the second derivative matrix and enforce Dirichlet bcs
#D2 = DM[1, 1:N, 1:N]                  
#
#    
#    
#    D2 = DM(2:N+1,2:N+1);                    % Enforce Dirichlet BCs
#                                     
#[x,D4] = cheb4c(N+2);                          % Compute fourth derivative
#     I = eye(size(D4));                        % Identity matrix
#
#A = (D4-2*D2+I)/R-2*i*I-i*diag(1-x.^2)*(D2-I); % Set up A and B matrices
#B = D2-I;
#
#e = eig(A,B);                                  % Compute eigenvalues
#
#[m,l] = max(real(e));                          % Find eigenvalue of largest
#disp('Eigenvalue with largest real part = ')   % real part
#disp(e(l))
#    
    
