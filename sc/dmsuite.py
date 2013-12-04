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

1.  poldif: General differentiation matrices.
2.  chebdif: Chebyshev differentiation matrices. 
3.  herdif: Hermite differentiation matrices. 
4.  lagdif: Laguerre differentiation matrices.

II. Differentiation Matrices (Non-Polynomial)

1.  fourdif: Fourier differentiation matrices.  
2.  sincdif: Sinc differentiation matrices. 

III. Boundary Conditions

1.  cheb2bc: Chebyshev 2nd derivative matrix incorporating Robin conditions.
2.  cheb4c: Chebyshev 4th derivative matrix incorporating clamped conditions.

IV. Interpolation

1.  polint: Barycentric polynomial interpolation on arbitrary distinct nodes
2.  chebint: Barycentric polynomial interpolation on Chebyshev nodes. 
3.  fourint: Barycentric trigonometric interpolation at equidistant nodes. 

V. Transform-based derivatives  

1.  chebdifft: Chebyshev derivative
2.  fourdifft: Fourier derivative
3.  sincdift: Sinc derivative 

VI. Roots of Orthogonal Polynomials

1.  legroots: Roots of Legendre polynomials.
2.  lagroots: Roots of Laguerre polynomials. 
3.  herroots: Roots of Hermite polynomials.  

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

import numpy as np
from scipy.linalg import toeplitz

__all__ = ['poldif', 'chebdif', 'herdif', 'lagdif', 'fourdif',
           'sincdif', 'cheb2bc', 'cheb4bc', 'polint', 'chebint',
           'fourint', 'chebdifft', 'fourdifft', 'sincdift', 
           'legroots', 'lagroots', 'herroots', 'cerfa', 'cerfb',
            'matplot', 'ce0', 'sineg', 'sgrhs', 'schrod', 'orrsom']




def poldif(x, malpha=0, B=0):
    """
    function DM = poldif(x, malpha, B)
    
      The function DM =  poldif(x, maplha, B) computes the
      differentiation matrices D1, D2, ..., DM on arbitrary nodes.
    
      The function is called with either two or three input arguments.
      If two input arguments are supplied, the weight function is assumed 
      to be constant.   If three arguments are supplied, the weights should 
      be defined as the second and third arguments.
    
      Input (constant weight):
    
      x:        Vector of N distinct nodes.
      malpha:   M, the number of derivatives required (integer).
      B:        Omitted.
    
      Note:     0 < M < N-1.
    
      Input (non-constant weight):
    
      x:        Vector of N distinct nodes.
      malpha:   Vector of weight values alpha(x), evaluated at x = x(k).
      B:        Matrix of size M x N,  where M is the highest 
                derivative required.  It should contain the quantities 
                B(ell,j) = beta(ell,j) = (ell-th derivative
                of alpha(x))/alpha(x),   evaluated at x = x(j).
    
      Output:
      DM:       DM(1:N,1:N,ell) contains ell-th derivative matrix, ell=1..M.
    
      J.A.C. Weideman, S.C. Reddy 1998
    """
    N= size(x)
    if malpha != 0:                     # Check if constant weight function
        M = malpha                      # is to be assumed.
        alpha = ones(N)              
        B = zeros((M,N))
        
    elif B != 0:
        alpha = malpha                  # Make sure alpha is a column vector
        M = size(B[:,1],1)              # First dimension of B is the number 
    I = eye(N)                       # Identity matrix.
    L = logical_or(I,zeros(N))    # Logical identity matrix. 
    XX = transpose(array([x,]*N))
    DX = XX-transpose(XX)            # DX contains entries x(k)-x(j). 
    DX[L] = ones(N)                  # Put 1's one the main diagonal. 
    c = alpha*prod(DX,1)             # Quantities c(j). 
    C = transpose(array([c,]*N)) 
    C = C/transpose(C)               # Matrix with entries c(k)/c(j).    
    Z = 1/DX                            # Z contains entries 1/(x(k)-x(j)
    Z[L] = 0 #eye(N)*ZZ;                  # with zeros on the diagonal.      
    X = transpose(copy(Z))                 # X is same as Z', but with 
    Xnew=X                
    for i in range(0,N):
        Xnew[i:N-1,i]=X[i+1:N,i]
    X=Xnew[0:N-1,:]                  # diagonal entries removed. 
    Y = ones([N-1,N])                    # Initialize Y and D matrices.
    D = eye(N);                         # Y is matrix of cumulative sums,
    DM=empty((N,N,M))                                   # D differentiation matrices.
    for ell in range(1,M+1):
        Y=cumsum(vstack((B[ell-1,:], ell*(Y[0:N-1,:])*X)),0) # Diagonals
        D=ell*Z*(C*transpose(tile(diag(D),(N,1))) - D)   # Off-diagonal         
        D[L]=Y[N-1,:]
        DM[:,:,ell-1] = D 
    return DM


def chebdif(N,M):
    '''
    Adopted from Weideman&Reddy's Matlab function chebdif.m.
    
    Input:
    N - size of diff matrix - np+1 where np is polynomial order.
    M - Highest derivative matrix order we need.
    Output:
    DM - (ell x N x N) - where ell=0..M-1.
    
    %  The code implements two strategies for enhanced 
    %  accuracy suggested by W. Don and S. Solomonoff in 
    %  SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    %  The two strategies are (a) the use of trigonometric 
    %  identities to avoid the computation of differences 
    %  x(k)-x(j) and (b) the use of the "flipping trick"
    %  which is necessary since sin t can be computed to high
    %  relative precision when t is small whereas sin (pi-t) cannot.

    Code from another-chebpy (http://code.google.com/p/another-chebpy) by Nikola Mirkov. 
    '''
    I = np.eye(N)  # Identity matrix
    DM = np.zeros((M,N,N))

    n1 = N/2; n2 = int(round(N/2.))  # Indices used for flipping trick

    k = np.arange(N)  # Compute theta vector
    th = k*np.pi/(N-1)

#    x = np.cos(np.pi*np.linspace(N-1,0,N)/(N-1))               # Old way with cos function.
    x = np.sin(np.pi*((N-1)-2*np.linspace(N-1,0,N))/(2*(N-1)))  # Compute Chebyshev points in the way W&R did it, with sin function.
    x = x[::-1]
    T = np.tile(th/2,(N,1))
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T)               # Trigonometric identity.
    DX[n1:,:] = -np.flipud(np.fliplr(DX[0:n2,:]))    # Flipping trick.!!!
    DX[range(N),range(N)]=1.                    # Put 1's on the main diagonal of DX.
    DX=DX.T

    C = toeplitz((-1.)**k)   # C i sthe matrix with entries c(k)/c(j)
    C[0,:]  *= 2
    C[-1,:] *= 2
    C[:,0] *= 0.5
    C[:,-1] *= 0.5

    Z = 1./DX                        # Z contains entries 1/(x(k)-x(j))
    Z[range(N),range(N)] = 0.        # with zeros on the diagonal.          

    D = np.eye(N)                    # D contains diff. matrices.
                                          
    for ell in range(M):
        D = (ell+1)*Z*(C*np.tile(np.diag(D),(N,1)).T - D)  # Off-diagonals    
        D[range(N),range(N)]= -np.sum(D,axis=1)       # Correct main diagonal of D - Negative sum trick!
        DM[ell,:,:] = D                               # Store current D in DM

    return x,DM

def herdif():
    pass

def lagdif():
    pass

def fourdif():
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

def legroots():
    pass

def lagroots():
    pass

def herroots():
    pass

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

def orrsom():
    pass
