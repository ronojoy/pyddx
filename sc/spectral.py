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

1.    poldif: General differentiation matrices.
2.    chebdif: Chebyshev differentiation matrices. 
3.    herdif: Hermite differentiation matrices. 
4.    lagdif: Laguerre differentiation matrices.

II. Differentiation Matrices (Non-Polynomial)

1.   fourdif: Fourier differentiation matrices.  
2.   sincdif: Sinc differentiation matrices. 

III. Boundary Conditions

1.   cheb2bc: Chebyshev 2nd derivative matrix
                incorporating Robin conditions.
2.   cheb4c: Chebyshev 4th derivative matrix
                incorporating clamped conditions.

IV. Interpolation

1.   polint: Barycentric polynomial interpolation on
                arbitrary distinct nodes
2.   chebint: Barycentric polynomial interpolation on
                Chebyshev nodes. 
3.   fourint: Barycentric trigonometric interpolation at
                equidistant nodes. 

V. Transform-based derivatives  

1. chebdifft: Chebyshev derivative
2. fourdifft: Fourier derivative
3.  sincdift: Sinc derivative 

VI. Roots of Orthogonal Polynomials

1.  legroots: Roots of Legendre polynomials.
2.  lagroots: Roots of Laguerre polynomials. 
3.  herroots: Roots of Hermite polynomials.  

VII. Examples

1.     cerfa.m: Function file for computing the complementary 
                error function.  Boundary condition (a) is used.
2.     cerfb.m: Same as cerfa.m but boundary condition (b) is used.
3.   matplot.m: Script file for plotting the characteristic curves 
                of Mathieu's equation.
4.       ce0.m: Function file for computing the Mathieu cosine
                elliptic function.
5.     sineg.m: Script file for solving the sine-Gordon equation.
6.     sgrhs.m: Function file for computing the right-hand side of 
                the sine-Gordon system.
7.    schrod.m: Script file for computing the eigenvalues of the 
                Schr\"odinger equation.
8.    orrsom.m: Script file for computing the eigenvalues of the 
                Orr-Sommerfeld equation.
"""

__all__ = ['poldif', 'chebdif', 'herdif', 'lagdif', 'fourdif',
           'sincdif', 'cheb2bc', 'cheb4bc', 'polint', 'chebint',
           'fourint', 'chebdifft', 'fourdifft', 'sincdift', 
           'legroots', 'lagroots', 'herroots', 'cerfa', 'cerfb',
            'matplot', 'ce0', 'sineg', 'sgrhs', 'schrod', 'orrsom']

def poldif():
    pass

def chebdif():
    pass

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
