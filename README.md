pyddx
=====

The finite difference method is a robust and easily implemented method for numerical solutions of partial differential equations in simple domains. Finite difference derivatives can be implemented efficiently as sparse matrices acting on dense vectors. Pyddx provides a collection of finite difference derivative matrices in one, two and three space dimensions, based on standard stencils and non-standard stencils derived from lattice kinetic theory. The collection of spectral derivative matrices in [DMSUITE](http://www.mathworks.com/matlabcentral/fileexchange/29-dmsuite) by Weidemann and Reddy is also made available through a native Numpy implementation. 

The utility of the finite difference method, and, of the pyddx library, is greatest when uncommon partial differential equations have to be solved in simple computational domains. The library provides a lot of flexibility in experimenting with spatial discretisations and temporal integrators, a likely requirement when encountering a new partial differential equation for which standard discretisations may not be suitable.

The approach adopted in the pyddx library avoids high levels of abstraction and works only with vectors (which represent the fields) and matrices (which represent discretised derivatives). Tensor products of one-dimensional grids are used for two and three space dimensions. Finite difference and spectral derivatives can be mixed, through the tensor product of derivative matrices, to construct hydrid finite difference - spectral methods. These hybrid methods can be useful in highly anisotropic problems where the function varies rapidly in some dimensions and slowly in others.

