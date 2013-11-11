pyddx
=====

The pyddx library provides a collection of finite difference and spectral collocation differentiation matrices. The library contains three parts :

* fd : standard finite difference matrices as described, for example, in [Scholarpedia](http://www.scholarpedia.org/article/Finite_difference_method).
* dk : discrete kinetic theory derived finite difference matrices with isotropic error, as described in [this paper](http://iopscience.iop.org/0295-5075/101/5/50006;jsessionid=1598A5ED2901FC9E6A693858AFBDB7BF.c2).
* sc : spectral collocation matrices, as provided in [DMSUITE](http://www.mathworks.com/matlabcentral/fileexchange/29-dmsuite) by Weidemann and Reddy.

The formulae for the finite difference and spectral collocation matrices follow, respectively, from local and global polynomial interpolation. The finite difference matrices can be applied to functions sampled on equispaced grids in bounded or periodic domains. The spectral collocation matrices can be applied to functions sampled on a variety of collocation grids (Chebyshev, Hermite, Laguerre and equispaced) in bounded, periodic, and infinite domains. Differentiation matricesfor two or more spatial dimensions are constructed from tensor producs of one dimensional matrices. The differentiation matrices derived from kinetic theory cannot be obtained as tensor products and are, thus,  provided explicitly for two and three dimensions.

The pyddx library avoids objects representing high levels of abstraction but prefers, instead, to work directly with vectors (which represent the solutions) and matrices (which represent the discrete derivative operators). The utility of this approach is greatest when uncommon partial differential equations have to be solved in simple computational domains. The library provides a lot of flexibility in experimenting with spatial discretisations and temporal integrators, a likely requirement when encountering a new partial differential equation for which standard discretisations may be unsuitable. 
In particular, finite difference and spectral derivatives can be mixed, through the tensor product of derivative matrices, to construct hydrid finite difference - spectral methods. These hybrid methods can be useful in highly anisotropic problems where the function varies rapidly in some dimensions and slowly in others. The low level of abstraction makes it possible to use [distributed global arrays](http://hpc.pnl.gov/globalarrays/) to move rapidly from prototyping to harnessing the power of high performance distributed memory architectures for the solution of large scale computational problems. 

