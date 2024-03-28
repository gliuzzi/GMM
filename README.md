# Gradient Method with Momentum (GMM)
A globally convergent gradient method with momentum (GMM)

### Authors
Matteo Lapucci, Giampaolo Liuzzi, Stefano Lucidi, Marco Sciandrone

#### Abstract
In this work, we consider smooth unconstrained optimization problems and we deal with the class of gradient methods with momentum, i.e., descent algorithms where the search direction is defined as a linear combination of the current gradient and the preceding search direction. This family of algorithms includes nonlinear conjugate gradient methods and Polyak's heavy-ball approach, and is thus of high practical and theoretical interest in large-scale nonlinear optimization. 
We propose a general framework where the scalars of the linear combination defining the search direction are computed simultaneously by minimizing the approximate quadratic model in the 2 dimensional subspace.
This strategy allows us to define a class of gradient methods with momentum enjoying global convergence guarantees and an optimal worst-case complexity bound in the nonconvex setting. Differently than all related works in the literature, the convergence conditions are stated in terms of the Hessian matrix of the bi-dimensional quadratic model. To the best of our knowledge, these results are novel to the literature. 
Moreover, extensive computational experiments show that the gradient methods with momentum here presented outperform classical conjugate gradient methods and are (at least) competitive with the state-of-art method for unconstrained optimization, i.e, L-BFGS method.

### Relevant paper and how to cite
The paper has been submitted for possible pubblication to SIOPT. The submitted paper can be found on arXiv.

M. Lapucci, G. Liuzzi, S. Lucidi, M. Sciandrone. A Globally Convergent Gradient Method with Momentum.  	arXiv:2403.17613 [math.OC]  	
https://doi.org/10.48550/arXiv.2403.17613