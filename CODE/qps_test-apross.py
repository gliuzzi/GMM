import numpy as np
import pycutest
from scipy.optimize import minimize, approx_fprime, fmin_bfgs, fmin_l_bfgs_b
from time import time, sleep
from scipy.linalg import hilbert
from tabulate import tabulate
import matplotlib.pyplot as plt

class Problem:
    def __init__(self, name, n=None):
        if n:
            self.__p = pycutest.import_problem(name, sifParams={'N':n})
        else:
            self.__p = pycutest.import_problem(name)
        self.n = self.__p.n
        self.x0 = self.__p.x0
        self.name = self.__p.name

    def f(self,x):
        return self.__p.obj(x)

    def f_g(self, x):
        return self.__p.obj(x, gradient=True)

    def g(self,x):
        _, gr = self.__p.obj(x,gradient=True)
        return gr

    def get_x0(self):
        return np.copy(self.x0)

    def get_n(self):
        return self.n

    def set_x0(self, x):
        self.x0 = x

    def hess(self,x):
        return self.__p.hess(x)


class QuadraticProblem:
    def __init__(self,Q,p):
        self.Q = np.array(Q)
        self.p = np.array(p)
        self.n = self.p.shape[0]
        assert(self.n == self.Q.shape[0])
        assert(self.n == self.Q.shape[1])
        self.x0 = np.zeros(self.n)

    def f(self,x):
        return 0.5*np.dot(x, np.matmul(self.Q, x)) + np.dot(x, self.p)

    def f_g(self, x):
        return 0.5*np.dot(x, np.matmul(self.Q, x)) + np.dot(x, self.p), np.matmul(self.Q, x) + self.p

    def g(self, x):
        return np.matmul(self.Q, x) + self.p


    def get_x0(self):
        return np.copy(self.x0)

    def get_n(self):
        return self.n

    def set_x0(self, x):
        self.x0 = x


    def hess(self,x):
        return self.Q


class HilbertProblem(QuadraticProblem):
    def __init__(self, n):
        Q = hilbert(n)
        p = np.array([0.1*(i+1) for i in range(n)])
        super().__init__(Q,p)
        self.set_x0(np.ones(n))



class Solver:
    def __init__(self, problem, method='Armijo', alpha0=1, gamma=1e-5, delta=0.5, min_step=1e-10, grad_tol=1e-10, max_iters=1000, beta0=0.1, sigma=0.1, epsilon=1e-10, gtol_ord=2, recovery_steps=4):
        self.method = method
        self.fevals = 0
        self.gevals = 0
        self.nnegeig = 0
        self.problem = problem
        self.alpha0 = alpha0 
        self.gamma=gamma 
        self.delta=delta 
        self.min_step=min_step 
        self.grad_tol=grad_tol 
        self.max_iters=max_iters
        self.beta0=beta0
        self.multistart=0
        self.sigma=sigma
        self.epsilon=epsilon
        self.strategy='base'
        self.inner_timer = 0
        self.gtol_ord = gtol_ord
        self.min_sample_val = 1e-8
        self.recovery_steps = recovery_steps

    def set_solver(self,method):
        self.method = method

    def set_problem(self,problem):
        self.problem = problem

        
    def solve(self, method=None, eps_grad=None, max_iters=None, gamma=None, min_step=None, gtol_ord=None, recovery_steps=None):
        if method is not None:
            self.set_solver(method)
        if eps_grad is not None:
            self.grad_tol = eps_grad
        if max_iters is not None:
            self.max_iters = max_iters
        if gamma is not None:
            self.gamma = gamma
        if min_step is not None:
            self.min_step = min_step
        if gtol_ord is not None:
            self.gtol_ord = gtol_ord
        if recovery_steps is not None:
            self.recovery_steps = recovery_steps
        self.fevals = 0
        self.gevals = 0
        tic = time()
        
        if self.method == 'Armijo':
            sol, info = self.solveArmijo()
        elif self.method == 'Extrapolation':
            sol, info = self.solveArmijoExtrapolation()
        elif self.method == 'Momentum':
            self.strategy = 'base'
            sol, info = self.solveMomentum()
        elif self.method == 'Momentum-quadratic':
            self.strategy = 'quadratic'
            sol, info = self.solveMomentum()
        elif self.method == 'Momentum-plane':
            self.strategy = 'd2search'
            sol, info = self.solveMomentum()
        elif self.method == 'Momentum-plane-deriv-free':
            #self.strategy = 'd2search'
            self.strategy = 'd2search-deriv-free'
            sol, info = self.solveMomentum()
        elif self.method == 'Momentum-plane-approx':
            self.strategy = 'inexact-d2s'
            sol, info = self.solveMomentum()
        elif self.method == 'Momentum-plane-multistart':
            self.strategy = 'd2search'
            self.multistart=5
            sol, info = self.solveMomentum()
        elif self.method == 'RandomMomentum':
            sol, info = self.solveRandomMomentum()
        elif self.method == 'QPS':
            sol, info = self.solvePlaneSearch()
        elif self.method == 'QPS-approx':
            sol, info = self.solvePlaneSearch(prova=True)
        elif self.method == 'QPS-iterative':
            sol, info = self.solvePlaneSearch(iterative=True)
        elif self.method == 'QPS-roma':
            sol, info = self.solvePlaneSearch_roma()
        elif self.method == 'CGlike':
            sol, info = self.solvePlaneSearch_CG()
        elif self.method == 'Barzilai':
            sol, info = self.solveBarzilaiBorwein()
        elif self.method == 'ConjGrad':
            sol, info = self.solveConjGrad()
        elif self.method == 'Quasi-Newton':
            sol, info = self.solveQuasiNewton()
        elif self.method == 'scipy_bfgs':
            bfgs = minimize(self.f, self.problem.get_x0(), jac=self.g, method="BFGS", options={"disp": False, "gtol": self.grad_tol, "maxiter": self.max_iters, 'norm': self.gtol_ord})
            info = {"iters": bfgs.nit, "f": bfgs.fun, "g_norm": np.linalg.norm(bfgs.jac, self.gtol_ord)}
            sol = bfgs.x
        elif self.method == 'scipy_lbfgs':
            if not np.isinf(self.gtol_ord):
                print('CANNOT SET DIFFERENT NORM THAN INFTY-NORM FOR L-BFGS')
            lbfgs = minimize(self.f, self.problem.get_x0(), jac=self.g, method="L-BFGS-B", options={"iprint": -1, "maxcor": 10, "gtol": self.grad_tol, "ftol": 1e-30, "maxiter": self.max_iters, "maxls": 20, 'maxfun': 1e15})
            info = {"iters": lbfgs.nit, "f": lbfgs.fun, "g_norm": np.linalg.norm(lbfgs.jac, self.gtol_ord)}
            sol = lbfgs.x
        elif self.method == 'scipy_cg':
            cg = minimize(self.f, self.problem.get_x0(), jac=self.g, method="CG", options={"disp": False, "gtol": self.grad_tol, "maxiter": self.max_iters, 'norm': self.gtol_ord})
            info = {"iters": cg.nit, "f": cg.fun, "g_norm": np.linalg.norm(cg.jac, self.gtol_ord)}
            sol = cg.x
        elif self.method == '':
            pass
        else:
            print('Solver unknown')
            
        info['fevals'] = self.fevals
        info['gevals'] = self.gevals
        info['time'] = time()-tic
        return sol, info
        

    def f(self, x):
        self.fevals +=1
        return self.problem.f(x)


    def f_g(self, x):
        self.fevals += 1
        self.gevals += 1
        return self.problem.f_g(x)

    def g(self,x):
        self.gevals += 1
        return self.problem.g(x)



    def armijoLS(self, x, f, g, d, alpha0=1, gamma=1e-5, min_step=1e-10, delta=0.5, extrapolation=False):
        alpha=alpha0
        f_trial = self.f(x+alpha*d)
        while(f_trial > f + gamma*alpha*np.dot(g,d) and alpha>min_step):
            alpha*=delta
            f_trial = self.f(x+alpha*d)
        if extrapolation and alpha == alpha0:
            f_best = f_trial
            f_trial = self.f(x+alpha/delta*d)
            while (f_trial < f + gamma*alpha/delta*np.dot(g,d) and f_trial > -np.inf and f_trial < f_best):
                alpha /= delta
                f_best = f_trial
                f_trial = self.f(x+alpha/delta*d)
        return alpha

    def solveArmijo(self):
        xk = self.problem.get_x0()
        n_iters=0
        while True:
            f,g = self.f_g(xk)
            g_norm = np.linalg.norm(g,self.gtol_ord)
            if g_norm > 1e6:
                g = g/g_norm
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break
            alpha = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=self.alpha0, gamma=self.gamma, min_step=self.min_step)
            xk -= alpha*g
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm}

    def solveArmijoExtrapolation(self):
        xk = self.problem.get_x0()
        n_iters=0
        alpha = self.alpha0
        while True:
            f,g = self.f_g(xk)
            g_norm = np.linalg.norm(g,self.gtol_ord)
            if g_norm > 1e6:
                g = g/g_norm
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break
            alpha = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha, gamma=self.gamma, min_step=self.min_step, extrapolation=True)
            xk -= alpha*g
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm}


    def bidimensional_search(self, x, d1, d2, alpha0, beta0=0, multistart=0, deriv_free=False, maxfev=10):
        def f2(ab):
            return self.f(x+ab[0]*d1+ab[1]*d2)
        def g2(ab):
            g = self.g(x+ab[0]*d1+ab[1]*d2)
            return [np.dot(g,d1), np.dot(g,d2)]
        def fg2(ab):
            g = self.g(x + ab[0] * d1 + ab[1] * d2)
            return self.f(x+ab[0]*d1+ab[1]*d2), [np.dot(g,d1), np.dot(g,d2)]
        def inner_solve(ab):
            if not deriv_free:
                return minimize(f2, ab, jac=g2, method="CG", options={"disp": False, "gtol": 1e-3, "maxiter": 10})
            else:
                return minimize(f2, ab, method="Nelder-Mead", options={"disp": False, "maxfev": maxfev})
        solution = inner_solve([alpha0, beta0])
        best, best_f = solution.x, solution.fun
        while multistart > 0:
            alpha0 = 0.1*np.random.uniform(low=0, high=1.0)
            beta0 = 4*np.random.uniform(low=0, high=1.0)
            solution = inner_solve([alpha0, beta0])
            if solution.fun < best_f:
                best = solution.x
                best_f = solution.fun
            multistart-=1
        return best

    def bidimensional_search_box(self, x, d1, d2, alpha0, beta0=0, multistart=0, deriv_free=False, maxfev=10):
        def f2(ab):
            return self.f(x + ab[0] * d1 + ab[1] * d2)

        def g2(ab):
            g = self.g(x + ab[0] * d1 + ab[1] * d2)
            return [np.dot(g, d1), np.dot(g, d2)]

        def fg2(ab):
            g = self.g(x + ab[0] * d1 + ab[1] * d2)
            return self.f(x + ab[0] * d1 + ab[1] * d2), np.array(np.dot(g, d1), np.dot(g, d2))

        def inner_BB(ab):
            BB = nmgrad2(2, ab, 1.e-5, 5, 0, fg2)
            x, f, ng, ifail, x_current, f_current, g_current = BB.minimize()
            return x, f

        def inner_solve(ab):
            if not deriv_free:
                return minimize(f2, ab, jac=g2, method="CG", options={"disp": False, "gtol": 1e-3, "maxiter": 10})
            else:
                return minimize(f2, ab, method="Nelder-Mead", bounds=[[ab[0]-10, ab[0]+10], [ab[1]-10, ab[1]+10]],
                                options={"disp": False, "maxfev": maxfev})

        #        alpha0=0.5
        #        beta0=0.
        solution = inner_solve([alpha0, beta0])
        best, best_f = solution.x, solution.fun
        # best, best_f = inner_BB(np.array([alpha0, beta0]))
        while multistart > 0:
            alpha0 = 0.1 * np.random.uniform(low=0, high=1.0)
            beta0 = 4 * np.random.uniform(low=0, high=1.0)
            solution = inner_solve([alpha0, beta0])
            if solution.fun < best_f:
                best = solution.x
                best_f = solution.fun
            multistart -= 1
        return best

    def solvePlaneSearch_roma_box(self, iterative=False):
        xk = self.problem.get_x0()
        n_iters = 0
        num_fails = 0
        xk_1 = np.copy(xk)
        f_1, g_1 = self.f_g(xk_1)
        alpha, beta = 0, 0
        aArm = 1
        cosmax = -np.inf
        cosmin = np.inf
        g_norm = np.inf
        count_barzilai = 0
        while True:
            f, g = self.f_g(xk)
            g_norm_prev = g_norm
            g_norm = np.linalg.norm(g, self.gtol_ord)
            if g_norm > 1e6:
                g = g / g_norm
            if g_norm < self.grad_tol or n_iters >= self.max_iters:
                break
            if count_barzilai < 1:
                # compute cosine of angle
                gnr = np.linalg.norm(g)
                momnr = np.linalg.norm(xk-xk_1)
                if momnr > 0:
                    cosphi = g.dot(xk-xk_1)/(gnr*momnr)
                    cosmax = cosphi if cosphi > cosmax else cosmax
                    cosmin = cosphi if cosphi < cosmin else cosmin
                if iterative:
                    ab, fExp = self.iterative_quadratic_plane_search(xk, xk_1, f, f_1, g, alpha, beta)
                else:
                    ab = self.quadratic_plane_search(xk, xk_1, f, f_1, g, alpha, beta)
                    #ab = np.zeros(2)
                    #ab[0]=alpha
                    #ab[0]=0.
                    #ab[1]=beta
                    #ab[1]=0.
                    ab[0]=np.maximum(0.,ab[0])
                    #print('alfa=',ab[0],'     beta=',ab[1] )
                    fExp = self.f(xk - ab[0] * g + ab[1] * (xk - xk_1))
                    #print('f=',f,'   fnew=',fExp)
                    #if True:
                    if fExp > f:
                        ab[0]=0.
                        ab[1]=0.
                        ab = self.bidimensional_search_box(xk, -g, xk - xk_1, alpha0=ab[0], beta0=ab[1], deriv_free=True, maxfev=10)
                        fExp = self.f(xk - ab[0] * g + ab[1] * (xk - xk_1))
                    #ab = self.bidimensional_search_box(xk, -g, xk - xk_1, alpha0=ab[0], beta0=ab[1], deriv_free=True, maxfev=10)
                    #fExp = self.f(xk - ab[0] * g + ab[1] * (xk - xk_1))
                    #print('alfa=',ab[0],'     beta=',ab[1] )
                    #print('f=',f,'   fnew=',fExp)
                    #print()
                    #input()
                if fExp < f:
                    alpha, beta = ab[0], ab[1]
                    aArm = max(np.abs(alpha), 10 * self.min_step)
                else:
                    #print('Barzilai')
                    num_fails += 1
                    count_barzilai = self.recovery_steps
                    alpha_start = self.bb_step(xk - xk_1, g - g_1, inverse=True)
                    aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma,
                                         min_step=self.min_step)

                    alpha, beta = aArm, 0
            else:
                count_barzilai -= 1
                alpha_start = self.bb_step(xk - xk_1, g - g_1, inverse=True)
                aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma, min_step=self.min_step)

                alpha, beta = aArm, 0

            self.alfas.append(alpha)
            self.betas.append(beta)
            new_x = xk - alpha * g + beta * (xk - xk_1)
            xk_1 = xk
            f_1, g_1 = f, g
            xk = new_x
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm, "nfails": num_fails, "nnegeig": self.nnegeig, "cosmax": cosmax}

    def solvePlaneSearch_roma(self, iterative=False):
        xk = self.problem.get_x0()
        n_iters = 0
        num_fails = 0
        xk_1 = np.copy(xk)
        f_1, g_1 = self.f_g(xk_1)
        alpha, beta = 0, 0
        aArm = 1
        cosmax = -np.inf
        cosmin = np.inf
        g_norm = np.inf
        count_barzilai = 0
        while True:
            f, g = self.f_g(xk)
            g_norm_prev = g_norm
            g_norm = np.linalg.norm(g, self.gtol_ord)
            if g_norm > 1e6:
                g = g / g_norm
            if g_norm < self.grad_tol or n_iters >= self.max_iters:
                break
            if count_barzilai < 1:
                # compute cosine of angle
                gnr = np.linalg.norm(g)
                momnr = np.linalg.norm(xk-xk_1)
                if momnr > 0:
                    cosphi = g.dot(xk-xk_1)/(gnr*momnr)
                    cosmax = cosphi if cosphi > cosmax else cosmax
                    cosmin = cosphi if cosphi < cosmin else cosmin
                if iterative:
                    ab, fExp = self.iterative_quadratic_plane_search(xk, xk_1, f, f_1, g, alpha, beta)
                else:
                    ab = self.quadratic_plane_search(xk, xk_1, f, f_1, g, alpha, beta)
                    #xk, f, f_1, g, g_1
                    #ab = self.CGlike_search(xk, f, f_1, g, g_1)
                    ab = self.bidimensional_search(xk, -g, xk - xk_1, alpha0=ab[0], beta0=ab[1], deriv_free=True, maxfev=10)

                    fExp = self.f(xk - ab[0] * g + ab[1] * (xk - xk_1))
                if fExp < f:
                    alpha, beta = ab[0], ab[1]
                    aArm = max(np.abs(alpha), 10 * self.min_step)
                else:
                    num_fails += 1
                    count_barzilai = self.recovery_steps
                    alpha_start = self.bb_step(xk - xk_1, g - g_1, inverse=True)
                    aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma,
                                         min_step=self.min_step)

                    alpha, beta = aArm, 0
            else:
                count_barzilai -= 1
                alpha_start = self.bb_step(xk - xk_1, g - g_1, inverse=True)
                aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma, min_step=self.min_step)

                alpha, beta = aArm, 0

            self.alfas.append(alpha)
            self.betas.append(beta)
            new_x = xk - alpha * g + beta * (xk - xk_1)
            xk_1 = xk
            f_1, g_1 = f, g
            xk = new_x
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm, "nfails": num_fails, "nnegeig": self.nnegeig, "cosmax": cosmax}

    def solvePlaneSearch_CG(self, iterative=False):
        xk = self.problem.get_x0()
        n_iters = 0
        num_fails = 0
        xk_1 = np.copy(xk)
        f_1, g_1 = self.f_g(xk_1)
        alpha, beta = 0, 0
        aArm = 1
        cosmax = -np.inf
        cosmin = np.inf
        g_norm = np.inf
        count_barzilai = 0
        while True:
            f, g = self.f_g(xk)
            g_norm_prev = g_norm
            g_norm = np.linalg.norm(g, self.gtol_ord)
            if g_norm > 1e6:
                g = g / g_norm
            if g_norm < self.grad_tol or n_iters >= self.max_iters:
                break
            if count_barzilai < 1:
                # compute cosine of angle
                gnr = np.linalg.norm(g)
                momnr = np.linalg.norm(xk-xk_1)
                if momnr > 0:
                    cosphi = g.dot(xk-xk_1)/(gnr*momnr)
                    cosmax = cosphi if cosphi > cosmax else cosmax
                    cosmin = cosphi if cosphi < cosmin else cosmin
                if iterative:
                    ab, fExp = self.iterative_quadratic_plane_search(xk, xk_1, f, f_1, g, alpha, beta)
                else:
                    #ab = self.quadratic_plane_search(xk, xk_1, f, f_1, g, alpha, beta)
                    #xk, f, f_1, g, g_1
                    ab = self.CGlike_search(xk, f, f_1, g, g_1)
                    fExp = self.f(xk - ab[0] * g + ab[1] * (xk - xk_1))
                    if fExp > f:
                        ab = self.bidimensional_search_box(xk, -g, xk - xk_1, alpha0=ab[0], beta0=ab[1], deriv_free=True, maxfev=10)
                        fExp = self.f(xk - ab[0] * g + ab[1] * (xk - xk_1))

                if fExp < f:
                    alpha, beta = ab[0], ab[1]
                    aArm = max(np.abs(alpha), 10 * self.min_step)
                else:
                    num_fails += 1
                    count_barzilai = self.recovery_steps
                    alpha_start = self.bb_step(xk - xk_1, g - g_1, inverse=True)
                    aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma,
                                         min_step=self.min_step)

                    alpha, beta = aArm, 0
            else:
                count_barzilai -= 1
                alpha_start = self.bb_step(xk - xk_1, g - g_1, inverse=True)
                aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma, min_step=self.min_step)

                alpha, beta = aArm, 0

            self.alfas.append(alpha)
            self.betas.append(beta)
            new_x = xk - alpha * g + beta * (xk - xk_1)
            xk_1 = xk
            f_1, g_1 = f, g
            xk = new_x
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm, "nfails": num_fails, "nnegeig": self.nnegeig, "cosmax": cosmax}

    def solvePlaneSearch(self, iterative=False, prova=False):
        xk = self.problem.get_x0()
        n_iters=0
        num_fails = 0
        xk_1 = np.copy(xk)
        f_1, g_1 = self.f_g(xk_1)
        alpha, beta = 0,0
        aArm = 1
        g_norm = np.inf
        count_barzilai = 0
        while True:
            f,g = self.f_g(xk)
            g_norm_prev = g_norm
            g_norm = np.linalg.norm(g,self.gtol_ord)
            if g_norm > 1e6:
                g = g/g_norm
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break
            if count_barzilai<1:
                if iterative:
                    ab, fExp = self.iterative_quadratic_plane_search(xk, xk_1, f, f_1, g, alpha, beta)
                else:
                    if prova:
                        ab = self.quadratic_plane_search_prova(xk, xk_1, f, f_1, g, alpha, beta)
                    else:
                        ab = self.quadratic_plane_search(xk, xk_1, f, f_1, g, alpha, beta)
                    fExp = self.f(xk-ab[0]*g+ab[1]*(xk-xk_1))
                if fExp < f:
                    alpha, beta = ab[0], ab[1]
                    aArm = max(np.abs(alpha), 10*self.min_step)
                else:
                    num_fails += 1
                    count_barzilai = self.recovery_steps
                    alpha_start = self.bb_step(xk-xk_1, g-g_1, inverse=True)
                    aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma, min_step=self.min_step)
                    

                    alpha, beta = aArm, 0
            else:
                count_barzilai -= 1
                alpha_start = self.bb_step(xk-xk_1, g-g_1, inverse=True)
                aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma, min_step=self.min_step)
                
                alpha, beta = aArm, 0
            new_x = xk - alpha*g + beta*(xk-xk_1)
            xk_1 = xk
            f_1, g_1 = f, g
            xk = new_x
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm, "nfails": num_fails, "nnegeig": self.nnegeig, "cosmax": "--"}

    def CGlike_search(self, xk, f, f_1, g, g_1):
        ab = np.zeros(2)
        eps = 1.e-6
        xeps = xk - eps*g
        Hg = (self.g(xeps) - g) / eps
        gnrm2 = g.dot(g)
        alpha0 = gnrm2 / (g.dot(Hg))
        beta0 = gnrm2 / g_1.dot(g_1)
        return np.array([alpha0,beta0])
    def quadratic_plane_search_prova(self, xk, xk_1, f, f_1, g, alpha, beta):
        # find quadratic approximating function and find d by solving system
        # Hd = -g
        d1 = -g
        d2 = xk - xk_1
        D = np.vstack((d1,d2)).T
        gab = g@D

        def f2(ab):
            return self.f(xk - ab[0] * g + ab[1] * (xk - xk_1))

        min_sample_val  = 1e-8
        if np.abs(alpha) <= min_sample_val:
            alpha = min_sample_val if alpha >= 0 else -min_sample_val
        if np.abs(beta) <= min_sample_val:
            beta = min_sample_val if beta >= 0 else -min_sample_val

        if False:
            print(alpha,' ',beta)
        fA = f
        fB = f_1
        #fC = self.f(xk+alpha*d1+beta*d2)
        #fD = self.f(xk+alpha*d1)

        #A = [0.5*alpha**2, 0.5*beta**2, alpha*beta]
        #A = np.vstack((A,[0.5*alpha**2, 0., 0.]))
        #A = np.vstack((A,[0., 0.5, 0.]))
        #A = np.vstack((A,[0., 0.5*beta**2, 0.]))
        #A = np.vstack((A,[0.5, 0., 0.]))
        #rhs = [fC-fA-gab[0]*alpha-gab[1]*beta]
        #rhs = np.vstack((rhs,[fD-fA-gab[0]*alpha]))
        #rhs = np.vstack((rhs,[fB-fA+gab[1]]))
        #rhs = np.vstack((rhs,[self.f(xk+beta*d2)-f-gab[1]*beta]))
        #rhs = np.vstack((rhs,[self.f(xk-d1)-f+gab[0]]))
        delta = 1.e-3
        #A = [0.5*delta**2, 0.0, 0.0]
        #rhs = [self.f(xk+delta*d1)-f-gab[0]*delta]
        A = [0.5*delta**2, 0.5*delta**2, delta*delta]
        rhs = [self.f(xk+delta*d1+delta*d2)-f-gab[0]*delta-gab[1]*delta]
        A = np.vstack((A,[0.5*delta**2, 0.5*delta**2, -delta*delta]))
        rhs = np.vstack((rhs,[self.f(xk-delta*d1+delta*d2)-f+gab[0]*delta-gab[1]*delta]))
        A = np.vstack((A,[0.0, 0.5, 0.0]))
        rhs = np.vstack((rhs,[f_1-f+gab[1]]))
        #A = np.vstack((A,[0.0, 0.5*delta**2, 0.0]))
        #rhs = np.vstack((rhs,[self.f(xk+delta*d2)-f-gab[1]*delta]))
        if False:
            A = np.vstack((A,[0.5*delta**2, 0.5*delta**2, -delta*delta]))
            rhs = np.vstack((rhs,[self.f(xk-delta*d1+delta*d2)-f+gab[0]*delta-gab[1]*delta]))
            A = np.vstack((A,[0.5*delta**2, 0.0, 0.0]))
            rhs = np.vstack((rhs,[self.f(xk-delta*d1)-f+gab[0]*delta]))
            A = np.vstack((A,[0.5*delta**2, 0.5*delta**2, delta*delta]))
            rhs = np.vstack((rhs,[self.f(xk-delta*d1-delta*d2)-f+gab[0]*delta+gab[1]*delta]))
            A = np.vstack((A,[0.0, 0.5*delta**2, 0.0]))
            rhs = np.vstack((rhs,[self.f(xk-delta*d2)-f+gab[1]*delta]))
            A = np.vstack((A,[0.5*delta**2, 0.5*delta**2, -delta*delta]))
            rhs = np.vstack((rhs,[self.f(xk+delta*d1-delta*d2)-f-gab[0]*delta+gab[1]*delta]))
        #print(A)
        #print(rhs)
        #print(self.f(xk-d1),' ',f,' ',gab[0])
        #input()
        #sol = np.linalg.pinv(A).dot(rhs)
        #print(sol)
        try:
            #sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        if False:
            print(A.dot(sol))
            print(rhs)
            #input()
        ba = sol[0,0]
        bc = sol[1,0]
        bb = sol[2,0]

        # bc1 = 2*(gab[1]+fB-fA)
        # ba1 = 2/(alpha**2)*(fD-alpha*gab[0]-fA)
        # bb1 = (fC-fA-gab[0]*alpha-gab[1]*beta-0.5*alpha*alpha*ba1-0.5*beta*beta*bc1)/(alpha*beta)

        # sol[0,0] = ba1
        # sol[1,0] = bc1
        # sol[2,0] = bb1
        if False:
            print(A.dot(sol))
            print(rhs)

        if False: #bc != bc1 or ba != ba1 or bb != bb1:
            print(bc,' ',bc1)
            print(ba, ' ', ba1)
            print(bb, ' ', bb1)
            input()

        Bab = np.array([[ba, bb], [bb, bc]])
        try:
            EIGV, EIGW = np.linalg.eig(Bab)
        except np.linalg.LinAlgError:
            print(A)
            print(rhs)
            print(Bab)
            EIGV = np.zeros(2)
            input()

        if min(EIGV) < 0:
            #print(EIGV)
            self.nnegeig += 1
            #input()
        #Bab = np.array([[ba1, bb1], [bb1, bc1]])

        try:
            solution_closed = np.linalg.solve(Bab, -gab)
        except np.linalg.LinAlgError:
            solution_closed = np.linalg.lstsq(Bab, -gab, rcond=None)[0]
        best = solution_closed
        return best

    def quadratic_plane_search(self, xk, xk_1, f, f_1, g, alpha, beta):
        ab0 = np.zeros(2)
        d1 = -g
        d2 = xk - xk_1
        D = np.vstack((d1,d2)).T
        gab = g@D

        min_sample_val  = 1e-8
        if np.abs(alpha) <= min_sample_val:
            alpha = min_sample_val if alpha >= 0 else -min_sample_val
        if np.abs(beta) <= min_sample_val:
            beta = min_sample_val if beta >= 0 else -min_sample_val

        fA = f
        fB = f_1
        fC = self.f(xk+alpha*d1+beta*d2)
        fD = self.f(xk+alpha*d1)

        bc = 2*(gab[1]+fB-fA)
        ba = 2/(alpha**2)*(fD-alpha*gab[0]-fA)
        bb = (fC-fA-gab[0]*alpha-gab[1]*beta-0.5*alpha*alpha*ba-0.5*beta*beta*bc)/(alpha*beta)

        Bab = np.array([[ba,bb],[bb,bc]])
        EIGV, EIGW = np.linalg.eig(Bab)
        if min(EIGV) < 0:
            #print(EIGV)
            self.nnegeig += 1
            #input()

        try:
            solution_closed = np.linalg.solve(Bab, -gab)
        except np.linalg.LinAlgError:
            solution_closed = np.linalg.lstsq(Bab, -gab, rcond=None)[0]
        best = solution_closed
        return best


    def iterative_quadratic_plane_search(self, xk, xk_1, f, f_1, g, alpha, beta):
        ab0 = np.zeros(2)
        d1 = -g
        d2 = xk - xk_1
        D = np.vstack((d1,d2)).T
        gab = g@D

        min_sample_val  = 1e-8
        if np.abs(alpha) <= min_sample_val:
            alpha = min_sample_val if alpha >= 0 else -min_sample_val
        if np.abs(beta) <= min_sample_val:
            beta = min_sample_val if beta >= 0 else -min_sample_val

        alphaC, betaC = alpha, beta
        alphaD = alpha

        fA = f
        fB = f_1
        fC = self.f(xk+alphaC*d1+betaC*d2)
        fD = self.f(xk+alphaD*d1)
        
        

        AM = np.array([[0,0,0.5], [alphaC**2/2,alphaC*betaC,betaC**2/2], [alphaD**2/2,0,0]])
        
        FS = np.array([fB-fA+gab[1],fC-fA-alphaC*gab[0]-betaC*gab[1],fD-fA-alphaD*gab[0]])



        count = 0
        while True: 
            count += 1
            
            
            try:
                qs = np.linalg.lstsq(AM, FS, rcond=None)
            except SystemError:
                f_new = np.inf
                best = [0,0]
                break

            if np.isnan(FS).any():
                return [0, 0], fA

            ba, bb, bc = qs[0][0], qs[0][1], qs[0][2]

        
            Bab = np.array([[ba,bb],[bb,bc]])
        
    
            try:
                solution_closed = np.linalg.solve(Bab, -gab)
            except np.linalg.LinAlgError:
                solution_closed = np.linalg.lstsq(Bab, -gab, rcond=None)[0]
            
            new_a, new_b = solution_closed[0], solution_closed[1]
            f_new = self.f(xk+new_a*d1+new_b*d2)
        
        
            if f_new < fA or count > 3:        
                best = solution_closed
                break
            else:
                FS = np.append(FS, f_new - fA - new_a*gab[0]-new_b*gab[1])
                AM = np.append(AM, [[new_a**2/2,new_a*new_b, new_b**2/2]], axis=0)
        
        return best, f_new


    def solveRandomMomentum(self):
        xk = self.problem.get_x0()
        n_iters=0
        while True:
            f,g = self.f_g(xk)
            g_norm = np.linalg.norm(g,self.gtol_ord)
            if g_norm > 1e6:
                g = g/g_norm
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break
            a0 = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=self.alpha0, gamma=self.gamma, min_step=self.min_step)
            d_mom = np.random.uniform(low=-1, high=1, size=xk.shape)
            d_mom = d_mom/np.linalg.norm(d_mom)
            ab = self.bidimensional_search(xk, -g, d_mom, alpha0=a0, beta0=0,multistart=self.multistart,maxfev=2*self.problem.n)
            alpha, beta = ab[0], ab[1]            
            new_x = xk - alpha*g + beta*d_mom
            xk_1 = xk
            xk = new_x
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm}

    def solveMomentum(self):
        xk = self.problem.get_x0()
        if self.strategy == 'quadratic':
            try:
                self.problem.Q
            except AttributeError:
                self.strategy='base'
            else:
                evs = np.linalg.eigvals(self.problem.Q)
                max_ev, min_ev = max(evs), min(evs)
                M, m = np.sqrt(max_ev), np.sqrt(min_ev)
        n_iters=0
        xk_1 = np.copy(xk)
        alpha = self.alpha0
        beta = 0
        while True:
            f,g = self.f_g(xk)
            g_norm = np.linalg.norm(g,self.gtol_ord)
            if g_norm > 1e6:
                g = g/g_norm
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break
            if self.strategy == 'd2search':
                a0 = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=self.alpha0, gamma=self.gamma, min_step=self.min_step)
                d_mom = xk-xk_1
                ab = self.bidimensional_search(xk, -g, d_mom, alpha0=a0, beta0=0,multistart=self.multistart,maxfev=2*self.problem.n)
                alpha, beta = ab[0], ab[1]
            elif self.strategy == 'd2search-deriv-free':
                d_mom = xk-xk_1
                #ab = self.bidimensional_search(xk, -g, d_mom, alpha0=self.alpha0, beta0=0,multistart=self.multistart, deriv_free=True, maxfev=100)
                ab = self.bidimensional_search(xk, -g, d_mom, alpha0=alpha, beta0=beta,multistart=self.multistart, deriv_free=True, maxfev=100)
                alpha, beta = ab[0], ab[1]
            elif self.strategy == 'quadratic':
                alpha = 4/(M+m)**2
                beta = ((M-m)/(M+m))**2
            elif self.strategy == 'inexact-d2s':
                aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=self.alpha0, gamma=self.gamma, min_step=self.min_step)
                ab = self.inexact_bidimensional_search(xk, -g, xk-xk_1, alpha0=aArm, beta0=0,multistart=self.multistart, maxfev=2*self.problem.n)
                if self.f(xk-ab[0]*g+ab[1]*(xk-xk_1)) < self.f(xk-aArm*g):
                    alpha, beta = ab[0], ab[1]
                    print('plane search', alpha)
                    input()
                else:
                    alpha, beta = aArm, 0
                    print('armijo', aArm)
            elif self.strategy == 'base':
                beta = self.beta0
                alpha = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=self.alpha0, gamma=self.gamma, min_step=self.min_step)
            new_x = xk - alpha*g + beta*(xk-xk_1)
            xk_1 = xk
            xk = new_x
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm}



    def bb_step(self,s,y,epsilon=1e-10, inverse=False):
        if inverse:
            aux = np.dot(y,s)
            mu = np.dot(y,y)/aux if np.abs(aux)>1e-8 else 0
        else:
            mu = np.dot(s,y)/np.dot(s,s) if np.linalg.norm(s)>1e-8 else 0
        return 1/mu if epsilon <= mu <= epsilon**(-1) else 1

    def solveBarzilaiBorwein(self):
        xk = self.problem.get_x0()
        n_iters=0
        xk_1 = np.copy(xk)
        _, gk_1 = self.f_g(xk)
        while True:
            f,g = self.f_g(xk)
            g_norm = np.linalg.norm(g,self.gtol_ord)
            if g_norm > 1e6:
                g = g/g_norm*1e6
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break
            alpha=self.bb_step(xk-xk_1, g-gk_1, inverse=True)
            alpha = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha, gamma=self.gamma, min_step=self.min_step)
            xk_1 = np.copy(xk)
            gk_1 = np.copy(g)
            xk -= alpha*g
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm}


    def solveConjGrad(self):
        xk = self.problem.get_x0()
        n_iters=0
        f, g = self.f_g(xk)
        dk = -g
        g_prev = np.copy(g)
        while True:
            g_norm = np.linalg.norm(g,2)
            g_norm_stop = g_norm if self.gtol_ord == 2 else np.linalg.norm(g,self.gtol_ord)
            if g_norm_stop<self.grad_tol or n_iters >= self.max_iters:
                break
            
            alpha=self.armijoLS(x=xk, f=f, g=g, d=dk, alpha0=self.alpha0, gamma=self.gamma, min_step=self.min_step)
            xk += alpha*dk
            g_prev = np.copy(g)
            f,g = self.f_g(xk)
            beta = np.dot((g-g_prev),g)/(g_norm**2) # Polyak-Polak-Ribiere
            # beta = np.dot(g,g)/(g_norm**2) # Fletcher-Reeves
            beta = max(beta, 0)
            dk = -g + beta*dk
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm_stop}


    def solveQuasiNewton(self):
        xk = self.problem.get_x0()
        n_iters=0
        xk_1 = np.copy(xk)
        f, gk_1 = self.f_g(xk)
        H = np.identity(len(xk))
        g = np.copy(gk_1)
        while True:
            g_norm = np.linalg.norm(g,self.gtol_ord)
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break
            dir =  - np.matmul(H, g)
            dTg = np.dot(g,dir)
            alpha_u = np.inf
            alpha_l = 0
            alpha = 1
            x_trial = xk+alpha*dir
            f_trial, g_trial = self.f_g(x_trial)
            while((f_trial > f +self.gamma*alpha*dTg or np.dot(g_trial,dir)< self.sigma*dTg) and alpha>self.min_step and alpha_u-alpha_l>self.min_step):
                if f_trial > f + self.gamma*alpha*dTg:
                    alpha_u = alpha
                    alpha = (alpha_u + alpha_l)/2
                    x_trial = xk+alpha*dir
                    f_trial, g_trial = self.f_g(x_trial)
                elif np.dot(g_trial,dir)< self.sigma*dTg:
                    alpha_l = alpha
                    alpha = min((alpha_u+alpha_l)/2, 2*alpha)
                    x_trial = xk+alpha*dir
                    f_trial, g_trial = self.f_g(x_trial)
            xk_1 = np.copy(xk)
            gk_1 = np.copy(g)
            xk = x_trial
            g = g_trial
            f = f_trial
            s = xk-xk_1
            y = g - gk_1
            sTy = np.dot(s,y)
            H = H + (1+np.matmul(y,np.matmul(H,y))/sTy)*np.outer(s,s)/sTy - (np.matmul(np.outer(s,y),H) + np.matmul(H,np.outer(y,s)))/sTy
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm}

    def test_problem(self, solvers, eps_grad=1e-3, max_iters=5000, gamma=1e-4, min_step=1e-10, gtol_ord=2, doplot=False):
        res_tab = []
        for solver in solvers:
            self.alfas = []
            self.betas = []
            sol, info = self.solve(method=solver, eps_grad=eps_grad, max_iters=max_iters, gamma=gamma, min_step=1e-10, gtol_ord=gtol_ord)
            if doplot:
                print(len(self.alfas),len(self.betas))
                k = len(self.alfas)
                xplt = [i for i in range(k)]
                #plot alpha and beta using pyplot
                plt.figure()
                plt.plot(xplt, self.alfas, 'b-',label='alpha')
                plt.plot(xplt, self.betas, 'r-',label='beta')
                plt.ylim(ymin = -2, ymax = 2)
                plt.legend()
                plt.show()
                plt.savefig(solver+" "+self.problem.name+".png")
                #input("Press RETURN to continue")

            if 'nfails' in info:
                res_tab.append([solver, self.problem.name, self.problem.n, info['time'], info['iters'], info['f'], info['g_norm'],
                                info['fevals'], info['gevals'], info['nfails'], info['nnegeig'], info['cosmax']])
            else:
                res_tab.append(
                    [solver, self.problem.name, self.problem.n, info['time'], info['iters'], info['f'], info['g_norm'], info['fevals'],
                     info['gevals'], np.nan, np.nan, np.nan])

        #table = tabulate(res_tab, headers=['Algorithm', 'n', 't', 'n_it', 'f_opt',
        #                                   'g_norm', 'fevals', 'gevals', 'nfails','cosmin','cosmax'], tablefmt='orgtbl')
        #print(table)
        return res_tab


def make_random_psd_matrix(size, ncond=None, eigenvalues=None):
    y = np.random.uniform(low=-1, high=1, size=(size,1))
    if ncond:
        ncond_ = np.log(ncond)
        d = np.array([np.exp((i)/(size-1)*ncond_) for i in range(size)])
    elif eigenvalues:
        d = np.array(eigenvalues)
    D = np.diag(d)
    Y = np.identity(size) - 2/(np.linalg.norm(y)**2)*(y@y.T)
    return Y@D@Y








#solvers = ['Armijo', 'Extrapolation', 'Barzilai', 'Momentum', 'Momentum-plane-deriv-free',
#           'Momentum-plane', 'RandomMomentum',  'Momentum-plane-multistart', 'QPS', 'QPS-iterative',  'scipy_cg',  'scipy_lbfgs']
#solvers = ['Momentum-plane-deriv-free', 'QPS', 'QPS-roma']
#solvers = ['QPS', 'QPS-roma', 'CGlike','scipy_lbfgs']
solvers = ['QPS', 'QPS-approx','scipy_lbfgs']

eps_grad = 1e-3


# TEST CUTEST
problems = ['GENROSE', 'ARWHEAD', 'BROYDN7D', 'CRAGGLVY', 'DIXMAANA1', 'DIXMAANB', 'DIXMAANC', 'DIXMAAND', 'DIXMAANE1', 'DIXMAANF', 'DIXMAANG', 'DIXMAANH', 'DIXMAANI1', 'DIXMAANJ', 'DIXMAANK', 'DIXMAANL', 'DIXMAANM1',
            'DIXMAANN', 'DIXMAANO', 'DIXMAANP', 'EDENSCH', 'ENGVAL1', 'FLETCBV3', "FLETCHCR", 'FMINSURF',
            'LIARWHD', 'MOREBV', 'NCB20', 'NONCVXU2', 'NONCVXUN', 'NONDIA', 'NONDQUAR', 'POWELLSG', 
            'POWER', 'SCHMVETT', 'SROSENBR', 'TOINTGSS', 'TQUARTIC', 'VAREIGVL', 'WOODS']

problems = ['GENROSE', 'ARWHEAD', 'BROYDN7D', 'CRAGGLVY', 'DIXMAANA', 'DIXMAANB', 'DIXMAANC', 'DIXMAAND', 'DIXMAANE',
            'DIXMAANF', 'DIXMAANG', 'DIXMAANH', 'DIXMAANI', 'DIXMAANJ', 'DIXMAANK', 'DIXMAANL', 'DIXMAANM',
            'DIXMAANN', 'DIXMAANO', 'DIXMAANP', 'EDENSCH', 'ENGVAL1', 'FLETCBV3', "FLETCHCR", 'FMINSURF',
            'LIARWHD', 'MOREBV', 'NCB20', 'NONCVXU2', 'NONCVXUN', 'NONDIA', 'NONDQUAR', 'POWELLSG',
            'POWER', 'SCHMVETT', 'SROSENBR', 'TOINTGSS', 'TQUARTIC', 'VAREIGVL', 'WOODS']

problems = ['GENROSE', 'ARWHEAD', 'BROYDN7D', 'CRAGGLVY', 'DIXMAANA1', 'DIXMAANB', 'DIXMAANC', 'DIXMAAND', 'DIXMAANE1', 'DIXMAANF', 'DIXMAANG', 'DIXMAANH', 'DIXMAANI1', 'DIXMAANJ', 'DIXMAANK', 'DIXMAANL', 'DIXMAANM1',
            'DIXMAANN', 'DIXMAANO', 'DIXMAANP', 'EDENSCH', 'ENGVAL1', 'FLETCBV3', "FLETCHCR", 'FMINSURF',
            'LIARWHD', 'MOREBV', 'NCB20', 'NONCVXU2', 'NONCVXUN', 'NONDIA', 'NONDQUAR', 'POWELLSG',
            'POWER', 'SCHMVETT', 'SROSENBR', 'TOINTGSS', 'TQUARTIC', 'VAREIGVL', 'WOODS']

#problems = ['ARWHEAD']


res_tutti = []
for p in problems:
    print('{}'.format(p))
    P = Problem(p)
    res_parz = []
    #res_tutti.append([p, P.n, '', '', '', '', '', '', '', '', ''])
    S = Solver(P)
    res = S.test_problem(solvers,max_iters=5000, eps_grad=eps_grad, gtol_ord=np.inf)
    for i,r in enumerate(res):
        res_tutti.append(r)
        res_parz.append(r)
    r=["--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--"]
    res_tutti.append(r)		
    print(tabulate(res_parz, headers=['Algorithm', 'prob', 'n', 't', 'n_it', 'f_opt',
                               'g_norm', 'fevals', 'gevals', 'nfails', 'nnegeig', 'cosmax'], tablefmt='orgtbl')
          )

table = tabulate(res_tutti, headers=['Algorithm','prob', 'n', 't', 'n_it', 'f_opt',
    'g_norm', 'fevals', 'gevals', 'nfails', 'nnegeig', 'cosmax'], tablefmt = 'orgtbl')
print(table)
print(table)
fr=open("risultati.txt","w")
print(table,file=fr)
fr.close()
table = tabulate(res_tutti, headers=['Algorithm','prob', 'n', 't', 'n_it', 'f_opt',
    'g_norm', 'fevals', 'gevals', 'nfails', 'nnegeig', 'cosmax'], tablefmt = 'latex')
print(table)

'''
# TEST QUADRATIC FUNCTION
cond_number = 1000
matrixSize=300
print('Quadratic - Cond = {}, size = {}'.format(cond_number, matrixSize))
Q = make_random_psd_matrix(matrixSize,ncond=cond_number)
p=np.random.uniform(low=-1, high=1, size=matrixSize)
P = QuadraticProblem(Q,p)
P.set_x0(np.random.uniform(low=-1, high=1, size=matrixSize))
S = Solver(P)

S.test_problem(solvers, max_iters=1000)
print("=====================================================================================================")
'''

''' TEST GENERALIZED ROSENBROCK
n=500
print('ROSENBROCK {} - x0 default'.format(n))
P = Problem('GENROSE',n)
S = Solver(P)
S.test_problem(solvers,max_iters=5000, eps_grad=eps_grad)
print('ROSENBROCK {} - x0 Matlab'.format(n))
P = Problem('GENROSE',n)
# pycutest.print_available_sif_params('GENROSE')
# print(P.get_x0())
x_init = np.array([2*(-1.)**(i+1) for i in range(len(P.get_x0()))])
P.set_x0(x_init)
# print(P.get_x0())
S = Solver(P)
S.test_problem(solvers,max_iters=5000, eps_grad=eps_grad)
'''



''' TEST HILBERT PROBLEMS
matrixSize=10
print('================================================================================================')
print('HILBERT - size: {}'.format(matrixSize))
p=np.random.uniform(low=-1, high=1, size=matrixSize)
P = QuadraticProblem(hilbert(matrixSize), p)
P.set_x0(np.random.uniform(low=-1, high=1, size=matrixSize))
P = HilbertProblem(matrixSize)
S.set_problem(P)
S.test_problem(solvers, max_iters=5000, eps_grad=eps_grad, gtol_ord=np.inf)
'''
