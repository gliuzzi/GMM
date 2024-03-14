import numpy as np
import pycutest
from scipy.optimize import minimize, approx_fprime, fmin_bfgs, fmin_l_bfgs_b
from time import time, sleep
from dfbox1 import der_free_method
from nesterov import nesterov
from scipy.linalg import hilbert
from tabulate import tabulate
from Newton_Nesterov import *
from Newton import *
import matplotlib.pyplot as plt

class Problem:
    def __init__(self, name, n=None, approxg=False):
        if n:
            self.__p = pycutest.import_problem(name, sifParams={'N':n})
        else:
            self.__p = pycutest.import_problem(name)
        self.n = self.__p.n
        self.m = self.__p.m
        self.x0 = self.__p.x0
        self.name = self.__p.name
        self.approxg = approxg

    def f(self,x):
        return self.__p.obj(x)

    def f_g(self, x):
    	if self.approxg:
    		f = self.__p.obj(x)
    		return f,self.gnum(x,f)

    	else:
	        return self.__p.obj(x, gradient=True)

    def g(self,x):
	    if self.approxg:
		    f = self.__p.obj(x)
		    gr = self.gnum(x,f)
	    else:
		    _, gr = self.__p.obj(x,gradient=True)
	    return gr

    def gnum(self,x,f):
        step = 1.e-3
        xp = np.copy(x)
        g = np.zeros(self.n)
        for i in range(self.n):
            xp[i] = x[i] + step
            fp = self.f(xp)
            g[i] = (fp - f)/step
            xp[i] = x[i]
        return g

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
        #print("method = ",self.method)        
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
        elif self.method == 'QPS_New':   
            sol, info = self.solvePlaneSearchConvergent()
        elif  self.method == 'QPS-Newton':
            sol, info = self.solvePlaneSearch_Newton()
        elif  self.method == 'QPS-Newton_Matteo':
            sol, info = self.solvePlaneSearch_Newton(var=1) 
        elif  self.method == 'QPS-Barzilai':
            sol, info = self.solvePlaneSearch_Newton(var=2)         
        elif self.method == 'QPS-approx':
            sol, info = self.solvePlaneSearch(prova=True)
        elif self.method == 'QPS-iterative':
            sol, info = self.solvePlaneSearch(iterative=True)
        elif self.method == 'QPS-roma':
            sol, info = self.solvePlaneSearch_roma()
        elif self.method == 'QPS-roma-box':
            sol, info = self.solvePlaneSearch_roma_box(prova=True)
        elif self.method == 'QPS-matteo-box':
            sol, info = self.solvePlaneSearch_roma_box()
        elif self.method == 'NESTEROV-bs_dfbox':
            sol, info = self.solvePlaneSearch_nesterov(der_free=True)
        elif self.method == 'NESTEROV-bs_dfbox1':
            sol, info = self.solvePlaneSearch_nesterov1(der_free=True)
        elif self.method == 'NESTEROV-bs_lbfgs':
            sol, info = self.solvePlaneSearch_nesterov(der_free=False)
        elif self.method == 'NESTEROV-bs_Newton':
            sol, info = self.solvePlaneSearch_Nesterov_Newton()			
        elif self.method == 'CGlike':
            sol, info = self.solvePlaneSearch_CG()
        elif self.method == 'Barzilai':
            sol, info = self.solveBarzilaiBorwein()
        elif self.method == 'ConjGrad':
            sol, info = self.solveConjGrad()
        elif self.method == 'Quasi-Newton':
            sol, info = self.solveQuasiNewton()
        elif self.method == 'DFBOX':
            DFBOX = der_free_method(self.f,-np.inf*np.ones(self.problem.n),np.inf*np.ones(self.problem.n),maxfev=25000,tol=self.grad_tol)
            sol, info = DFBOX.sd_box(self.problem.get_x0())
        elif self.method == 'NESTEROV':
            NESTEROV = nesterov(self.f,self.g,self.problem.n,self.grad_tol,self.max_iters)
            sol, info = NESTEROV.run(self.problem.get_x0())
        elif self.method == 'scipy_bfgs':
            bfgs = minimize(self.f, self.problem.get_x0(), jac=self.g, method="BFGS", options={"disp": False, "gtol": self.grad_tol, "maxiter": self.max_iters, 'norm': self.gtol_ord})
            info = {"iters": bfgs.nit, "f": bfgs.fun, "g_norm": np.linalg.norm(bfgs.jac, self.gtol_ord)}
            sol = bfgs.x
        elif self.method == 'scipy_lbfgs':
            if not np.isinf(self.gtol_ord):
                print('CANNOT SET DIFFERENT NORM THAN INFTY-NORM FOR L-BFGS')
            lbfgs = minimize(self.f, self.problem.get_x0(), jac=self.g, method="L-BFGS-B", options={"iprint": -1, "maxcor": 10, "gtol": self.grad_tol, "ftol": 1e-50, "maxiter": self.max_iters, "maxls": 20, 'maxfun': 1e15})
            #print(lbfgs)
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

    def bidimensional_search_box(self, x, d1, d2, alpha0, beta0=0, multistart=0, deriv_free=False, maxfev=10,nesterov=False):
        nrd1 = np.linalg.norm(d1)
        nrd2 = np.linalg.norm(d2)
        def f2(ab):
            return self.f(x + ab[0] * d1 + ab[1] * d2)
        
        def f2nesterov(ab):
            if ab[1] == 0:
                y = x
                g = -d1
            else:
                y = x + ab[1]*d2
                g = self.g(y)
            return self.f(y-ab[0]*g)
            
        def g2(ab):
            g = self.g(x + ab[0] * d1 + ab[1] * d2)
            return [np.dot(g, d1), np.dot(g, d2)]

        def fg2(ab):
            g = self.g(x + ab[0] * d1 + ab[1] * d2)
            return self.f(x + ab[0] * d1 + ab[1] * d2), np.array([np.dot(g, d1), np.dot(g, d2)])

        def fg2_approx(ab):
            fk = self.f(x + ab[0] * d1 + ab[1] * d2)
            f1 = self.f(x + ab[0] * d1 + ab[1] * d2 + 1.e-5*d1/nrd1)
            f2 = self.f(x + ab[0] * d1 + ab[1] * d2 + 1.e-5*d2/nrd2)
            gk = np.array([(f1 - fk), (f2 - fk)]) / 1.e-5
            return fk, gk
        def inner_BB(ab):
            BB = nmgrad2(2, ab, 1.e-5, 5, 0, fg2)
            x, f, ng, ifail, x_current, f_current, g_current = BB.minimize()
            return x, f
        def inner_lbfgsb(ab):
            return minimize(fg2, ab, jac=True, method="L-BFGS-B",
                             options={"iprint": -1, "maxcor": 5, "gtol": self.grad_tol, "ftol": 1e-30,"maxiter": 5})
            #return minimize(f2, ab, jac=g2, bounds=[[ab[0]-10, ab[0]+10], [ab[1]-10, ab[1]+10]], method="L-BFGS-B",
            #                 options={"iprint": -1, "maxcor": 5, "gtol": self.grad_tol, "ftol": 1e-30,"maxiter": 10})
            #return minimize(f2, ab, jac=g2, bounds=[[0,10], [0,+10]], method="L-BFGS-B",
            #                 options={"iprint": -1, "maxcor": 5, "gtol": self.grad_tol, "ftol": 1e-30,"maxiter": 10})

        class my_solution:
            def __init__(self,n):
                self.x = np.nan*np.ones(n)
                self.fun = np.inf

        def inner_nesterov(ab):
            DFBOX2 = der_free_method(f2nesterov,-np.inf*np.ones(2),np.inf*np.ones(2),maxfev=100,tol=1.e-3)
            sol, info = DFBOX2.sd_box(ab)
            solution = my_solution(2)
            solution.x = np.copy(sol)
            solution.fun = info["f"]
            return solution
            #return minimize(f2nesterov, ab, method="Nelder-Mead", options={"disp": False, "maxfev": maxfev})
            
        def inner_solve(ab):
            if not deriv_free:
                return minimize(f2, ab, jac=g2, method="CG", options={"disp": False, "gtol": 1e-3, "maxiter": 10})
            else:
                return minimize(f2, ab, method="Nelder-Mead", options={"disp": False, "maxfev": maxfev})
                #return minimize(f2, ab, method="Nelder-Mead", bounds=[[ab[0]-10, ab[0]+10], [ab[1]-10, ab[1]+10]],
                #                options={"disp": False, "maxfev": maxfev})

        #        alpha0=0.5
        #        beta0=0.
        #solution = inner_solve([alpha0, beta0])
             
        if nesterov:
            #print('call Nesterov')
            solution = inner_nesterov([alpha0, beta0])

            #input()
        else:
            solution = inner_lbfgsb([alpha0,beta0])
            
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
        return best, best_f

    def bidimensional_search_nesterov(self, x, x1, g, g1, g_norm,alpha1, alpha0, beta0=0, maxfev=10, der_free=True):
        
        def f2nesterov(ab):
            d2 = x-x1
            y = x - ab[0]*g
            return self.f(y + ab[1]*(y - x1 + alpha1*g1))

        def fg2nesterov(ab):
            d2 = x-x1
            y = x - ab[0]*g
            yy = y + ab[1]*(y - x1 + alpha1*g1)
            f = self.f(yy)
            
            eps = 1.e-6
            z = np.copy(ab)
            z[0] += eps
            y = x - z[0]*g
            yy = y + z[1]*(y - x1 + alpha1*g1)
            f1 = self.f(yy)
            z[0] = ab[0]
            z[1] += eps
            y = x - z[0]*g
            yy = y + z[1]*(y - x1 + alpha1*g1)
            f2 = self.f(yy)
            
            return f,np.array([(f1-f)/eps,(f2-f)/eps])
            
        class my_solution:
            def __init__(self,n):
                self.x = np.nan*np.ones(n)
                self.fun = np.inf

        def inner_lbfgsb(ab):
            return minimize(fg2nesterov, ab, jac=True, method="L-BFGS-B",
                             options={"iprint": -1, "maxcor": 5, "gtol": self.grad_tol, "ftol": 1e-30,"maxiter": 1})

        def inner_nesterov(ab):
            DFBOX2 = der_free_method(f2nesterov,-np.inf*np.ones(2),np.inf*np.ones(2),g_norm,maxfev=10,tol=1.e-3)        
            #DFBOX2 = der_free_method(f2nesterov,-np.inf*np.ones(2),np.inf*np.ones(2),maxfev=100,tol=1.e-3)
            sol, info = DFBOX2.sd_box(ab)
            solution = my_solution(2)
            solution.x = np.copy(sol)
            solution.fun = info["f"]
            return solution
            #return minimize(f2nesterov, ab, method="Nelder-Mead", options={"disp": False, "maxfev": maxfev})

        #print('call Nesterov')
        if der_free:
            solution = inner_nesterov([alpha0, beta0])
        else:
            alfa0=0.
            beta0=0.
            solution = inner_lbfgsb([alpha0, beta0])
        #input()
            
        best, best_f = solution.x, solution.fun
        # best, best_f = inner_BB(np.array([alpha0, beta0]))
        return best, best_f

    def bidimensional_search_nesterov1(self, x, x1, g, g1, g_norm,alpha1, alpha0, beta0=0, maxfev=10, der_free=True):
        
        def f2nesterov(ab):           
            g = self.g(x + ab[1]*(x-x1))
            new_x = x - ab[0]*g + ab[1]*(x-x1)
            return self.f(new_x)

        def fg2nesterov(ab):
            d2 = x-x1
            y = x - ab[0]*g
            yy = y + ab[1]*(y - x1 + alpha1*g1)
            f = self.f(yy)
            
            eps = 1.e-6
            z = np.copy(ab)
            z[0] += eps
            y = x - z[0]*g
            yy = y + z[1]*(y - x1 + alpha1*g1)
            f1 = self.f(yy)
            z[0] = ab[0]
            z[1] += eps
            y = x - z[0]*g
            yy = y + z[1]*(y - x1 + alpha1*g1)
            f2 = self.f(yy)
            
            return f,np.array([(f1-f)/eps,(f2-f)/eps])
            
        class my_solution:
            def __init__(self,n):
                self.x = np.nan*np.ones(n)
                self.fun = np.inf

        def inner_lbfgsb(ab):
            return minimize(fg2nesterov, ab, jac=True, method="L-BFGS-B",
                             options={"iprint": -1, "maxcor": 5, "gtol": self.grad_tol, "ftol": 1e-30,"maxiter": 1})

        def inner_nesterov(ab):
            DFBOX2 = der_free_method(f2nesterov,-np.inf*np.ones(2),np.inf*np.ones(2),g_norm,maxfev=10,tol=1.e-3)
            #DFBOX2 = der_free_method(f2nesterov,np.zeros(2),np.inf*np.ones(2),maxfev=100,tol=1.e-6)
            #print(DFBOX2.lb," ",DFBOX2.ub)
            sol, info = DFBOX2.sd_box(ab)
            solution = my_solution(2)
            solution.x = np.copy(sol)
            solution.fun = info["f"]
            return solution
            #return minimize(f2nesterov, ab, method="Nelder-Mead", options={"disp": False, "maxfev": maxfev})

        #print('call Nesterov')
        if der_free:
            solution = inner_nesterov([alpha0, beta0])
        else:
            alfa0=0.
            beta0=0.
            solution = inner_lbfgsb([alpha0, beta0])
        #input()
            
        best, best_f = solution.x, solution.fun
        # best, best_f = inner_BB(np.array([alpha0, beta0]))
        return best, best_f

    def bidimensional_search_Newton(self,finiz, f_1, g_1, x, d1, d2,n_iter_glob , alpha0, beta0, alpha_1, beta_1, var,multistart=0, deriv_free=False, maxfev=10):
        def f2(ab):
            return self.f(x + ab[0] * d1 + ab[1] * d2)

        def g2(ab):
            g = self.g(x + ab[0] * d1 + ab[1] * d2)
            return [np.dot(g, d1), np.dot(g, d2)]

        def fg2(ab):
            g = self.g(x + ab[0] * d1 + ab[1] * d2)
            return self.f(x + ab[0] * d1 + ab[1] * d2), np.array(np.dot(g, d1), np.dot(g, d2))
			
        def inner_NN(ab,ab_1,f_1,g_1):
            # print('d1=',d1)
            # gapp=-np.dot(d1.T,d1)
            #print('alpha0=',alpha0)
            # print('grad app',gapp)			
            NN=Newton(2,finiz,f_1,g_1,ab,ab_1,1.e-3*self.grad_tol*self.grad_tol,1,False,f2,d1,d2,n_iter_glob,var)
            f,x=NN.minimize()
            return f,x
        #print('alpha0=',alpha0)
        #print('beta0=',beta0)
        #print('alpha_1=',alpha_1)
        #print('beta_1=',beta_1)
        best_f,best = inner_NN([alpha0,beta0],[alpha_1, beta_1],f_1,g_1)
        return best,best_f 

    def solvePlaneSearch_Newton(self, var=0):
        xk = self.problem.get_x0()
        n_iters = 0
        num_fails = 0
        xk_1 = np.copy(xk)
        f_1, g_1 = self.f_g(xk_1)
        alpha, beta = 0, 0
        alpha_1, beta_1 = 0, 0
        aArm = 1
        g_norm = np.inf
        while True:
            if n_iters ==0:
                f, g = f_1, g_1		
            else:
                f = fExp 
                g = self.g(xk)
            g_norm_prev = g_norm
            g_norm = np.linalg.norm(g, self.gtol_ord)
            #norma_g=np.sqrt(np.dot(g.T,g))
            #print(' gnorm=',g_norm)
            #print(' gnorm2=',norma_g)
            #input()
            # if np.dot(g.T,(xk-xk_1)) > 0.:
                #xk_1=xk
                # xk_1=2.*xk-xk_1
            #if g_norm >  1.e+9:
                ##print(' gnorm=',g_norm)
                #g = g / g_norm*1.e+9
            if g_norm < self.grad_tol or n_iters >= self.max_iters or f == -np.inf or np.isnan(f):
                break
       
            #gnr = np.linalg.norm(g)
            #print(' gnr2=',gnr*gnr)
            ab = np.zeros(2)
            ab[0]=0.
            ab[1]=0.  
			
            #if prova:
                #ab = self.quadratic_plane_search_prova(xk, xk_1, f, f_1, g, alpha, beta)
            #else:
                #ab = self.quadratic_plane_search(xk, xk_1, f, f_1, g, alpha, beta)
                #ab[0]=np.maximum(0.,ab[0])

            if True:
            #if fExp >= f:
                fExp=f
                ab,fExp = self.bidimensional_search_Newton(fExp,f_1, g_1, xk, -g, xk - xk_1,n_iters, ab[0], ab[1], alpha_1, beta_1, var, deriv_free=True, maxfev=5)
                #input()
                #if (ab[0]< 1.e-12):
                #    ab[0]=1.e-12
                    #print('alfa piccolo')
                #    fExp = self.f(xk - ab[0] * g + ab[1] * (xk - xk_1))
                if fExp == -np.inf:
                    f=fExp
                    break
            if fExp <= f:
                alpha_1, beta_1 = ab[0], ab[1]
                #alpha_1, beta_1 = alpha, beta 
                alpha, beta = ab[0], ab[1]
                #print('alpha dopo=',alpha)
                #print('beta dopo=',beta)
                #print('alpha_1 dopo =',alpha_1)
                #print('beta_1 dopo=',beta_1)
                aArm = max(np.abs(alpha), 10 * self.min_step)
                #aArm = alpha
            else:
                num_fails += 1
                count_barzilai = self.recovery_steps
                alpha_start = self.bb_step(xk - xk_1, g - g_1, inverse=True)
                aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma,
                                         min_step=self.min_step)
                alpha_1, beta_1 = alpha, beta 
                alpha, beta = aArm, 0         

            self.alfas.append(alpha)
            self.betas.append(beta)
            new_x = xk - alpha * g + beta * (xk - xk_1)
            xk_1 = xk
            f_1, g_1 = f, g
            xk = new_x
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm, "nfails": num_fails, "nnegeig": self.nnegeig, "cosmax": '----'}

    def solvePlaneSearch_roma_box(self, iterative=False, prova=False):
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
                    if prova:
                        ab = self.quadratic_plane_search_prova(xk, xk_1, f, f_1, g, alpha, beta)
                        #ab = np.zeros(2)
                        #ab[0]=0.
                        #ab[1]=0.
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
                        #ab[0]=0.
                        #ab[1]=0.
                        ab, fExp = self.bidimensional_search_box(xk, -g, xk - xk_1, alpha0=ab[0], beta0=ab[1], deriv_free=True, maxfev=20)
                        #fExp = self.f(xk - ab[0] * g + ab[1] * (xk - xk_1))
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
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm, "nfails": num_fails, "nnegeig": "--", "cosmax": "--"}
    def solvePlaneSearchConvergent(self):
        xk = self.problem.get_x0()
        n_iters=0
        xk_1 = np.copy(xk)
        f_1, g_1 = self.f_g(xk_1)
        alpha, beta = 0,0
        aArm = 1
        g_norm = np.inf


        while True:
            self.prev_g = g_1
            f,g = self.f_g(xk)

            g_norm_prev = g_norm
            g_norm = np.linalg.norm(g,self.gtol_ord)
            if g_norm > 1e6:
                g = g/g_norm
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break
            ab = self.quadratic_plane_search_robust(xk, xk_1, f, f_1, g, alpha, beta)

            alpha, beta = ab[0], ab[1]
            new_x = xk - alpha*g + beta*(xk-xk_1)
            xk_1 = xk
            f_1, g_1 = f, g
            xk = new_x
            n_iters += 1

        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm}


    def quadratic_plane_search_robust(self, xk, xk_1, f, f_1, g, alpha, beta):
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
        if np.isinf(fC):
            fC = 1e35
        if np.isinf(fD):
            fD = 1e35

        bc = 2*(gab[1]+fB-fA)
        ba = 2/(alpha**2)*(fD-alpha*gab[0]-fA)
        bb = (fC-fA-gab[0]*alpha-gab[1]*beta-0.5*alpha*alpha*ba-0.5*beta*beta*bc)/(alpha*beta)



        Bab = np.array([[ba,bb],[bb,bc]])

        nonconvex = False
        '''try:
            eigvals, eigvects = eigh(Bab)
            if np.min(eigvals)<1e-8:
                Bab = np.identity(2)
            # print(eigvals)

            if np.min(eigvals)<1e-8:
                eigvects = eigvects.T
                nonconvex = True
                # Bab = sum(max(1e-8,eigvals[i])*np.outer(eigvects[i],eigvects[i]) for i in range(len(eigvals)))
        except ValueError:
            # print('error case')
            Bab = np.identity(2)
        '''

        try:
            eigvals = np.linalg.eigvals(Bab)
            if np.min(eigvals)<1e-8:
                nonconvex = True
        except:
            Bab = np.identity(2)


        if not nonconvex:
            try:
                solution_closed = np.linalg.solve(Bab, -gab)
            except np.linalg.LinAlgError:
                solution_closed = np.linalg.lstsq(Bab, -gab, rcond=None)[0]
            best = solution_closed

        else:
            best = np.array([self.bb_step(xk-xk_1, g-self.prev_g, inverse=True), 0.])

        stepsize = 1
        stepsize_red = 0.5
        best*=stepsize

        dot_g_d = np.dot(gab, best)
        f_trial = self.f(xk+best[0]*d1+best[1]*d2)
        while(f_trial > f + 1e-5*stepsize*dot_g_d and stepsize>1e-30):
            stepsize *= stepsize_red
            best *= stepsize_red
            f_trial = self.f(xk+best[0]*d1+best[1]*d2)

        return best
        
        
        
        

    def solvePlaneSearch_nesterov(self,der_free=True):
        xk = self.problem.get_x0()
        iterative=False
        prova=False
        n_iters=0
        num_fails = 0
        xk_1 = np.copy(xk)
        f_1, g_1 = self.f_g(xk_1)
        alpha, beta = 0,0
        alpha1 = alpha
        aArm = 1
        g_norm = np.inf
        count_barzilai = 0
        while True:
            f,g = self.f_g(xk)
            #print('f = ',f)
            g_norm_prev = g_norm
            g_norm = np.linalg.norm(g,self.gtol_ord)
            #if g_norm > 1.e6:
                #g = g/g_norm
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break
            if count_barzilai<1:
                # alfa=0.
                # beta=0.
                #ab, fExp = self.bidimensional_search_box(xk, -g, xk-xk_1, alpha0=alpha,beta0=beta,maxfev=100,nesterov=True)
                ab, fExp = self.bidimensional_search_nesterov(xk, xk_1, g, g_1, g_norm, alpha1, alpha0=0.0, maxfev=100, der_free=der_free)
                #ab, fExp = self.bidimensional_search_nesterov(xk, xk_1, g, g_1, alpha1, alpha0=0.0, maxfev=10, der_free=der_free)
                #ab, fExp = self.bidimensional_search_nesterov(xk, xk_1, g, g_1, alpha1, alpha0=alpha,beta0=beta, maxfev=10, der_free=der_free)
                #print(f,fExp)
                #input()
                #print(ab,np.shape(g),np.shape(xk-xk_1),fExp)
                #fExp = self.f(xk-ab[0]*g+ab[1]*(xk-xk_1))
                # if (ab[0]< 1.e-7):
                    # ab[0]=1.e-7
                    # ab[1]=0.
                    #print('alfa piccolo')
                    # fExp = self.f(xk - ab[0] * g)
                if fExp < f:
                    alpha, beta = ab[0], ab[1]
                    aArm = max(np.abs(alpha), 10*self.min_step)
                else:
                    num_fails += 1
                    count_barzilai = self.recovery_steps
                    alpha_start = self.bb_step(xk-xk_1, g-g_1, inverse=True)
                    aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma, min_step=self.min_step)
                    alpha, beta = aArm, 0
                    #print('alpha, beta = ',alpha, beta )
                    #print('g_norm=',g_norm)
                    #input()
            else:
                count_barzilai -= 1
                alpha_start = self.bb_step(xk-xk_1, g-g_1, inverse=True)
                aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma, min_step=self.min_step)
                
                alpha, beta = aArm, 0
                
            #g = self.g(xk + beta*(xk-xk_1))
            #new_x = xk - alpha*g + beta*(xk-xk_1)
            y = xk - alpha*g
            new_x = y + beta*(y - xk_1 + alpha1*g_1)
            alpha1 = alpha
            xk_1 = xk
            f_1, g_1 = f, g
            xk = new_x
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm, "nfails": num_fails, "nnegeig": "--", "cosmax": "--"}

    def solvePlaneSearch_nesterov1(self,der_free=True):
        xk = self.problem.get_x0()
        iterative=False
        prova=False
        n_iters=0
        num_fails = 0
        xk_1 = np.copy(xk)
        f_1, g_1 = self.f_g(xk_1)
        alpha, beta = 0,0
        alpha1 = alpha
        aArm = 1
        g_norm = np.inf
        count_barzilai = 0
        while True:
            f,g = self.f_g(xk)
            #print('f = ',f)
            g_norm_prev = g_norm
            #g_norm = np.linalg.norm(g,self.gtol_ord)
            g_norm = np.linalg.norm(g,2)
            #print('g_norm 1=',g_norm)
            #if g_norm > 1.e6:
                #print('g_norm 1=',g_norm)
                #g = g/g_norm
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break
            if count_barzilai<1:
                #print("--------------------------------")
                # alfa=0.
                # beta=0.
                #ab, fExp = self.bidimensional_search_box(xk, -g, xk-xk_1, alpha0=alpha,beta0=beta,maxfev=100,nesterov=True)
                ab, fExp = self.bidimensional_search_nesterov1(xk, xk_1, g, g_1, g_norm, alpha1, alpha0=0.0, maxfev=100, der_free=der_free)
                #ab, fExp = self.bidimensional_search_nesterov(xk, xk_1, g, g_1, alpha1, alpha0=alpha,beta0=beta, maxfev=10, der_free=der_free)
                #input()
                #print(ab,np.shape(g),np.shape(xk-xk_1),fExp)
                #fExp = self.f(xk-ab[0]*g+ab[1]*(xk-xk_1))
                #if (ab[0]< 1.e-7):
                    #ab[0]=1.e-7
                    #ab[1]=0.
                    #print('alfa piccolo')
                    #fExp = self.f(xk - ab[0] * g)
                    #print('fExp =',fExp,fp)
                if fExp < f:
                    alpha, beta = ab[0], ab[1]
                    aArm = max(np.abs(alpha), 10*self.min_step)
                else:
                    num_fails += 1
                    count_barzilai = self.recovery_steps
                    alpha_start = self.bb_step(xk-xk_1, g-g_1, inverse=True)
                    aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma, min_step=self.min_step)
                    alpha, beta = aArm, 0
                    #print('alpha, beta = ',alpha, beta )
                    #print('g_norm=',g_norm)
            else:
                count_barzilai -= 1
                alpha_start = self.bb_step(xk-xk_1, g-g_1, inverse=True)
                aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma, min_step=self.min_step)
                #print('aArm=',aArm)
                alpha, beta = aArm, 0
            #print('alpha beta=',alpha,beta)   
            g = self.g(xk + beta*(xk-xk_1))
            new_x = xk - alpha*g + beta*(xk-xk_1)
            #y = xk - alpha*g
            #new_x = y + beta*(y - xk_1 + alpha1*g_1)
            alpha1 = alpha
            xk_1 = xk
            f_1, g_1 = f, g
            xk = new_x
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm, "nfails": num_fails, "nnegeig": "--", "cosmax": "--"}

    def solvePlaneSearch_Nesterov_Newton(self):
        xk = self.problem.get_x0()
        n_iters = 0
        num_fails = 0
        xk_1 = np.copy(xk)
        d1 = 0.
        f_1, g_1 = self.f_g(xk_1)
        alpha, beta = 0., 0.
        aArm = 1

        g_norm = np.inf
        while True:
            if n_iters ==0:
                f, g = f_1, g_1		
            else:
                f, g = self.f_g(xk)
            g_norm_prev = g_norm
            g_norm = np.linalg.norm(g, self.gtol_ord)
            #print('g main=',g)			
            # if np.dot(g.T,(xk-xk_1)) > 0.:
                #xk_1=xk
                # xk_1=2.*xk-xk_1
            #if g_norm >  1.e+9:
                ##print(' gnorm=',g_norm)
                #g = g / g_norm*1.e+9
            if g_norm < self.grad_tol or n_iters >= self.max_iters:
                break
       
            #gnr = np.linalg.norm(g)
            ab = np.zeros(2)
            
            #if prova:
                #ab = self.quadratic_plane_search_prova(xk, xk_1, f, f_1, g, alpha, beta)
            #else:
                #ab = self.quadratic_plane_search(xk, xk_1, f, f_1, g, alpha, beta)
                #ab[0]=np.maximum(0.,ab[0])

            if True:
            #if fExp >= f:
                fExp=f
                ab,fExp = self.bidimensional_search_Nesterov_Newton(fExp,xk, g, d1,n_iters, alpha0=ab[0], beta0=ab[1],  maxfev=5)
                #print('ab[0]=',ab[0],'   ab[1]=',ab[1])
                if (ab[0]< 1.e-9):
                    ab[0]=1.e-9
                    ab[1]=0.
                    #print('alfa piccolo')
                    fExp = self.f(xk - ab[0] * g + ab[1] * (d1 - ab[0]*g))
                    #input()
            if fExp <= f:
                alpha, beta = ab[0], ab[1]
                aArm = max(np.abs(alpha), 10 * self.min_step)
            else:
                num_fails += 1
                #print('la minimizzazione bidiemnsionale non trova discesa')
                #print('f=',f,'    fExp=',fExp)
                #input()
                count_barzilai = self.recovery_steps
                alpha_start = self.bb_step(xk - xk_1, g - g_1, inverse=True)
                aArm = self.armijoLS(x=xk, f=f, g=g, d=-g, alpha0=alpha_start, gamma=self.gamma,
                                         min_step=self.min_step)
                alpha, beta = aArm, 0         

            self.alfas.append(alpha)
            self.betas.append(beta)
 						
            y = xk - alpha*g
            new_x = y + beta*(d1 - alpha*g)			
            xk = np.copy(new_x)
            d1=xk-y
            #('d1 main=',d1)
            n_iters += 1
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm, "nfails": num_fails, "nnegeig": "--", "cosmax": "--"}

    def bidimensional_search_Nesterov_Newton(self, finiz, x, g, d1, n_iter_glob, alpha0, beta0=0, maxfev=10):
 
        def f2nesterov(ab):
            y = x - ab[0]*g
            return self.f(y + ab[1]*(d1- ab[0]*g))

        def fg2nesterov(ab):
            y = x - ab[0]*g
            yy = y + ab[1]*(d1- ab[0]*g)
            f = self.f(yy)
         
            eps = 1.e-6
            z = np.copy(ab)
            z[0] += eps
            y = x - z[0]*g
            yy = y + z[1]*(d1- ab[0]*g)
            f1 = self.f(yy)
            z[0] = ab[0]
            z[1] += eps
            y = x - z[0]*g
            yy = y + z[1]*(d1- ab[0]*g)
            f2 = self.f(yy)           
            return f,np.array([(f1-f)/eps,(f2-f)/eps])

        def g2nesterov(ab):
            y = x - ab[0]*g
            yy = d1- ab[0]*g
            gtot = self.g(y + ab[1]*yy)
            return [-(1.+ab[1])*np.dot(gtot, g), np.dot(gtot, yy)]
			
        def inner_NN(ab):
            # print('d1=',d1)
            # gapp=-np.dot(d1.T,d1)
            # #print('d1=',d1)
            # print('grad app',gapp)			
            NN=Newton_Nesterov(2,finiz,ab,1.e-3*self.grad_tol*self.grad_tol,2,False,f2nesterov,g,d1,n_iter_glob)
            f,x=NN.minimize()
            return f,x

        best_f,best = inner_NN([alpha0,beta0])
        return best,best_f 

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
        
        # min_sample_val  = 1e-8
        # if np.abs(alpha) <= min_sample_val:
            # alpha = min_sample_val if alpha >= 0 else -min_sample_val
        # if np.abs(beta) <= min_sample_val:
            # beta = min_sample_val if beta >= 0 else -min_sample_val

        # if False:
            # print(alpha,' ',beta)
        # fA = f
        # fB = f_1
		# #fC = self.f(xk+alpha*d1+beta*d2)
		# #fD = self.f(xk+alpha*d1)

		# #A = [0.5*alpha**2, 0.5*beta**2, alpha*beta]
		# #A = np.vstack((A,[0.5*alpha**2, 0., 0.]))
		# #A = np.vstack((A,[0., 0.5, 0.]))
		# #A = np.vstack((A,[0., 0.5*beta**2, 0.]))
		# #A = np.vstack((A,[0.5, 0., 0.]))
		# #rhs = [fC-fA-gab[0]*alpha-gab[1]*beta]
		# #rhs = np.vstack((rhs,[fD-fA-gab[0]*alpha]))
		# #rhs = np.vstack((rhs,[fB-fA+gab[1]]))
		# #rhs = np.vstack((rhs,[self.f(xk+beta*d2)-f-gab[1]*beta]))
		# #rhs = np.vstack((rhs,[self.f(xk-d1)-f+gab[0]]))
        # delta = 1.e-3
        # A = [0.5*delta**2, 0.0, 0.0]
        # rhs = [self.f(xk+delta*d1)-f-gab[0]*delta]
        # A = np.vstack((A,[0.5*delta**2,0.5*delta**2, delta*delta]))
        # rhs = np.vstack((rhs,[self.f(xk+delta*d1+delta*d2)-f-gab[0]*delta-gab[1]*delta]))
		# #rhs = np.vstack((rhs,[self.f(xk-delta*d1-delta*d2)-f+gab[0]*delta+gab[1]*delta]))
		# #A = np.vstack((A,[0.5*delta**2, 0.5*delta**2, -delta*delta]))
		# #rhs = np.vstack((rhs,[self.f(xk-delta*d1+delta*d2)-f+gab[0]*delta-gab[1]*delta]))
		# #A = np.vstack((A,[0.0, 0.5, 0.0]))
		# #rhs = np.vstack((rhs,[f_1-f+gab[1]]))
        # A = np.vstack((A,[0.0, 0.5*delta**2, 0.0]))
        # rhs = np.vstack((rhs,[self.f(xk+delta*d2)-f-gab[1]*delta]))
        # if False:
            # A = np.vstack((A,[0.5*delta**2, 0.5*delta**2, -delta*delta]))
            # rhs = np.vstack((rhs,[self.f(xk-delta*d1+delta*d2)-f+gab[0]*delta-gab[1]*delta]))
            # A = np.vstack((A,[0.5*delta**2, 0.0, 0.0]))
            # rhs = np.vstack((rhs,[self.f(xk-delta*d1)-f+gab[0]*delta]))
            # A = np.vstack((A,[0.5*delta**2, 0.5*delta**2, delta*delta]))
            # rhs = np.vstack((rhs,[self.f(xk-delta*d1-delta*d2)-f+gab[0]*delta+gab[1]*delta]))
            # A = np.vstack((A,[0.0, 0.5*delta**2, 0.0]))
            # rhs = np.vstack((rhs,[self.f(xk-delta*d2)-f+gab[1]*delta]))
            # A = np.vstack((A,[0.5*delta**2, 0.5*delta**2, -delta*delta]))
            # rhs = np.vstack((rhs,[self.f(xk+delta*d1-delta*d2)-f-gab[0]*delta+gab[1]*delta]))
		# #print(A)
		# #print(rhs)
		# #print(self.f(xk-d1),' ',f,' ',gab[0])
		# #input()
		# #sol = np.linalg.pinv(A).dot(rhs)
		# #print(sol)
        # try:
			# #sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
            # sol = np.linalg.solve(A, rhs)
        # except np.linalg.LinAlgError:
            # sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        # if False:
            # print(A.dot(sol))
            # print(rhs)
			# #input()
        # ba = sol[0,0]
        # bc = sol[1,0]
        # bb = sol[2,0]
		
        # ba2 = sol[0,0]
        # bc2 = sol[1,0]
        # bb2 = sol[2,0]		

		# # bc1 = 2*(gab[1]+fB-fA)
		# # ba1 = 2/(alpha**2)*(fD-alpha*gab[0]-fA)
		# # bb1 = (fC-fA-gab[0]*alpha-gab[1]*beta-0.5*alpha*alpha*ba1-0.5*beta*beta*bc1)/(alpha*beta)

		# # sol[0,0] = ba1
		# # sol[1,0] = bc1
		# # sol[2,0] = bb1
        # if False:
            # print(A.dot(sol))
            # print(rhs)

        # if False: #bc != bc1 or ba != ba1 or bb != bb1:
            # print(bc,' ',bc1)
            # print(ba, ' ', ba1)
            # print(bb, ' ', bb1)
            # input()
				
        delta1=1.e-3
        delta2=1.e-3		
        f1=self.f(xk+delta1*d1)
        f2=self.f(xk+delta2*d2)
        f3=self.f(xk+delta1*d1+delta2*d2)
        ba = (2./delta1)*((f1-f)/delta1-gab[0])
        bc = (2./delta2)*((f2-f)/delta2-gab[1])
        bb = (1./delta2)*((f3-f)/delta1)-(1./delta2)*((f1-f)/delta1)-(1./delta1)*((f2-f)/delta2)
        if False: #bc != bc1 or ba != ba1 or bb != bb1:
            print(bc,' ',bc2)
            print(ba, ' ', ba2)
            print(bb, ' ', bb2)
            input()			
        Bab = np.array([[ba, bb], [bb, bc]])

        #try:
        #    EIGV, EIGW = np.linalg.eig(Bab)
        #except np.linalg.LinAlgError:
        #    print(A)
        #    print(rhs)
        #    print(Bab)
        #    EIGV = np.zeros(2)
        #    input()

        #if min(EIGV) < 0:
            #print(EIGV)
        #    self.nnegeig += 1
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
            #print('aux =',aux)
            #print('np.dot(y,y)=',np.dot(y,y))
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
            sol, info = self.solve(method=solver, eps_grad=eps_grad, max_iters=max_iters, gamma=gamma, min_step=1e-24, gtol_ord=gtol_ord)
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
solvers = ['QPS', 'QPS-approx', 'QPS-roma-box', 'scipy_lbfgs']
solvers = ['scipy_cg']
solvers = ['QPS', 'QPS-approx', 'QPS-matteo-box', 'QPS-roma-box', 'scipy_lbfgs', 'scipy_cg']
solvers = ['DFBOX']
solvers = ['QPS', 'QPS-matteo-box', 'scipy_lbfgs', 'scipy_cg']
solvers = ['NESTEROV']
#solvers = ['NESTEROV','NESTEROV-bs_dfbox', 'NESTEROV-bs_lbfgs', 'NESTEROV-bs_Newton','scipy_lbfgs']
solvers = ['NESTEROV','NESTEROV-bs_dfbox','NESTEROV-bs_dfbox1', 'NESTEROV-bs_lbfgs', 'NESTEROV-bs_Newton','scipy_lbfgs']
#solvers = ['NESTEROV-bs_dfbox','NESTEROV-bs_dfbox1']
#solvers = ['NESTEROV','NESTEROV-bs_dfbox', 'NESTEROV-bs_lbfgs', 'scipy_lbfgs']
solvers = ['NESTEROV-bs_dfbox']
solvers = ['QPS', 'QPS-Newton', 'NESTEROV','NESTEROV-bs_dfbox','NESTEROV-bs_dfbox1', 'NESTEROV-bs_lbfgs', 'NESTEROV-bs_Newton', 'scipy_lbfgs']
solvers = ['QPS', 'QPS_New','QPS-Newton', 'QPS-Newton_Matteo', 'QPS-Barzilai','scipy_lbfgs', 'scipy_cg']
#solvers = ['NESTEROV-bs_dfbox']
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


#'BA-L16LS','BA-L21LS','BA-L49LS','BA-L52LS','BA-L73LS',
            
problems = ['ARGLINA_10']
problems = ['GENROSE100']

############ problemi cutest medio-piccoli (Morteza)
problems = ['AKIVA','ALLINITU','ARGLINA_10','ARGLINA_50','ARGLINB_10','ARGLINB_50',
            'ARGLINC_10','ARGLINC_50','BARD','BDQRTIC','BEALE','BOX_100','BOX_10','BOX3','BOXPOWER',
            'BRKMCC','BROWNBS','BROWNDEN','BROYDN7D_10','BROYDN7D_50','BRYBND_100','BRYBND_50','BRYBND',
            'CHAINWOO_100','CHAINWOO_4','CHNROSNB_25','CHNROSNB_50','CHNROSNB','CHNRSNBM_25','CHNRSNBM_50','CHNRSNBM','CLIFF',
            'COSINE_10','CRAGGLVY_100','CRAGGLVY_10','CRAGGLVY_50','CRAGGLVY','CUBE','DECONVU','DENSCHNA','DENSCHNB','DENSCHNC',
            'DENSCHND','DENSCHNE','DENSCHNF','DIXMAANA_90','DIXMAANA','DIXMAANB_90','DIXMAANB','DIXMAANC_90','DIXMAANC','DIXMAAND_90',
            'DIXMAAND','DIXMAANE_90','DIXMAANE','DIXMAANF_90','DIXMAANF','DIXMAANG_90','DIXMAANG','DIXMAANH_90','DIXMAANH','DIXMAANI_90',
            'DIXMAANI','DIXMAANJ_90','DIXMAANJ','DIXMAANK_90','DIXMAANK','DIXMAANL_90','DIXMAANL','DIXMAANM_90','DIXMAANM','DIXMAANN_90',
            'DIXMAANN','DIXMAANO_90','DIXMAANO','DIXMAANP_90','DIXMAANP','DIXON3DQ_100','DIXON3DQ','DJTL','DQDRTIC_100','DQDRTIC_50',
            'DQDRTIC','DQRTIC_100','DQRTIC_10','DQRTIC_50','EDENSCH','EIGENALS_6','EIGENBLS_6','EIGENCLS_30','ENGVAL1_100','ENGVAL1_50',
            'ENGVAL1','ENGVAL2','ERRINROS_10','ERRINROS_25','ERRINROS_50','ERRINRSM_10','ERRINRSM_25','ERRINRSM_50','EXPFIT','EXTROSNB_100',
            'EXTROSNB_10','EXTROSNB','FLETBV3M','FLETCBV2_100','FLETCBV2','FLETCBV3_100','FLETCHBV_100','FLETCHCR','FMINSRF2_16','FMINSRF2_49',
            'FMINSRF2','FMINSURF_49','FMINSURF_64','FMINSURF','FREUROTH_100','FREUROTH_10','FREUROTH_50','FREUROTH','GENHUMPS_100',
            'GENHUMPS_10','GENHUMPS_5','GENROSE_10','GENROSE_5','GROWTHLS','GULF','HAIRY','HATFLDD','HATFLDE','HELIX','HIELOW','HILBERTA_10',
            'HILBERTA_2','HILBERTA_4','HILBERTA_5','HILBERTA_6','HILBERTB_10','HILBERTB_50','HILBERTB','HIMMELBB','HIMMELBF','HIMMELBG',
            'HIMMELBH','HUMPS','HYDC20LS','INDEF_100','INDEF_10','INDEF_50','INDEFM_100','INDEFM_10','INDEFM_50','JENSMP','KOWOSB',
            'LIARWHD_100','LIARWHD','LOGHAIRY','MANCINO_100','MANCINO_10','MANCINO_20','MANCINO_30','MANCINO_50','MARATOSB','MEXHAT',
            'MODBEALE_4','MODBEALE','MOREBV_100','MOREBV_50','MOREBV','MSQRTALS_49','MSQRTALS','MSQRTBLS_49','MSQRTBLS','NCB20B_100',
            'NCB20B_22','NCB20B_50','NCB20B','NONCVXU2_100','NONCVXU2_10','NONCVXUN_10','NONDIA_100','NONDIA_10','NONDIA_20','NONDIA_30',
            'NONDIA_50','NONDIA_90','NONDQUAR','OSBORNEA','OSBORNEB','OSCIGRAD_10','OSCIGRAD_15','OSCIGRAD_25','OSCIGRAD_2','OSCIPATH_100',
            'OSCIPATH_25','OSCIPATH_2','OSCIPATH_5','PALMER1C','PALMER1D','PALMER2C','PALMER3C','PALMER4C','PALMER5C','PALMER6C','PALMER7C',
            'PALMER8C','PARKCH','PENALTY1_100','PENALTY1_10','PENALTY1_4','PENALTY1_50','PENALTY2_100','PENALTY2_10','PENALTY2_50','PENALTY2',
            'PENALTY3_100','PENALTY3','POWELLSG_100','POWELLSG_16','POWELLSG_20','POWELLSG_36','POWELLSG_40','POWELLSG_60','POWELLSG_80',
            'POWELLSG_8','POWELLSG','POWER_100','POWER_20','POWER_30','POWER_50','POWER_75','POWER','QUARTC_100','QUARTC','ROSENBR','S308',
            'SBRYBND_100','SBRYBND_10','SBRYBND_50','SCHMVETT_100','SCHMVETT_10','SCHMVETT','SCOSINE_100','SCOSINE_10','SCURLY10_100',
            'SCURLY10_10','SENSORS_100','SENSORS_10','SENSORS_2','SENSORS_3','SINEVAL','SINQUAD_100','SINQUAD_50','SINQUAD','SISSER',
            'SNAIL','SPARSINE_100','SPARSINE_10','SPARSINE_50','SPARSQUR_100','SPARSQUR_10','SPARSQUR_50','SPMSRTLS','SROSENBR_100',
            'SROSENBR_50','SROSENBR','SSBRYBND_100','SSBRYBND_10','SSBRYBND_50','SSCOSINE_100','SSCOSINE_10','STRATEC','TOINTGOR','TOINTGSS_100',
            'TOINTGSS_50','TOINTGSS','TOINTPSP','TOINTQOR','TQUARTIC_100','TQUARTIC_10','TQUARTIC_50','TQUARTIC','TRIDIA_100','TRIDIA_10',
            'TRIDIA_20','TRIDIA_50','TRIDIA','VARDIM_100','VARDIM_50','VARDIM','VAREIGVL_100','VAREIGVL_10','VAREIGVL','VIBRBEAM',
            'WATSON_12','WATSON_31','WOODS_100','WOODS_4','ZANGWIL2',]



#problems = ['ARWHEAD']
#problems = ['BA-L16LS','BA-L21LS','BA-L49LS','BA-L52LS','BA-L73LS']

problems = ['ARWHEAD','BDQRTIC','BROYDN7D','BRYBND','CHAINWOO','CLPLATEB','CLPLATEC','COSINE','CRAGGLVY','CURLY10',
            'CURLY20','DIXMAANA1','DIXMAANB','DIXMAANC','DIXMAAND','DIXMAANE1','DIXMAANF','DIXMAANG','DIXMAANH','DIXMAANI1','DIXMAANJ',
            'DIXMAANK','DIXMAANL','DIXMAANM1','DIXMAANN','DIXMAANO','DIXMAANP','DIXON3DQ','DQDRTIC','DQRTIC','EDENSCH','ENGVAL1',
            'EXTROSNB','FLETCBV3','FLETCHCR','FMINSRF2','FMINSURF','FREUROTH','GENROSE','LIARWHD','LMINSURF','MANCINO','MOREBV',
            'NCB20','NCB20B','NLMSURF','NONCVXU2','NONCVXUN','NONDIA','NONDQUAR','ODC','PENALTY1','PENALTY2','PENALTY3','POWELLSG',
            'POWER','QUARTC','RAYBENDL','RAYBENDS','SCHMVETT','SCOSINE','SENSORS','SPARSINE','SPARSQUR','SROSENBR','SSC','TESTQUAD',
            'TOINTGSS','TQUARTIC','TRIDIA','VARDIM','VAREIGVL','WOODS']            


problems = ['10FOLDTR','ARGLINA','ARGLINB','ARGLINC','ARGTRIGLS','BDEXP','BOX','BOXPOWER','BRATU1D','BROWNAL','BROWNALE','BROYDN3DLS','BROYDNBDLS','CHEBYQAD',
            'CHEBYQADNE','CHNROSNB','CHNRSNBM','CLPLATEA','CURLY30','CVXBQP1','CYCLIC3','CYCLIC3LS','CYCLOOCFLS',
            'CYCLOOCTLS','DEGDIAG','DEGTRID','ARWHEAD','BDQRTIC','BROYDN7D','BRYBND','CHAINWOO','CLPLATEB','CLPLATEC',
            'COSINE','CRAGGLVY','CURLY10','CURLY20','DEGTRID2','DIAGIQB','DIAGIQE','DIAGIQT','DIAGNQB','DIAGNQE',
            'DIAGNQT','DIAGPQB','DIAGPQE','DIAGPQT','DIXMAANA','DIXMAANB','DIXMAANC','DIXMAAND','DIXMAANE','DIXMAANF',
            'DIXMAANG','DIXMAANH','DIXMAANI','DIXMAANJ','DIXMAANK','DIXMAANL','DIXMAANM','DIXMAANN','DIXMAANO','DIXMAANP',
            'DIXON3DQ','DQDRTIC','DQRTIC','EDENSCH','EIGENALS','EIGENBLS','EIGENCLS','ENGVAL1','ERRINROS','ERRINRSM',
            'EXTROSNB','FLETBV3M','FLETCBV2',
            'FLETCBV3','FLETCHBV','FLETCHCR','FMINSRF2','FMINSURF','FREUROTH','GENHUMPS','GENROSE','GENROSEB','GENROSEBNE',
            'GRIDGENA','HILBERTA','HILBERTB','INDEF','INDEFM','INTEQNELS','JNLBRNG1','JNLBRNG2','JNLBRNGA','JNLBRNGB',
            'KSSLS','LIARWHD','LINVERSE','LINVERSENE','LMINSURF','LUKSAN11LS','LUKSAN12LS','LUKSAN13LS','LUKSAN14LS',
            'LUKSAN15LS','LUKSAN16LS','LUKSAN17LS','LUKSAN21LS','LUKSAN22LS','MANCINO','MCCORMCK','MODBEALE','MOREBV',
            'NCB20','NCB20B','NCVXBQP1','NCVXBQP2','NCVXBQP3','NLMSURF','NOBNDTOR','NONCVXU2','NONCVXUN','NONDIA','NONDQUAR',
            'NONMSQRT','NONSCOMP','NONSCOMPNE','OBSTCLAE','OBSTCLAL','OBSTCLBL','OBSTCLBM','OBSTCLBU','ODC','ODNAMUR','OSCIGRAD',
            'OSCIPATH','PENALTY1','PENALTY2','PENALTY3','PENTDI','POWELLBC','POWELLSG','POWER','POWERSUM','PRICE3','PROBPENL',
            'QING','QRTQUAD','QUARTC','RAYBENDL','RAYBENDS','S368','SBRYBND','SCHMVETT','SCOSINE','SCURLY10','SCURLY20','SCURLY30',
            'SENSORS','SINEALI','SINQUAD','SPARSINE','SPARSQUR','SPECAN','SPIN2LS','SPINLS','SROSENBR','SSBRYBND',
            'SSC','SSCOSINE','STRTCHDV','TESTQUAD','TOINTGSS','TORSION1','TORSION2','TORSION3','TORSION4','TORSION5','TORSION6',
            'TORSIONA','TORSIONB','TORSIONC','TORSIOND','TORSIONE','TORSIONF','TQUARTIC','TRIDIA','TRIGON1','TRIGON2',
            'VANDANMSLS','VARDIM','VAREIGVL','WATSON','WOODS']
            
    
 # problemi ridotti           
problems = ['ARGLINA','ARGLINB','ARGLINC','ARGTRIGLS','BDEXP','BOX','BOXPOWER','BROWNAL','BROYDN3DLS','BROYDNBDLS','CHEBYQAD',
            'CHNRSNBM','CLPLATEA','CURLY30','CVXBQP1','CYCLIC3LS','CYCLOOCFLS',
            'CYCLOOCTLS','DEGDIAG','DEGTRID','ARWHEAD','BDQRTIC','BROYDN7D','BRYBND','CHAINWOO','CLPLATEB','CLPLATEC',
            'COSINE','CRAGGLVY','CURLY10','CURLY20','DEGTRID2','DIAGPQB','DIAGPQE','DIAGPQT','DIXMAANA','DIXMAANB','DIXMAANC','DIXMAAND','DIXMAANE',
            'DIXMAANF','DIXMAANG','DIXMAANH','DIXMAANI','DIXMAANJ','DIXMAANK','DIXMAANL','DIXMAANM','DIXMAANN','DIXMAANO','DIXMAANP',
            'DIXON3DQ','DQDRTIC','DQRTIC','EDENSCH','EIGENALS','EIGENBLS','EIGENCLS','ENGVAL1',
            'EXTROSNB','FLETBV3M','FLETCBV2','FLETCHCR','FMINSRF2','FMINSURF','FREUROTH','GENHUMPS','GENROSE','GENROSEB',
            'GRIDGENA','HILBERTA','HILBERTB','INDEFM','INTEQNELS','JNLBRNG1','JNLBRNG2','JNLBRNGA','JNLBRNGB',
            'KSSLS','LIARWHD','LINVERSE','LMINSURF','LUKSAN11LS','LUKSAN12LS','LUKSAN13LS','LUKSAN14LS',
            'LUKSAN15LS','LUKSAN16LS','LUKSAN17LS','LUKSAN21LS','LUKSAN22LS','MANCINO','MCCORMCK','MODBEALE','MOREBV',
            'NCB20','NCB20B','NLMSURF','NOBNDTOR','NONCVXU2','NONCVXUN','NONDIA','NONDQUAR',
            'NONMSQRT','NONSCOMP','OBSTCLAE','OBSTCLAL','OBSTCLBL','OBSTCLBM','OBSTCLBU','ODC','ODNAMUR','OSCIGRAD',
            'OSCIPATH','PENALTY1','PENALTY2','PENALTY3','PENTDI','POWELLBC','POWELLSG','POWER','POWERSUM','PRICE3','PROBPENL',
            'QING','QRTQUAD','QUARTC','RAYBENDL','RAYBENDS','SBRYBND','SCHMVETT','SCOSINE','SCURLY10','SCURLY20','SCURLY30',
            'SENSORS','SINEALI','SINQUAD','SPARSINE','SPARSQUR','SPIN2LS','SPINLS','SROSENBR','SSBRYBND',
            'SSC','SSCOSINE','STRTCHDV','TESTQUAD','TOINTGSS','TORSION1','TORSION2','TORSION3','TORSION4','TORSION5','TORSION6',
            'TORSIONA','TORSIONB','TORSIONC','TORSIOND','TORSIONE','TORSIONF','TQUARTIC','TRIDIA','TRIGON1','TRIGON2',
            'VANDANMSLS','VARDIM','VAREIGVL','WOODS']         
            
            

problems = ['ARGLINA','ARGLINB','ARGLINC','ARGTRIGLS','BDEXP','BOX','BOXPOWER','BRATU1D','BROWNAL','BROYDN3DLS','BROYDNBDLS','CHEBYQAD',
            'CHNRSNBM','CLPLATEA','CURLY30','CVXBQP1','CYCLIC3LS','CYCLOOCFLS',
            'CYCLOOCTLS','DEGDIAG','DEGTRID','ARWHEAD','BDQRTIC','BROYDN7D','BRYBND','CHAINWOO','CLPLATEB','CLPLATEC',
            'COSINE','CRAGGLVY','CURLY10','CURLY20','DEGTRID2','DIAGIQB','DIAGIQE','DIAGIQT','DIAGNQB','DIAGNQE',
            'DIAGNQT','DIAGPQB','DIAGPQE','DIAGPQT','DIXMAANA','DIXMAANB','DIXMAANC','DIXMAAND','DIXMAANE','DIXMAANF',
            'DIXMAANG','DIXMAANH','DIXMAANI','DIXMAANJ','DIXMAANK','DIXMAANL','DIXMAANM','DIXMAANN','DIXMAANO','DIXMAANP',
            'DIXON3DQ','DQDRTIC','DQRTIC','EDENSCH','EIGENALS','EIGENBLS','EIGENCLS','ENGVAL1',
            'EXTROSNB','FLETBV3M','FLETCBV2',
            'FLETCBV3','FLETCHBV','FLETCHCR','FMINSRF2','FMINSURF','FREUROTH','GENHUMPS','GENROSE','GENROSEB',
            'GRIDGENA','HILBERTA','HILBERTB','INDEF','INDEFM','INTEQNELS','JNLBRNG1','JNLBRNG2','JNLBRNGA','JNLBRNGB',
            'KSSLS','LIARWHD','LINVERSE','LMINSURF','LUKSAN11LS','LUKSAN12LS','LUKSAN13LS','LUKSAN14LS',
            'LUKSAN15LS','LUKSAN16LS','LUKSAN17LS','LUKSAN21LS','LUKSAN22LS','MANCINO','MCCORMCK','MODBEALE','MOREBV',
            'NCB20','NCB20B','NCVXBQP1','NCVXBQP2','NCVXBQP3','NLMSURF','NOBNDTOR','NONCVXU2','NONCVXUN','NONDIA','NONDQUAR',
            'NONMSQRT','NONSCOMP','OBSTCLAE','OBSTCLAL','OBSTCLBL','OBSTCLBM','OBSTCLBU','ODC','ODNAMUR','OSCIGRAD',
            'OSCIPATH','PENALTY1','PENALTY2','PENALTY3','PENTDI','POWELLBC','POWELLSG','POWER','POWERSUM','PRICE3','PROBPENL',
            'QING','QRTQUAD','QUARTC','RAYBENDL','RAYBENDS','S368','SBRYBND','SCHMVETT','SCOSINE','SCURLY10','SCURLY20','SCURLY30',
            'SENSORS','SINEALI','SINQUAD','SPARSINE','SPARSQUR','SPIN2LS','SPINLS','SROSENBR','SSBRYBND',
            'SSC','SSCOSINE','STRTCHDV','TESTQUAD','TOINTGSS','TORSION1','TORSION2','TORSION3','TORSION4','TORSION5','TORSION6',
            'TORSIONA','TORSIONB','TORSIONC','TORSIOND','TORSIONE','TORSIONF','TQUARTIC','TRIDIA','TRIGON1','TRIGON2',
            'VANDANMSLS','VARDIM','VAREIGVL','WOODS']
            
              

       
#problems = ['SINEALI']
#problems = ['SPARSINE']
#problems = ['INDEFM ']
#problems = ['CHAINWOO']
#problems = ['CHEBYQAD']
#problems = ['DIAGIQB']
#problems = ['INDEFM ']
#problems = ['COSINE']
#problems = ['SENSORS']
#problems = ['SINQUAD']
#problems = ['MODBEALE']
#problems = ['MANCINO']
#problems = ['PENALTY2']
#problems = ['VARDIM']
#problems = ['SSBRYBND ']
#problems = ['CYCLIC3LS']
#problems = ['CURLY30']
#problems = ['BOXPOWER']
#problems = ['BOX']
#problems = ['POWERSUM']
#print(len(problems))
#problems = ['DIXON3DQ']
#problems = ['CRAGGLVY']#
problems = ['VAREIGVL']
solvers = ['QPS-Newton_Matteo']
#solvers = ['QPS-Newton']
#solvers = ['QPS_New']
#solvers = ['QPS_New']

#res_tutti = []
#for p in problems:
#    #print('{}'.format(p))
#    P = Problem(p)
#    print('{} {} {}'.format(P.name,P.n,P.m))
#    res_tutti.append([P.name, P.n, P.m])
#table = tabulate(res_tutti, headers=['Problem','n', 'm'], tablefmt = 'orgtbl')
#print(table)
#input()

res_tutti = []
for p in problems:
    print('{}'.format(p))
    P = Problem(p,approxg=False)
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

