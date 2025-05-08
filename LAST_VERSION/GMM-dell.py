import numpy as np
import pycutest
import pycgdescent as cg
from time import time, sleep
from tabulate import tabulate
from Newton import *
from scipy.optimize import minimize
import sys
import argparse

class Problem:
    def __init__(self, name, n=None):
        if n:
            self.__p = pycutest.import_problem(name, sifParams={'N':n})
        else:
            self.__p = pycutest.import_problem(name)
        self.n = self.__p.n
        self.m = self.__p.m
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

class Solver:
    def __init__(self, problem, method='GMM2', alpha0=1, gamma=1e-5, delta=0.5, \
                 min_step=1e-10, grad_tol=1e-10, max_iters=1000, beta0=0.1, sigma=0.1, \
                 epsilon=1e-10, gtol_ord=np.inf):
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

    def set_solver(self,method):
        self.method = method

    def set_problem(self,problem):
        self.problem = problem

    def solve(self, method=None, eps_grad=None, max_iters=None, gamma=None, min_step=None, gtol_ord=np.inf):
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

        self.fevals = 0
        self.gevals = 0
        tic = time()

        if self.method == 'GMM2':
            sol, info = self.solvePlaneSearch_Newton()
        elif  self.method == 'GMM3':
            sol, info = self.solvePlaneSearch_Newton(var=5)
        elif  self.method == 'GMM1':
            sol, info = self.solvePlaneSearch_Newton(var=11)
        elif self.method == 'Hager':
            sol, info = self.solveHager()    
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

    def bidimensional_search_Newton(self,finiz, f_1, g_1, x, d1, d2,n_iter_glob , alpha0, beta0, alpha_1, beta_1, var,multistart=0, deriv_free=False, maxfev=10):
        def f2(ab):
            return self.f(x + ab[0] * d1 + ab[1] * d2)
        def g2(ab):
            g = self.g(x + ab[0] * d1 + ab[1] * d2)
            return [np.dot(g, d1), np.dot(g, d2)]

        def fg2(ab):
            g = self.g(x + ab[0] * d1 + ab[1] * d2)
            return self.f(x + ab[0] * d1 + ab[1] * d2), np.array(np.dot(g, d1), np.dot(g, d2))

        def inner_NN(ab,ab_1,f_1,g_1,x):
            NN=Newton(2,finiz,f_1,g_1,ab,ab_1,1.e-3*self.grad_tol*self.grad_tol,1,False,f2,self.g,x,d1,d2,n_iter_glob,var)
            f,x=NN.minimize()
            return f,x
        best_f,best = inner_NN([alpha0,beta0],[alpha_1, beta_1],f_1,g_1,x)
        return best,best_f

    def solveHager(self):
        options = cg.OptimizeOptions(StopRule=True, StopFac=0, maxit=self.max_iters)

        def obf(t):
            return self.f(t)

        def obg(jac_store, t):
            j = self.g(t)
            jac_store[:] = j[:]
            return jac_store

        r = cg.minimize(fun=obf, x0=self.problem.get_x0(), jac=obg, tol=self.grad_tol, options=options)

        return r.x, {"iters": r.nit, "f": r.fun, "g_norm": np.linalg.norm(self.problem.g(r.x), self.gtol_ord)}

    def solvePlaneSearch_Newton(self, var=1):
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
                #-------------------------------------------------------
                # stopping criterion on the variation of function values
                #-------------------------------------------------------
                testnew=abs(f- fExp)/np.maximum(np.maximum(abs(f),abs(fExp)),1)
                if(testnew <= 1.e-50):
                    break
                f = fExp
                g = self.g(xk)
            #--------------------------------------------------------------------------------------------------------
            # stopping criteria on the infinity norm of the gradient or on the maximum number of function evaluations
            #--------------------------------------------------------------------------------------------------------
            g_norm_prev = g_norm
            g_norm = np.linalg.norm(g, self.gtol_ord)            
            if g_norm < self.grad_tol or n_iters >= self.max_iters or f == -np.inf or np.isnan(f):
                break

            ab = np.zeros(2)
            ab[0]=0.
            ab[1]=0.

            if True:
                fExp=f
                ab,fExp = self.bidimensional_search_Newton(fExp,f_1, g_1, xk, -g, xk - xk_1,n_iters, ab[0], ab[1], alpha_1, beta_1, var, deriv_free=True, maxfev=5)
                if fExp == -np.inf:
                    f=fExp
                    break
            if fExp <= f:
                alpha_1, beta_1 = ab[0], ab[1]
                alpha, beta = ab[0], ab[1]
                aArm = max(np.abs(alpha), 10 * self.min_step)
            else:
                num_fails += 1
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
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm}

    def bb_step(self,s,y,epsilon=1e-10, inverse=False):
        if inverse:
            aux = np.dot(y,s)
            mu = np.dot(y,y)/aux if np.abs(aux)>1e-8 else 0
        else:
            mu = np.dot(s,y)/np.dot(s,s) if np.linalg.norm(s)>1e-8 else 0
        return 1/mu if epsilon <= mu <= epsilon**(-1) else 1

    def test_problem(self, solvers, eps_grad=1e-3, max_iters=5000):
        gtol_ord = np.inf
        gamma = 1e-4
        res_tab = []
        for solver in solvers:
            self.alfas = []
            self.betas = []
            sol, info = self.solve(method=solver, eps_grad=eps_grad, max_iters=max_iters, gamma=gamma, min_step=1e-24, gtol_ord=gtol_ord)

            res_tab.append(
                [solver, self.problem.name, self.problem.n, info['time'], info['iters'], info['f'], info['g_norm'], info['fevals'],
                 info['gevals']])

        return res_tab

parser = argparse.ArgumentParser(prog='GMM',description='Algorithm GMM')
parser.add_argument('-v','--version',action='store_true',help='Print version number')
parser.add_argument('-p','--problem',nargs=1,type=str,
                             default=['ARGLINA'],
                             help='SIF name of the problem to be solved')
args = parser.parse_args(sys.argv[1:])
if args.version:
    print('\nGMM.py version 0.1\n')

probname = args.problem[0]

if probname == '':
    problems = ['ARGLINB','ARGLINC','ARGTRIGLS','BDEXP','BOX','BOXPOWER','BROWNAL',
                'BROYDN3DLS','BROYDNBDLS','CHEBYQAD','CHNRSNBM','CLPLATEA','CURLY30','CVXBQP1',
                'CYCLIC3LS','CYCLOOCFLS','CYCLOOCTLS','DEGDIAG','DEGTRID','ARWHEAD','BDQRTIC',
                'BROYDN7D','BRYBND','CHAINWOO','CLPLATEB','CLPLATEC','COSINE','CRAGGLVY','CURLY10',
                'CURLY20','DEGTRID2','DIAGPQB','DIAGPQE','DIAGPQT','DIXMAANA','DIXMAANB','DIXMAANC',
                'DIXMAAND','DIXMAANE','DIXMAANF','DIXMAANG','DIXMAANH','DIXMAANI','DIXMAANJ','DIXMAANK',
                'DIXMAANL','DIXMAANM','DIXMAANN','DIXMAANO','DIXMAANP','DIXON3DQ','DQDRTIC','DQRTIC',
                'EDENSCH','EIGENALS','EIGENBLS','EIGENCLS','ENGVAL1','EXTROSNB','FLETBV3M','FLETCBV2',
                'FLETCHCR','FMINSRF2','FMINSURF','FREUROTH','GENHUMPS','GENROSE','GENROSEB','GRIDGENA',
                'HILBERTA','HILBERTB','INDEFM','INTEQNELS','JNLBRNG1','JNLBRNG2','JNLBRNGA','JNLBRNGB',
                'KSSLS','LIARWHD','LINVERSE','LMINSURF','LUKSAN11LS','LUKSAN12LS','LUKSAN13LS','LUKSAN14LS',
                'LUKSAN15LS','LUKSAN16LS','LUKSAN17LS','LUKSAN21LS','LUKSAN22LS','MANCINO','MCCORMCK',
                'MODBEALE','MOREBV','NCB20','NCB20B','NLMSURF','NOBNDTOR','NONCVXU2','NONCVXUN','NONDIA',
                'NONDQUAR','NONMSQRT','NONSCOMP','OBSTCLAE','OBSTCLAL','OBSTCLBL','OBSTCLBM','OBSTCLBU',
                'ODC','ODNAMUR','OSCIGRAD','OSCIPATH','PENALTY1','PENALTY2','PENALTY3','PENTDI','POWELLBC',
                'POWELLSG','POWER','PRICE3','PROBPENL','QING','QRTQUAD','QUARTC','SBRYBND','SCHMVETT',
                'SCOSINE','SCURLY10','SCURLY20','SCURLY30','SENSORS','SINEALI','SINQUAD','SPARSINE',
                'SPARSQUR','SPIN2LS','SPINLS','SROSENBR','SSBRYBND','SSCOSINE','STRTCHDV','TESTQUAD',
                'TOINTGSS','TORSION1','TORSION3','TORSION4','TORSION5','TORSION6','TORSIONA','TORSIONC',
                'TORSIOND','TORSIONE','TORSIONF','TQUARTIC','TRIDIA','TRIGON1','TRIGON2','VANDANMSLS',
                'VARDIM','VAREIGVL','WOODS']

    problems = ['CYCLOOCFLS']
else:
    problems = [probname]

solvers = ['GMM2']
solvers = ['scipy_lbfgs', 'Hager']
solvers = ['GMM1','GMM3','GMM2','scipy_lbfgs', 'Hager']
solvers = ['Hager']
solvers = ['Hager']
solvers = ['GMM1','GMM3','GMM2','scipy_lbfgs', 'scipy_cg']
solvers = ['GMM1','GMM3','GMM2','scipy_lbfgs', 'scipy_cg', 'Hager']

print(len(problems))
res_tutti = []
for p in problems:
    print('{}'.format(p))
    P = Problem(p)

    res_parz = []
    S = Solver(P)
    res = S.test_problem(solvers, max_iters=5000, eps_grad=1e-6)
    for i,r in enumerate(res):
        res_tutti.append(r)
        res_parz.append(r)
    r=["--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--"]
    res_tutti.append(r)
    r=["--", "--", "--", "--", "--", "--", "--", "--", "--"]
    res_parz.append(r)
    print(tabulate(res_parz, headers=['Algorithm', 'prob', 'n', 't', 'n_it', 'f_opt', \
                'g_norm', 'fevals', 'gevals'], tablefmt='orgtbl')
          )
    fr=open("risultati.txt","a")
    #print(tabulate(res_parz, headers=['Algorithm', 'prob', 'n', 't', 'n_it', 'f_opt', \
    #            'g_norm', 'fevals', 'gevals'], tablefmt='orgtbl'),file=fr)
    print(tabulate(res_parz, tablefmt='orgtbl'),file=fr)
    fr.close()

table = tabulate(res_tutti, headers=['Algorithm','prob', 'n', 't', 'n_it', 'f_opt', \
    'g_norm', 'fevals', 'gevals'], tablefmt = 'orgtbl')
print(table)

#fr=open("risultati.txt","w")
#print(table,file=fr)
#fr.close()
table = tabulate(res_tutti, headers=['Algorithm','prob', 'n', 't', 'n_it', 'f_opt', \
    'g_norm', 'fevals', 'gevals'], tablefmt = 'latex')
print(table)

