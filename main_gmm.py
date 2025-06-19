import numpy as np
import pycutest
from scipy.optimize import minimize
from time import time
from tabulate import tabulate
import pycgdescent as cg

from datetime import datetime
import os
import csv
import argparse

from Problems.TorchProblem import MLPProblem


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


class Solver:
    def __init__(self, problem, method='Armijo', alpha0=1, gamma=1e-5, delta=0.5, min_step=1e-30, 
                 grad_tol=1e-10, max_iters=1000, beta0=0.1, sigma=0.1, epsilon=1e-10, gtol_ord=2, nm_param=1):
        self.method = method
        self.fevals = 0
        self.gevals = 0
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
        self.gtol_ord = gtol_ord
        self.min_sample_val = 1e-8
        self.Bab = None
        self.nm_param = nm_param

    def set_solver(self, method):
        self.method = method

    def set_problem(self, problem):
        self.problem = problem

    def solve(self, method=None, eps_grad=None, max_iters=None, gamma=None, min_step=None, gtol_ord=None, nm_param=None):
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
        if nm_param is not None:
            self.nm_param = nm_param
        self.fevals = 0
        self.gevals = 0
        tic = time()
        
        if self.method == 'Armijo':
            sol, info = self.solveArmijo()
        elif self.method == 'GMM':
            sol, info = self.solveGMM()
        elif self.method == 'scipy_lbfgs':
            if not np.isinf(self.gtol_ord):
                print('CANNOT SET DIFFERENT NORM THAN INFTY-NORM FOR L-BFGS')
            lbfgs = minimize(self.f, self.problem.get_x0(), jac=self.g, method="L-BFGS-B", options={"iprint": -1, "maxcor": 10, "gtol": self.grad_tol, "ftol": 1e-50, "maxiter": self.max_iters, "maxls": 20, 'maxfun': 1e15})
            info = {"iters": lbfgs.nit, "f": lbfgs.fun, "g_norm": np.linalg.norm(lbfgs.jac, self.gtol_ord)}
            sol = lbfgs.x
        elif self.method == 'scipy_cg':
            cg = minimize(self.f, self.problem.get_x0(), jac=self.g, method="CG", options={"disp": False, "gtol": self.grad_tol, "maxiter": self.max_iters, 'norm': self.gtol_ord})
            info = {"iters": cg.nit, "f": cg.fun, "g_norm": np.linalg.norm(cg.jac, self.gtol_ord)}
            sol = cg.x
        elif self.method == 'cg_descent':
            sol, info = self.solve_cg_descent()
        else:
            print('Solver unknown')
            
        info['fevals'] = self.fevals
        info['gevals'] = self.gevals
        info['time'] = time()-tic

        return sol, info
        
    def f(self, x):
        self.fevals +=1
        fv = self.problem.f(x)
        return fv

    def f_g(self, x):
        self.fevals += 1
        self.gevals += 1
        return self.problem.f_g(x)

    def g(self,x):
        self.gevals += 1
        gr = self.problem.g(x)
        return gr

    def armijoLS(self, x, f, g, d, alpha0=1, gamma=1e-5, min_step=1e-30, delta=0.5):
        alpha=alpha0
        f_trial = self.f(x+alpha*d)
        dg = np.dot(d,g)
        while(f_trial > f + gamma*alpha*dg and alpha>min_step) or f_trial > 2*np.abs(f)  or f_trial == np.inf or np.isnan(f_trial):
            alpha = alpha * delta if f_trial-f-gamma*alpha*dg  < 1e6 else alpha*delta/5
            f_trial = self.f(x+alpha*d)
        return alpha, f_trial

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


    def solve_cg_descent(self):
        options = cg.OptimizeOptions(StopRule=True, StopFac=0, maxit=self.max_iters)

        def obf(t):
            return self.f(t)

        def obg(jac_store, t):
            j = self.g(t)
            jac_store[:] = j[:]
            return jac_store

        r = cg.minimize(
            fun=obf,
            x0=self.problem.get_x0(),
            jac=obg,
            tol=self.grad_tol,
            options=options,
        )

        return r.x, {"iters": r.nit, "f": r.fun, "g_norm": np.linalg.norm(self.problem.g(r.x), self.gtol_ord)}


    def solveGMM(self):
        xk = self.problem.get_x0()
        n_iters=0
        xk_1 = np.copy(xk)
        f_1, _ = self.f_g(xk_1)
        C_k = f_1
        alpha, beta = 0,0
        aArm = 1
        g_norm = np.inf
        consecutive_no_dec = 0
        while True:
            g = self.g(xk)
            f = new_f if n_iters > 0 else f_1
           
            g_norm = np.linalg.norm(g,self.gtol_ord)
           
            if g_norm<self.grad_tol or n_iters >= self.max_iters:
                break

            if n_iters == 0:
                a_init = self.quad_step_unid(-g,xk,f)
                aArm, new_f = self.armijoLS(x=xk, f=C_k, g=g, d=-g, alpha0=a_init, gamma=self.gamma, min_step=self.min_step)
                C_k = (self.nm_param*C_k+new_f)/(self.nm_param+1)
                new_x = xk - aArm*g
                alpha, beta = aArm, 0
            else:
                ab = self.quadratic_plane_searchGMM(xk, xk_1, f, f_1, g, alpha, beta)
                alpha, beta = ab[0], ab[1]
                dir = -alpha*g + beta*(xk-xk_1)
                eta, new_f = self.armijoLS(x=xk, f=C_k, g=g, d=dir, alpha0=1, gamma=self.gamma, min_step=self.min_step)
                C_k = (self.nm_param*C_k+new_f)/(self.nm_param+1)
                new_x = xk + eta*dir
            xk_1 = xk
            f_1, g_1 = f, g
            xk = new_x
            n_iters += 1
            if f - new_f < 1e-50:
                consecutive_no_dec += 1
                if consecutive_no_dec > 200:
                    break
            elif consecutive_no_dec:
                consecutive_no_dec = 0
        return xk, {"iters": n_iters, "f": f, "g_norm": g_norm}


    def quadratic_plane_searchGMM(self, xk, xk_1, f, f_1, g, alpha, beta):
        d1 = -g
        d2 = xk - xk_1
        D = np.vstack((d1,d2)).T
        gab = g@D

        if True: 
            d1_norm = np.linalg.norm(g)
            d2_norm = np.linalg.norm(d2)

            if np.isnan(d1_norm):
                min_sample_val  = 1.e-16
            else:
                min_sample_val =  1.e-3/min(np.maximum(d1_norm,1.e-16),1.e+16)
            if np.abs(alpha) <= min_sample_val:
                alpha = min_sample_val if alpha >= 0 else -min_sample_val 

            if np.isnan(d2_norm):
                min_sample_val  = 1.e-16
            else:
                min_sample_val =  1.e-3/min(np.maximum(d2_norm,1.e-16),1.e+16)
            if np.abs(beta) <= min_sample_val:
                beta = min_sample_val if beta >= 0 else  -min_sample_val   

            fA = f
            fB = f_1
            fC = self.f(xk+alpha*d1+beta*d2)
            fD = self.f(xk+alpha*d1)

            bc = 2*(gab[1]+fB-fA)
            ba = 2/(alpha**2)*(fD-alpha*gab[0]-fA)
            bb = (fC-fA-gab[0]*alpha-gab[1]*beta-0.5*alpha*alpha*ba-0.5*beta*beta*bc)/(alpha*beta)


            self.Bab = np.array([[ba,bb],[bb,bc]])
            if np.isinf(self.Bab).any() or np.isnan(self.Bab).any():
                return np.array([self.quad_step_unid(-g,xk,f), 0]) # np.array([1,0])

        try:
            solution_closed = np.linalg.solve(self.Bab, -gab)
        except np.linalg.LinAlgError:
            solution_closed = np.linalg.lstsq(self.Bab, -gab, rcond=None)[0]

    
        best = solution_closed
        dir = best[0]*d1 + best[1]*d2
        dtg = np.dot(dir,g)
        g_norm2 = np.linalg.norm(g,2)
        if dtg>0:
            best = -best
            dtg = -dtg
            
        if dtg >= -1e-18*min((g_norm2**2), 1) or np.linalg.norm(dir) >= 1e25*g_norm2:
            best = self.perturb_Cholesky(gab,self.Bab,g_norm2,np.linalg.norm(d2))

        return best


    def quad_step_unid(self,d,xk,f):
        normd2 = np.linalg.norm(d)**2
        return max(1e-4,normd2/min(1e9,max(1e-9,np.abs((self.f(xk-1e+3*d)+1e-3*normd2-f)/(1e-3)**2))))


    def perturb_Cholesky(self, g, H, norma_g, d2_norm, eps_H=1e-6):
         
           print('Cholesky')
           d=np.zeros(2)
         
           if H[0,0] > 1.e-12*min(norma_g**2,1.e0):
              l11=np.sqrt(H[0,0]/d2_norm**2)
           else:
              l11=eps_H

           l21=(H[1,0]/(norma_g*d2_norm))/l11
            
           if H[1,1]-l21*l21 > 1.e-12*min(norma_g**2,1.e0):
              l22=np.sqrt(H[1,1]-l21*l21)
           else:
              l22=eps_H
                
           y1=-g[0]/l11
           y2=-(g[1]+l21*y1)/l22
           
           d[1]=y2/l22
           d[0]=(y1-l21*d[1])/l11           

           return d


    def test_problem(self, solvers, eps_grad=1e-3, max_iters=5000, gamma=1e-4, min_step=1e-30, gtol_ord=2, out_folder="res_matlap"):

        res_tab = []
        for solver in solvers:
            _, info = self.solve(method=solver, eps_grad=eps_grad, max_iters=max_iters, gamma=gamma, min_step=min_step, gtol_ord=gtol_ord)
            res_tab.append([solver, self.problem.name, info['time'], info['iters'], info['f'], info['g_norm'], info['fevals'], info['gevals']])

            with open(os.path.join(out_folder, solver + '.csv'), mode='a', newline='') as out_csv:
                writer = csv.writer(out_csv)
                writer.writerow([self.problem.name, info['time'], info['iters'], info['f'], info['g_norm'], info['fevals'], info['gevals']])

        table = tabulate(res_tab, headers=['Algorithm', 'prob', 't', 'n_it', 'f_opt', 'g_norm', 'fevals', 'gevals', 'tf', 'tg', 'to'], tablefmt='orgtbl')
        print(table)
        fr=open(os.path.join(out_folder, "res_matlap.txt"), "a")
        print(tabulate(res_tab, tablefmt='orgtbl'), file=fr)
        fr.close()



CUTEST_PROBLEMS =   ['ARGLINA', 'ARGLINB', 'ARGLINC', 'ARGTRIGLS', 'ARWHEAD', 'BDEXP', 'BDQRTIC', 'BOX', 'BOXPOWER', 'BROWNAL',
                     'BROYDN3DLS', 'BROYDN7D', 'BROYDNBDLS', 'BRYBND', 'CHAINWOO', 'CHEBYQAD', 'CHNRSNBM', 'CLPLATEA', 'CLPLATEB', 'CLPLATEC',
                     'COSINE', 'CRAGGLVY', 'CURLY10', 'CURLY20', 'CURLY30', 'CVXBQP1', 'CYCLIC3LS', 'CYCLOOCFLS', 'CYCLOOCTLS', 'DEGDIAG',
                     'DEGTRID', 'DEGTRID2', 'DIAGPQB', 'DIAGPQE', 'DIAGPQT', 'DIXMAANA1', 'DIXMAANB', 'DIXMAANC', 'DIXMAAND', 'DIXMAANE1',
                     'DIXMAANF', 'DIXMAANG', 'DIXMAANH', 'DIXMAANI1', 'DIXMAANJ', 'DIXMAANK', 'DIXMAANL', 'DIXMAANM1', 'DIXMAANN', 'DIXMAANO',
                     'DIXMAANP', 'DIXON3DQ', 'DQDRTIC', 'DQRTIC', 'EDENSCH', 'EIGENALS', 'EIGENBLS', 'EIGENCLS', 'ENGVAL1', 'EXTROSNB',
                     'FLETBV3M', 'FLETCBV2', 'FLETCHCR', 'FMINSRF2', 'FMINSURF', 'FREUROTH', 'GENHUMPS', 'GENROSE', 'GENROSEB', 'GRIDGENA', 
                     'HILBERTA', 'HILBERTB', 'INDEFM', 'INTEQNELS', 'JNLBRNG1', 'JNLBRNG2', 'JNLBRNGA', 'JNLBRNGB', 'KSSLS', 'LIARWHD',
                     'LINVERSE', 'LMINSURF', 'LUKSAN11LS', 'LUKSAN12LS', 'LUKSAN13LS', 'LUKSAN14LS', 'LUKSAN15LS', 'LUKSAN16LS', 'LUKSAN17LS', 'LUKSAN21LS', 
                     'LUKSAN22LS', 'MANCINO', 'MCCORMCK', 'MODBEALE', 'MOREBV', 'NCB20', 'NCB20B', 'NLMSURF', 'NOBNDTOR', 'NONCVXU2',
                     'NONCVXUN', 'NONDIA', 'NONDQUAR', 'NONMSQRT', 'NONSCOMP', 'OBSTCLAE', 'OBSTCLAL', 'OBSTCLBL', 'OBSTCLBM', 'OBSTCLBU',
                     'ODC', 'ODNAMUR', 'OSCIGRAD', 'OSCIPATH', 'PENALTY1', 'PENALTY2', 'PENALTY3', 'PENTDI', 'POWELLBC', 'POWELLSG',
                     'POWER', 'PRICE3', 'PROBPENL', 'QING', 'QRTQUAD', 'QUARTC', 'RAYBENDL', 'RAYBENDS', 'SBRYBND', 'SCHMVETT',
                     'SCOSINE', 'SCURLY10', 'SCURLY20', 'SCURLY30', 'SENSORS', 'SINEALI', 'SINQUAD', 'SPARSINE', 'SPARSQUR', 'SPIN2LS',
                     'SPINLS', 'SROSENBR', 'SSBRYBND', 'SSC', 'SSCOSINE', 'STRTCHDV', 'TESTQUAD', 'TOINTGSS', 'TORSION1', 'TORSION2',
                     'TORSION3', 'TORSION4', 'TORSION5', 'TORSION6', 'TORSIONA', 'TORSIONB', 'TORSIONC', 'TORSIOND', 'TORSIONE', 'TORSIONF',
                     'TQUARTIC', 'TRIDIA', 'TRIGON1', 'TRIGON2', 'VANDANMSLS', 'VARDIM', 'VAREIGVL', 'WOODS']

torch_problems = ['a1a_MLP_TANH', 'a2a_MLP_TANH', 'a3a_MLP_TANH', 'a4a_MLP_TANH', 'a5a_MLP_TANH', 'a6a_MLP_TANH', 'a7a_MLP_TANH', 'a8a_MLP_TANH', 'a9a_MLP_TANH', 
                  'breast-cancer_scale_MLP_TANH', 'cod-rna_MLP_TANH', 'covtype.libsvm.binary.bz2_MLP_TANH', 'diabetes_scale_MLP_TANH', 'fourclass_scale_MLP_TANH',
                  'german.numer_scale_MLP_TANH', 'gisette_scale.bz2_MLP_TANH', 'ijcnn1.bz2_MLP_TANH', 'liver-disorders_scale_MLP_TANH', 'phishing_MLP_TANH',
                  'sonar_scale_MLP_TANH', 'svmguide1_MLP_TANH', 'svmguide3_MLP_TANH',
                  'w1a_MLP_TANH', 'w2a_MLP_TANH', 'w3a_MLP_TANH', 'w4a_MLP_TANH', 'w5a_MLP_TANH', 'w6a_MLP_TANH', 'w7a_MLP_TANH', 'w8a_MLP_TANH']

TORCH_PROBLEMS = []
for n_seed in ['42', '420', '16007']:
    for p in torch_problems:
        TORCH_PROBLEMS.append(p + "_" + n_seed)


parser = argparse.ArgumentParser(prog='QPS', description='Algorithm QPS')
parser.add_argument('-p', '--problem', type=str, help='Problem name')
parser.add_argument('-g_tol', '--grad_tol', default=1e-6, type=float)
parser.add_argument('-l_reg', default=1e-3, type=float)
parser.add_argument('-m_iter', '--max_iter', default=5000, type=int)

args = parser.parse_args()

if args.problem == 'cutest':
    problems = CUTEST_PROBLEMS

elif 'cutest_' in args.problem:
    cutest_part = int(args.problem.split("_")[1])
    assert cutest_part >= 1 and cutest_part <= 12
    problems = CUTEST_PROBLEMS[14*(cutest_part-1):14*(cutest_part)]
    print(len(problems), problems)

elif args.problem == 'torch':
    problems = TORCH_PROBLEMS

else:
    problems = [args.problem]

solvers = ['GMM', 'cg_descent', 'scipy_lbfgs']

# Initialize Logs
now = datetime.now()
out_folder = os.path.join('logs', f"res_{now.strftime('%Y-%m-%d_%H-%M-%S')}")
os.makedirs(out_folder)

for solver in solvers:
    with open(os.path.join(out_folder, solver + '.csv'), mode='a', newline='') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(['prob', 't', 'n_it', 'f_opt', 'g_norm', 'fevals', 'gevals'])

fr=open(os.path.join(out_folder, "res_matlap.txt"), "a")
print(f"g_tol={args.grad_tol}, l_reg={args.l_reg}, m_iter={args.max_iter}", file=fr)
fr.close()


for p in problems:
    # print('{}'.format(p))
    if p in CUTEST_PROBLEMS:
        P = Problem(p)
    elif p in TORCH_PROBLEMS:
        P = MLPProblem(p, l_reg=args.l_reg)
    S = Solver(P)
    S.test_problem(solvers, max_iters=args.max_iter, eps_grad=args.grad_tol, gtol_ord=np.inf, out_folder=out_folder)    

