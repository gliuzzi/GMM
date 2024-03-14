import numpy as np
import pycutest
from tabulate import tabulate

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



# TEST CUTEST


problems = ['ARWHEAD','BDQRTIC','BROYDN7D','BRYBND','CHAINWOO','CLPLATEB','CLPLATEC','COSINE','CRAGGLVY','CURLY10',
            'CURLY20','DIXMAANA1','DIXMAANB','DIXMAANC','DIXMAAND','DIXMAANE1','DIXMAANF','DIXMAANG','DIXMAANH','DIXMAANI1','DIXMAANJ',
            'DIXMAANK','DIXMAANL','DIXMAANM1','DIXMAANN','DIXMAANO','DIXMAANP','DIXON3DQ','DQDRTIC','DQRTIC','EDENSCH','ENGVAL1',
            'EXTROSNB','FLETCBV3','FLETCHCR','FMINSRF2','FMINSURF','FREUROTH','GENROSE','LIARWHD','LMINSURF','MANCINO','MOREBV',
            'NCB20','NCB20B','NLMSURF','NONCVXU2','NONCVXUN','NONDIA','NONDQUAR','ODC','PENALTY1','PENALTY2','PENALTY3','POWELLSG',
            'POWER','QUARTC','RAYBENDL','RAYBENDS','SCHMVETT','SCOSINE','SENSORS','SPARSINE','SPARSQUR','SROSENBR','SSC','TESTQUAD',
            'TOINTGSS','TQUARTIC','TRIDIA','VARDIM','VAREIGVL','WOODS']            


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


#problems = ['PRICE3']
#print(len(problems))
#input()  
          
res_tutti = []
for p in problems:
    #print('{}'.format(p))
    P = Problem(p)
    print('{} {} {}'.format(P.name,P.n,P.m))
    res_tutti.append([P.name, P.n, P.m])
table = tabulate(res_tutti, headers=['Problem','n', 'm'], tablefmt = 'orgtbl')
print(table)
#input()

