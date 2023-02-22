#============================================================================================
#    SDBOX - FORTRAN90 implementation of a Derivative-Free algorithm for bound 
#    constrained optimization problems 
#    Copyright (C) 2011  G.Liuzzi, S. Lucidi
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    S. Lucidi, M. Sciandrone. A Derivative-Free Algorithm for Bound Constrained Optimization, 
#    Computational Optimization and Applications, 21(2): 119-142 (2002)
#    DOI: 10.1023/A:1013735414984
#
#============================================================================================
import numpy as np
from scipy.optimize import minimize

class der_free_method:
	def __init__(self,funct,lb,ub,maxiter=10000,maxfev=1000,tol=1.e-6,iprint=0):
		self.funct = funct
		self.lb = lb
		self.ub = ub
		self.maxfev = maxfev
		self.maxiter = maxiter
		self.tol = tol
		self.iprint = iprint

	#     *********************************************************
	#     *
	#     *                 stop condition verify
	#     *
	#     *********************************************************
	def stop(self,n,alfa_d,nf,ni,fstop,f,alfa_stop,nf_max, flag_fail):

		istop = 0
		alfa_max = 0.0

		for i in range(0,n):
			if alfa_d[i] > alfa_max:
				alfa_max = alfa_d[i]

		if ni >= (n+1):
			ffm = f
			for i in range(0,n):
				ffm += fstop[i]

			ffm /= float(n+1)

			ffstop = (f-ffm)**2

			for i in range(0,n):
				ffstop += (fstop[i]-ffm)**2

			ffstop = np.sqrt(ffstop/float(n+1))


		if alfa_max <= alfa_stop:
			istop = 1

		if nf > nf_max:
			istop = 2

		return istop, alfa_max

	#     *********************************************************
	#     *
	#     *                 Continuous Linesearch
	#     *
	#     *********************************************************
	def linesearchbox_cont(self,n,x,f,d,alfa_d,j,alfa_max,iprint,bl,bu,nf):

		#z = [a for a in x]
		z = np.copy(x)

		gamma = 1.e-6
		delta =0.5
		delta1=0.5
		i_corr_fall =0
		ifront      =0

		if iprint >= 1:
			print('j =%d    d(j) =%f alfa=%e' % (j,d[j-1],alfa_d[j-1]))

		if abs(alfa_d[j-1]) <= 1.e-3*min(1.0,alfa_max):
			alfa = 0.0
			if iprint >= 1:
				print('  alfa piccolo')
				print(' alfa_d(j)=%e    alfamax=%e' % (alfa_d[j-1],alfa_max))
			return alfa, f, nf, i_corr_fall

		for ielle in range(1,3):

			if d[j-1] > 0.0:

				if (alfa_d[j-1]-(bu[j-1]-x[j-1])) < -1.e-6:
					alfa = max(1.e-24,alfa_d[j-1])
				else:
					alfa = bu[j-1]-x[j-1]
					ifront=1
					if iprint >= 1:
						print(' point on the boundary. *')
			else:

				if (alfa_d[j-1]-(x[j-1]-bl[j-1])) < -1.e-6:
					alfa = max(1.e-24,alfa_d[j-1])
				else:
					alfa = x[j-1]-bl[j-1]
					ifront=1
					if iprint >= 1:
						print(' point on the boundary. *')

			if abs(alfa) <= 1.e-3*min(1.0,alfa_max):

				d[j-1] = -d[j-1]
				i_corr_fall += 1
				ifront = 0

				if iprint >= 1:
					print(' direzione opposta per alfa piccolo')
					print(' j =%d    d(j) =%f' % (j,d[j-1]))
					print(' alfa=%e    alfamax=%e' % (alfa,alfa_max))

				alfa = 0.0
				continue

			alfaex = alfa
			z[j-1] = x[j-1] + alfa*d[j-1]

			fz  = self.funct(z)
			nf += 1

			if iprint >= 1:
				print(' fz =%f   alfa =%e' % (fz,alfa))

			if iprint >= 2:
				for i in range(0,n):
					print(' z(%d)=%f' % (i,z[i]))

			fpar = f - gamma*alfa**2

			if fz < fpar:

			# expansion step

				while True:

					if ifront==1:

						if iprint >= 1:
							print(' accetta punto sulla frontiera fz =%f   alfa =%f' % (fz,alfa))

						alfa_d[j-1] = delta*alfa
						return alfa, fz, nf, i_corr_fall

					if d[j-1] > 0.0:

						if (alfa/delta1-(bu[j-1]-x[j-1])) < -1.e-6:
							alfaex = alfa/delta1
						else:
							alfaex = bu[j-1]-x[j-1]
							ifront = 1
							if iprint >= 1:
								print(' punto espan. sulla front.')

					else:

						if (alfa/delta1-(x[j-1]-bl[j-1])) < -1.e-6:
							alfaex = alfa/delta1
						else:
							alfaex = x[j-1]-bl[j-1]
							ifront = 1
							if iprint >= 1:
								print(' punto espan. sulla front.')

					z[j-1] = x[j-1] + alfaex*d[j-1]

					fzdelta = self.funct(z)
					nf     += 1

					if iprint >= 1:
						print(' fzex=%f  alfaex=%f' % (fzdelta,alfaex))

					fpar = f - gamma*alfaex**2

					if fzdelta < fpar:
						fz   = fzdelta
						alfa = alfaex
					else:
						alfa_d[j-1] = delta*alfa

						if iprint>= 1:
							print(' accetta punto fz =%f   alfa =%f' % (fz,alfa))

						return alfa, fz, nf, i_corr_fall

			else:   #opposite direction

				d[j-1] = -d[j-1]
				ifront = 0

				if iprint >= 1:
					print(' direzione opposta')
					print(' j =%d    d(j) =%f' % (j,d[j-1]))

		if i_corr_fall != 2:
			alfa_d[j-1] = delta*alfa_d[j-1]

		alfa = 0.0

		if iprint >= 1:
			print(' failure along the direction')

		return alfa, f, nf, i_corr_fall

	def QPS(self,xk,f,fk,fk_1,d1,d2):

		fA = f
		fB = fk_1
		fC = fk
		fD = self.funct(xk - d1 + d2)
		fE = self.funct(xk + 0.1*d1 + 0.2*d2)
		fF = self.funct(xk - 0.1*d1 - 0.1*d2)

		#bc = 2 * (gab[1] + fB - fA)
		#ba = 2 / (alpha ** 2) * (fD - alpha * gab[0] - fA)
		#bb = (fC - fA - gab[0] * alpha - gab[1] * beta - 0.5 * alpha * alpha * ba - 0.5 * beta * beta * bc) / (
		#			alpha * beta)
		#ga =
		#gb =

		AM = np.array([[0., -1, 0, 0, 0.5],
					   [1.,  0, 0.5, 0, 0],
					   [-1, 1., 0.5, -1, 0.5],
					   [ 0.1,  0.1, 0.5*0.1*0.1, 0.1*0.1, 0.5*0.1*0.1],
					   [-0.1, -0.1, 0.5*0.1*0.1, 0.1*0.1, 0.5*0.1*0.1]])

		FS = np.array([fB-fA, fC-fA, fD-fA, fE-fA, fF-fA])

		qs = np.linalg.lstsq(AM, FS, rcond=None)
		ga, gb, ba, bb, bc = qs[0][0], qs[0][1], qs[0][2], qs[0][3], qs[0][4]
		Bab = np.array([[ba,bb],[bb,bc]])
		gab = np.array([ga,gb])

		try:
			solution_closed = np.linalg.solve(Bab, -gab)
		except np.linalg.LinAlgError:
			solution_closed = np.linalg.lstsq(Bab, -gab, rcond=None)[0]
		best = solution_closed
		return best

	#     *********************************************************
	#     *
	#     *                 DF_BOX outer loop
	#     *
	#     *********************************************************
	def sd_box(self,x):
	#     initialization
		n = len(x)
		nfails = 0
		alfa_stop = self.tol
		bl = self.lb
		bu = self.ub
		nf_max = self.maxfev
		maxiter = self.maxiter
		iprint = self.iprint
		eta       = 1.e-6
		num_fal   = 0
		istop     = 0
		flag_fail = [0]*n
		fstop     = [0.0]*(n+1)
		alfa_d    = [0.0]*n
		d         = [1.0]*n
		nf = 0

		format100 = ' ni=%4d  nf=%5d   f=%12.5e   alfamax=%12.5e a=%12.5e b=%12.5e'

	#---- choice of the starting stepsizes along the directions --------

		for i in range(0,n):
			alfa_d[i] = max(1.e-3,min(1.0,abs(x[i])))

			if iprint >= 1:
				print(' alfainiz(%d)=%e' % (i,alfa_d[i]))

		alfa_max = max(alfa_d)
		f   = self.funct(x)
		nf += 1
		i_corr = 1
		fstop[i_corr-1] = f

		dm   = np.zeros(n)
		xk   = np.copy(x)
		xk_1 = np.copy(x)
		fk   = f
		fk_1 = f
		ab   = [0., 0.]

	#---------------------------
	#     main loop
	#---------------------------

		for ni in range(1,maxiter+1):

			if iprint >= 0:
				print(format100 % (ni,nf,f,alfa_max,ab[0],ab[1]))

	#-------------------------------------
	#    sampling along coordinate i_corr
	#-------------------------------------
			alfa, fz, nf, i_corr_fall = self.linesearchbox_cont(n,x,f,d,alfa_d,i_corr,alfa_max,iprint,bl,bu,nf)

			if abs(alfa) >= 1.e-12:
				flag_fail[i_corr-1] = 0
				x[i_corr-1] = x[i_corr-1] + alfa*d[i_corr-1]
				f = fz
				fstop[i_corr-1] = f
				num_fal = 0
				ni += 1

			else:

				flag_fail[i_corr-1] = 1
				if i_corr_fall < 2:
					fstop[i_corr-1] = fz
					num_fal += 1
					ni += 1

			istop, alfa_max = self.stop(n,alfa_d,nf,ni,fstop,f,alfa_stop,nf_max,flag_fail)

			if istop >= 1:
				if iprint >= 0:
					print(format100 % (ni,nf,f,alfa_max,ab[0],ab[1]))
				break

			if i_corr < n:
				i_corr += 1
			else:
				i_corr = 1

				if True:
					g = x - xk

					def f2(ab):
					    return self.funct(xk+ab[0]*g+ab[1]*dm)

					# STEP MOMENTUM CON PLANAR SEARCH
					# (alfa,beta) = (1,0) --> f(x)
					# (alfa,beta) = (0,-1) --> f(xk_1)

					sol = minimize(f2, [2.,0.], method="Nelder-Mead", options={"disp": False, "maxfev": 10})
					ab = sol.x

					#ab = self.QPS(xk,f,fk,fk_1,g,dm)

					xhat = xk + ab[0]*g+ab[1]*dm
					fhat = self.funct(xhat)
					xk_1 = np.copy(xk)
					fk_1 = fk
					print("fhat = {}, f = {}".format(fhat,f))
					if fhat < f:
						xk = np.copy(xhat)
						fk = fhat
						x = np.copy(xk)
						f = fk
						#input()
					else:
						xk = np.copy(x)
						fk = f
						nfails += 1
					dm = xk - xk_1


		return x, {"iters": ni, "f": f, "g_norm": np.max(alfa_d), "nfails": nfails, "nnegeig": np.nan, "cosmax": np.nan}


#-----------------------------------------------------------------------
#      Starting point and bound calculation
#-----------------------------------------------------------------------

'''
bl, bu = setbounds()	
x = startp()
n = len(x)

exit = False

for i in range(0,n):
	if (x[i] < bl[i]) or (x[i] > bu[i]):
		print('ERROR: initial point is out of the feasible box')
		exit = True		

if not exit:
	num_funct   = 0 
	alfa_stop   = 1.e-6
	maxiter     = 20000
	nf_max      = 20000
	iprint      = 0

	fob   = funct(x)
	finiz = fob
	num_funct += 1

	print(' ------------------------------------------------- ')
	print(' objective function at xo = %f' % fob)
	print(' ------------------------------------------------- ')

	tbegin = time.clock()

	xstar, f = sd_box(funct,n,x,fob,bl,bu,alfa_stop,nf_max,maxiter,num_funct,iprint)

	tend = time.clock()

	format = ' & %3d & %14.7e & %14.7e & %5d & %9.2e'
	print(format % (n,finiz,f,num_funct,(tend-tbegin)))


	print('------------------------------------------------------------------------------')
	print(' total time:%f' % (tend-tbegin))
	print('------------------------------------------------------------------------------')
'''
