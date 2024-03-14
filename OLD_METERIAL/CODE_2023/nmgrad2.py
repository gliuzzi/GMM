# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np

class nmgrad2:
    def __init__(self,n,x,eps,maxgrad,iprint,funct_grad):
        self.n = n
        self.x = x
        self.eps = eps
        self.maxgrad = maxgrad
        self.iprint = iprint
        self.funct_grad = funct_grad

    def minimize(self):
        n = self.n
        x = self.x
        eps = self.eps
        maxgrad = self.maxgrad
        iprint = self.iprint
        funct_grad = self.funct_grad

        """----------------------
          starting values
          ---------------------- """
        M=10   #originale 20
        nn=10
        # M=1
        # nn=1
        w=np.zeros((M,1))
        g=np.zeros(n)
        file_10=open('stampeNMBB.txt',"w")
        dfmax=10**4
        epsilon=10**-10
        maxrest=maxgrad
        maxiter=maxgrad



        f,g  = funct_grad(x)


        ng=1
        #print  ("ng=",ng,"f=",f,file=file_10)
        xmin=x
        fmin=f
        gmin=g

        x_current = x.copy()
        f_current = f
        g_current = g.copy()

        "------------------------------------"
        for i in range(1,maxrest):
            ifail=-1
    #        print("--------------")
    #        print("i=", i)

            norma2gq=np.dot(g.T,g)
            norma2g=np.sqrt(norma2gq)
            norma2x=np.sqrt(np.dot(x.T,x))
            dnrx0=1.+norma2x
            epsupp=norma2g/(epsilon*dnrx0)
            w[:]=-10**20
            w[0]=f
            dspost=np.zeros((nn,1))
            """----------------------
            stopping  criterion
            ---------------------- """
            criterio=eps*(1.+np.abs(f))
            if(norma2g <= criterio):
              ifail=0
              return  x,f,ng,ifail,x_current, f_current, g_current
              break
    #        iter=0
            """----------------------
            intial stepsize
            ---------------------- """
            alpha=min(10**20,norma2g)
            lambda_var=1/alpha
    #        print("lambda iniziale", lambda_var)
            if iprint >=1 :
                print('Starting values')
                print(f'ng ={ng}')
    #            print(f'f ={f:.8f}   gnr={norma2g:.8f}')
                print(f'f ={f:.8f}   gnr={norma2g}')
                print(f'**************************************')
                print('Starting values',file=file_10)
                print(f'ng ={ng}',file=file_10)
    #            print(f'f ={f:.8f}   gnr={norma2g:.8f}',file=file_10)
                print(f'f ={f:.8f}   gnr={norma2g}',file=file_10)
                print(f'**************************************',file=file_10)

            newg = 0

            x,f,w,ng,newg,xmin,fmin,gmin, iline, lambda_var=self.lsexpq2(funct_grad,n,x,f,w,g,lambda_var,M,dnrx0,norma2gq,ng,nn,xmin,fmin,newg,gmin)

            lnl=0
            xbest=x.copy()
            fbest=f
            ibest=1
            if(iline > 0) :
                ifail=iline
                return x,f,ng,ifail,x_current, f_current, g_current

            """-------------------------------------------------
                   inner cycle
            --------------------------------------------------"""
            for n_iter in range(1,maxiter):
    #            print("***** \t, j=", n_iter)
                imod1=0
                imod2=0
                imodc=0

                """---------------------------------------------------
                search direction
                -----------------------------------------------------"""
                y=newg-g
    #            print("lambda: ", lambda_var)
                sy=np.dot(-lambda_var*g.T,y)
                den=lambda_var*lambda_var*norma2gq
                g=newg.copy()
                norma2gq=np.dot(g.T,g)
                norma2g=np.sqrt(norma2gq)
                criterio=eps*(1+abs(f))
                if(norma2g <= criterio):
    #                print("norma < criterio")

                    if((fmin < f) and (abs(fmin-f)/(1+abs(fmin)) > 1**-10)) :
                        if(iprint >= 1) :
                            print(f'f ={f:.8f}  fmin={fmin:.8f}')
                            print(f'restart')
                            print(f'f ={f:.8f}  fmin={fmin:.8f}',file=file_10)
                            print(f'restart',file=file_10)


                        x=xmin.copy()
                        f=fmin
                        g=gmin.copy()


                        norma2gq=np.dot(g.T,g)
                        norma2g=np.sqrt(norma2gq)
                        criterio=eps*(1+abs(f))
                        if(norma2g <= criterio) :
                            if(iprint >= 1) :
                                print(f'Termination criterion satisfied')
                                print(f'ng ={ng}  ')
    #                            print(f'f ={f:.8f}  norma2g={norma2g:.8f}')
                                print(f'f ={f:.8f}  norma2g={norma2g}')
                                print(f'Termination criterion satisfied',file=file_10)
                                print(f'ng ={ng}  ',file=file_10)
    #                            print(f'f ={f:.8f}  norma2g={norma2g:.8f}',file=file_10)
                                print(f'f ={f:.8f}  norma2g={norma2g}',file=file_10)

                            ifail=0
                            return x,f,ng,ifail,x_current, f_current, g_current
                        else:
                            ifail=-2
                            break

                    else:
                        if(iprint >= 1) :


                            print(f'Termination criterion satisfied')
                            print(f'ng ={ng}  ')
    #                        print(f'f ={f:.8f}  norma2g={norma2g:.8f}')
                            print(f'f ={f:.8f}  norma2g={norma2g}')
                            print(f'Termination criterion satisfied',file=file_10)
                            print(f'ng ={ng}  ',file=file_10)
    #                        print(f'f ={f:.8f}  norma2g={norma2g:.8f}',file=file_10)
                            print(f'f ={f:.8f}  norma2g={norma2g}',file=file_10)

                        ifail=0
                        return  x,f,ng,ifail,x_current, f_current, g_current

                if iprint >= 1:
                    print(f'ng ={ng}  ')
        #                print(f'f ={f:.8f}  norma2g={norma2g:.8f}')
                    print(f'f ={f:.8f}  norma2g={norma2g}')
                    print(f'ng ={ng}  ',file=file_10)
    #                print(f'f ={f:.8f}  norma2g={norma2g:.8f}',file=file_10)
                    print(f'f ={f:.8f}  norma2g={norma2g}',file=file_10)


                """---------------------------------------------------------------------
                computation of the safeguarded Barzilai-Borwein stepsizes
                ---------------------------------------------------------------------"""

                """---------------------------------------------------------------------
                computation of the BB stepsize alpha=s^Ty/s^Ts
                ---------------------------------------------------------------------"""

                if(den >= 10**-25):
                    alpha=sy/den
                else:
                    imod1=1
                    alpha=min(norma2g,10**20)

                """----------------------------------------------------------------------
                    computation of the BB stepsize alpha=y^Ty/s^Ty
                    ----------------------------------------------------------------------"""
                if(sy >= 10**-25) :
                    alpha2=np.dot(y.T,y)/sy
                else:
                    imod2=1
                    alpha2=min(norma2g,10**20)

    #            print("alpha,", alpha, "alpha2,", alpha2)
    #            print("den,", den, "sy,", sy)

                epslow=max(epsilon,norma2g/((10**5)*dnrx0))
                """----------------------------------------------------------------------
                    safeguarded BB stepsize alpha=s^Ty/s^Ts
                    ----------------------------------------------------------------------"""
                epsupp1=epsupp
                if((alpha <= epslow) or (alpha > epsupp1)):
                    imod1=1
                    alpha=min(norma2g,10**+20)

                """----------------------------------------------------------------------
                    safeguarded BB stepsize alpha=y^Ty/s^Ty
                    ----------------------------------------------------------------------"""

                if((alpha2 <= epslow) or (alpha2 > epsupp1)):
                    imod2=1
                    alpha2=min(norma2g,10**+20)

                    """"----------------------------------------------------------------------
                        choice of the BB stepsize
                        ----------------------------------------------------------------------"""

                if(n_iter%2 == 0) :

                    if((imod2 == 0) or (imod1 == 1)) :
                        alpha=alpha2
                        imodc=imod2
                    else:
                        imodc=imod1

                else:

                   if((imod1 == 0) or (imod2 == 1)) :
                       imodc=imod1
                   else:
                       alpha=alpha2
                       imodc=imod2

                lambda_var=1/alpha

                if(lnl==0) :
                    gbest=g.copy()
                    dnrbest=norma2gq
                    lambdabest=lambda_var*0.75

                """----------------------------------------------------------------------------
                    watchdog test
                    ----------------------------------------------------------------------------"""

                x=x-lambda_var*g
    #            print("lambda: ", lambda_var)
                dist=lambda_var*norma2g
                dspost[lnl]=dist
                f,newg=funct_grad(x)


                if(f < fmin) :

                    xmin=x.copy()
                    fmin=f
                    gmin=newg.copy()

                ng=ng+1

    #                    aux=maxval(w)-1.d-6*maxval(dspost(1:lnl))/dfloat(lnl)
                aux=np.max(w)-(10**-6)*np.max(dspost[0:lnl+1])/(lnl+1)
                if(f <= aux) :
                    for  i in range(M-1,0,-1):
                        w[i]=w[i-1]

                    w[0]=f
                    xbest=x.copy()
                    fbest=f
                    lnl=0
                else:
                    if((lnl == nn-1) or (f>fbest+dfmax*(1+abs(fbest)))) :
                        lnl=0
                        x=xbest.copy()
                        f=fbest
                        lambda_var=lambdabest
                        g=gbest.copy()
                        norma2gq=dnrbest.copy()
                        x,f,w,ng,newg,xmin,fmin,gmin, iline, lambda_var=self.lsexpq2(funct_grad,n,x,f,w,g,lambda_var,M,dnrx0,norma2gq,ng,nn,xmin,fmin,newg,gmin)

                        if(iline > 0) :
                            ifail=iline

                            x_current = x.copy()
                            f_current = f
                            g_current = g.copy()


                            x=xmin.copy()
                            f=fmin
                            g=gmin.copy()
                            return  x,f,ng,ifail,x_current, f_current, g_current

                        xbest=x.copy()
                        fbest=f

                    else:

                        lnl=lnl+1


            if(ifail ==  -2) :
                if(maxiter > ng) :
                    maxiter=maxiter-ng
                else:
                    ifail=1
                    if (iprint >= 1) :
                        print(f'Maximum number of gradient evaluations')
                        print(f'ng ={ng}  ')
    #                    print(f'f ={f:.8f}  norma2g={norma2g:.8f}')
                        print(f'f ={f:.8f}  norma2g={norma2g}')
                        print(f'Maximum number of gradient evaluations',file=file_10)
                        print(f'ng ={ng}  ',file=file_10)
    #                    print(f'f ={f:.8f}  norma2g={norma2g:.8f}',file=file_10)
                        print(f'f ={f:.8f}  norma2g={norma2g}',file=file_10)

                    return  x,f,ng,ifail,x_current, f_current, g_current

            else:

                if(fmin > f) :


                    x=xmin.copy()
                    f=fmin
                    g=gmin.copy()


                ifail=1
                if (iprint >= 1) :
                        print(f'Maximum number of gradient evaluations')
                        print(f'ng ={ng}  ')
    #                    print(f'f ={f:.8f}  norma2g={norma2g:.8f}')
                        print(f'f ={f:.8f}  norma2g={norma2g}')
                        print(f'Maximum number of gradient evaluations',file=file_10)
                        print(f'ng ={ng}  ',file=file_10)
    #                    print(f'f ={f:.8f}  norma2g={norma2g:.8f}',file=file_10)
                        print(f'f ={f:.8f}  norma2g={norma2g}',file=file_10)

                return  x,f,ng,ifail,x_current, f_current, g_current

    #        print(f'ifail ={ifail}  norma2g={norma2g:.8f}')
    #        print(f'ifail ={ifail}  norma2g={norma2g:.8f}',file=file_10)
            #print(f'ifail ={ifail}  norma2g={norma2g}')
            #print(f'ifail ={ifail}  norma2g={norma2g}',file=file_10)

        if(fmin < f) :


            x=xmin.copy()
            f=fmin
            g=gmin.copy()

        if (iprint >= 1) :



            print(f'Maximum number of gradient evaluations')
            print(f'ng ={ng}  ')
    #        print(f'f ={f:.8f}  norma2g={norma2g:.8f}')
            print(f'f ={f:.8f}  norma2g={norma2g}')
            print(f'Maximum number of gradient evaluations',file=file_10)
            print(f'ng ={ng}  ',file=file_10)
    #        print(f'f ={f:.8f}  norma2g={norma2g:.8f}',file=file_10)
            print(f'f ={f:.8f}  norma2g={norma2g}',file=file_10)
        file_10.close()
        return  x,f,ng,ifail, x_current, f_current, g_current
 

#def funct_grad(x):
#
#    from problem import problem
#
#    name,dim,init_point, function,gradient =problem()
#
#    f=function(x,dim())
#    g=gradient(x,dim())
#
#    return f,g


    def lsexpq2(self,funct_grad,n,x,f,w,g,lambda_var,M,dnrx,dnr,nf,nn,xmin,fmin,newg,gmin):
        import numpy as np
        iline=3
        maxiter=10
        gamma=10**-6
        dnrlam=lambda_var*np.sqrt(dnr)/dnrx
        sigma1=10**-1
        sigma2=5*10**-1
        gd=-dnr
        for j in range(1,maxiter):
            newx=x-lambda_var*g
            newf,newg = funct_grad(newx)
            if newf<fmin :
                xmin=x.copy()
                fmin=newf
                gmin=newg.copy()
            nf=nf+1
            wmax=np.max(w)
            lamgd=lambda_var*gd
            aux=wmax+(gamma*lambda_var*lamgd)
            if newf<aux:
                newgd=np.dot(-g.T,newg)
                if((j==1) and (dnrlam<10**-2) and (newf<f) and (nn>1) and (newgd<0)) :
                    """------------------------------------------------------------------------------------
                    !          extrapolation 
                    !------------------------------------------------------------------------------------"""
                    iline=2
                    for jj in range(1,maxiter):
                        den=2*(gd*lambda_var+f-newf)
                        if(np.abs(den)>10**-8) :

                            sigma=max(1.5,gd*lambda_var/den)
                            sigma=min(5,sigma)
                        else:
                            sigma=2

                        lambdaesp=sigma*lambda_var
                        xesp=x-lambdaesp*g
                        fesp,gesp= funct_grad(xesp)

                        if(fesp < fmin) :
                            xmin=xesp.copy()
                            fmin=fesp
                            gmin=gesp.copy()

                        nf=nf+1

                        if(fesp>min(newf,f+gamma*lambdaesp*lambdaesp*gd)) :
                            x=newx.copy()
                            iline=0
                            break
                        else:
                            newx=xesp.copy()
                            lambda_var=lambdaesp.copy()
                            newf=fesp
                            newg=gesp.copy()

                else:
                      x=newx.copy()
                iline=0
                break
            else:
                """--------------------------------------------------------------------------------------
                           interpolation
                --------------------------------------------------------------------------------------"""
                den=2*(gd*lambda_var+f-newf)
                if(abs(den)>10**-8):
                    sigma=max(sigma1,gd*lambda_var/den)
                    sigma=min(sigma2,sigma)
                else:
                    sigma=0.5

                lambda_var=sigma*lambda_var

        if(iline==0):
            f=newf
            for i in range(M-1,0,-1):
                w[i]=w[i-1]

            w[0]=f

        return x,f,w,nf,newg,xmin,fmin,gmin, iline, lambda_var





            
            

