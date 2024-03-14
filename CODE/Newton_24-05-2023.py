
"""
Created on Tue Nov 23 13:10:39 2021

@author: Stefa
"""
import numpy as np

class Newton:

        def __init__(self,n,f0,x,eps,maxiter,iprint,funct,d1,d2,n_iter_glob):
            self.n = n
            self.f0=f0
            self.x = x
            self.eps = eps
            self.maxiter = maxiter
            self.iprint = iprint
            self.funct = funct
            self.d1=d1
            self.d2=d2
            self.n_iter_glob=n_iter_glob
            
        def gradiente_Hessiano(self,n,x,f,prec,funct,d1,d2,n_iter,n_iter_glob):
        #def gradiente_Hessiano(self,n,x,f,prec,funct,d1,d2,n_iter,n_iter_glob):
            d1_norm=np.linalg.norm(d1,2)
            d2_norm=np.linalg.norm(d2,2)
            prec1=prec/min(np.maximum(d1_norm,1.e-16),1.e+16)
            prec2=prec/min(np.maximum(d2_norm,1.e-6),1.e+6)
            #prec1=prec/min(np.maximum(d1_norm,1.e-12),1.e+12)
            #prec2=prec/min(np.maximum(d2_norm,1.e-12),1.e+12)            
            # prec1=1.e-6
            # prec2=1.e-6
            gapp=np.zeros(n)
            HessApp=np.zeros(shape=(n,n))
            f_min=f
            x_min=np.copy(x)
            if d2_norm==0.:
                if (n_iter==1):
                    x1=np.copy(x)
                    gapp[0]=-d1_norm*d1_norm
                    x1[0]=x1[0]+prec1
                    f1=funct(x1)
                    #if f_min>f1:
                    #	f_min=f1
                    #	x_min=np.copy(x1)
                else:
                    x1=np.copy(x)
                    xm=np.copy(x)
                    x1[0]=x1[0]+prec1
                    xm[0]=xm[0]-prec1			
                    f1=funct(x1)
                    #if f_min>f1:
                    #	f_min=f1
                    #	x_min=np.copy(x1)
                    fm=funct(xm)
                    #if f_min>fm:
                    #	f_min=fm
                    #	x_min=np.copy(xm)                    	
                    gapp[0]=(f1-fm)/(2.*prec1)
                    gapp[1]=0.0
                ba = (2./prec1)*((f1-f)/prec1-gapp[0])
                bc = 0.
                bb = 0.
                HessApp = np.array([[ba, bb], [bb, bc]])
                return gapp,HessApp #,f_min,x_min			
            if (n_iter==1):
                gapp[0]=-d1_norm*d1_norm
                gapp[1]=-np.dot(d1.T,d2)
                x1=np.copy(x)
                x1[0]=x1[0]+prec1			
                f1=funct(x1)
                #if f_min>f1:
                #	f_min=f1
                #	x_min=np.copy(x1)
                x1=np.copy(x)
                x1[1]=x1[1]+prec2		
                f2=funct(x1)
                #if f_min>f2:
                #	f_min=f2
                #	x_min=np.copy(x1)  
            else:			
                x1=np.copy(x)
                xm=np.copy(x)
                x1[0]=x1[0]+prec1
                xm[0]=xm[0]-prec1			
                f1=funct(x1)
                #if f_min>f1:
                #	f_min=f1
                #	x_min=np.copy(x1)
                fm=funct(xm)
                #if f_min>fm:
                #	f_min=fm
                #	x_min=np.copy(xm)
                gapp[0]=(f1-fm)/(2.*prec1)
                x1=np.copy(x)
                xm=np.copy(x)
                x1[1]=x1[1]+prec2
                xm[1]=xm[1]-prec2			
                f2=funct(x1)
                #if f_min>f2:
                #	f_min=f1
                #	x_min=np.copy(x1)
                fm=funct(xm)
                #if f_min>fm:
                #	f_min=fm
                #	x_min=np.copy(xm)
                gapp[1]=(f2-fm)/(2.*prec2)
            x1=np.copy(x)
            x1[0]=x1[0]+prec1
            x1[1]=x1[1]+prec2
            f3=funct(x1)
            #if f_min>f3:
            #	f_min=f3
            #	x_min=np.copy(x1) 			
            ba = (2./prec1)*((f1-f)/prec1-gapp[0])
            bc = (2./prec2)*((f2-f)/prec2-gapp[1])
            bb = (1./prec1)*((f3-f)/prec2)-(1./prec2)*((f1-f)/prec1)-(1./prec1)*((f2-f)/prec2)
            if False: #bc != bc1 or ba != ba1 or bb != bb1:
                print(bc,' ',bc2)
                print(ba, ' ', ba2)
                print(bb, ' ', bb2)
                input()			
            HessApp = np.array([[ba, bb], [bb, bc]])

            return gapp,HessApp #,f_min,x_min		
		
        def perturb_Hessian(self,eps_H,H):
            HP=np.zeros(shape=(2,2))
            HP=np.copy(H)
            #while (eps_H-1./eps_H >= -2.*abs(H[0,1])):
            #    eps_H=(10.**-1)*eps_H
            print('H=',H)
            print('eps_H=',eps_H)
            delta1=eps_H+abs(H[0,1])-H[0,0]
            delta2=eps_H+abs(H[0,1])-H[1,1]
            HP[0,0]=H[0,0]+delta1
            HP[1,1]=H[1,1]+delta2
            print('delta1=',delta1)
            print('delta2=',delta2)
            print('HP=',HP)
            return HP
            
        def perturb_Cholesky(self,g,H,norma_g,eps_H):
         
            d=np.zeros(2)
         
            if H[0,0] > 1.e-12*min(norma_g**2,1):
                l11=np.sqrt(H[0,0])
            else:
           	    l11=eps_H

            l21=H[1,0]/l11
            
            if H[1,1]-l21*l21 > 1.e-12*min(norma_g**2,1):
                l22=np.sqrt(H[1,1]-l21*l21)
            else:
                l22=eps_H
                
            y1=-g[0]/l11
            y2=-(g[1]+l21*y1)/l22
            
            d[1]=y2/l22
            d[0]=(y1-l21*d[1])/l11           
            
            return d          
		
		
		    
        def linesearch_armijo(self,n,x,f,d,gd,gamma,nf,funct):
    
            alfa=1.
	
            y=np.zeros(n)
    
            y=x+alfa*d
            f_alfa=funct(y)
            f_min=f_alfa
            alfa_min=alfa
            #print(' f_alfa iniz=',f_alfa)
            #print(' alfa iniz=',alfa)  
            nf=nf+1
   
    #        while(f_alfa >1.e+20 ):
            while(f_alfa >(f+alfa*gamma*gd) ):
                if f_alfa-(f+alfa*gamma*gd) > 1.e6:
                    alfa=alfa/10.
                else:					
                    alfa=alfa/2.
                y=x+alfa*d
                f_alfa=funct(y)
                #if f_min>f_alfa:
                #	f_min=f_alfa
                #	alfa_min=alfa
                #print(' f_alfa =',f_alfa)
                #print(' alfa =',alfa)                
                nf=nf+1
            #print(' alfa*gamma*gd =',alfa*gamma*gd)
            #print(' f+alfa*gamma*gd  =',f+alfa*gamma*gd) 
            #alfa=alfa_min
            #f_alfa=f_min 
            return alfa,f_alfa,nf
            
        def linesearch_armijo_nc(self,n,x,f,d,gd,gamma,nf,funct):
    
            alfa=1.
	
            y=np.zeros(n)
    
            y=x+alfa*d
            f_alfa=funct(y)
            f_min=f_alfa
            alfa_min=alfa
            #print(' f_alfa iniz=',f_alfa)
            #print(' alfa iniz=',alfa)  
            nf=nf+1
   
    #        while(f_alfa >1.e+20 ):
            while(f_alfa >(f+alfa*gamma*gd) ):
                if f_alfa-(f+alfa*gamma*gd) > 1.e6:
                    alfa=alfa/10.
                else:					
                    alfa=alfa/2.
                y=x+alfa*d
                f_alfa=funct(y)
                if f_min>f_alfa:
                	f_min=f_alfa
                	alfa_min=alfa
                #print(' f_alfa =',f_alfa)
                #print(' alfa =',alfa)                
                nf=nf+1
            if alfa==1:
            	y=x+(2.*alfa)*d
            	f_ex=funct(y)
            	nf=nf+1
            	while (f_ex <=(f+2.*alfa*gamma*gd)):
                    alfa=2.*alfa
                    f_alfa=f_ex
                    y=x+2.*alfa*d
                    f_ex=funct(y)
                    if f_min>f_alfa:
                   	    f_min=f_alfa
                   	    alfa_min=alfa
                    #print(' f_alfa =',f_alfa)
                    #print(' alfa =',alfa)                
                    nf=nf+1            	
            #print(' alfa*gamma*gd =',alfa*gamma*gd)
            #print(' f+alfa*gamma*gd  =',f+alfa*gamma*gd) 
            alfa=alfa_min
            f_alfa=f_min 
            return alfa,f_alfa,nf            

        def direction(self,g,H,norma_g,n,x,n_iter_glob):
            nc=0
            eps_H=10**-6
            d=np.zeros(n)
            
            if (H[1,1]==0) and (H[0,1]==0):
                d[0]=-g[0]/min(np.maximum(abs(H[0,0]),1.e-16),1.e16)
                #d[0]=-g[0]/min(np.maximum(H[0,0],1.e-9),1.e9)
                               
                d[1]=0.
                #print('H[0,0],=',H[0,0])
                #print('d[0]=',d[0])
                #print('d[1]=',d[1])
                gd=np.dot(g.T,d)
                #print('gd=',gd)
                # input()	
                return d, gd, nc
            # try:
                # d= np.linalg.solve(H,-g)
            # except np.linalg.LinAlgError:
                # print("An exception occurred")
                # d = np.linalg.lstsq(H, -g, rcond=None)[0]
            # gd=np.dot(g.T,d)
            # if (gd > 0 ):
                # d=-d
                # gd=-gd
            # norma_d=np.sqrt(np.dot(d.T,d))
            # # if((gd>-(1.e-7*norma_g)*(1.e-7*norma_g)) or (norma_d>10**15*norma_g)):
                # # for j in range(0,n-1):
                    # # d[j]=-g[j]/min(np.maximum(abs(H[j,j]),10**-3),1.e+3)
                # # gd=np.dot(g.T,d)                	
            # # d[0]=-g[0]/H[0,0]
            # # d[1]=0.
    #        try:
    #        solution_closed = np.linalg.solve(Bab, -gab)
    #    except np.linalg.LinAlgError:
    #        solution_closed = np.linalg.lstsq(Bab, -gab, rcond=None)[0]
    #    best = solution_closed
            
            
            try:
                #d= scipy.linalg.solve(H,-g,check_finite=False, assume_a='gen', )
                d= np.linalg.solve(H,-g)
            except:
                print("An exception occurred")    
                #for j in range(0,n):
                #  d[j]=-g[j]/min(np.maximum(abs(H[j,j]),10**-3),1.e+3)
                #HP=self.perturb_Hessian(eps_H,H)
                #d= np.linalg.solve(HP,-g)
                d = np.linalg.lstsq(H, -g, rcond=None)[0]
                #d = self.perturb_Cholesky(g,H,norma_g,eps_H)
                  
            gd=np.dot(g.T,d)
            #if  False:
            if (gd > 0 ):
                #print(gd)
                d=-d
                gd=-gd
            #    nc=1
            #    #print("Newton non di discesa")
            norma_d=np.sqrt(np.dot(d.T,d))
            #if False:
            #print(gd)
            #print(norma_g)
            #print(norma_d)
            if((gd>-(1.e-12*np.minimum(norma_g,1.))*(1.e-12*np.minimum(norma_g,1.))) or (norma_d>10**18*norma_g)):
                #for j in range(0,n):
                #   d[j]=-g[j]/min(np.maximum(abs(H[j,j]),10**-3),1.e+3)
                #gd=np.dot(g.T,d)
                #HP=self.perturb_Hessian(eps_H,H)
                #d= np.linalg.solve(HP,-g)	
                d = self.perturb_Cholesky(g,H,norma_g,eps_H)                			
                #print("Newton non gr.related")
				
            #if (d[0]< 1.e-6):
            if False:
                    d[0]=1.e1*d[0]
                    d[1]=1.e1*d[1]
                    gd=np.dot(g.T,d)
                    if (gd > 0 ):
                        d=-d
                        gd=-gd
            #print(' H=',H)
            #print('d[0]=',d[0])
            #print('d[1]=',d[1])	
            #print('gd=',gd)				
            return d, gd, nc
			
        def minimize(self):
            n = self.n
            f0=self.f0
            x = self.x
            eps = self.eps
            maxiter = self.maxiter
            iprint = self.iprint
            funct= self.funct
            d1=self.d1
            d2=self.d2
            n_iter_glob=self.n_iter_glob
            f_min=1.e36
            #iprint=True
			
            if iprint:
            #if True:
                file_10=open('risultati.txt',"w")

           
            gamma=10**-6
            prec=10**-3
            eps_H=10**-6

#            f=funct(x)

            f=f0			
            nf=0
                             
            d=np.zeros(n)
            # print(x)
            # print(f)
            n_iter=0

            while True:
    
                n_iter=n_iter+1
                if(f == -np.inf ):
                    #if True:
                    if iprint:
                        print("         ")
                        print("problema illimitato inferiormente  nf=",nf)
                        print("         ",file=file_10)
                        print("problema illimitato inferiormente  nf=",nf,file=file_10)
                        for i in range(0,n):
                            print  ("x(",i+1,") =",x[i])
                            print("         ")
                        input()
                    break      
                if(n_iter > maxiter):
                    #if True:
                    if iprint:
                        print("         ")
                        print("Algoritmo terminato per massimo numero di iterazioni  nf=",nf)
                        print("         ",file=file_10)
                        print("Algoritmo terminato per massimo numero di iterazioni  nf=",nf,file=file_10)
                        for i in range(0,n):
                            print  ("x(",i+1,") =",x[i])
                            print("         ")
                        input()
                    break 
               
                # test=np.sqrt(np.dot(d1.T,d1))
                # if(f <= f0-1.e-3*test):
                if False:
                    if iprint:
                    #if True:					
                        print("         ")
                        print("Algoritmo terminato con un punto che produce una sufficiente riduzione",f-f0)
                        print("         ")
                        print("         ",file=file_10)
                        print("Algoritmo terminato con un punto che produce una sufficiente riduzione",f-f0,norma_g,file=file_10)
                        print("         ",file=file_10)
                        for i in range(0,n):
                           print  ("x(",i+1,") =",x[i])
                           print("         ")
                        input()
                    break
                #g,H,f_min,x_min=self.gradiente_Hessiano(n,x,f,prec,funct,d1,d2,n_iter,n_iter_glob)
                g,H=self.gradiente_Hessiano(n,x,f,prec,funct,d1,d2,n_iter,n_iter_glob)
                norma_g = np.linalg.norm(g,2)
                # g_norm = np.linalg.norm(g,np.inf)
                # norma_g=np.sqrt(np.dot(g.T,g))
                #if True:
                if iprint:
                   print('grad=',g)
                   print  ("n_iter=",n_iter,"  nf",nf,"  f=",f,"  norma_grad=",norma_g)
                   print  ("n_iter=",n_iter,"  nf",nf,"  f=",f,"  norma_grad=",norma_g,file=file_10)
#                   input()
                if(norma_g <= eps):
                    if iprint:
                        print("         ")
                        print("Algoritmo terminato con un punto stazionario  norma del gradiente =",norma_g)
                        print("         ")
                        print("         ",file=file_10)
                        print("Algoritmo terminato con un punto stazionario  norma del gradiente =",norma_g,file=file_10)
                        print("         ",file=file_10)
                        for i in range(0,n):
                           print  ("x(",i+1,") =",x[i])
                           print("         ")
                        #input()
                    break				
 
    
                d,gd,nc = self.direction(g,H,norma_g,n,x,n_iter_glob)
                nc=0
                if nc==0:    
                    alfa,f_alfa,nf=self.linesearch_armijo(n,x,f,d,gd,gamma,nf,funct)
                else:
                    alfa,f_alfa,nf=self.linesearch_armijo_nc(n,x,f,d,gd,gamma,nf,funct)
                x=x+alfa*d
    
                f=f_alfa
                
                #if f_min< f:
                #	f=f_min
                #	x=x_min
				
            return f,x			
    
    
