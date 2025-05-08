"""
Created on Tue Nov 23 13:10:39 2021

@author: Stefa
"""
import numpy as np

class Newton:

        def __init__(self,n,f0,f_1,g_1,x,x_1,eps,maxiter,iprint,funct,grad,x_glob,d1,d2,n_iter_glob,var):
            self.n = n
            self.f0=f0
            self.f_1=f_1
            self.g_1 = g_1
            self.x = x
            self.x_1 = x_1
            self.eps = eps
            self.maxiter = maxiter
            self.iprint = iprint
            self.funct = funct
            self.grad=grad
            self.x_glob=x_glob
            self.d1=d1
            self.d2=d2
            self.n_iter_glob=n_iter_glob
            self.var=var
            
        def gradiente_Hessiano_B(self, n ,f, f_1, g_1, funct, d1,d2, alpha, beta,n_iter_glob,d1_norm):

            eps=1.e-24
            ab = np.zeros(2)
            ab1 = np.zeros(2)

            gab = np.zeros(2)
            gab1 = np.zeros(2)

            gab[0]=-d1_norm*d1_norm
            gab[1]=-np.dot(d1.T,d2)
            ab1[0]=0.
            ab1[1]=-1.
            gab1[0]=np.dot(g_1.T,d1)
            gab1[1]=np.dot(g_1.T,d2)

            if n_iter_glob == 0:
                mu=d1_norm
                mu_1=mu
                mu_2=mu
            else:
                y=gab1-gab
                s=ab1-ab
                if n_iter_glob % 2 == 0:
                   if abs(np.dot(s.T,y)) < 1.e-24:
                      mu=1.e1*1./eps
                   else:
                      mu=np.dot(y.T,y)/np.dot(s.T,y)
                   if (mu < eps) or (mu > 1./eps):
                      if abs(np.dot(s.T,s)) < 1.e-24:
                         mu=1.e1*1./eps
                      else:
                         mu=np.dot(s.T,y)/np.dot(s.T,s)
                      if (mu < eps) or (mu > 1./eps):
                         mu=d1_norm
                else:
                   if abs(np.dot(s.T,s)) < 1.e-24:
                      mu=1.e1*1./eps
                   else:
                      mu=np.dot(s.T,y)/np.dot(s.T,s)
                   if (mu < eps) or (mu > 1./eps):
                      if abs(np.dot(s.T,y)) < 1.e-24:
                         mu=1.e1*1./eps
                      else:
                         mu=np.dot(s.T,y)/np.dot(s.T,s)
                      if (mu < eps) or (mu > 1./eps):
                         mu=d1_norm

            Bab = np.array([[1./mu,0.],[0.,1./mu]])
  
            return gab, Bab
            
        def gradiente_Hessiano_D(self, n ,f, f_1,x, x_1, g_1, funct, d1,d2, alpha, beta,n_iter_glob,var,d1_norm):

            eps=1.e-12
            ab = np.zeros(2)
            n_glob=np.size(d1)
            yv = np.zeros(n_glob)
            sv =np.zeros(n_glob)
            muv =np.zeros(n_glob)

            gab = np.zeros(2)

            gab[0]=-d1_norm*d1_norm
            gab[1]=-np.dot(d1.T,d2)             
            if var < 7:     
                if n_iter_glob == 0:
                    if np.isnan(d1_norm):
                        min_sample_val  = 1.e-16
                    else:
                        min_sample_val =  1.e-3/min(np.maximum(d1_norm,1.e-16),1.e+16)
                    if np.abs(alpha) <= min_sample_val:
                        alpha = min_sample_val if alpha >= 0 else -min_sample_val 
                    f1=funct([alpha,0.])     
                    ba = (2./alpha)*((f1-f)/alpha-gab[0])
                    bc = 0.
                    bb = 0.
                    Bab = np.array([[ba, bb], [bb, bc]])
                    return gab,Bab
                else:
                    yv=-d1-g_1
                    sv1=d2
                    sv2=np.where(sv1>0,sv1,np.where(np.abs(sv1)<eps,-eps,sv1))
                    sv=np.where(sv2<0,sv2,np.where(np.abs(sv2)<eps,eps,sv2))
                    
                    if var==3:
                        muv=np.abs(yv/sv)
                        muv=np.where(np.abs(muv)<eps,eps,muv)
                        muv=np.where(np.abs(muv)>1./eps,1./eps,muv)
                        ba=np.sum(muv*d1**2)
                        bc=np.sum(muv*d2**2)
                        bb=np.sum(muv*d1*d2)
                    if var==4:
                        muv=yv/sv
                        muv=np.where(muv>=0,muv,np.where(np.abs(muv)<eps,-eps,muv))
                        muv=np.where(muv<=0,muv,np.where(np.abs(muv)<eps,eps,muv))
                        muv=np.where(muv>=0,muv,np.where(np.abs(muv)>1./eps,-1./eps,muv))
                        muv=np.where(muv<=0,muv,np.where(np.abs(muv)>1./eps,1./eps,muv)) 
                        ba=np.sum(muv*d1**2)
                        bc=np.sum(muv*d2**2)
                        bb=np.sum(muv*d1*d2)
                    if var==5:
                        muv=np.abs(yv/sv)
                        muv=np.where(np.abs(muv)<eps,eps,muv)
                        muv=np.where(np.abs(muv)>1./eps,1./eps,muv) 
                        ba=np.sum(muv*d1**2)
                        bc=np.dot(d2.T,yv)
                        bb=np.dot(d1.T,yv)
                    if var==6:
                        muv=yv/sv
                        muv=np.where(muv>=0,muv,np.where(np.abs(muv)<eps,-eps,muv))
                        muv=np.where(muv<=0,muv,np.where(np.abs(muv)<eps,eps,muv))
                        muv=np.where(muv>=0,muv,np.where(np.abs(muv)>1./eps,-1./eps,muv))
                        muv=np.where(muv<=0,muv,np.where(np.abs(muv)>1./eps,1./eps,muv)) 
                        ba=np.sum(muv*d1**2)
                        bc=np.dot(d2.T,yv)
                        bb=np.dot(d1.T,yv)
                    #print(ba,bb,bc)
                    Bab = np.array([[ba,bb],[bb,bc]])
                    return gab, Bab
            else:
                if n_iter_glob == 0:
                    if np.isnan(d1_norm):
                        min_sample_val  = 1.e-16
                    else:
                        min_sample_val =  1.e-3/min(np.maximum(d1_norm,1.e-16),1.e+16)
                    if np.abs(alpha) <= min_sample_val:
                        alpha = min_sample_val if alpha >= 0 else -min_sample_val 
                    f1=funct([alpha,0.])
                    ba = (2./alpha)*((f1-f)/alpha-gab[0])
                    bc = 0.
                    bb = 0.
                    Bab = np.array([[ba, bb], [bb, bc]])
                    return gab,Bab
                else:
                    ab = np.zeros(2)
                    ab1 = np.zeros(2)
                    gab = np.zeros(2)
                    gab1 = np.zeros(2)
                    ab1[0]=0.
                    ab1[1]=-1.
                    gab[0]=-d1_norm*d1_norm
                    gab[1]=-np.dot(d1.T,d2) 

                    y=gab1-gab
                    s=ab1-ab

                    if True:
                        if var==7 :
                            mu=min(np.maximum(np.dot(s.T,y)/np.maximum(np.dot(s.T,s),eps),eps),1./eps)
                        if var==8 :    
                            den=np.dot(s.T,s)
                            if den >= 0:
                                den= np.maximum(den,eps)
                            if den < 0:
                                den = min(den,-eps)
                            mu=np.dot(s.T,y)/den
                            if mu >= 0 :
                                mu = min(np.maximum(mu,eps),1./eps)
                            if mu < 0 :
                                mu = min(np.maximum(mu,-1./eps),-eps)                               
                        if var==9:
                            mu=min(np.maximum(np.dot(y.T,y)/np.maximum(np.dot(s.T,y),eps),eps),1./eps)
                        if var==10:
                            den=np.dot(s.T,y)
                            if den >= 0:
                                den= np.maximum(den,eps)
                            if den < 0:
                                den = min(den,-eps)
                            mu=np.dot(y.T,y)/den
                            if mu >= 0 :
                                mu = min(np.maximum(mu,eps),1./eps)
                            if mu < 0 :
                                mu = min(np.maximum(mu,-1./eps),-eps)
                yv=-d1-g_1
                ba=mu*d1_norm**2
                bc=np.dot(d2.T,yv)
                bb=np.dot(d1.T,yv)
                Bab = np.array([[ba,bb],[bb,bc]])    
         
                return gab, Bab
  
        def gradiente_Hessiano_M(self, n ,f, f_1, funct, d1,d2, alpha, beta,n_iter_glob,d1_norm,d2_norm):
        
            ab = np.zeros(2)
            gab= np.zeros(2)

            gab[0]=-d1_norm*d1_norm
            gab[1]=-np.dot(d1.T,d2)
            
            if np.isnan(d1_norm):
                min_sample_val  = 1.e-16
            else:
                min_sample_val =  1.e-3/min(np.maximum(d1_norm,1.e-16),1.e+16)
            if np.abs(alpha) <= min_sample_val:
                alpha = min_sample_val if alpha >= 0 else -min_sample_val 

            if n_iter_glob == 0:
                f1=funct([alpha,0.])
                ba = (2./alpha)*((f1-f)/alpha-gab[0])
                bc = 0.
                bb = 0.
                Bab = np.array([[ba, bb], [bb, bc]])
                return gab,Bab
                
            if np.isnan(d2_norm):
                min_sample_val  = 1.e-16
            else:               
                min_sample_val =  1.e-3/min(np.maximum(d2_norm,1.e-16),1.e+16)                
            if np.abs(beta) <= min_sample_val:
                beta = min_sample_val if beta >= 0 else  -min_sample_val                	
            fA = f
            fB = f_1
            fC = self.funct([alpha,beta])

            if fC == np.inf or np.isnan(fC)  or abs(fC-f) > np.maximum(abs(f),1.)*1.e3:
                alpha=1.e-3*alpha
                beta=1.e-3*beta
                fC = self.funct([alpha,beta])
            fD = self.funct([alpha,0.])

            if fD == np.inf or np.isnan(fD) or abs(fD-f) > np.maximum(abs(f),1.)*1.e3:
                alpha=1.e-3*alpha
                fD = self.funct([alpha,0])

            bc = 2*(gab[1]+fB-fA)
            ba = (2./alpha)*((fD-fA)/alpha-gab[0])
            bb = ((fC-fA)/alpha-gab[0])/beta-gab[1]/alpha-0.5*alpha*ba/beta-0.5*beta*bc/alpha
           
            Bab = np.array([[ba,bb],[bb,bc]])
  
            return gab, Bab  

        def gradiente_Hessiano_Hd(self, n, f, f_1, funct,x_glob, grad, d1,d2, alpha, beta,n_iter_glob,d1_norm,d2_norm):
            eta=1.e-6
            ab = np.zeros(2)
            gab= np.zeros(2)
            gab[0]=-d1_norm*d1_norm
            gab[1]=-np.dot(d1.T,d2)
            
            if np.isnan(d1_norm):
                min_sample_val  = 1.e-16
            else:
                min_sample_val =  1.e-3/min(np.maximum(d1_norm,1.e-16),1.e+16)
            if np.abs(alpha) <= min_sample_val:
                alpha = min_sample_val if alpha >= 0 else -min_sample_val 
                                      
            if n_iter_glob == 0:
                f1=funct([alpha,0.])
                ba = (2./alpha)*((f1-f)/alpha-gab[0])
                bc = 0.
                bb = 0.
                Bab = np.array([[ba, bb], [bb, bc]])
                return gab,Bab
                
            n_glob=np.size(d1)
            g_p= np.zeros(n_glob)
            g_p = grad(x_glob+eta*(-d1/d1_norm))
            ba = np.dot((-d1).T,(g_p+d1)/(eta/d1_norm))
            if np.isnan(d2_norm) or d2_norm==0.:
                bb = 0.
                bc = 0.
            else:
                bb = - np.dot((d2).T,(g_p+d1)/(eta/d1_norm))
                g_p = grad(x_glob+eta*(d2/d2_norm))
                bb =(bb-np.dot((-d1).T,(g_p+d1)/(eta/d2_norm)))/2.
                bc = np.dot((d2).T,((g_p+d1)/eta)*d2_norm)
            Bab = np.array([[ba,bb],[bb,bc]])
            
            return gab, Bab

        def gradiente_Hessiano(self,n,x,f,prec,funct,d1,d2,n_iter,n_iter_glob,d1_norm,d2_norm):

            prec1=prec/min(np.maximum(d1_norm,1.e-16),1.e+16)
            prec2=prec/min(np.maximum(d2_norm,1.e-6),1.e+6)

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
                else:
                    x1=np.copy(x)
                    xm=np.copy(x)
                    x1[0]=x1[0]+prec1
                    xm[0]=xm[0]-prec1			
                    f1=funct(x1)
                    fm=funct(xm)
                    gapp[0]=(f1-fm)/(2.*prec1)
                gapp[1]=0.0
                ba = (2./prec1)*((f1-f)/prec1-gapp[0])
                bc = 0.
                bb = 0.
                HessApp = np.array([[ba, bb], [bb, bc]])
                return gapp,HessApp
            if (n_iter==1):
                gapp[0]=-d1_norm*d1_norm
                gapp[1]=-np.dot(d1.T,d2)
                x1=np.copy(x)
                x1[0]=x1[0]+prec1			
                f1=funct(x1)
                x1=np.copy(x)
                x1[1]=x1[1]+prec2		
                f2=funct(x1)
            else:
                x1=np.copy(x)
                xm=np.copy(x)
                x1[0]=x1[0]+prec1
                xm[0]=xm[0]-prec1			
                f1=funct(x1)
                fm=funct(xm)
                gapp[0]=(f1-fm)/(2.*prec1)
                x1=np.copy(x)
                xm=np.copy(x)
                x1[1]=x1[1]+prec2
                xm[1]=xm[1]-prec2			
                f2=funct(x1)
                fm=funct(xm)
                gapp[1]=(f2-fm)/(2.*prec2)
            x1=np.copy(x)
            x1[0]=x1[0]+prec1
            x1[1]=x1[1]+prec2
            f3=funct(x1)
            ba = (2./prec1)*((f1-f)/prec1-gapp[0])
            bc = (2./prec2)*((f2-f)/prec2-gapp[1])
            bb = (1./prec1)*((f3-f)/prec2)-(1./prec2)*((f1-f)/prec1)-(1./prec1)*((f2-f)/prec2)

            HessApp = np.array([[ba, bb], [bb, bc]])

            return gapp,HessApp #,f_min,x_min		
		
        def perturb_Hessian(self,eps_H,H):
            HP=np.zeros(shape=(2,2))
            HP=np.copy(H)
            delta1=eps_H+abs(H[0,1])-H[0,0]
            delta2=eps_H+abs(H[0,1])-H[1,1]
            HP[0,0]=H[0,0]+delta1
            HP[1,1]=H[1,1]+delta2
            return HP
            
        def perturb_Cholesky(self,g,H,norma_g,eps_H,d1_norm,d2_norm):
            #print(' Cholesky')
            #input()
            HT=np.zeros(shape=(2,2))
            
            HT[0,0]=H[0,0]/(d1_norm*d1_norm)
            HT[1,0]=H[1,0]/(d1_norm*d2_norm)
            HT[1,1]=H[1,1]/(d2_norm*d2_norm)
            
            try:
               d=np.zeros(2)
         
               if HT[0,0] > 1.e-12*min(norma_g**2,1.e0):
                  l11=np.sqrt(HT[0,0])
               else:
           	      l11=eps_H

               l21=HT[1,0]/l11
            
               if HT[1,1]-l21*l21 > 1.e-12*min(norma_g**2,1.e0):
                  l22=np.sqrt(HT[1,1]-l21*l21)
               else:
                  l22=eps_H
                  
               lt11=l11*d1_norm
               lt21=l21*sqrt(d1_norm*d2_norm)
               lt22=l22*d2_norm
               y1=-g[0]/lt11
               y2=-(g[1]+lt21*y1)/lt22
            
               d[1]=y2/lt22
               d[0]=(y1-lt21*d[1])/lt11           
            except:
               print('except Cholesky')
               d=-g/norma_g
               #print(' d =',d)
            return d
            
  
        def linesearch_armijo(self,n,x,f,d,gd,gamma,nf,funct):
    
            alfa=1.
	
            y=np.zeros(n)
    
            y=x+alfa*d
            f_alfa=funct(y)
            f_min=f_alfa
            alfa_min=alfa
            nf=nf+1
            while(f_alfa >(f+alfa*gamma*gd) or np.isnan(f_alfa) ):
                if (f_alfa-(f+alfa*gamma*gd)  > 1.e6):
                    alfa=alfa/10.
                else:
                    alfa=alfa/2.
                if (alfa == 0.0):
                   break
                y=x+alfa*d
                f_alfa=funct(y)
                nf=nf+1
            return alfa,f_alfa,nf
            
        def linesearch_armijo_nc(self,n,x,f,d,gd,gamma,nf,funct):
            alfa=1.
	
            y=np.zeros(n)
            y=x+alfa*d
            f_alfa=funct(y)
            f_min=f_alfa
            alfa_min=alfa
            nf=nf+1
   
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
                    nf=nf+1
            alfa=alfa_min
            f_alfa=f_min 
            return alfa,f_alfa,nf            

        #def direction(self,g,H,norma_g,n,x,n_iter_glob,var,d2,d1_norm):
        def direction(self,g,H,norma_g,n,x,n_iter_glob,var,d1_norm,d2_norm):
            eps_H=10**-6
            iprintdir=True
            iprintdir=False
            d=np.zeros(n)
            if (H[1,1]==0) and (H[0,1]==0):
                if iprintdir:
                    print(' h(1,1)=h(0,1)=0')
                d[0]=-g[0]/min(np.maximum(abs(H[0,0]),1.e-9),1.e9)
                d[1]=0.
                gd=np.dot(g.T,d)
                return d, gd
            #H = np.reshape(H,(2,2))
            #print(H,H.shape)
            #print(g)
            try:
                d= np.linalg.solve(H,-g)
            except:
                if iprintdir:
                    print("An exception occurred")    
                d = np.linalg.lstsq(H, -g, rcond=None)[0]

            gd=np.dot(g.T,d)
                      
            if (gd > 0 ):
                d=-d
                gd=-gd
                if iprintdir:
                    print("Newton non di discesa")
           
            norma_d=np.sqrt(np.dot(d.T,d))

            if((gd>-(1.e-12*np.minimum(norma_g,1.e0))*(1.e-12*np.minimum(norma_g,1.e0))) or (norma_d>10**24*norma_g)):

                #if var >= 2 and (var <=10):
                    #d2_norm=np.linalg.norm(d2,2)	
                d = self.perturb_Cholesky(g,H,norma_g,eps_H,d1_norm,d2_norm)

                gd=np.dot(g.T,d)

                if iprintdir:
                    print("Newton non gr.related")
            return d, gd
			
        def minimize(self):
            n = self.n
            f0=self.f0
            f_1=self.f_1
            g_1=self.g_1
            x = self.x
            x_1=self.x_1
            eps = self.eps
            maxiter = self.maxiter
            iprint = self.iprint
            funct= self.funct
            grad=self.grad
            x_glob=self.x_glob
            d1=self.d1
            d2=self.d2
            n_iter_glob=self.n_iter_glob
            var=self.var

            d1_norm=np.linalg.norm(d1,2)
            d2_norm=0
            
            f_min=1.e36
            #iprint=True
            iprint=False
			
            gamma=1.e-6
            prec=1.e-3
            eps_H=1.e-6

            f=f0
            nf=0

            d=np.zeros(n)
            n_iter=0

            while True:
                n_iter=n_iter+1
                if np.isnan(f):
                    if iprint:
                        print("         ")
                        print("problema mal definito=",nf)
                        print("         ",file=file_10)
                        print("problema mal definito nf=",nf,file=file_10)
                        for i in range(0,n):
                            print  ("x(",i+1,") =",x[i])
                            print("         ")
                    break
                if(f == -np.inf ):
                    if iprint:
                        print("         ")
                        print("problema illimitato inferiormente  nf=",nf)
                        print("         ",file=file_10)
                        print("problema illimitato inferiormente  nf=",nf,file=file_10)
                        for i in range(0,n):
                            print  ("x(",i+1,") =",x[i])
                            print("         ")
                    break
                if(n_iter > maxiter):
                    if iprint:
                        print("         ")
                        print("Algoritmo terminato per massimo numero di iterazioni  nf=",nf)
                        print("         ",file=file_10)
                        print  ("fin=",n_iter,"  nf",nf,"  f=",f,"  norma_grad=",norma_g)
                        print('iterazion globali =',n_iter_glob)
                        for i in range(0,n):
                            print  ("x(",i+1,") =",x[i])
                            print("         ")
                    break
               
                if (var == 0):
                   d2_norm=np.linalg.norm(d2,2) 
                   g,H=self.gradiente_Hessiano(n,x,f,prec,funct,d1,d2,n_iter,n_iter_glob,d1_norm,d2_norm)
                   
                if (var == 1): 
                   d2_norm=np.linalg.norm(d2,2) 
                   g,H=self.gradiente_Hessiano_M( n, f, f_1, funct, d1,d2, x_1[0], x_1[1],n_iter_glob,d1_norm,d2_norm)
                if (var == 11): 
                   d2_norm=np.linalg.norm(d2,2) 
                   #print('did2: ',d1,d2)
                   g,H=self.gradiente_Hessiano_Hd( n, f, f_1, funct,x_glob, grad,d1,d2, x_1[0], x_1[1],n_iter_glob,d1_norm,d2_norm)               
                if (var == 2):
                   g,H=self.gradiente_Hessiano_B( n, f, f_1,g_1,funct, d1,d2, x_1[0], x_1[1],n_iter_glob,d1_norm)
                   
                if (var >= 3) and (var <=10):
                   g,H=self.gradiente_Hessiano_D( n, f, f_1, x, x_1, g_1, funct, d1,d2, x_1[0], x_1[1],n_iter_glob,var,d1_norm)
                 
                norma_g = np.linalg.norm(g,2)
                file_10 = open("risulati.txt","w") 
                if iprint:
                   print('grad=',g)
                   print  ("n_iter=",n_iter,"  nf=",nf,"  f=",f,"  norma_grad=",norma_g)
                   print  ("n_iter=",n_iter,"  nf=",nf,"  f=",f,"  norma_grad=",norma_g,file=file_10)
                   #input()
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
                        input()
                    break				
 
                if (var == 2):                
                    d=-np.dot(H,g)
                    gd=np.dot(g.T,d)
                else:
                    if var >= 2 and (var <=10):
                        d2_norm=np.linalg.norm(d2,2)
                    d,gd = self.direction(g,H,norma_g,n,x,n_iter_glob,var,d1_norm,d2_norm)
                    #if var >= 3:
                        #d,gd = self.direction(g,H,norma_g,n,x,n_iter_glob,var,d2,d1_norm)
                    #else:
                        #d,gd = self.direction(g,H,norma_g,n,x,n_iter_glob,var,d2,d2_norm)
                    
                alfa,f_alfa,nf=self.linesearch_armijo(n,x,f,d,gd,gamma,nf,funct)

                x=x+alfa*d
    
                f=f_alfa
                
            return f,x
