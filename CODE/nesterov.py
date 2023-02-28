
"""
Created on Tue Febr 17 13:10:39 2022

@author: Stefa
"""
import numpy as np
#from sys import exit
#from linesearch import linesearch    
#from problem import problem


class nesterov:
    def __init__(self,funct,grad,n,tol,maxiter):
        self.funct   = funct
        self.grad    = grad
        self.n       = n
        self.tol     = tol
        self.maxiter = maxiter
        
    #derivata=True
    #name,dim,init_point, function,gradient =problem(i,derivata)
    #n = dim()

    def run(self,init_point):
        #file_10=open('risultati.txt',"w")
        i=1
        n_iter=0
        ng=0
        n = self.n
        z=np.zeros(n)
        xp=np.copy(init_point)
        y=xp.copy()
        fy=self.funct(y) 
        nf=1
        gy=self.grad(y)
        gz=self.grad(z)
        ng=ng+2
        norma_g=np.sqrt(np.dot(gy.T,gy))

        print  ("n_iter=",n_iter,"  nf",nf,"  f=",fy,"  norma_grad=",norma_g)
        #print  ("n_iter=",n_iter,"  nf",nf,"  f=",fy,"  norma_grad=",norma_g,file=file_10)

        #input()

        if( np.sqrt(np.dot((gy-gz).T,gy-gz)) <= 10**-12):
            print  ("gy = gz")
         #   print  ("gy = gz",file=file_10)
         #   exit("gy = gz")
            return xp, {"iters": 0, "f": fy, "g_norm": norma_g, "nfails": np.nan, "nnegeig": np.nan, "cosmax": np.nan}
            
        alfap=np.sqrt(np.dot((y-z).T,y-z))/np.sqrt(np.dot((gy-gz).T,gy-gz))
        #alfap=10**-3

        a=1

        gamma=10**-6

        eps = self.tol
        maxiter=self.maxiter

        d=np.zeros(n)


        while True:
            
            d=-gy.copy()
            gd=np.dot(gy.T,d)
            
            alfa,f_alfa,nf=self.linesearch(n,y,fy,d,gd,alfap,nf)
            
            x=y+alfa*d
            
            f=f_alfa  
            
            n_iter=n_iter+1
            g=self.grad(x)
            ng=ng+1
            norma_g=np.sqrt(np.dot(g.T,g))
            
            #print  ("n_iter=",n_iter,"  nf",nf,"  f=",f,"  norma_grad=",norma_g)
        #    print  ("n_iter=",n_iter,"  nf",nf,"  f=",f,"  norma_grad=",norma_g,file=file_10)
        #    input()
            if(norma_g <= eps):
                #print("         ")
                #print("Algoritmo terminato con un punto stazionario  norma del gradiente =",norma_g)
        #        print("         ",file=file_10)
        #        print("Algoritmo terminato con un punto stazionario  norma del gradiente =",norma_g,file=file_10)
                break

            if(n_iter >= maxiter):
                #print("         ")
                #print("Algoritmo terminato per massimo numero di iterazioni")
        #        print("         ",file=file_10)
        #        print("Algoritmo terminato per massimo numero di iterazioni",file=file_10)
                break
            
        #    input()
            an=(1.+np.sqrt(1.+4.*a*a))/2.
            y=x+(a-1.)*(x-xp)/an
            a=an
            xp=x
            alfap=alfa
            fy=self.funct(y) 
            nf=nf+1
            gy=self.grad(y)
            ng=ng+1

        return x, {"iters": n_iter, "f": f, "g_norm": norma_g, "nfails": np.nan, "nnegeig": np.nan, "cosmax": np.nan}
        
        
    def linesearch(self,n,y,fy,d,gd,alfap,nf):
         
        yy=np.zeros(n)
        alfa=alfap
        yy=y+alfa*d
        f_alfa=self.funct(yy)
        nf=nf+1
        
        while(f_alfa >(fy+(alfa/2.)*gd) ):
            alfa=alfa/2.
            yy=y+alfa*d
            f_alfa=self.funct(yy)
            nf=nf+1
    #        print("     primo    ",gd,fy,f_alfa,alfa)
    #        input()
            

        
        if False :
           
    #    while(f_2alfa <=(fy+2*alfa*gamma*gd) ):
            yy=y+2.*alfa*d
            f_2alfa=self.funct(yy)
            nf=nf+1  
            while(f_2alfa <=(fy+2.*(alfa/2.)*gd) ):    
                alfa=2.*alfa
                f_alfa=f_2alfa
                yy=y+2*alfa*d
                f_2alfa=self.funct(yy)
                nf=nf+1
    #            print("     secondo    ")

        return alfa,f_alfa,nf

        
