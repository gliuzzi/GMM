import numpy as np
import pandas as pd
from pp_wild import plot_performance_profiles_wild

data = pd.read_csv('risultati.txt', sep="|", header=None)


print(data.shape)
print(data.iloc[1][1],data.iloc[1][2])
nrows,ncols = data.shape

# get solver names
S = []
for i in range(nrows):
    name = data.iloc[i][1]
    name = name.strip()
    #print(name)
    if name == '--':
        break
    S.append(name)

print(S)

# get problem names
P = []
lastp = ''
for i in range(nrows):
    name = data.iloc[i][2]
    name = name.strip()
    if name == lastp:
        continue

    if name == '--':
        continue

    #print(name)
    P.append(name)
    lastp = name

print(P)
npp = len(P)
nss = len(S)
print(npp,nss)

Htime = np.zeros((npp,nss));
Hiter = np.zeros((npp,nss));
Hfval = np.zeros((npp,nss));
Hgrad = np.zeros((npp,nss));

#print(P.index('WOODS'))
for row in range(nrows):
    solver = (data.iloc[row][1]).strip()
    problem = (data.iloc[row][2]).strip()
    if solver == '--':
        continue

    iip = P.index(problem)
    iis = S.index(solver)
    Hfval[iip,iis] = data.iloc[row][6]
    Hgrad[iip,iis] = data.iloc[row][7]
    if Hgrad[iip,iis] <= 1.e-3:
        Htime[iip,iis] = data.iloc[row][4]
        Hiter[iip,iis] = data.iloc[row][5]
    else:
        Htime[iip,iis] = np.nan
        Hiter[iip,iis] = np.nan

Istaz = []
I = []
for iip in range(npp):
    if max(Hfval[iip,:]) - min(Hfval[iip,:]) < 1.e-3:
        I.append(iip)

CS = [
    [0, 0.5, 1], #GMM1
    [0, 0.5, 0.5], #GMM3
    [0, 0, 1], #GMM2
    [1, 0, 0], #L-BFGS
    [0.5, 1, 0] #CG
    ]

compare = [[0,1,2,4],[0,1,2,3],[2,3]]

for confronti in compare:
    print(confronti)
    pp = confronti
    pair = pp
    lgd = np.array(S)
    clr = np.array(CS)
    plot_performance_profiles_wild(Htime[:,pair],lgd[pair],clr[pair],'Time')
    plot_performance_profiles_wild(Hiter[:,pair],lgd[pair],clr[pair],'Iter')

    I = []
    nbest = np.zeros((1,len(pair)))
    for ip in range(npp):
        bestf = min(Hfval[ip,pair])
        worsf = max(Hfval[ip,pair])

        if worsf - bestf < 1.e-3:
            I.append(ip)
        else:
            v = min(Hfval[ip,pair])
            for ii in range(len(pair)):
                if abs(v-Hfval[ip,pair[ii]]) < 1.e-3:
                    nbest[0,ii] = nbest[0,ii]+1

    print(lgd[pair])
    plot_performance_profiles_wild(Htime[I,:][:,pair],lgd[pair],clr[pair],'Time (eq)')
    plot_performance_profiles_wild(Hiter[I,:][:,pair],lgd[pair],clr[pair],'Iter (eq)')

#    nu = len(I)
#    for p in range(len(pair)):
        #print("%20s wins on %3d/%3d\n",SS{pair(p)},nbest(1,p),np)




