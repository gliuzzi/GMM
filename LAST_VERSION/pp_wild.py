import numpy as np
import matplotlib.pyplot as plt

def plot_performance_profiles_wild(H,labels,colors,title):
    npp, ns = H.shape # Grab the dimensions

    # Compute ratios and divide by smallest element in each row.
    T = np.copy(H)
    T = T[~np.isinf(T)[:, 0], :]
    npp, ns = T.shape

    T[np.where(T >= 1.e+9)] = np.nan
    best = np.nanmin(T,axis=1)
    worst= np.nanmax(T,axis=1)
    bestM = np.tile(np.reshape(best,(npp,1)),(1,ns))
    r = T/bestM

    max_ratio = np.max(r[~np.isnan(r)])
    r[np.isnan(r)] = 2 * max_ratio
    r = np.sort(r,axis=0)

    plt.figure(figsize=(4.68, 4.68), dpi=300)
    plt.title("{}".format(title))

    edges = [(i+1)/npp for i in range(npp)]
    edges = edges[1:]
    for s in range(ns):
        lgd = labels[s]
        clr = colors[s] #(1/(s+1),1-1/(s+1),0.5)
        lst = '-'

        # aggiunta la riga di sotto per fare si che il plot non torni a 0
        r[-1, s] = 2 * max_ratio
        plt.stairs(edges,r[:,s],linestyle=lst,label=lgd,color=clr)

    # Axis properties are set so that failures are not shown, but with the
    # max_ratio data points shown. This highlights the "flatline" effect.
    plt.ylabel("Cumulative")
    plt.xlabel(r'$\tau$')
    if max_ratio < 11:
        plt.xlim((1, 1.1*max_ratio))
    else:
        plt.xlim((1, 11))
    plt.ylim((0, 1))
    plt.legend(loc='lower right',ncol=1)
    plt.show()
    plt.close()
