import numpy as np
from numpy.lib.function_base import append
from PlotFunctions import *

#F = np.array([[0,0,0],[0,1,0],[0,0,0]])
#F = np.array([[0,0,0],[0,0,1],[0,0,0]])
#F = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
F = (1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]])

print(F)
k, l = np.shape(F)
print(k)
print(l)

I = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(I)

i, j = np.shape(I)
print(i)
print(j)

I_pad = np.pad(I, [[int(np.floor(k/2)),int(np.floor(k/2))],[int(np.floor(l/2)),int(np.floor(l/2))]])
print(I_pad)

G = np.zeros(0)
print(G)
for row in range(i):
    for col in range(j):
        g = 0
        for u in range(k):
            for v in range(l):
                g += F[u,v]*I_pad[u+row,v+col]
        print(g)
        G = np.append(G,g)
G = np.reshape(G,(i,j))
print(G)        



