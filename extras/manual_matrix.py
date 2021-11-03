import numpy as np
from numpy.lib.function_base import append

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


A = np.array([1,2,1])

B = np.vstack(A)

print(A)
print(B)


print(A*np.vstack(A))
print(A*np.vstack(A)*A)

C = np.array([1,2,3])
D = np.array([4,5,6])

print(C)
print(np.vstack(D))

print(C*np.vstack(D))

print('################################')


filt4 =           np.array([[1,  4,  7,  4, 1],
                            [4, 16, 26, 16, 4],
                            [7, 26, 41, 26, 7],
                            [4, 16, 26, 16, 4],
                            [1,  4,  7,  4, 1]])


filt1 = np.zeros((3, 3))
filt1[1, 1] = 1

# _, single_value ,_ = np.linalg.svd(filt4)
print(filt1)
print(np.linalg.matrix_rank(filt1))
u, s, vh = np.linalg.svd(filt1)
np.dot(u[:, :3] * s, vh)
print(single_value)
print(single_value.flatten()*single_value)





##########################
