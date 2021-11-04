import numpy as np

def idxFromTo(Array, index_from, index_to):
    # includes from and to indexes
    return Array[index_from:index_to+1]

def removeEnds(Array, remove_from_start, remove_from_end):
    return Array[remove_from_start:len(Array)-remove_from_end]



# #TEST HARNESS
A = np.array([1,2,3,4,5,6,7,8,9])
B = idxFromTo(A,1,4)
C = removeEnds(A,1,3)

print(B)

print(C)
