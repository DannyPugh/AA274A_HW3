import numpy as np

def idxFromTo(Array, index_from, index_to):
    return Array[index_from:index_to+1]

def removeEnds(Array, remove_from_start, remove_from_end):
    return Array[remove_from_start:len(Array)-remove_from_end]

def removeIdx(Array,index):
    return np.delete(Array,index)

def replaceIdx(Array,index,replacement):
    Array[index] = replacement
    return Array

def getEndIdx(Array):
    return len(Array)-1

# def idxAfterAdd(current_number_of_entries, number_of_entries_to_add):


# # # #TEST HARNESS
# A = np.array([1,2,3,4,5,6,7,8,9])
# B = idxFromTo(A,1,4)
# C = removeEnds(A,1,3)

# print(B)

# print(C)

# print(idxFromTo(A,8,8))

# print(replaceIdx(A,4,324342))
# print(removeIdx(A,0))

# print(getEndIdx(A))
