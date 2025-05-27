import numpy as np

# ---------- Vectors ----------
def vector_add(v, w):
    '''
    Vector addition work like this 
    (2,3) + (1,1)
    vi_01 = 2  wi_01 = 1
    vi_02 = 3 , wi_02 = 1
    (vi_01,vi_02) + (wi_01,wi_02)
    (vi_01 + wi_01,vi_02 + wi_02)
    '''
    # print("this is how multiplication works")
    # for vi , wi in zip(v,w):
    #     print(vi,"this is vi")
    #     print(wi,"this is wi")

    # Similar thing can be done by List Comprehension
    
    return [vi + wi for vi, wi in zip(v, w)]

def scalar_multiply(c, v):
    '''
    Multiply the scalar withe each value.
    ''' 

    return [c * vi for vi in v]

def dot(v, w):
    print("this is ",v,"and this w ",w)
    for st,end in zip(v,w):
        print(st,'this is a')
        print(end,'this is b')
    return sum(vi * wi for vi, wi in zip(v, w))

def vector_norm(v):
    '''
    '''
    return np.sqrt(dot(v, v))

# ---------- Matrices ----------
def shape(matrix):
    return len(matrix), len(matrix[0])

def get_row(matrix, i):
    return matrix[i]

def get_col(matrix, j):
    return [row[j] for row in matrix]

def mat_add(A, B):
    return [[a_ij + b_ij for a_ij, b_ij in zip(row_a, row_b)] 
            for row_a, row_b in zip(A, B)]

def mat_multiply(A, B):
    B_T = list(zip(*B))
    return [[dot(row, col) for col in B_T] for row in A]

# ---------- Determinant & Inverse ----------
def determinant_2x2(matrix):
    assert shape(matrix) == (2, 2)
    return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

def inverse_2x2(matrix):
    det = determinant_2x2(matrix)
    assert det != 0, "Matrix not invertible"
    a, b = matrix[0]
    c, d = matrix[1]
    return [[d/det, -b/det], [-c/det, a/det]]

# ---------- Eigenvalues & Eigenvectors (via numpy) ----------
def eigen_decomposition(matrix):
    vals, vecs = np.linalg.eig(np.array(matrix))
    return vals, vecs

if __name__ == "__main__":
    vector_01 = (2,3)
    vector_02 = (1,1)
    product = vector_add(vector_01,vector_02)
    print(product)
    print(vector_norm((3,4)))
