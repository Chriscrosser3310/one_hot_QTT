import numpy as np
from one_hot_basis import *
from scipy.linalg import block_diag
from functools import reduce

I = np.array([[1, 0], [0, 1]])
Q = np.array([[0, 0], [1, 0]])
P1 = np.array([[0, 0], [0, 1]])

def proj_to_subspace(H, basis):
    l = []
    for b in basis:
        l.append(int(b, 2))
    return H[np.ix_(l, l)]

def laplacian_2n(n, q):
    H = -2 * np.identity(2**(q*n))

    for l in range(n):
        for i in range(q-1):
            term = [I] * n * q
            for lp in range(l):
                term[lp*q + 0] = Q.T
                term[lp*q + q-1] = Q
            term[l*q + i+1] = Q.T
            term[l*q + i] = Q
            term_mat = reduce(np.kron, term[::-1])
            H += term_mat
            H += term_mat.T
    
    return H

def g_end(q, l, i):
    g_first = [0]*l
    if q % 2 == 0:
        g_last = [0]*(l-1) + [q-1]
    else:
        g_last = [q-1]*l
    if i % 2 == 0:
        return g_last
    else:
        return g_first

def laplacian_np1(n, q):
    H = -2 * np.identity(2**(q*n))

    for l in range(n):
        for i in range(q-1):
            term = [I] * n * q
            for lp in range(l):
                term[lp*q + g_end(q,l,i)[lp]] = P1
            term[l*q + i+1] = Q.T
            term[l*q + i] = Q
            term_mat = reduce(np.kron, term[::-1])
            H += term_mat
            H += term_mat.T
    
    return H

def shift_np2(n, q, p):

    H = np.zeros((2**(q*n), 2**(q*n)))
    for i in range(q-p):
        term = [I] * n * q
        term[i+p] = Q.T
        term[i] = Q
        term_mat = reduce(np.kron, term[::-1])
        H += term_mat
        H += term_mat.T
    
    for l in range(1, n):
        for i in range(q-1):
            for k in range(p):
                term = [I] * n * q
                if g_end(q, l, i)[0] == q-1:
                    if q-p+k == q-k-1:
                        term[q-p+k] = P1
                    else:
                        term[q-k-1] = Q.T
                        term[q-p+k] = Q
                elif g_end(q, l, i)[0] == 0:
                    print((k, p-1-k))
                    if k == p-1-k:
                        term[k] = P1
                    else:
                        term[k] = Q.T
                        term[p-1-k] = Q
                else:
                    raise NotImplementedError
                for lp in range(1, l):
                    term[lp*q + g_end(q,l,i)[lp]] = P1
                term[l*q + i+1] = Q.T
                term[l*q + i] = Q
                term_mat = reduce(np.kron, term[::-1])
                H += term_mat
                H += term_mat.T
    return H

def toeplitz_np2(n, q, a):
    w = len(a)-1
    H = a[0] * np.identity(2**(q*n))
    for p in range(1, w+1):
        H += a[p] * shift_np2(n, q, p)
    return H

def reversal_matrix(n, q, dtype=int):
    N = q**n 
    R = np.zeros((N, N), dtype=dtype)
    R[np.arange(N), N - 1 - np.arange(N)] = 1
    return R

def gray_code_permutation(n, q, dtype=int):
    if n == 1:
        return np.identity(q, dtype=dtype)
    P = gray_code_permutation(n-1, q, dtype=dtype)
    R = reversal_matrix(n-1, q, dtype=dtype)
    RP = R @ P
    P_list = [RP if i % 2 == 1 else P for i in range(q)]
    return block_diag(*P_list)

if __name__ == "__main__":

    np.set_printoptions(linewidth=300)

    '''
    # Test permutation
    n = 3
    q = 10
    P = gray_code_permutation(n, q)
    perm = P @ np.array(range(q**n), dtype=int)

    basis = []
    for i in range(q**n):
        basis.append(ith_lex_onehot(n, q, i, False))

    for p in perm:
        print(basis[p])
    '''

    '''
    # Test 2n Laplacian
    n = 3
    q = 3
    basis = [ith_lex_onehot(n, q, i, False) for i in range(q**n)]
    print(basis)
    print(proj_to_subspace(laplacian_2n(n, q), basis))
    '''

    # Test (n+1) Laplacian
    n = 3
    q = 2
    basis = [ith_gray_onehot(n, q, i, False) for i in range(q**n)]
    print(basis)
    print(proj_to_subspace(laplacian_np1(n, q), basis))
   
    '''
    # Test (n+2) shift
    n = 2
    q = 5
    p = 3
    basis = [ith_gray_onehot(n, q, i, False) for i in range(q**n)]
    print(basis)
    print(proj_to_subspace(shift_np2(n, q, p), basis))
    '''

    '''
    # Test (n+2) toeplitz
    n = 2
    q = 5
    a = [1, 2, 3, 4, 5, 6]
    basis = [ith_gray_onehot(n, q, i, False) for i in range(q**n)]
    print(basis)
    print(proj_to_subspace(toeplitz_np2(n, q, a), basis))
    '''