import numpy as np
from one_hot_basis import *
from scipy.linalg import block_diag
from functools import reduce

# === utilities ===

I = np.array([[1, 0], [0, 1]])
Q = np.array([[0, 0], [1, 0]])
P1 = np.array([[0, 0], [0, 1]])

def proj_i(i: int, q: int, dtype=float) -> np.ndarray:
    """
    Return the q x q projector |i><i|.
    """
    if not (0 <= i < q):
        raise ValueError("i must satisfy 0 <= i < q")
    P = np.zeros((q, q), dtype=dtype)
    P[i, i] = 1
    return P

def sym_outer_ij(i: int, j: int, q: int, dtype=float) -> np.ndarray:
    """
    Return the q x q matrix |i><j| + |j><i|.
    """
    if not (0 <= i < q and 0 <= j < q):
        raise ValueError("i, j must satisfy 0 <= i, j < q")
    if i == j:
        raise ValueError("i and j must be different for |i><j| + |j><i|")

    M = np.zeros((q, q), dtype=dtype)
    M[i, j] = 1
    M[j, i] = 1
    return M

def proj_to_subspace(H, basis, base):
    l = []
    for b in basis:
        l.append(int(b, base=base))
    return H[np.ix_(l, l)]

def kron(mat_list):
    return reduce(np.kron, mat_list)

def is_banded_toeplitz(A, w: int, tol: float = 1e-12) -> bool:
    """
    Check whether A is a banded Toeplitz matrix with half-bandwidth w.

    Conditions:
      1) Toeplitz: A[i,j] depends only on (i-j) -> constant along diagonals
      2) Banded:  A[i,j] = 0 for |i-j| > w

    Parameters
    ----------
    A : array-like, shape (n, n)
    w : int
        Half-bandwidth (w=0 means diagonal only, w=1 means tri-diagonal, etc.)
    tol : float
        Absolute tolerance for floating comparisons

    Returns
    -------
    bool
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    n = A.shape[0]
    if w < 0:
        return False

    # 1) Band check: entries outside the band must be ~0
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    outside = np.abs(i - j) > w
    if np.any(np.abs(A[outside]) > tol):
        return False

    # 2) Toeplitz check: each diagonal must be constant (within tol)
    for k in range(-(n - 1), n):  # diagonal offset (i-j)=k
        diag = np.diagonal(A, offset=-k)  # note: np.diagonal uses offset=j-i
        # If diag has length 0 or 1 it's trivially constant
        if diag.size > 1:
            if np.max(np.abs(diag - diag[0])) > tol:
                return False

    return True

# === basis ===

# Basis for exponentially compressed Laplacian
def laplacian_exp_comp_basis(n, q):
    if n == 1:
        return [str(i) for i in range(q)]
    new_basis = ["0"*n]
    prev_basis = laplacian_exp_comp_basis(n-1, q)
    rev_prev_basis = prev_basis[::-1]
    for i in range(1, q-1):
        if i % 2 == 1:
            new_basis += [str(i)+s for s in prev_basis]
        else:
            new_basis += [str(i)+s for s in rev_prev_basis]
    if q % 2 == 0:
        new_basis += [str(q-1) + "0"*(n-1)]
    else:
        new_basis += [str(q-1) * n]
    return new_basis

# === Hamitlonians ===

# laplacian 2n-local
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
            term_mat = kron(term[::-1])
            H += term_mat
            H += term_mat.T
    
    return H

# endpoint configuration of gray-code
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

# laplacian n+1 local
def laplacian_np1(n, q):
    H = -2 * np.identity(2**(q*n))

    for l in range(n):
        for i in range(q-1):
            term = [I] * n * q
            for lp in range(l):
                term[lp*q + g_end(q,l,i)[lp]] = P1
            term[l*q + i+1] = Q.T
            term[l*q + i] = Q
            term_mat = kron(term[::-1])
            H += term_mat
            H += term_mat.T
    
    return H

# shift operator n+2 local
def shift_np2(n, q, p):

    H = np.zeros((2**(q*n), 2**(q*n)))
    for i in range(q-p):
        term = [I] * n * q
        term[i+p] = Q.T
        term[i] = Q
        term_mat = kron(term[::-1])
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

# Toeplitz matrix n+2 local
def toeplitz_np2(n, q, a):
    w = len(a)-1
    H = a[0] * np.identity(2**(q*n))
    for p in range(1, w+1):
        H += a[p] * shift_np2(n, q, p)
    return H

# reversal permutation
def reversal_matrix(n, q, dtype=int):
    N = q**n 
    R = np.zeros((N, N), dtype=dtype)
    R[np.arange(N), N - 1 - np.arange(N)] = 1
    return R

# gray code permutation
def gray_code_permutation(n, q, dtype=int):
    if n == 1:
        return np.identity(q, dtype=dtype)
    P = gray_code_permutation(n-1, q, dtype=dtype)
    R = reversal_matrix(n-1, q, dtype=dtype)
    RP = R @ P
    P_list = [RP if i % 2 == 1 else P for i in range(q)]
    return block_diag(*P_list)

# exponential compression of Laplacian matrix
def laplacian_exp(n, q):
    if n == 1:
        L = np.zeros((q, q))
        np.fill_diagonal(L, 0)
        if q > 1:
            idx = np.arange(q - 1)
            L[idx, idx + 1] = 1
            L[idx + 1, idx] = 1
        return L
    Hp = laplacian_exp(n-1, q)
    H = kron([np.identity(q), Hp])
    
    boundary_1 = [-proj_i(0, q), sym_outer_ij(0, 1, q)] + [np.identity(q)] * (n - 2)
    if q % 2 == 0:
        boundary_2 = [-proj_i(q-1, q), sym_outer_ij(0, 1, q)] + [np.identity(q)] * (n - 2)
    else:
        boundary_2 = [-proj_i(q-1, q), sym_outer_ij(q-1, q-2, q)] + [np.identity(q)] * (n - 2)
    
    H += kron(boundary_1)
    H += kron(boundary_2)
    
    for i in range(0, q-1):
        if i % 2 == 0:
            term = [sym_outer_ij(i, i+1, q), proj_i(0, q)] + [np.identity(q)] * (n - 2)
        else:
            term = [sym_outer_ij(i, i+1, q), proj_i(q-1, q)] + [np.identity(q)] * (n - 2)
        H += kron(term)
    
    return H
    
    

if __name__ == "__main__":

    np.set_printoptions(linewidth=np.inf)

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

    '''
    # Test (n+1) Laplacian
    n = 3
    q = 2
    basis = [ith_gray_onehot(n, q, i, False) for i in range(q**n)]
    print(basis)
    print(proj_to_subspace(laplacian_np1(n, q), basis))
    '''
    
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
    
    # Test exponentially compressed Laplaican
    n = 3
    q = 4
    basis = laplacian_exp_comp_basis(n, q)
    print(basis)
    M = proj_to_subspace(laplacian_exp(n, q), basis, q)
    print(M)
    print(is_banded_toeplitz(M, 1))