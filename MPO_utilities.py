import quimb as qu
import quimb.tensor as qtn
import numpy as np

def kronMPSs(mps_array):
    num_mps = len(mps_array)
    array = []
    for i, mps in enumerate(mps_array):
        mps = mps.copy(deep=True)
        n = len(mps.sites)
        mps.permute_arrays("lrp")
        datas = [t.data for t in mps]
        if i != 0:
            t0 = datas[0]
            r, p = t0.shape
            datas[0] = np.reshape(t0, (1, r, p))
        if i != num_mps-1:
            tlast = datas[-1]
            l, p = tlast.shape
            datas[-1] = np.reshape(tlast, (l, 1, p))
        array += datas
    return qtn.MatrixProductState(array)

def kronMPOs(mpo_array):
    num_mps = len(mpo_array)
    array = []
    for i, mpo in enumerate(mpo_array):
        mpo = mpo.copy(deep=True)
        mpo.permute_arrays("lrud")
        datas = [t.data for t in mpo]
        if i != 0:
            t0 = datas[0]
            r, u, d = t0.shape
            datas[0] = np.reshape(t0, (1, r, u, d))
        if i != num_mps-1:
            tlast = datas[-1]
            l, u, d = tlast.shape
            datas[-1] = np.reshape(tlast, (l, 1, u, d))
        array += datas
    return qtn.MatrixProductOperator(array)

# insert Identity into middle of mpo
# ""ind 0 will be ind start in mpo1""
def insertIdMPO(mpo, start, L):
    if L == 0:
        return mpo
    mpo = mpo.copy(deep=True)
    mpo.permute_arrays("lrud")
    datas = [t.data for t in mpo]
    mpo_len = len(datas)
    if start == 0:
        return kronMPOs([qtn.MPO_identity(L), mpo])
    elif start == mpo_len:
        return kronMPOs([mpo, qtn.MPO_identity(L)])
    else:   
        dim = mpo[start-1].data.shape[-3]
        A = np.zeros((dim, dim, 2, 2))
        for i in range(dim):
            for j in range(2):
                A[i, i, j, j] = 1.
        datas_id = datas[:start]
        for _ in range(L):
            datas_id.append(A)
        datas_id += datas[start:]
        return qtn.MatrixProductOperator(datas_id)
    

#===============================================
#================== utilities ==================
#===============================================

# ---- compatibility patch for some quimb/autoray versions ----
if not hasattr(qu, "transpose"):
    qu.transpose = np.transpose

def _as_np(x):
    return np.asarray(x, dtype=complex)

def mpo_identity_arrays(L, d=2):
    """Identity MPO with *rank-3 boundaries*: left is (r,u,d), right is (l,u,d)."""
    I = _as_np(np.eye(d))
    arrs = []
    for s in range(L):
        if s == 0:
            arrs.append(I.reshape(1, d, d).copy())      # (Dr=1, u, d) == "rud"
        elif s == L - 1:
            arrs.append(I.reshape(1, d, d).copy())      # (Dl=1, u, d) == "lud"
        else:
            arrs.append(I.reshape(1, 1, d, d).copy())   # (Dl, Dr, u, d) == "lrud"
    return arrs


def mpo_from_arrays(arrs, *, upper_ind_id="k{}", lower_ind_id="b{}"):
    """
    Build MPO from a mix of:
      left boundary:  (Dr, u, d)
      middle:         (Dl, Dr, u, d)
      right boundary: (Dl, u, d)
    with shape='lrud' telling quimb what order these correspond to.
    """
    L = len(arrs)
    arrs = [_as_np(A) for A in arrs]
    return qtn.MatrixProductOperator(
        arrs,
        sites=range(L),
        shape="lrud",
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
    )


def set_site_tensor_with_boundary(arrs, s, W):
    """
    Put a rank-4 site tensor W[Dl,Dr,u,d] into arrs[s], but if s is a boundary,
    drop the missing bond index to make it rank-3 as desired.
    """
    W = _as_np(W)
    if s == 0:
        # want (Dr,u,d) -> drop Dl (assumed 1)
        if W.shape[0] != 1:
            raise ValueError("Left boundary expects Dl=1.")
        arrs[s] = W[0, :, :, :]          # (Dr,u,d)
    elif s == len(arrs) - 1:
        # want (Dl,u,d) -> drop Dr (assumed 1)
        if W.shape[1] != 1:
            raise ValueError("Right boundary expects Dr=1.")
        arrs[s] = W[:, 0, :, :]          # (Dl,u,d)
    else:
        arrs[s] = W                      # (Dl,Dr,u,d)

def mpo_product_unitaries(U_list, *, upper_ind_id="k{}", lower_ind_id="b{}"):
    """
    Build a rank-1 MPO (bond dim = 1) representing ⊗_i U_list[i].

    Parameters
    ----------
    U_list : list of (2,2) arrays
        One-qubit unitaries (or any 2x2 operators) for each site, in site order.
    """
    L = len(U_list)
    arrs = []

    for i, U in enumerate(U_list):
        U = np.asarray(U, dtype=complex)
        if U.shape != (2, 2):
            raise ValueError(f"U_list[{i}] must have shape (2,2), got {U.shape}.")

        if L == 1:
            # single-site MPO can just be rank-2, but we keep boundary convention consistent:
            arrs.append(U.reshape(1, 2, 2))          # "rud" (Dr,u,d)
        elif i == 0:
            arrs.append(U.reshape(1, 2, 2))          # left boundary: (Dr,u,d) == "rud"
        elif i == L - 1:
            arrs.append(U.reshape(1, 2, 2))          # right boundary: (Dl,u,d) == "lud"
        else:
            arrs.append(U.reshape(1, 1, 2, 2))       # middle: (Dl,Dr,u,d) == "lrud"

    return qtn.MatrixProductOperator(
        [np.asarray(A, dtype=complex) for A in arrs],
        sites=range(L),
        shape="lrud",
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
    )

def mpo_two_site_gate_nonlocal(L, j, k, U2, *, cutoff=1e-12):
    if j == k:
        raise ValueError("Need two distinct sites.")
    if j > k:
        j, k = k, j

    d = 2
    U2 = _as_np(U2)

    # U2 -> (out_j,out_k,in_j,in_k) then to (out_j,in_j) x (out_k,in_k)
    U = U2.reshape(d, d, d, d)
    U = np.transpose(U, (0, 2, 1, 3))
    M = U.reshape(d * d, d * d)

    Uu, Ss, Vh = np.linalg.svd(M, full_matrices=False)
    keep = np.where(Ss > cutoff)[0]
    Uu = Uu[:, keep]
    Ss = Ss[keep]
    Vh = Vh[keep, :]

    r = len(Ss)
    sqrtS = np.sqrt(Ss)

    Aops = [(Uu[:, a] * sqrtS[a]).reshape(d, d) for a in range(r)]
    Bops = [(Vh[a, :] * sqrtS[a]).reshape(d, d) for a in range(r)]

    arrs = mpo_identity_arrays(L, d=d)
    I = _as_np(np.eye(d))

    # site j core tensor: (Dl=1, Dr=r, u, d)
    Aj = np.zeros((1, r, d, d), dtype=complex)
    for a in range(r):
        Aj[0, a, :, :] = Aops[a]
    set_site_tensor_with_boundary(arrs, j, Aj)

    # middle propagation tensors: (Dl=r, Dr=r, u, d)
    for s in range(j + 1, k):
        W = np.zeros((r, r, d, d), dtype=complex)
        for a in range(r):
            W[a, a, :, :] = I
        # NOTE: if s is boundary (can only happen if L=2, but then no middle), safe anyway
        set_site_tensor_with_boundary(arrs, s, W)

    # site k core tensor: (Dl=r, Dr=1, u, d)
    Bk = np.zeros((r, 1, d, d), dtype=complex)
    for a in range(r):
        Bk[a, 0, :, :] = Bops[a]
    set_site_tensor_with_boundary(arrs, k, Bk)

    return mpo_from_arrays(arrs)