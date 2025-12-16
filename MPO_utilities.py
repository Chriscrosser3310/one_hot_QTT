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