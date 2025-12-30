import numpy as np
import quimb as qu
import quimb.tensor as qtn
from MPO_utilities import kronMPOs

# compat patch (matches your setup)
if not hasattr(qu, "transpose"):
    qu.transpose = np.transpose


def mpo_one_hot_projector(q, *, upper_ind_id="k{}", lower_ind_id="b{}"):
    """
    Diagonal MPO on q qubits projecting onto the one-hot subspace:
      P = sum_{m=0}^{q-1} |0...010...0><0...010...0|
    i.e. exactly one '1' among the q qubits.

    Returns: quimb.tensor.MatrixProductOperator
    Bond dimension: q+1 (counts number of 1s seen so far, capped at 2 then killed)
    """
    I = np.eye(2, dtype=complex)
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|

    # states: c in {0, 1, 2} meaning "seen c ones so far (2 = >=2 / dead)"
    D = 3

    arrs = []
    for s in range(q):
        if q == 1:
            # one-hot on 1 qubit means exactly |1>
            W = np.zeros((1, 1, 2, 2), dtype=complex)
            W[0, 0] = P1
            arrs.append(W.reshape(1, 2, 2))  # boundary
            break

        if s == 0:
            W = np.zeros((1, D, 2, 2), dtype=complex)
            # if qubit=0 keep count, if qubit=1 increment
            W[0, 0] = P0          # 0 -> stay at 0
            W[0, 1] = P1          # 1 -> go to 1
            arrs.append(W[0])     # (Dr,u,d) == "rud"
        elif s == q - 1:
            W = np.zeros((D, 1, 2, 2), dtype=complex)
            # accept only paths that end with total count == 1
            W[0, 0] = P1          # from 0 with last=1 -> total 1
            W[1, 0] = P0          # from 1 with last=0 -> total 1
            W[2, 0] = 0.0         # dead stays dead
            arrs.append(W[:, 0])  # (Dl,u,d) == "lud"
        else:
            W = np.zeros((D, D, 2, 2), dtype=complex)

            # on a 0: keep count (including dead)
            W[0, 0] = P0
            W[1, 1] = P0
            W[2, 2] = P0

            # on a 1: increment count; 1->dead if already saw one
            W[0, 1] = P1
            W[1, 2] = P1
            W[2, 2] = P1  # dead stays dead

            arrs.append(W)  # (Dl,Dr,u,d) == "lrud"

    return qtn.MatrixProductOperator(
        [np.asarray(A, dtype=complex) for A in arrs],
        sites=range(q),
        shape="lrud",
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
    )

def mpo_prod_one_hot_projector(n, q, *, upper_ind_id="k{}", lower_ind_id="b{}"):
    P = mpo_one_hot_projector(q, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    return kronMPOs([P]*n)


# ----------------
# quick sanity check (small q)
# ----------------
if __name__ == "__main__":
    n = 2
    q = 3
    P = mpo_prod_one_hot_projector(n, q)
    P.show()
    Pd = P.to_dense()

    print(np.nonzero(np.diag(np.round(np.real(Pd), 1))))