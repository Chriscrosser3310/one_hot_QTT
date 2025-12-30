import numpy as np
import quimb as qu
import quimb.tensor as qtn
from pathlib import Path
from one_hot_MPO_projector import mpo_prod_one_hot_projector
from MPO_utilities import kronMPOs
from one_hot_basis import ith_lex_binary, ith_gray_binary, ith_lex_onehot, ith_gray_onehot

#===============================================
#================== utilities ==================
#===============================================

# ---- compatibility patch for some quimb/autoray versions ----
if not hasattr(qu, "transpose"):
    qu.transpose = np.transpose
    

def _as_np(x):
    return np.asarray(x, dtype=complex)

P0 = _as_np([[1, 0],
             [0, 0]])
P1 = _as_np([[0, 0],
             [0, 1]])
PauliI = _as_np([[1, 0],
                 [0, 1]]) 
PauliX = _as_np([[0, 1],
                 [1, 0]]) 
PauliY = _as_np([[0, -1j],
                 [1j, 0]]) 
PauliZ = _as_np([[1, 0],
                 [0, -1]]) 

def exp_i_theta_X(theta):
    c = np.cos(theta)
    s = 1j * np.sin(theta)
    return _as_np([
        [c,   s],
        [s,   c]
    ])

def exp_i_theta_Z(theta):
    return _as_np([
        [np.exp(1j* theta), 0.0],
        [0.0,  np.exp(-1j* theta)],
    ])

#exp(-i theta (XX + YY)/2)
def exp_i_theta_xx_yy_over_two(theta):
    c = np.cos(theta)
    s = 1j * np.sin(theta)
    return _as_np([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,   c,   s, 0.0],
        [0.0,   s,   c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    
def exp_i_theta_ZZ(theta):
    return _as_np([
        [np.exp(1j* theta), 0.0, 0.0, 0.0],
        [0.0, np.exp(-1j* theta), 0.0, 0.0],
        [0.0, 0.0, np.exp(-1j* theta), 0.0],
        [0.0, 0.0, 0.0, np.exp(1j* theta)],
    ])


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


def _set_site_tensor_with_boundary(arrs, s, W):
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
    _set_site_tensor_with_boundary(arrs, j, Aj)

    # middle propagation tensors: (Dl=r, Dr=r, u, d)
    for s in range(j + 1, k):
        W = np.zeros((r, r, d, d), dtype=complex)
        for a in range(r):
            W[a, a, :, :] = I
        # NOTE: if s is boundary (can only happen if L=2, but then no middle), safe anyway
        _set_site_tensor_with_boundary(arrs, s, W)

    # site k core tensor: (Dl=r, Dr=1, u, d)
    Bk = np.zeros((r, 1, d, d), dtype=complex)
    for a in range(r):
        Bk[a, 0, :, :] = Bops[a]
    _set_site_tensor_with_boundary(arrs, k, Bk)

    return mpo_from_arrays(arrs)

def mpo_RXY_pairs(L, pairs, theta, *, max_bond=None, cutoff=1e-12):
    U2 = exp_i_theta_xx_yy_over_two(theta)
    mpo = mpo_from_arrays(mpo_identity_arrays(L))
    for (j, k) in pairs:
        mpo_jk = mpo_two_site_gate_nonlocal(L, j, k, U2, cutoff=cutoff)
        mpo = mpo_jk.apply(mpo, compress=True, max_bond=max_bond, cutoff=cutoff)
    return mpo

def mpo_rand_RZZ_pairs(L, pairs, W, max_bond=None, cutoff=1e-12, rng=np.random):
    mpo = mpo_from_arrays(mpo_identity_arrays(L))
    for (j, k) in pairs:
        theta = W*np.pi*(2*rng.random()-1.)
        U2 = exp_i_theta_ZZ(theta)
        mpo_jk = mpo_two_site_gate_nonlocal(L, j, k, U2, cutoff=cutoff)
        mpo = mpo_jk.apply(mpo, compress=True, max_bond=max_bond, cutoff=cutoff)
    return mpo

def mpo_rand_RZ(L, W, rng=np.random):
    U_list = []
    for _ in range(L):
        theta = W*np.pi*(2*rng.random()-1.)
        U1 = exp_i_theta_Z(theta)
        U_list.append(U1)
    return mpo_product_unitaries(U_list)

def mpo_X(L, inds):
    U_list = [PauliI] * L
    for i in inds:
        U_list[i] = PauliX
    return mpo_product_unitaries(U_list)

def mpo_RX(L, inds, theta):
    U_list = [PauliI] * L
    for i in inds:
        U_list[i] = exp_i_theta_X(theta)
    return mpo_product_unitaries(U_list)

def multi_controlled_RXY_pairs_mpo(L, controls, pairs, theta, *,
                               max_bond=None, cutoff=1e-12):
    """
    MPO for:
      (I - Π_c P1(c)) ⊗ I   +   (Π_c P1(c)) ⊗ U_pairs
    equivalently:
      Π_c P0(c) ⊗ I + Π_c P1(c) ⊗ U_pairs  only if there is exactly one control,
    but for multiple controls the 'else' branch should be identity whenever any control is 0.

    Here we implement:
      mpo = mpo_else + mpo_then
    where:
      mpo_then = (Π_c P1(c)) ⊗ U_pairs
      mpo_else = I - (Π_c P1(c)) ⊗ I
    so total is identity unless all controls are 1.
    """
    controls = tuple(controls)   
    if len(controls) == 0:
        # no controls: just return U_pairs
        return mpo_RXY_pairs(L, pairs, theta, max_bond=max_bond, cutoff=cutoff)

    # validate controls
    if len(set(controls)) != len(controls):
        raise ValueError("Duplicate control indices.")
    for c in controls:
        if not (0 <= c < L):
            raise ValueError(f"Control index {c} outside [0, L).")

    # validate pairs (disjoint + don't touch controls)
    used = set()
    for (j, k) in pairs:
        if j == k:
            raise ValueError(f"Pair {(j, k)} has j==k.")
        if j in controls or k in controls:
            raise ValueError(f"Control appears in pair {(j, k)}.")
        if j in used or k in used:
            raise ValueError("Pairs overlap (share a site).")
        used.add(j); used.add(k)

    # U_pairs
    mpoU = mpo_RXY_pairs(L, pairs, theta, max_bond=max_bond, cutoff=cutoff)

    # Π_c P1(c) as an MPO (diagonal projectors placed locally)
    arrP1 = mpo_identity_arrays(L)
    for c in controls:
        W = P1.reshape(1, 1, 2, 2)
        _set_site_tensor_with_boundary(arrP1, c, W)
    mpoAll1 = mpo_from_arrays(arrP1)

    # THEN branch: (Π_c P1(c)) ⊗ U_pairs
    mpo_then = mpoAll1.apply(mpoU, max_bond=max_bond, cutoff=cutoff)

    # ELSE branch: I - (Π_c P1(c)) ⊗ I
    # Build I as an MPO:
    mpoI = mpo_from_arrays(mpo_identity_arrays(L))
    # The (Π_c P1(c)) ⊗ I is just mpoAll1 itself (since other sites are identity ops)
    mpo_else = mpoI - mpoAll1

    mpo = mpo_else + mpo_then
    mpo.compress(max_bond=max_bond, cutoff=cutoff)
    return mpo

def multi_controlled_RX_mpo(L, controls, targets, theta, *,
                               max_bond=None, cutoff=1e-12):
    """
    MPO for:
      (I - Π_c P1(c)) ⊗ I   +   (Π_c P1(c)) ⊗ U_pairs
    equivalently:
      Π_c P0(c) ⊗ I + Π_c P1(c) ⊗ U_pairs  only if there is exactly one control,
    but for multiple controls the 'else' branch should be identity whenever any control is 0.

    Here we implement:
      mpo = mpo_else + mpo_then
    where:
      mpo_then = (Π_c P1(c)) ⊗ U_pairs
      mpo_else = I - (Π_c P1(c)) ⊗ I
    so total is identity unless all controls are 1.
    """
    controls = tuple(controls)   
    if len(controls) == 0:
        # no controls
        return mpo_RX(L, targets, theta)

    # validate controls
    if len(set(controls)) != len(controls):
        raise ValueError("Duplicate control indices.")
    for c in controls:
        if not (0 <= c < L):
            raise ValueError(f"Control index {c} outside [0, L).")

    # U_pairs
    mpoU = mpo_RX(L, targets, theta)

    # Π_c P1(c) as an MPO (diagonal projectors placed locally)
    arrP1 = mpo_identity_arrays(L)
    for c in controls:
        W = P1.reshape(1, 1, 2, 2)
        _set_site_tensor_with_boundary(arrP1, c, W)
    mpoAll1 = mpo_from_arrays(arrP1)

    # THEN branch: (Π_c P1(c)) ⊗ U_pairs
    mpo_then = mpoAll1.apply(mpoU, max_bond=max_bond, cutoff=cutoff)

    # ELSE branch: I - (Π_c P1(c)) ⊗ I
    # Build I as an MPO:
    mpoI = mpo_from_arrays(mpo_identity_arrays(L))
    # The (Π_c P1(c)) ⊗ I is just mpoAll1 itself (since other sites are identity ops)
    mpo_else = mpoI - mpoAll1

    mpo = mpo_else + mpo_then
    mpo.compress(max_bond=max_bond, cutoff=cutoff)
    return mpo

#=============================================
#================== one-hot ==================
#=============================================

def MC_RXY_trotter_step_mpos(D, n, q, dt, max_bond=None, cutoff=1e-10):
    Q = n * q

    even_mpos = []
    odd_mpos = []
    
    even_pairs = [(i, i+1) for i in range(0, q-1, 2)]
    odd_pairs = [(i, i+1) for i in range(1, q-1, 2)]
    
    even_mpo = mpo_RXY_pairs(Q, even_pairs, dt, max_bond=max_bond, cutoff=cutoff)
    odd_mpo = mpo_RXY_pairs(Q, odd_pairs, dt, max_bond=max_bond, cutoff=cutoff)

    even_mpo_kron = [even_mpo] * D
    odd_mpo_kron = [odd_mpo] * D

    even_mpos.append(kronMPOs(even_mpo_kron))
    odd_mpos.append(kronMPOs(odd_mpo_kron))
    
    
    if q % 2 == 0:
        for l in range(1, n):
            even_controls = [q*lp for lp in range(l-1)] + [q*l - 1]
            odd_controls = [q*lp for lp in range(l)] 
            
            even_pairs = [(q*l+i, q*l+i+1) for i in range(0, q-1, 2)]
            odd_pairs = [(q*l+i, q*l+i+1) for i in range(1, q-1, 2)]

            even_mpo = multi_controlled_RXY_pairs_mpo(Q, even_controls, even_pairs, dt, max_bond=max_bond, cutoff=cutoff)
            odd_mpo = multi_controlled_RXY_pairs_mpo(Q, odd_controls, odd_pairs, dt, max_bond=max_bond, cutoff=cutoff)

            even_mpo_kron = [even_mpo] * D
            odd_mpo_kron = [odd_mpo] * D
            
            # for even q, both are globally odd
            odd_mpos.append(kronMPOs(even_mpo_kron))
            odd_mpos.append(kronMPOs(odd_mpo_kron))
    
    elif q % 2 == 1:
        for l in range(1, n):
            even_controls = [q*lp-1 for lp in range(1, l+1)]
            odd_controls = [q*lp for lp in range(l)] 
            
            even_pairs = [(q*l+i, q*l+i+1) for i in range(0, q-1, 2)]
            odd_pairs = [(q*l+i, q*l+i+1) for i in range(1, q-1, 2)]
            
            even_mpo = multi_controlled_RXY_pairs_mpo(Q, even_controls, even_pairs, dt, max_bond=max_bond, cutoff=cutoff)
            odd_mpo = multi_controlled_RXY_pairs_mpo(Q, odd_controls, odd_pairs, dt, max_bond=max_bond, cutoff=cutoff)

            even_mpo_kron = [even_mpo] * D
            odd_mpo_kron = [odd_mpo] * D
            
            even_mpos.append(kronMPOs(even_mpo_kron))
            odd_mpos.append(kronMPOs(odd_mpo_kron))
    
    return even_mpos, odd_mpos

# W = disorder strength
def randZZ_trotter_step_mpos(D, n, q, W, dt, max_bond=None, cutoff=1e-12, seed=None):
    DQ = D * n * q
    
    even_pairs = [(i, i+1) for i in range(0, DQ-1, 2)]
    odd_pairs = [(i, i+1) for i in range(1, DQ-1, 2)]

    rng = np.random.default_rng(seed=seed)
    
    even_mpos = [mpo_rand_RZZ_pairs(DQ, even_pairs, W*dt, max_bond=max_bond, cutoff=cutoff, rng=rng)] 
    odd_mpos = [mpo_rand_RZZ_pairs(DQ, odd_pairs, W*dt, max_bond=max_bond, cutoff=cutoff, rng=rng)]
    
    return even_mpos, odd_mpos

# W = disorder strength
def randZ_trotter_step_mpos(D, n, q, dt, W, seed=None):
    DQ = D * n * q

    rng = np.random.default_rng(seed=seed)

    return [mpo_rand_RZ(DQ, W*dt, rng=rng)]

# W = disorder strength
def one_hot_gray_trotter_evolution(D, n, q, W, dt, n_steps,
                      init_state="center",
                      disorder_type="Z",
                      seed=0,
                      max_bond=None, 
                      cutoff=1e-12, 
                      fidelity_type="one_hot",
                      save_to_disk=True,
                      load_if_exists=True):
    """
    Yield the MPS after each full Trotter step.
    First yield is the initial state.
    """
    
    if init_state == "center":
        ind = q**n//2
        init_bitstring = ith_gray_onehot(n, q, ind) * D
        init_mps = qtn.MPS_computational_state(init_bitstring)
        init_state_str = f"({', '.join([str(ind)] * D)})"
    elif type(init_state) == tuple:
        assert len(init_state) == D, f"len(init_state)={len(init_state)} is not equal to D={D}."
        init_state_str_list = []
        init_bitstring = ""
        for ind_d in init_state:
            init_state_str_list.append(str(ind_d))
            init_bitstring += ith_gray_onehot(n, q, ind_d)
        init_mps = qtn.MPS_computational_state(init_bitstring)
        init_state_str = f"({', '.join(init_state_str_list)})"
        '''
        elif init_state == "random":
            init_bitstring = ith_gray_onehot(D*n, q, np.random.randint(0, q**(D*n)))
            init_mps = qtn.MPS_computational_state(init_bitstring)
        elif type(init_state) == int:
            init_bitstring = ith_gray_onehot(D*n, q, init_state)
            init_mps = qtn.MPS_computational_state(init_bitstring)
        elif type(init_state) == str:
            init_bitstring = init_state
            init_mps = qtn.MPS_computational_state(init_bitstring)
        '''
    else:
        raise NotImplementedError(f"init_state = {init_state}")
        
    
    folder = f"MPS/one_hot_gray/D={D}/n={n}/q={q}/dt={dt}/n_steps={n_steps}/max_bond={max_bond}/cutoff={cutoff}/disorder_type={disorder_type}/W={W}/seed={seed}/init_state={init_state_str}"

    if load_if_exists:
        try:
            print("Try to load from files.")
            fidelity_list = []
            with open(folder + f"/fidelity_type={fidelity_type}.txt", "r") as f:
                for line in f:
                    fidelity_list.append(float(line.rstrip("\n").split(" ")[1]))
            for step in range(n_steps+1):
                mps = qu.load_from_disk(folder + f"/step={step}.mps")
                yield mps, fidelity_list[step]
            print("Finished loading from files.")
            return
        except FileNotFoundError:
            print("Files not found, continue to generate MPS evolutions.")
            pass

    XY_even_mpos, XY_odd_mpos = MC_RXY_trotter_step_mpos(
        D, n, q, dt,
        max_bond=None,
        cutoff=cutoff,
    )

    if disorder_type == "Z":
        Z_mpos = randZ_trotter_step_mpos(D, n, q, W, dt, seed=seed)
        trotter_mpos = XY_even_mpos + XY_odd_mpos + Z_mpos
    elif disorder_type == "ZZ":
        ZZ_even_mpos, ZZ_odd_mpos = randZZ_trotter_step_mpos(
            D, n, q, W, dt, 
            max_bond=None,
            cutoff=cutoff,
            seed=seed)
        trotter_mpos = XY_even_mpos + ZZ_even_mpos + XY_odd_mpos + ZZ_odd_mpos
    elif disorder_type == None:
        trotter_mpos = XY_even_mpos + XY_odd_mpos
    else:
        raise NotImplementedError(f"disorder_type = {disorder_type}")

    if fidelity_type == "one_hot":
        proj_mpo = mpo_prod_one_hot_projector(D*n, q)
        fidelity_fn = lambda mps1, mps2: np.abs(mps1.H @ (proj_mpo.apply(mps2)))
    elif fidelity_type == "all":
        fidelity_fn = lambda mps1, mps2: np.abs(mps1.H @ mps2)
    else:
        raise NotImplementedError(f"fidelity_type = {fidelity_type}")

    if save_to_disk:
        Path(folder).mkdir(parents=True, exist_ok=True)

    cur_mps = init_mps
    if save_to_disk:
        Path(folder).mkdir(parents=True, exist_ok=True)
        with open(folder + f"/fidelity_type={fidelity_type}.txt", "w") as f:
            f.write(f"step=0: 1.0\n")
        qu.save_to_disk(cur_mps, folder + f"/step=0.mps")
    yield (cur_mps, 1.0)   # t = 0

    for step in range(1, n_steps+1):
        perfect_mps = cur_mps.copy()
        for mpo in trotter_mpos:
            perfect_mps = mpo.apply(
                perfect_mps,
                compress=False
            )
        perfect_mps.normalize()
        
        new_mps = cur_mps.copy()
        for mpo in trotter_mpos:
            new_mps = mpo.apply(
                new_mps,
                compress=True,
                max_bond=max_bond,
                cutoff=cutoff,
            )
        new_mps.normalize()
        
        cur_mps = new_mps
        fidelity = fidelity_fn(perfect_mps, cur_mps)

        if save_to_disk:
            with open(folder + f"/fidelity_type={fidelity_type}.txt", "a") as f:
                f.write(f"step={step}: {fidelity}\n")
            qu.save_to_disk(cur_mps, folder + f"/step={step}.mps")

        yield (cur_mps, fidelity)
    print("Fnished generating MPS evolutions.")

#============================================
#================== binary ==================
#============================================

def MC_X_trotter_step_mpos(D, n, dt, max_bond=None, cutoff=1e-12):
    
    even_mpos = []
    odd_mpos = []

    even_mpo = mpo_RX(n, [0], dt)
    even_mpo_kron = [even_mpo] * D
    even_mpos.append(kronMPOs(even_mpo_kron))
    
    for l in range(1, n):
        X_mpo = mpo_X(n, range(l-1))
        control_mpo = multi_controlled_RX_mpo(n, range(l), [l], dt, max_bond=max_bond, cutoff=cutoff)
        odd_mpo = X_mpo.apply(control_mpo.apply(X_mpo))
        odd_mpo.compress(max_bond=max_bond, cutoff=cutoff)

        odd_mpo_kron = [odd_mpo] * D
        odd_mpos.append(kronMPOs(odd_mpo_kron))
    
    return even_mpos, odd_mpos

# W = disorder strength
def binary_gray_trotter_evolution(D, n, W, dt, n_steps,
                      init_state="center",
                      disorder_type=None,
                      seed=0,
                      max_bond=None, 
                      cutoff=1e-12, 
                      fidelity_type="all",
                      save_to_disk=True,
                      load_if_exists=True):
    """
    Yield the MPS after each full Trotter step.
    First yield is the initial state.
    """

    if init_state == "center":
        ind = 2**n//2 
        init_bitstring = ith_gray_binary(n, ind) * D
        init_mps = qtn.MPS_computational_state(init_bitstring)
        init_state_str = f"({', '.join([str(ind)] * D)})"
    elif type(init_state) == tuple and all(isinstance(x, int) for x in init_state):
        assert len(init_state) == D, f"len(init_state)={len(init_state)} is not equal to D={D}."
        init_state_str_list = []
        init_bitstring = ""
        for ind_d in init_state:
            init_state_str_list.append(str(ind_d))
            init_bitstring += ith_gray_binary(n, ind_d)
        init_mps = qtn.MPS_computational_state(init_bitstring)
        init_state_str = f"({', '.join(init_state_str_list)})"
#    elif type(init_state) == tuple and init_state[0] == "MPS_rand_state":
#        bd = init_state[1]
#        mps_seed = init_state[2]
#        init_mps = qtn.MPS_rand_state(n, bd)
#        init_state_str = f"(MPS_rand_state, bd, seed={mps_seed})"
    else:
        raise NotImplementedError(f"init_state = {init_state}")
    
    folder = f"MPS/binary_gray/D={D}/n={n}/dt={dt}/n_steps={n_steps}/max_bond={max_bond}/cutoff={cutoff}/disorder_type={disorder_type}/W={W}/seed={seed}/init_state={init_state_str}"

    if load_if_exists:
        try:
            print("Try to load from files.")
            fidelity_list = []
            with open(folder + f"/fidelity_type={fidelity_type}.txt", "r") as f:
                for line in f:
                    fidelity_list.append(float(line.rstrip("\n").split(" ")[1]))
            for step in range(n_steps+1):
                mps = qu.load_from_disk(folder + f"/step={step}.mps")
                yield mps, fidelity_list[step]
            print("Finished loading from files.")
            return
        except FileNotFoundError:
            print("Files not found, continue to generate MPS evolutions.")
            pass

    X_even_mpos, X_odd_mpos = MC_X_trotter_step_mpos(
        D, n, dt,
        max_bond=None,
        cutoff=cutoff,
    )

    if disorder_type == "Z":
        Z_mpos = randZ_trotter_step_mpos(D, n, q, W, dt, seed=seed)
        trotter_mpos = X_even_mpos + X_odd_mpos + Z_mpos
    elif disorder_type == "ZZ":
        ZZ_even_mpos, ZZ_odd_mpos = randZZ_trotter_step_mpos(
            D, n, q, W, dt, 
            max_bond=None,
            cutoff=cutoff,
            seed=seed)
        trotter_mpos = X_even_mpos + ZZ_even_mpos + X_odd_mpos + ZZ_odd_mpos
    elif disorder_type == None:
        trotter_mpos = X_even_mpos + X_odd_mpos
    else:
        raise NotImplementedError(f"disorder_type = {disorder_type}")
    
    if fidelity_type == "all":
        fidelity_fn = lambda mps1, mps2: np.abs(mps1.H @ mps2)
    else:
        raise NotImplementedError(f"fidelity_type = {fidelity_type}")

    if save_to_disk:
        Path(folder).mkdir(parents=True, exist_ok=True)

    cur_mps = init_mps
    if save_to_disk:
        Path(folder).mkdir(parents=True, exist_ok=True)
        with open(folder + f"/fidelity_type={fidelity_type}.txt", "w") as f:
            f.write(f"step=0: 1.0\n")
        qu.save_to_disk(cur_mps, folder + f"/step=0.mps")
    yield (cur_mps, 1.0)   # t = 0

    for step in range(1, n_steps+1):
        perfect_mps = cur_mps.copy()
        for mpo in trotter_mpos:
            perfect_mps = mpo.apply(
                perfect_mps,
                compress=False
            )
        perfect_mps.normalize()
        
        new_mps = cur_mps.copy()
        for mpo in trotter_mpos:
            new_mps = mpo.apply(
                new_mps,
                compress=True,
                max_bond=max_bond,
                cutoff=cutoff,
            )
        new_mps.normalize()
        
        cur_mps = new_mps
        fidelity = fidelity_fn(perfect_mps, cur_mps)

        if save_to_disk:
            with open(folder + f"/fidelity_type={fidelity_type}.txt", "a") as f:
                f.write(f"step={step}: {fidelity}\n")
            qu.save_to_disk(cur_mps, folder + f"/step={step}.mps")

        yield (cur_mps, fidelity)
    print("Fnished generating MPS evolutions.")

if __name__ == "__main__":

    D = 1
    q = 4
    n = 2
    dt = 0.5
    W = 1
    n_steps = 50

    nb = 4

    #'''
    
    for mps, _ in one_hot_gray_trotter_evolution(D, n, q, W, dt, n_steps):
        mps.show()

    for mps, _ in binary_gray_trotter_evolution( D, nb, W, dt, n_steps):
        mps.show()

    for mps, _ in one_hot_gray_trotter_evolution(D, n, q, W, dt, n_steps, init_state=(0,)):
        mps.show()

    for mps, _ in binary_gray_trotter_evolution( D, nb, W, dt, n_steps, init_state=(0,)):
        mps.show()
    
    #'''
    # I'm Pen. Pen is here.