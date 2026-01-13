import numpy as np
import itertools
import matplotlib.pyplot as plt
import re
from typing import Dict, Hashable, List, Tuple, Optional, Any

from qiskit_aer.noise import NoiseModel, PauliError, ReadoutError, phase_damping_error, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit import CouplingMap

from qiskit_trotter_circuit import *

from qiskit.quantum_info import state_fidelity

noise_params = dict(
    # gate faults
    p1=2.8e-5,
    p2=8.3e-4,

    # measurement bitflip (use asymmetric tuple form (p0->1, p1->0))
    # using SPAM(0/1) as a proxy for measurement flip rates:
    p_meas=(6.7e-4, 1.2e-3),

    # init faults: not separately published as a single number in the benchmark table,
    # but using the same scale as SPAM(0) is a reasonable starting proxy:
    p_init=6.7e-4,

    # crosstalk during measurement/init
    p_crosstalk_meas=2.2e-5,
    # not separately listed in the benchmark table; start smaller than meas-crosstalk:
    p_crosstalk_init=1.0e-5,

    # spontaneous emission fraction of p1/p2
    # docs: emission is ~order-of-magnitude smaller than the depolarizing component
    # so set emission_ratio ~ 0.1 as a starting point:
    p1_emission_ratio=0.10,
    p2_emission_ratio=0.10,

    # scaling knobs (leave at 1 unless you want to sweep)
    scale=1.0,
    p1_scale=1.0,
    p2_scale=1.0,
    meas_scale=1.0,
    init_scale=1.0,
    memory_scale=1.0,
    emission_scale=1.0,
    crosstalk_scale=1.0,
)

def quantinuum_params_to_aer_noise_model(
    noise_params: dict = noise_params,
    entangling_gates=("rzz",),      # set to ("ms",) or ("rxx","ryy","rzz") if you use those
    one_qubit_gates=("u"),
    include_1q_gate_noise=False,    # emulator has p1; set True if you want it in Aer
):
    """
    Map a Quantinuum-emulator-style parameter dict to a Qiskit Aer NoiseModel.

    Implemented:
      - p2: 2-qubit gate error as a Pauli channel (Z-biased surrogate by default)
      - p_meas=(p01,p10): asymmetric readout flips
      - p1 (optional): 1-qubit depolarizing error

    Not implemented here (needs custom modeling):
      - init error p_init
      - measurement/init crosstalk
      - transport/idle dephasing models
      - leakage/spontaneous emission as a true non-Pauli channel
    """
    nm = NoiseModel()

    # ---- scales (if present) ----
    p1 = noise_params.get("p1", 0.0) * noise_params.get("p1_scale", 1.0) * noise_params.get("scale", 1.0)
    p2 = noise_params.get("p2", 0.0) * noise_params.get("p2_scale", 1.0) * noise_params.get("scale", 1.0)

    p_meas = noise_params.get("p_meas", (0.0, 0.0))
    meas_scale = noise_params.get("meas_scale", 1.0) * noise_params.get("scale", 1.0)
    p01 = float(p_meas[0]) * meas_scale
    p10 = float(p_meas[1]) * meas_scale

    # ---- 2q gate noise: anisotropic Pauli channel (Z-biased surrogate) ----
    # If you have your own pauli2q_probs dict, replace this block with it.
    pauli2q_probs = noise_params.get("pauli2q_probs", {
        "ZI": 0.22, "IZ": 0.22, "ZZ": 0.20,
        "XI": 0.06, "IX": 0.06, "YI": 0.06, "IY": 0.06,
        "XX": 0.04, "YY": 0.04,
        "XZ": 0.01, "ZX": 0.01, "YZ": 0.01, "ZY": 0.01
    })

    # build PauliError with total error prob p2
    paulis = ["II"]
    probs = [max(0.0, 1.0 - p2)]
    for P, w in pauli2q_probs.items():
        paulis.append(P)
        probs.append(p2 * float(w))

    # normalize tiny numerical drift
    s = sum(probs)
    probs = [x / s for x in probs]

    err2q = PauliError(paulis, probs)
    nm.add_all_qubit_quantum_error(err2q, entangling_gates)

    # ---- 1q gate noise (optional): depolarizing proxy ----
    if include_1q_gate_noise and p1 > 0:
        err1q = depolarizing_error(p1, 1)
        nm.add_all_qubit_quantum_error(err1q, one_qubit_gates)

    # ---- readout noise ----
    if p01 > 0 or p10 > 0:
        ro = ReadoutError([[1 - p01, p01],
                           [p10, 1 - p10]])
        nm.add_all_qubit_readout_error(ro)

    return nm

def depolarizing_noise_model(p, basis_2q=["rzz"]):
    """
    Create a NoiseModel with 2-qubit depolarizing noise applied
    after every 2-qubit gate.

    Parameters
    ----------
    p : float
        Total depolarizing probability.
        Each non-identity 2-qubit Pauli occurs with probability p/15.

    Returns
    -------
    NoiseModel
        Qiskit Aer noise model.
    """
    noise_model = NoiseModel()

    # 2-qubit depolarizing channel
    dep2 = depolarizing_error(p, num_qubits=2)

    # Apply to *all* 2-qubit gates
    noise_model.add_all_qubit_quantum_error(
        dep2,
        basis_2q 
    )

    return noise_model


'''
def make_noise_model(p2, pauli2q_probs, p01=0.0, p10=0.0, idle_gamma=0.0):
    nm = NoiseModel()

    # --- 2-qubit anisotropic Pauli noise after every 2q gate ---
    paulis = ["II"]
    probs  = [1.0 - p2]
    for P, w in pauli2q_probs.items():
        paulis.append(P)
        probs.append(p2 * w)

    err2q = PauliError(paulis, probs)
    nm.add_all_qubit_quantum_error(err2q, ["cx", "cz", "ecr"])  # <- key change

    # --- idle dephasing (optional) on id gates ---
    if idle_gamma > 0:
        nm.add_all_qubit_quantum_error(phase_damping_error(idle_gamma), ["id"])

    # --- readout error (optional) ---
    if p01 > 0 or p10 > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p01, p01],
                          [p10, 1 - p10]])
        )

    return nm
'''

def run_with_noise(qc, noise_model, shots=2000):
    sim = AerSimulator(noise_model=noise_model)
    tqc = transpile(qc, sim)
    return sim.run(tqc, shots=shots).result()

def normalize(counts):
    s = sum(counts.values())
    return {k: v/s for k,v in counts.items()}

'''
def tvd(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k,0) - q.get(k,0)) for k in keys)
'''

def run_counts(qc, sim, shots=5000, one_hot_params=None):
    tqc = transpile(qc, sim)
    counts = sim.run(tqc, shots=shots).result().get_counts()
    if one_hot_params is None:
        return counts
    else:
        D, n, q = one_hot_params
        assert all(len(k) == D*n*q for k in counts.keys()), "QC qubit number does not match one_hot parameters."
        new_counts = {}
        total = 0
        subspace = 0
        for k in counts.keys():
            total += counts[k]
            accept = True
            for d in range(D):
                for l in range(n):
                    if k[d*q*n + l*q: d*q*n + (l+1)*q].count("1") != 1:
                        accept = False
                        break
                if accept == False:
                    break
            if accept:
                subspace += counts[k]
                new_counts[k] = counts[k]
        prob = subspace/total
        print(f"Success probability: {prob}")
        return counts, new_counts, prob
    

from typing import Dict, Hashable, List, Tuple, Optional, Any

PMF = Dict[Hashable, float]
PMFPair = Tuple[PMF, PMF]

# ---------- TVD core ----------

def tv_from_pmfs(pmf_P: PMF, pmf_Q: PMF) -> float:
    support = set(pmf_P) | set(pmf_Q)
    return 0.5 * sum(abs(pmf_P.get(k, 0.0) - pmf_Q.get(k, 0.0)) for k in support)


def tv_point_bootstrap(
    pmf_P: PMF,
    pmf_Q: PMF,
    *,
    n_eff: int,
    n_boot: int = 500,
    ci=(2.5, 97.5),
    rng=None,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(rng)
    support = list(set(pmf_P) | set(pmf_Q))

    pP = np.array([pmf_P.get(k, 0.0) for k in support])
    pQ = np.array([pmf_Q.get(k, 0.0) for k in support])
    pP /= pP.sum()
    pQ /= pQ.sum()

    tv_hat = 0.5 * np.abs(pP - pQ).sum()

    boots = np.empty(n_boot)
    for b in range(n_boot):
        cP = rng.multinomial(n_eff, pP)
        cQ = rng.multinomial(n_eff, pQ)
        boots[b] = 0.5 * np.abs(cP / n_eff - cQ / n_eff).sum()

    lo, hi = np.percentile(boots, ci)
    return tv_hat, max(0.0, lo), min(1.0, hi)

# ---------- Curve construction ----------

def tv_curve_from_pmf_pairs(
    pmf_pairs: List[PMFPair],
    *,
    t_values: Optional[List[float]] = None,
    n_eff: Optional[int] = None,
    n_boot: int = 500,
    rng=None,
) -> Dict[str, np.ndarray]:

    L = len(pmf_pairs)
    t = np.arange(L) if t_values is None else np.asarray(t_values, float)
    if len(t) != L:
        raise ValueError("t_values length mismatch")

    y = np.empty(L)
    y_lo = np.full(L, np.nan)
    y_hi = np.full(L, np.nan)

    base_rng = np.random.default_rng(rng)
    seeds = base_rng.integers(0, 2**63 - 1, size=L)

    for i, (pP, pQ) in enumerate(pmf_pairs):
        if n_eff is None:
            y[i] = tv_from_pmfs(pP, pQ)
        else:
            y[i], y_lo[i], y_hi[i] = tv_point_bootstrap(
                pP, pQ, n_eff=n_eff, n_boot=n_boot, rng=int(seeds[i])
            )

    order = np.argsort(t)
    out = {"t": t[order], "y": y[order]}
    if n_eff is not None:
        out.update({"y_lo": y_lo[order], "y_hi": y_hi[order]})
    return out


#----- fidelity test -----

def end_state_fidelity(qc, noise_model, *, optimization_level: int = 0, seed: int | None = None) -> float:
    """
    Args:
        qc: QuantumCircuit (no measurements recommended; they will be stripped if present)
        noise_model: qiskit_aer.noise.NoiseModel
        optimization_level: transpile optimization level (0-3)
        seed: simulator seed for reproducibility (can be None)

    Returns:
        Fidelity F = <psi| rho |psi> between ideal and noisy final states.
    """
    # --- make a copy and remove measurements (fidelity is about quantum state) ---
    qc0 = qc.copy()
    try:
        qc0.remove_final_measurements(inplace=True)
    except Exception:
        # older qiskit: if remove_final_measurements isn't available, just proceed
        pass

    # --- Ideal statevector simulation ---
    ideal_backend = AerSimulator(method="statevector", seed_simulator=seed)
    qc_ideal = qc0.copy()
    qc_ideal.save_statevector()

    tqc_ideal = transpile(qc_ideal, ideal_backend, optimization_level=optimization_level)
    ideal_res = ideal_backend.run(tqc_ideal).result()
    psi = ideal_res.get_statevector(tqc_ideal)

    # --- Noisy density-matrix simulation ---
    noisy_backend = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        seed_simulator=seed,
    )
    qc_noisy = qc0.copy()
    qc_noisy.save_density_matrix()

    tqc_noisy = transpile(qc_noisy, noisy_backend, optimization_level=optimization_level)
    noisy_res = noisy_backend.run(tqc_noisy).result()
    rho = noisy_res.data(0)["density_matrix"]  # qiskit.quantum_info.DensityMatrix

    # --- Fidelity: <psi| rho |psi> ---
    return float(state_fidelity(psi, rho))

def fidelity_in_product_onehot_subspace(
    qc,
    noise_model,
    q: int,
    n: int,
    optimization_level: int = 0,
    seed: int | None = None,
    renormalize: bool = False,
    return_extras: bool = True,
):
    """
    Compute fidelity restricted to the subspace (one-hot)^⊗n, where each block has q qubits.

    Total qubits must be nq = n*q.

    Steps:
      1) Ideal: simulate |psi> with Aer statevector.
      2) Noisy: simulate rho with Aer density_matrix + noise_model.
      3) Restrict to the product-one-hot subspace by slicing basis indices.
      4) Return conditional fidelity inside subspace:
            F_cond = <psi_sub| rho_sub |psi_sub>   (with psi_sub optionally renormalized)
         Also optionally return leakage probability:
            p_in = Tr(rho_sub) = Tr(P rho)

    Args:
        qc: QuantumCircuit (measurements will be removed if present)
        noise_model: qiskit_aer.noise.NoiseModel
        q, n: block size and number of blocks
        optimization_level: 0 is recommended to avoid altering your pre-optimized circuit
        seed: simulator seed
        renormalize: if True, renormalize projected ideal state within the subspace
        return_extras: if True, return (F_cond, p_in, dim_sub, inds)

    Returns:
        F_cond if return_extras=False
        else (F_cond, p_in, dim_sub, inds)
    """
    nq = n * q
    if qc.num_qubits != nq:
        raise ValueError(f"qc has {qc.num_qubits} qubits, but expected n*q = {nq} (n={n}, q={q}).")

    # ----- copy circuit and remove measurements -----
    qc0 = qc.copy()
    try:
        qc0.remove_final_measurements(inplace=True)
    except Exception:
        pass

    # ----- Ideal simulation (pure state) -----
    ideal_backend = AerSimulator(method="statevector", seed_simulator=seed)
    qc_ideal = qc0.copy()
    qc_ideal.save_statevector()
    tqc_ideal = transpile(qc_ideal, ideal_backend, optimization_level=optimization_level)
    psi = ideal_backend.run(tqc_ideal).result().get_statevector(tqc_ideal)

    # ----- Noisy simulation (mixed state) -----
    noisy_backend = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        seed_simulator=seed,
    )
    qc_noisy = qc0.copy()
    qc_noisy.save_density_matrix()
    tqc_noisy = transpile(qc_noisy, noisy_backend, optimization_level=optimization_level)
    rho = noisy_backend.run(tqc_noisy).result().data(0)["density_matrix"]

    # Convert to numpy arrays
    psi_vec = np.asarray(psi, dtype=complex)
    rho_mat = np.asarray(rho, dtype=complex)

    # ----- Build indices for (one-hot)^⊗n subspace -----
    # One-hot basis in a q-qubit block: basis states with exactly one '1' bit
    onehot_block = [1 << j for j in range(q)]  # integers 1,2,4,...,2^(q-1)
    inds = []
    for xs in itertools.product(onehot_block, repeat=n):
        idx = 0
        for l, x in enumerate(xs):
            idx |= (x << (l * q))  # block l occupies bits [l*q, ..., (l+1)q-1]
        inds.append(idx)
    dim_sub = len(inds)  # = q^n

    # ----- Restrict to subspace (slice vector and density matrix) -----
    psi_sub = psi_vec[inds]
    rho_sub = rho_mat[np.ix_(inds, inds)]

    # Leakage probability into subspace: p_in = Tr(P rho) = Tr(rho_sub)
    p_in = float(np.real(np.trace(rho_sub)))

    # Optionally renormalize the projected ideal state within the subspace
    if renormalize:
        norm = np.vdot(psi_sub, psi_sub).real
        if norm <= 0:
            # ideal state has zero support in subspace
            F_cond = 0.0
            return (F_cond, p_in, dim_sub, inds) if return_extras else F_cond
        psi_sub = psi_sub / np.sqrt(norm)

    # Conditional fidelity inside subspace:
    # F_cond = <psi_sub| rho_sub |psi_sub>
    F_cond = float(np.real(np.vdot(psi_sub, rho_sub @ psi_sub)))

    # (Optional) If you want a true conditional-on-being-in-subspace fidelity, divide by p_in:
    # F_post = F_cond / p_in  (only meaningful if p_in>0)
    # Here we return F_cond as defined above; p_in is returned for interpretation.

    if return_extras:
        return F_cond, p_in, dim_sub, inds
    return F_cond

class TrotterNoisySim:
    def __init__(self, D, n, logq, dt, n_steps_list, noise_rate, apply_transpilation=True, optimization_level=3, basis=["rzz", "u"]):
        self.D = D
        self.n = n
        self.dt = dt
        self.n_steps_list = n_steps_list
        self.logq = logq
        self.q = 2**logq
        self.nb = n*logq
        self.noise_rate = noise_rate

        self.binary_circuits = [] 
        self.one_hot_circuits = [] 

        self.basis = basis
        for n_steps in n_steps_list:
            bc = binary_gray_trotter_circuit(D, n*logq, 0, dt, n_steps)
            oc = one_hot_gray_trotter_circuit(D, n, self.q, 0, dt, n_steps)
            if apply_transpilation:
                bc = transpile(bc, basis_gates=self.basis, optimization_level=optimization_level)
                oc = transpile(oc, basis_gates=self.basis, optimization_level=optimization_level)
            self.binary_circuits.append(bc)
            self.one_hot_circuits.append(oc)

        self.binary_hist_pairs = []
        self.one_hot_hist_pairs = []
        self.one_hot_projected_hist_pairs = []
        self.TVD_sucess_probs = []

        self.binary_tv_curve = None
        self.one_hot_tv_curve = None
        self.one_hot_projected_tv_curce = None

        self.binary_fidelities = []
        self.one_hot_fidelities = []
        self.one_hot_projected_fidelities = []
        self.fidelity_sucess_probs = []

    def get_tvd_with_qiskit(self, shots):
        print(f"====Running: D={self.D}, n={self.n}, q={self.q}, dt={self.dt}====")

        noise_model = depolarizing_noise_model(self.noise_rate)

        for i, n_steps in enumerate(self.n_steps_list):
            print(f"Running: n_steps={n_steps}")
            ideal_sim = AerSimulator()
            noisy_sim = AerSimulator(noise_model=noise_model)

            ideal_binary = normalize(run_counts(self.binary_circuits[i], ideal_sim, shots=shots))
            noisy_binary = normalize(run_counts(self.binary_circuits[i], noisy_sim, shots=shots))

            bt = (ideal_binary, noisy_binary)
            self.binary_hist_pairs.append(bt)
            #print("Binary TVD =", bt)

            ideal_one_hot = normalize(run_counts(self.one_hot_circuits[i], ideal_sim, shots=shots))
            counts, proj_counts, prob = run_counts(self.one_hot_circuits[i], noisy_sim, shots=shots, one_hot_params=(self.D, n, self.q))
            noisy_one_hot = normalize(counts)
            proj_noisy_one_hot = normalize(proj_counts)
            self.TVD_sucess_probs.append(prob)

            ot = (ideal_one_hot, noisy_one_hot)
            opt = (ideal_one_hot, proj_noisy_one_hot)
            self.one_hot_hist_pairs.append(ot)
            self.one_hot_projected_hist_pairs.append(opt)

        self.binary_tv_curve = tv_curve_from_pmf_pairs(self.binary_hist_pairs, n_eff=shots, n_boot=shots//10)
        self.one_hot_tv_curve = tv_curve_from_pmf_pairs(self.one_hot_hist_pairs, n_eff=shots, n_boot=shots//10)
        self.one_hot_projected_tv_curce = tv_curve_from_pmf_pairs(self.one_hot_projected_hist_pairs, n_eff=shots, n_boot=shots//10)

    def get_fidelity_with_qiskit(self):
        print(f"====Running: D={self.D}, n={self.n}, q={self.q}, dt={self.dt}====")

        noise_model = depolarizing_noise_model(self.noise_rate)

        for i, n_steps in enumerate(self.n_steps_list):
            print(f"Running: n_steps={n_steps}")
            self.binary_fidelities.append(end_state_fidelity(self.binary_circuits[i], noise_model))
            
            #self.one_hot_fidelities.append(end_state_fidelity(self.one_hot_circuits[i], noise_model))
            
            F_cond, p_in, _, _ = fidelity_in_product_onehot_subspace(self.one_hot_circuits[i], noise_model, self.q, self.n)
            self.one_hot_fidelities.append(F_cond)
            self.one_hot_projected_fidelities.append(F_cond/p_in)
            self.fidelity_sucess_probs.append(p_in)

def plot_tv_curve(curves, 
                  xlabel: str = "Number of Trotter steps",
                  ylabel: str | None = None,
                  title: str | None = None,
                  alpha_band: float = 0.25,
                  lw: float = 2.0,):
    
    fig, ax = plt.subplots()

    for label, data in curves.items():
        label = re.sub(r",\s*p\s*=.*$", "", label)
        line, = ax.plot(data["t"], data["y"], lw=lw, label=label)
        color = line.get_color()
        if "y_lo" in data:
            ax.fill_between(
                data["t"], data["y_lo"], data["y_hi"],
                color=color, alpha=alpha_band, linewidth=0
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.show()

    return fig, ax