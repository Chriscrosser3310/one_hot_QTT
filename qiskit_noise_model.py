from qiskit_aer.noise import NoiseModel, PauliError, ReadoutError, phase_damping_error
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit import QuantumCircuit

from qiskit_aer.noise import NoiseModel, PauliError, ReadoutError, depolarizing_error

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

def run_with_noise(qc, noise_model, shots=2000):
    sim = AerSimulator(noise_model=noise_model)
    tqc = transpile(qc, sim)
    return sim.run(tqc, shots=shots).result()

def normalize(counts):
    s = sum(counts.values())
    return {k: v/s for k,v in counts.items()}

def tvd(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k,0) - q.get(k,0)) for k in keys)

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
