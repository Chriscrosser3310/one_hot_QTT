from typing import List
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from one_hot_basis import ith_gray_binary, ith_gray_onehot

def pauli_string_from_ops(num_qubits: int, ops_by_qubit: dict[int, str]) -> str:
    """
    Build a Qiskit Pauli string of length `num_qubits` from a dict mapping
    qubit index -> {'I','X','Y','Z'}.

    Qiskit convention: the *leftmost* character acts on the *highest-index* qubit.
    """
    chars: List[str] = ['I'] * num_qubits
    for q, op in ops_by_qubit.items():
        if op not in ("I", "X", "Y", "Z"):
            raise ValueError(f"Invalid Pauli op {op} on qubit {q}")
        chars[q] = op
    return "".join(chars)  # reverse for Qiskit string order


def build_H_xxyy_projected(n: int) -> SparsePauliOp:
    r"""
    Construct H_n = (XX/2 + YY/2) ⊗ P1^{⊗(n-1)} where P1=(I-Z)/2.

    Interpretation:
    - total qubits N = n + 1
    - qubits 0,1 carry XX/2 + YY/2
    - qubits 2..n each carry P1

    Returns:
    SparsePauliOp on N qubits.

    Note:
    Expanding P1^{⊗(n-1)} gives 2^(n-1) Z/I terms, so total Pauli terms = 2^n.
    """
    if n < 1:
        raise ValueError("n must be >= 1 (so that n-1 projectors is >= 0).")

    N = n + 1  # total qubits
    proj_qubits = list(range(2, N))  # qubits with P1, size n-1

    labels: List[str] = []
    coeffs: List[complex] = []

    # P1 = (I - Z)/2, so tensor product over (n-1) qubits expands as:
    # (1/2^(n-1)) * sum_{S subset} (-1)^{|S|} Z_S
    # Multiply by (XX+YY)/2 gives overall factor 1/2^n.
    base_factor = 1.0 / (2**n)

    m = len(proj_qubits)  # = n-1

    for mask in range(1 << m):
        # Choose subset S of projector qubits where we place Z (else I)
        ops = {}
        parity = 0
        for j in range(m):
            if (mask >> j) & 1:
                ops[proj_qubits[j]] = 'Z'
                parity ^= 1  # track |S| mod 2

        sign = -1.0 if parity else 1.0
        c = sign * base_factor

        # XX term on qubits 0,1
        ops_xx = dict(ops)
        ops_xx[0] = 'X'
        ops_xx[1] = 'X'
        labels.append(pauli_string_from_ops(N, ops_xx))
        coeffs.append(c)

        # YY term on qubits 0,1
        ops_yy = dict(ops)
        ops_yy[0] = 'Y'
        ops_yy[1] = 'Y'
        labels.append(pauli_string_from_ops(N, ops_yy))
        coeffs.append(c)

    return SparsePauliOp(labels, coeffs)

def build_H_x_projected(n: int) -> SparsePauliOp:
    r"""
    Construct H_n = X ⊗ P1^{⊗(n-1)} where P1=(I-Z)/2.

    Interpretation:
    - total qubits N = n 
    - qubits 0 carry X
    - qubits 1..n each carry P1

    Returns:
    SparsePauliOp on N qubits.

    Note:
    Expanding P1^{⊗(n-1)} gives 2^(n-1) Z/I terms, so total Pauli terms = 2^(n-1).
    """
    if n < 1:
        raise ValueError("n must be >= 1 (so that n-1 projectors is >= 0).")

    N = n  # total qubits
    proj_qubits = list(range(1, N))  # qubits with P1, size n-1

    labels: List[str] = []
    coeffs: List[complex] = []

    # P1 = (I - Z)/2, so tensor product over (n-1) qubits expands as:
    # (1/2^(n-1)) * sum_{S subset} (-1)^{|S|} Z_S
    # Multiply by (XX+YY)/2 gives overall factor 1/2^n.
    base_factor = 1.0 / (2**n)

    m = len(proj_qubits)  # = n-1

    for mask in range(1 << m):
        # Choose subset S of projector qubits where we place Z (else I)
        ops = {}
        parity = 0
        for j in range(m):
            if (mask >> j) & 1:
                ops[proj_qubits[j]] = 'Z'
                parity ^= 1  # track |S| mod 2

        sign = -1.0 if parity else 1.0
        c = sign * base_factor

        # XX term on qubits 0,1
        ops_x = dict(ops)
        ops_x[0] = 'X'
        labels.append(pauli_string_from_ops(N, ops_x))
        coeffs.append(c)

    return SparsePauliOp(labels, coeffs)

def binary_gray_trotter_circuit(D, n, W, dt, n_steps,
                            init_state="center",
                            disorder_type=None,
                            seed=0,
                            measure=True):
    qc = QuantumCircuit(D*n, D*n)

    if init_state == "center":
        for d in range(D):
            bs = ith_gray_binary(n, 2**n//2)
            for i, b in enumerate(bs):
                if b == "1":
                    qc.x(d*n + i)

    for _ in range(n_steps):
        for d in range(D):
            qc.rx(-2*dt, d*n)
            for l in range(1, n):
                for lp in range(l-1):
                    qc.x(d*n + lp)
                qc.mcrx(-2*dt, range(d*n, d*n + l), d*n + l)
                for lp in range(l-1):
                    qc.x(d*n + lp)
    if measure:
        qc.measure(range(D*n), range(D*n))
    return qc

def one_hot_gray_trotter_circuit(D, n, q, W, dt, n_steps,
                            init_state="center",
                            disorder_type=None,
                            seed=0,
                            measure=True):
    
    qc = QuantumCircuit(D*q*n, D*q*n)

    if init_state == "center":
        for d in range(D):
            bs = ith_gray_onehot(n, q, q**n//2)
            for i, b in enumerate(bs):
                if b == "1":
                    qc.x(d*q*n + i)

    for _ in range(n_steps):
        for d in range(D):
            if q % 2 == 0:
                for m in range(0, q-1, 2):
                     qc.append(XXPlusYYGate(-2*dt), (d*q*n + m, d*q*n + m+1))
                for m in range(1, q-1, 2):
                     qc.append(XXPlusYYGate(-2*dt), (d*q*n + m, d*q*n + m+1))
                for l in range(1, n):
                    Pauli_op = build_H_xxyy_projected(l+1)
                    mc_rxy = PauliEvolutionGate(Pauli_op, -dt)
                    even_controls = [d*q*n + q*lp for lp in range(l-1)] + [d*q*n + q*l - 1]
                    odd_controls = [d*q*n + q*lp for lp in range(l)] 
                    for m in range(0, q-1, 2):
                        qc.append(mc_rxy, even_controls + [d*q*n + q*l+m, d*q*n + q*l+m+1])
                    for m in range(1, q-1, 2):
                        qc.append(mc_rxy, odd_controls + [d*q*n + q*l+m, d*q*n + q*l+m+1])
        
    if measure:
        qc.measure(range(D*q*n), range(D*q*n))
    return qc

if __name__ == "__main__":

    from qiskit import transpile

    # test multi particle move gate count
    n = 5
    Pauli_op = build_H_xxyy_projected(n)
    mc_rxy = PauliEvolutionGate(Pauli_op, 1)

    qc = QuantumCircuit(n+1)
    qc.append(mc_rxy, range(n+1))
    qc = transpile(qc, basis_gates=["rzz", "u"], optimization_level=3)

    # test pauli evolution vs multi-controlled-rotation
    n = 6
    qc = QuantumCircuit(n)
    qc.mcrx(2, range(n-1), n-1)
    qc = transpile(qc, basis_gates=["rzz", "u"], optimization_level=3)
    print(qc.count_ops())

    qc2 = QuantumCircuit(n)
    qc2.append(PauliEvolutionGate(build_H_x_projected(n), 1), range(n))
    qc2 = transpile(qc2, basis_gates=["rzz", "u"], optimization_level=3)
    print(qc2.count_ops())