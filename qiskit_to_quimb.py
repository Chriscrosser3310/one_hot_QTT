# -*- coding: utf-8 -*-
from __future__ import annotations

from tqdm import tqdm
import quimb as qu
from gate_matrices import *
from MPO_utilities import *

import itertools
import random

from typing import Callable, Dict, List, Tuple, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Gate
from qiskit.quantum_info import Operator

class MPOGate:
    def __init__(self, nq, qubits, mat, mpo):
        self.nq = nq
        self.qubits = qubits
        self.mat = mat
        self.mpo = mpo

def qc_to_mpos(qc):
    nq = qc.num_qubits
    mpo_list = []
    for inst in qc.data:

        if not isinstance(inst.operation, Gate):
            continue

        qubit_index = {q: i for i, q in enumerate(qc.qubits)}
        qubits = tuple(qubit_index[q] for q in inst.qubits)
        
        mat = Operator(inst.operation).data

        if len(qubits) == 1:
            U_list = [PauliI] * nq
            U_list[qubits[0]] = mat
            mpo = mpo_product_unitaries(U_list)
            mpo_list.append(MPOGate(nq, qubits, mat, mpo))
        elif len(qubits) == 2:
            mat = mat.reshape(2,2,2,2).transpose(1,0,3,2).reshape(4,4)
            i, j = qubits
            mpo = mpo_two_site_gate_nonlocal(nq, i, j, mat, cutoff=1E-15)
            mpo_list.append(MPOGate(nq, qubits, mat, mpo))
        else:
            raise NotImplementedError("Not implemented for gate acting on > 2 qubits.")
        
    return mpo_list

def depolarize_2q_mpo(mpo, i, j, p, max_bond=None, cutoff=1E-15):

    nq = mpo.L
    mats = {}
    mats["I"] = PauliI
    mats["X"] = PauliX
    mats["Y"] = PauliY
    mats["Z"] = PauliZ

    mpo_new = (1-p)*mpo.copy()
    for p1 in ["I", "X", "Y", "Z"]:
        for p2 in ["I", "X", "Y", "Z"]:
            if p1 + p2 != "II" :
                U_list = [PauliI] * nq
                U_list[i] = mats[p1]
                U_list[j] = mats[p2]
                noise_mpo = mpo_product_unitaries(U_list)
                mpo_new += noise_mpo.apply(mpo.apply(noise_mpo)) * p/15
                mpo_new.compress(max_bond=max_bond, cutoff=cutoff)

    return mpo_new
 
def depolarize_2q_mps(mps, i, j, p, max_bond=None, cutoff=1E-15, rng=None):
    if random.random() >= p:
        return mps
    
    nq = mps.L
    mats = {}
    mats["I"] = PauliI
    mats["X"] = PauliX
    mats["Y"] = PauliY
    mats["Z"] = PauliZ

    PAULIS_2Q = [
        (p1, p2)
        for p1 in ['I','X','Y','Z']
        for p2 in ['I','X','Y','Z']
        if not (p1 == 'I' and p2 == 'I')
    ]
    if rng is None:
        p1, p2 = random.choice(PAULIS_2Q)
    else:
        p1, p2 = rng.choice(PAULIS_2Q)
    U_list = [PauliI] * nq
    U_list[i] = mats[p1]
    U_list[j] = mats[p2]
    noise_mpo = mpo_product_unitaries(U_list)
    mps_new = noise_mpo.apply(mps, compress=True, max_bond=max_bond, cutoff=cutoff)

    return mps_new

def noisy_qc_mpo_sim(qc, p, max_bond=None, cutoff=1E-15, use_tqdm=True, show=False):
    nq = qc.num_qubits
    init_mpo = qtn.MPO_product_operator([P0]*nq)
    mpo_gate_list = qc_to_mpos(qc)

    cur_mpo = init_mpo
    if show:
        cur_mpo.show()
    if use_tqdm:
        it = tqdm(mpo_gate_list)
    else:
        it = mpo_gate_list
    for mpo_gate in it:
        cur_mpo = mpo_gate.mpo.apply(cur_mpo,
                                     compress=True,
                                     max_bond=max_bond,
                                     cutoff=cutoff)
        cur_mpo = cur_mpo.apply(mpo_gate.mpo.conj().partial_transpose(range(nq)), 
                                  compress=True,
                                  max_bond=max_bond,
                                  cutoff=cutoff)
        if len(mpo_gate.qubits) == 2:
            if p > 0:
                i, j = mpo_gate.qubits
                cur_mpo = depolarize_2q_mpo(cur_mpo, i, j, p, 
                                        max_bond=max_bond, 
                                        cutoff=cutoff)
        if show:
            cur_mpo.show()
    return cur_mpo

def noisy_qc_mps_sim(qc, p, max_bond=None, cutoff=1E-15, use_tqdm=True, show=False, seed=None):
    nq = qc.num_qubits
    init_mps = qtn.MPS_computational_state("0"*nq)
    mpo_gate_list = qc_to_mpos(qc)

    rng = random.Random(seed)

    cur_mps = init_mps
    if show:
        cur_mps.show()
    if use_tqdm:
        it = tqdm(mpo_gate_list)
    else:
        it = mpo_gate_list
    for mpo_gate in it:
        cur_mps = mpo_gate.mpo.apply(cur_mps,
                                     compress=True,
                                     max_bond=max_bond,
                                     cutoff=cutoff)
        if len(mpo_gate.qubits) == 2:
            if p > 0:
                i, j = mpo_gate.qubits
                cur_mps = depolarize_2q_mps(cur_mps, i, j, p, 
                                        max_bond=max_bond, 
                                        cutoff=cutoff,
                                        rng=rng)
        if show:
            cur_mps.show()
    return cur_mps

# ------------------- example -------------------
if __name__ == "__main__":
    from qiskit import QuantumRegister
    from qiskit.circuit import Parameter
    from qiskit import transpile

    qr = QuantumRegister(10, "q")
    qc = QuantumCircuit(qr)
    qc.h(0)
    qc.cx(0, 9)
    qc.ry(0.45, 2)

    #qc = transpile(qc, basis_gates=["rzz", "u"])
    #for mpo in qc_to_mpos(qc):
    #    mpo.show()

    mpo = noisy_qc_mpo_sim(qc, 0.001)
    mpo.show()