"""
This module contains functions to create quantum circuits for spin lattice simulations.
"""

import os
import cirq
import numpy as np
from scipy.linalg import expm
from algorithms.state_preparation import apply_field, apply_coupling

px = np.array([[0.0, 1.0], [1.0, 0.0]])
py = np.array([[0.0, -1.0j], [1.0j, 0.0]])
pz = np.array([[1.0, 0.0], [0.0, -1.0]])

pxpx = np.kron(px, px)
pypy = np.kron(py, py)
pzpz = np.kron(pz, pz)


def spin_lattice_circuit(coupling, field, dt, num_steps, order, paulis_coupling, qubits):

    n, m = np.shape(field)
    field = field.reshape(-1)
    observable = cirq.Z(qubits[-1])

    circuit = cirq.Circuit()

    H = coupling[0] * pxpx + coupling[1] * pypy + coupling[2] * pzpz
    two_u = expm(-1.0j * dt * H)
    g2 = cirq.MatrixGate(two_u, name="H")

    for nn in range(num_steps):

        if order == 1:
            circuit = apply_field(circuit, qubits, field, dt)
            circuit = apply_coupling(circuit, qubits, g2, paulis_coupling)
        else:
            circuit = apply_field(circuit, qubits, field, dt / 2)
            circuit = apply_coupling(circuit, qubits, g2, paulis_coupling)
            circuit = apply_field(circuit, qubits, field, dt / 2)

    return circuit, observable, qubits


def apply_field(circuit, qubits, field, dt):

    for i in range(len(field)):
        one_u = expm(-1.0j * dt * field[i] * pz)
        g1 = cirq.MatrixGate(one_u, name="Z")
        circuit.append(g1.on(qubits[i]))

    return circuit


def apply_coupling(circuit, qubits, g2, paulis_coupling):

    for p in paulis_coupling:
        circuit.append(g2.on(qubits[p[0]], qubits[p[1]]))

    return circuit
