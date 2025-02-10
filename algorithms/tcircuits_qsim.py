"""
Title: Early Fault-Tolerant Quantum Algorithms in Practice: Application to Ground-State Energy Est.
Authors: O. Kiss, U. Azad, B. Requena, A. Roggero, D. Wakeham, J. M. Arrazola
Paper: arXiv:2405.03754 
Year: 2024
Description: This module contains functions to construct cirq circuits for simulating the Hamiltonian systems.
"""
import cirq
import numpy as np
from scipy.linalg import expm

px = np.array([[0.0, 1.0], [1.0, 0.0]])
py = np.array([[0.0, -1.0j], [1.0j, 0.0]])
pz = np.array([[1.0, 0.0], [0.0, -1.0]])

pxpx = np.kron(px, px)
pypy = np.kron(py, py)
pzpz = np.kron(pz, pz)


def spin_lattice_circuit(coupling, field, dt, num_steps, order, paulis_coupling, qubits):
    """Simulates the evolution of a spin lattice system using a quantum circuit.

    Args:
        coupling (list or array): Coupling constants for the spin interactions.
        field (array): External field applied to the spins.
        wf (array): Initial wavefunction of the system.
        dt (float): Time step for the simulation.
        num_steps (int): Number of time steps to simulate.
        order (int): Order of the Trotter-Suzuki decomposition.
        paulis_coupling (list): List of Pauli operators for the coupling terms.

    Returns:
        array: Final wavefunction of the system after evolution.
    """
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
    """Applies a time-evolved field to a set of qubits.

    Args:
        qubits (list[int]): List of qubit indices to which the field is applied.
        field (list[float]): List of field values corresponding to each qubit.
        dt (float): Time step for the evolution.
    """
    for i, field_value in enumerate(field):
        one_u = expm(-1.0j * dt * field_value * pz)
        matrix_gate_z = cirq.MatrixGate(one_u, name="Z")
        circuit.append(matrix_gate_z.on(qubits[i]))

    return circuit


def apply_coupling(circuit, qubits, g2, paulis_coupling):
    """Apply a coupling operation to a set of qubits using a specified unitary matrix.

    Args:
        qubits (list[int]): List of qubit indices to which the coupling operation will be applied.
        g2 (array): Unitary matrix representing the coupling operation.
        paulis_coupling (list[tuple[int, int]]): List of tuples, where each tuple contains two qubit indices
                                                 indicating the qubits to which the unitary matrix will be applied.
    """
    for p in paulis_coupling:
        circuit.append(g2.on(qubits[p[0]], qubits[p[1]]))

    return circuit
