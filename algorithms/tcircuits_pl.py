"""
Title: Early Fault-Tolerant Quantum Algorithms in Practice: Application to Ground-State Energy Est.
Authors: O. Kiss, U. Azad, B. Requena, A. Roggero, D. Wakeham, J. M. Arrazola
Paper: arXiv:2405.03754 
Year: 2024
Description: This module contains functions to construct PL circuits for simulating the Hamiltonian systems.
"""

import pennylane as qml
from scipy.linalg import expm
import numpy as np


px = np.array([[0.0, 1.0], [1.0, 0.0]])
py = np.array([[0.0, -1.0j], [1.0j, 0.0]])
pz = np.array([[1.0, 0.0], [0.0, -1.0]])

pxpx = np.kron(px, px)
pypy = np.kron(py, py)
pzpz = np.kron(pz, pz)


def spin_lattice_circuit(coupling, field, wf, dt, num_steps, order, paulis_coupling):
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

    H = coupling[0] * pxpx + coupling[1] * pypy + coupling[2] * pzpz
    two_u = expm(-1.0j * dt * H)

    field = field.reshape(-1)
    n = len(field)

    dev = qml.device("lightning.qubit", wires=n)

    @qml.qnode(dev)
    def my_circ(wf, dt, n, two_u, paulis_coupling, field):
        qml.StatePrep(wf, wires=range(n))
        for _ in range(num_steps):
            apply_field(range(n), field, dt)
            apply_coupling(range(n), two_u, paulis_coupling)
        return qml.state()

    wf_final = my_circ(wf, dt, n, two_u, sorted(paulis_coupling), field)

    return wf_final


def apply_field(qubits, field, dt):
    """Applies a time-evolved field to a set of qubits.

    Args:
        qubits (list[int]): List of qubit indices to which the field is applied.
        field (list[float]): List of field values corresponding to each qubit.
        dt (float): Time step for the evolution.
    """

    for i, f in enumerate(field):
        unitary = expm(-1.0j * dt * f * pz)
        qml.QubitUnitary(unitary, wires=qubits[i], id="Z")


def apply_coupling(qubits, g2, paulis_coupling):
    """Apply a coupling operation to a set of qubits using a specified unitary matrix.

    Args:
        qubits (list[int]): List of qubit indices to which the coupling operation will be applied.
        g2 (array): Unitary matrix representing the coupling operation.
        paulis_coupling (list[tuple[int, int]]): List of tuples, where each tuple contains two qubit indices
                                                 indicating the qubits to which the unitary matrix will be applied.
    """

    for p in paulis_coupling:
        qml.QubitUnitary(g2, wires=[p[0], p[1]], id="U")
