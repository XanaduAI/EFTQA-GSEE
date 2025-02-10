"""
Title: Early Fault-Tolerant Quantum Algorithms in Practice: Application to Ground-State Energy Est.
Authors: O. Kiss, U. Azad, B. Requena, A. Roggero, D. Wakeham, J. M. Arrazola
Paper: arXiv:2405.03754 
Year: 2024
Description: This module contains functions to construct the Hamiltonian for a lattice system, 
calculate the exact dynamics of a quantum state, and execute a dynamics simulation
for a given initial state and number of steps.
"""

import os
import numpy as np
import tqdm

import pennylane as qml

from algorithms.trotter_circs_pl import get_tevol_circ
from algorithms.utils import gen_int_list


def Hamiltonian(coupling, field):
    """Constructs the Hamiltonian for a given lattice system with specified coupling constants and field.

    Args:
        coupling (dict): A dictionary containing the coupling constants for the Heisenberg model.
        field (numpy.ndarray): A 2D array representing the external field applied to the lattice.

    Returns:
        tuple (): A tuple containing the Hamiltonian operator of the system and a list of tuples representing
        the indices of the Pauli X operators in the Hamiltonian.
    """
    bc = True

    n_cells = [n, m]

    spin_ham = qml.spin.heisenberg("square", n_cells, coupling=coupling, boundary_condition=bc)
    spin_ham += qml.ops.sum(*[qml.Z(wire) * val for wire, val in enumerate(np.ravel(field))])

    lattice = qml.spin.generate_lattice("square", n_cells, boundary_condition=bc)
    coupling_edges = [edge[:2] for edge in lattice.edges]

    return spin_ham, coupling_edges


def exact_dynamics(state, ew, ev, times):
    """Calculate the exact dynamics of a quantum state over time.

    Args:
        state (numpy.ndarray): The initial state vector.
        ew (numpy.ndarray): The eigenvalues of the Hamiltonian.
        ev (numpy.ndarray): The eigenvectors of the Hamiltonian.
        times (list or numpy.ndarray): The time points at which to evaluate the dynamics.

    Returns:
        numpy.ndarray: The values of the state at the specified times.
    """

    state = np.dot(ev.conjugate().transpose(), state.copy())

    values = []
    for t in times:
        op = np.diag(np.exp(-1.0j * ew * t))
        values.append(np.dot(state.copy().conjugate().transpose(), np.dot(op, state.copy())))

    return np.array(values)


def execute(number, initial_state, num_steps):
    """Executes the dynamics simulation for a given initial state and number of steps.

    Args:
        number (int): Identifier for the simulation run.
        initial_state (str): The initial state of the system. If it starts with "dmrg", additional parameters are appended.
        num_steps (int): Number of steps for the simulation. If 0, exact dynamics are computed.

    Developer Notes:
    - The Hamiltonian is constructed using predefined coupling constants and a random field.
    - The norm of the Hamiltonian is used to compute the time step `tau`.
    - The dynamics are computed either exactly (if num_steps is 0) or using a spin lattice circuit.
    - Results are saved as numpy arrays in the "results/{number}/" directory.
    """

    ######## HAMILTONIAN #######
    n, m = 2, 2
    Jx, Jz = -1, 1
    coupling = -np.array(([Jx, Jx, Jz]))

    if initial_state[:4] == "dmrg":
        initial_state += "_{}_{}_{}".format(n * m, Jx, Jz)

    wf = np.load("data/{}.npy".format(initial_state))

    # For random initial state:
    # wf = np.random.uniform(-1, 1, size = (2**(n*m)))
    # wf = wf / np.linalg.norm(wf)

    # For all zero initial state:
    # wf = np.zeros(2**(n*m))
    # wf[0] = 1

    np.random.seed(42)
    field = np.random.uniform(-1, 1, (n, m))

    ###########################
    Ham, edges = Hamiltonian(coupling, field)

    field = field / 2
    coupling = coupling / (n * m)
    ###########################

    print(np.sum(np.absolute(field)))

    norm = 20
    # We can consider other norms for the Hamiltonian:
    # norm = np.sum(np.absolute(field)) + (2*abs(Jx)+abs(Jz))*len(edges)
    # norm = scipy.sparse.linalg.norm(ham, ord = 1)
    # norm = np.linalg.norm(ham,ord=2)
    print(f"norm {norm}")

    tau = np.pi / (2 * norm)
    print("tau:", tau)

    epsilon = 10**-2

    d = (
        np.sqrt(2)
        / (tau * epsilon)
        * np.log(4 * np.sqrt(2 * np.pi / (tau * epsilon)) * (2 + tau / epsilon))
    )
    print("d :", d, min(d, 4000))
    d = min(d, 4000)

    time_interval = tau * np.arange(1, 2 * int(d) + 1, 2)

    if num_steps == 0:
        ham = Ham.to_matrix()
        ew, ev = np.linalg.eigh(ham)

        res = exact_dynamics(wf, ew, ev, time_interval)
        results = np.array([res.real, res.imag])

        np.save(f"results/{number}/ew.npy", ew)
    else:

        # Trotter
        L = n * m
        qubits = qml.wires.Wires(range(L))
        labels = np.arange(L)

        wf = np.array(wf, dtype="complex64")
        wf_initial = wf.copy()

        vij = coupling
        int_terms, _, _ = gen_int_list(L, vij)

        circ_params = {
            "dt": 0,
            "num_steps": num_steps,
            "J": coupling,
            "num_spins": L,
            "int_terms": int_terms,
            "labels": labels,
            "qubit_list": qubits,
            "echo": False,
            "verbose": False,
        }

        dev = qml.device("lightning.qubit", wires=L)
        full_qscript = qml.QuantumScript(
            [qml.StatePrep(wf_initial, wires=qubits), *get_tevol_circ(circ_params)],
            [qml.expval(Ham)],
        )
        res = qml.execute(full_qscript, dev, diff_method=None)
        print(f"initial_state energy: {res}")
        np.savetxt(f"results/{number}/is_energy.txt", np.array(res).reshape(-1))

        results = np.zeros((2, len(time_interval)))
        for tx in tqdm.tqdm(range(d)):
            circ_params["dt"] = 2 ** min(1, tx) / num_steps

            full_qscript = qml.QuantumScript(
                [qml.StatePrep(wf, wires=qubits), *get_tevol_circ(circ_params)],
                [qml.state()],
            )

            wf = qml.execute(full_qscript, dev, diff_method=None)
            overlap = np.dot(wf.conjugate().transpose(), wf_initial)
            results[0, tx] = float(overlap.real)
            results[1, tx] = float(overlap.imag)

    np.save(f"results/{number}/moments_{initial_state}_{num_steps}.npy", results)
    np.save(f"results/{number}/tau.npy", tau)


if __name__ == "__main__":
    number = 5

    os.makedirs(f"results/{number}", exist_ok=True)

    name = "dmrg_"
    for state in [2]:
        for num_steps in [1, 5, 10]:
            execute(number, name + str(state), num_steps)
