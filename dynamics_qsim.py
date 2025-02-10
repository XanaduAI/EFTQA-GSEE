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

import cirq
import qsimcirq
import GPUtil
from qiskit.opflow import X, Z, Y, I

from algorithms.trotter_circs_qsim import get_tevol_circ
from algorithms.utils import gen_int_list


if len(GPUtil.getAvailable()) == 0:
    print("CPU")
    options = {"t": 8, "f": 3}
    qsim_simulator = qsimcirq.QSimSimulator(options)

else:
    print("GPU available :) !")
    options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=0, max_fused_gate_size=3)
    qsim_simulator = qsimcirq.QSimSimulator(options)


def heisenberg_ham(L, J):
    """Constructs the Heisenberg Hamiltonian for a 1D chain of length L with interaction strengths given by J.

    Args:
        L (int): The number of sites (qubits) in the chain.
        J (numpy.ndarray): A 2D array of shape (3, L*(L-1)/2) containing the interaction strengths for the
                           X, Y, and Z terms. J[0, it] corresponds to the X interaction strength,
                           J[1, it] to the Y interaction strength, and J[2, it] to the Z interaction strength.

    Returns:
        A tuple containing the Heisenberg Hamiltonian, a list of cirq operations representing the Hamiltonian,
        and a list of coefficients corresponding to the Cirq operations.
    """
    H = 0 * (I ^ L)

    qubits = cirq.LineQubit.range(L)
    H_cirq = []
    coefficients = []

    print(H_cirq)

    it = 0
    for i in range(L - 1):
        for j in range(i + 1, L):
            l = j - i
            op_x = (I ^ (i)) ^ X ^ (I ^ (l - 1)) ^ X ^ (I ^ (L - 1 - i - l))
            op_y = (I ^ (i)) ^ Y ^ (I ^ (l - 1)) ^ Y ^ (I ^ (L - 1 - i - l))
            op_z = (I ^ (i)) ^ Z ^ (I ^ (l - 1)) ^ Z ^ (I ^ (L - 1 - i - l))

            H = H + (J[0, it] * op_x + J[1, it] * op_y + J[2, it] * op_z)

            H_cirq.append(cirq.X(qubits[i]) * cirq.X(qubits[j]))
            coefficients.append(J[0, it])
            H_cirq.append(cirq.Y(qubits[i]) * cirq.Y(qubits[j]))
            coefficients.append(J[1, it])
            H_cirq.append(cirq.Z(qubits[i]) * cirq.Z(qubits[j]))
            coefficients.append(J[2, it])
            it += 1

    return H, H_cirq, coefficients


def exact_dynamics(state, ew, ev, times):
    """Compute the exact dynamics of a quantum state over time.

    Args:
        state (numpy.ndarray): The initial quantum state vector.
        ew (numpy.ndarray): The eigenvalues of the Hamiltonian.
        ev (numpy.ndarray): The eigenvectors of the Hamiltonian.
        times (list or numpy.ndarray): The time points at which to evaluate the dynamics.

    Returns:
        numpy.ndarray: The values of the state at the specified time points.
    """

    state = np.dot(ev.conjugate().transpose(), state.copy())

    values = []
    for t in times:
        op = np.diag(np.exp(-1.0j * ew * t))
        values.append(np.dot(state.copy().conjugate().transpose(), np.dot(op, state.copy())))
    return np.array(values)


def main(number, initial_state, num_steps):

    ######## HAMILTONIAN #######
    L = 10  # number of spins

    np.random.seed(23)
    wf = np.random.uniform(-1, 1, size=(2 ** (L)))
    wf = wf / np.linalg.norm(wf)

    if initial_state[:4] == "dmrg":
        # fix the name with your initial state
        Jx, Jz = 1.0, 1.0  # Define Jx and Jz with appropriate values
        initial_state += f"_{L}_{Jx}_{Jz}"
        wave_function = np.load(f"data/{initial_state}.npy").reshape(-1)
        pass

    # build the Hamiltonian
    dim = int(L * (L - 1) / 2)
    np.random.seed(42)
    J = np.random.normal(0, 1, size=3 * dim).reshape(3, dim) / L

    coupling = np.zeros((dim, dim, 3))
    it = 0
    for i in range(L - 1):
        for j in range(i + 1, L):
            coupling[i, j, :] = J[:, it]
            it += 1

    norm = np.sum(np.absolute(J))
    H, H_cirq, H_coeff = heisenberg_ham(L, J)
    print("norm {}".format(norm))

    tau = np.pi / (2 * norm)
    print("tau:", tau)

    epsilon = 10**-2

    d = (
        np.sqrt(2)
        / (tau * epsilon)
        * np.log(4 * np.sqrt(2 * np.pi / (tau * epsilon)) * (2 + tau / epsilon))
    )
    print("d :", d, min(d, 20000))
    d = min(d, 20000)

    time_interval = tau * np.arange(1, 2 * int(d) + 1, 2)

    if num_steps == 0:
        ### exact evolution ###
        ham = H.to_matrix()
        ew, ev = np.linalg.eigh(ham)
        print("EW: ", ew)

        p = [np.dot(wf.copy().transpose().conjugate(), ev[:, i]) ** 2 for i in range(2**L)]
        np.save("results/{}/overlap.npy".format(number), p)

        res = exact_dynamics(wf.copy(), ew, ev, time_interval)
        results = np.array([res.real, res.imag])
        np.save("results/{}/ew.npy".format(number), ew)

    else:
        # Trotter
        qubits = cirq.LineQubit.range(L)
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

        results = np.zeros((2, len(time_interval)))

        full_circuit = get_tevol_circ(circ_params)
        res = qsim_simulator.simulate_expectation_values(
            full_circuit, observables=H_cirq, initial_state=wf.copy()
        )
        res = np.dot(np.array(H_coeff), res)
        print("initial_state energy: ", res)
        np.savetxt(f"results/{number}/is_energy.txt", np.array(res).reshape(-1))

        for tx in tqdm.tqdm(range(d)):
            circ_params["dt"] = 2 ** min(1, tx) / num_steps
            full_circuit = get_tevol_circ(circ_params)
            res = qsim_simulator.simulate(full_circuit, initial_state=wf.copy())

            wf = res.final_state_vector
            overlap = np.dot(wf, wf_initial.conjugate().transpose())
            results[0, tx] = overlap.real
            results[1, tx] = overlap.imag

    np.save(f"results/{number}/moments_{initial_state}_{num_steps}.npy", results)
    np.save(f"results/{number}/tau.npy", tau)


if __name__ == "__main__":
    number = 32

    os.makedirs(f"results/{number}", exist_ok=True)

    name = "dmrg_"
    for state in [0]:
        for num_steps in [0, 10]:
            main(number, name + str(state), num_steps)
