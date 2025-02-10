"""
Title: Early Fault-Tolerant Quantum Algorithms in Practice: Application to Ground-State Energy Est.
Authors: O. Kiss, U. Azad, B. Requena, A. Roggero, D. Wakeham, J. M. Arrazola
Paper: arXiv:2405.03754 
Year: 2024
Description: This module contains functions to construct cirq circuits for simulating the Hamiltonian systems.
"""
import numpy as np
from scipy.linalg import expm
import cirq

px = np.matrix([[0.0, 1.0], [1.0, 0.0]])
py = np.matrix([[0.0, -1.0j], [1.0j, 0.0]])
pz = np.matrix([[1.0, 0.0], [0.0, -1.0]])

sigmas = np.array(np.array([px, py, pz]))  # Vector of Pauli matrices
sds = np.array(
    np.array(
        [
            np.kron(px, px),
            np.kron(py, py),
            np.kron(pz, pz),
        ]  # Vector of tensor products of Pauli matrices
    )
)

mSWAP = np.matrix(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def get_tevol_circ(circ_parameters):
    """Generates a time evolution circuit based on the provided parameters.

    Args:
        circ_parameters (dict): A dictionary containing the parameters for the circuit.
            - "num_steps" (int): Number of time steps.
            - "dt" (float): Time step duration.
            - "num_spins" (int): Number of spins in the system.
            - "J" (float): Coupling constant.
            - "int_terms" (list): Interaction terms.
            - "labels" (list): Labels for the qubits.
            - "qubit_list" (list): List of qubits.
            - "verbose" (bool): Verbosity flag.
        labels (list, optional): Labels for the qubits. Defaults to None.
        slice (optional): Not used in the current implementation. Defaults to None.

    Returns:
        cirq.Circuit: The generated time evolution circuit.
    """

    tevol = cirq.Circuit()
    for _ in range(circ_parameters["num_steps"]):
        tcirc, _ = get_tstep_circ_2nd(
            circ_parameters["dt"],
            circ_parameters["num_spins"],
            circ_parameters["int_terms"],
            circ_parameters["labels"],
            circ_parameters["qubit_list"],
            circ_parameters["verbose"],
        )
        tevol.append(tcirc)

    return tevol

# setup 2nd order Trotter step with swap_network
def get_tstep_circ_2nd(tstep, num_spins, int_terms, labels, qubit_list, verbose=False):
    """Generates a second-order Trotter step circuit for a given time step and interaction terms.

    Args:
        tstep (float): The time step for the Trotterization.
        num_spins (int): The number of spins (qubits) in the system.
        int_terms (list of dict): List of interaction terms, where each term is a
            dictionary with keys "sui", "suj", and "hij".
        labels (list of int): List of spin labels.
        qubit_list (list of cirq.Qid): List of qubits corresponding to the spins.
        verbose (bool, optional): If True, prints additional information for debugging.
            Default is False.

    Returns:
        Tuple[cirq.Circuit, List[List[int]]]: the generated circuit for the given time step and
            the sequence of spin interactions for each layer.
    """
    nhalf = num_spins >> 1
    sequence = []
    tstep_so = cirq.Circuit()

    # FORWARD step
    for _ in range(nhalf):
        ## make an even layer
        layer_e = []
        for nn in range(nhalf):
            # fetch indices
            in0 = labels[2 * nn]
            in1 = labels[2 * nn + 1]
            # corresponding qubit
            iq0 = qubit_list[2 * nn]
            iq1 = qubit_list[2 * nn + 1]
            # fetch interaction element
            int_01 = next(
                (item for item in int_terms if ((item["sui"] == in0) and (item["suj"] == in1))),
                None,
            )

            if int_01 is None:
                if verbose:
                    print("WARNING: ordered pair not found, trying reverse")
                int_01 = next(
                    (item for item in int_terms if ((item["sui"] == in1) and (item["suj"] == in0))),
                    None,
                )
            if verbose:
                print(nn, int_01)
            ## TODO: try with explicit implementation
            # build two-qubit unitary
            mat = np.einsum("i,ijk", int_01["hij"], sds)
            twob_u = expm(-tstep * 1.0j * mat / 2)
            # add SWAP
            full_gate = twob_u * mSWAP
            # make gate
            g01 = cirq.MatrixGate(full_gate, name="U_" + str(in0) + str(in1))
            # add gate in the right place
            tstep_so.append(g01.on(iq0, iq1))
            # save index info
            layer_e.append([in0, in1])
            labels[2 * nn] = in1
            labels[2 * nn + 1] = in0
        sequence.append(layer_e)
        ## make an odd layer
        layer_o = []
        for nn in range(nhalf - 1):
            # fetch indices
            in0 = labels[2 * nn + 1]
            in1 = labels[2 * nn + 2]
            # corresponding qubit
            iq0 = qubit_list[2 * nn + 1]
            iq1 = qubit_list[2 * nn + 2]
            # fetch interaction element
            int_01 = next(
                (item for item in int_terms if ((item["sui"] == in0) and (item["suj"] == in1))),
                None,
            )
            if int_01 is None:
                if verbose:
                    print("WARNING: ordered pair not found, trying reverse")
                int_01 = next(
                    (item for item in int_terms if ((item["sui"] == in1) and (item["suj"] == in0))),
                    None,
                )
            if verbose:
                print(nn, int_01)
            # build two-qubit unitary
            mat = np.einsum("i,ijk", int_01["hij"], sds)
            twob_u = expm(-tstep * 1.0j * mat / 2)
            # add SWAP
            full_gate = twob_u * mSWAP
            # make gate
            g01 = cirq.MatrixGate(full_gate, name="U_" + str(in0) + str(in1))
            # add gate in the right place
            tstep_so.append(g01.on(iq0, iq1))
            # save index info
            layer_o.append([in0, in1])
            labels[2 * nn + 1] = in1
            labels[2 * nn + 2] = in0
        sequence.append(layer_o)

    # BACKWARD step
    for nl in range(nhalf):
        ## make an odd layer
        layer_o = []
        for nn in range(nhalf - 1):
            # fetch indices
            in0 = labels[2 * nn + 2]
            in1 = labels[2 * nn + 1]
            # corresponding qubit
            iq0 = qubit_list[2 * nn + 1]
            iq1 = qubit_list[2 * nn + 2]
            # fetch interaction element
            int_01 = next(
                (item for item in int_terms if ((item["sui"] == in0) and (item["suj"] == in1))),
                None,
            )
            if int_01 is None:
                if verbose:
                    print("WARNING: ordered pair not found, trying reverse")
                int_01 = next(
                    (item for item in int_terms if ((item["sui"] == in1) and (item["suj"] == in0))),
                    None,
                )
            if verbose:
                print(nn, int_01)
            # build two-qubit unitary
            mat = np.einsum("i,ijk", int_01["hij"], sds)
            twob_u = expm(-tstep * 1.0j * mat / 2)
            # add SWAP
            full_gate = twob_u * mSWAP
            # make gate
            g01 = cirq.MatrixGate(full_gate, name="U_" + str(in0) + str(in1))
            # add gate in the right place
            tstep_so.append(g01.on(iq0, iq1))
            # save index info
            layer_o.append([in0, in1])
            labels[2 * nn + 1] = in0
            labels[2 * nn + 2] = in1
        sequence.append(layer_o)
        ## make an even layer
        layer_e = []
        for nn in range(nhalf):
            # fetch indices
            in0 = labels[2 * nn + 1]
            in1 = labels[2 * nn]
            # corresponding qubit
            iq0 = qubit_list[2 * nn]
            iq1 = qubit_list[2 * nn + 1]
            # fetch interaction element
            int_01 = next(
                (item for item in int_terms if ((item["sui"] == in0) and (item["suj"] == in1))),
                None,
            )
            if int_01 is None:
                if verbose:
                    print("WARNING: ordered pair not found, trying reverse")
                int_01 = next(
                    (item for item in int_terms if ((item["sui"] == in1) and (item["suj"] == in0))),
                    None,
                )
            if verbose:
                print(nn, int_01)

            # build two-qubit unitary
            mat = np.einsum("i,ijk", int_01["hij"], sds)
            twob_u = expm(-tstep * 1.0j * mat / 2)
            # add SWAP
            full_gate = twob_u * mSWAP
            # make gate
            g01 = cirq.MatrixGate(full_gate, name="U_" + str(in0) + str(in1))
            # add gate in the right place
            tstep_so.append(g01.on(iq0, iq1))
            # save index info
            layer_e.append([in0, in1])
            labels[2 * nn] = in0
            labels[2 * nn + 1] = in1
        sequence.append(layer_e)

    if verbose:
        print(tstep_so)
    return tstep_so, sequence
