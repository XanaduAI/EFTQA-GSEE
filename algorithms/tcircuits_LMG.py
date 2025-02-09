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


def get_tevol_circ(circ_parameters, labels=None, slice=None):

    # dumb version first
    tevol = cirq.Circuit()
    for nn in range(circ_parameters["num_steps"]):

        if circ_parameters["order"] == 1:
            tcirc, tseq = get_tstep_circ_1st(
                circ_parameters["dt"],
                circ_parameters["neutrino_number"],
                circ_parameters["J"],
                circ_parameters["int_terms"],
                circ_parameters["labels"],
                circ_parameters["qubit_list"],
                verbose=circ_parameters["verbose"],
            )
        elif circ_parameters["order"] == 2:
            tcirc, tseq = get_tstep_circ_2nd(
                circ_parameters["dt"],
                circ_parameters["neutrino_number"],
                circ_parameters["J"],
                circ_parameters["int_terms"],
                circ_parameters["labels"],
                circ_parameters["qubit_list"],
                verbose=circ_parameters["verbose"],
            )
        else:
            print("order undefined")

        tevol.append(tcirc)

    return tevol


# setup 1st order Trotter step with swap_network
def get_tstep_circ_1st(
    tstep, neutrino_number, J, int_terms, labels, qubit_list, verbose=False, cut=None, vij=None
):

    nhalf = neutrino_number >> 1

    sequence = []
    tstep_fo = cirq.Circuit()

    for n in range(neutrino_number):
        tstep_fo.append(cirq.rz(B * tstep).on(qubit_list[n]))

    for nl in range(nhalf):
        ## make an even layer
        layer_e = []
        for nn in range(nhalf):
            # fetch neutrino indices
            in0 = labels[2 * nn]
            in1 = labels[2 * nn + 1]
            # corresponding qubit indices
            iq0 = qubit_list[2 * nn]
            iq1 = qubit_list[2 * nn + 1]

            # fetch interaction element
            int_01 = next(
                (item for item in int_terms if ((item["nui"] == in0) and (item["nuj"] == in1))),
                None,
            )

            if int_01 is None:
                if verbose:
                    print("WARNING: ordered pair not found, trying reverse")
                int_01 = next(
                    (item for item in int_terms if ((item["nui"] == in1) and (item["nuj"] == in0))),
                    None,
                )
            if verbose:
                print(nn, int_01)

            twob_u = expm(-tstep * 1.0j * sds * J)

            full_gate = np.matmul(twob_u, mSWAP)

            g01 = cirq.MatrixGate(full_gate, name="U_" + str(in0) + str(in1))

            tstep_fo.append(g01.on(iq0, iq1))

            layer_e.append([in0, in1])

            labels[2 * nn] = in1
            labels[2 * nn + 1] = in0
        sequence.append(layer_e)
        ## make an odd layer
        layer_o = []
        for nn in range(nhalf - 1):
            # fetch neutrino indices
            in0 = labels[2 * nn + 1]
            in1 = labels[2 * nn + 2]
            # corresponding qubit indices
            iq0 = qubit_list[2 * nn + 1]
            iq1 = qubit_list[2 * nn + 2]
            # fetch interaction element
            int_01 = next(
                (item for item in int_terms if ((item["nui"] == in0) and (item["nuj"] == in1))),
                None,
            )
            if int_01 is None:
                if verbose:
                    print("WARNING: ordered pair not found, trying reverse")
                int_01 = next(
                    (item for item in int_terms if ((item["nui"] == in1) and (item["nuj"] == in0))),
                    None,
                )
            if verbose:
                print(nn, int_01)
            # build two-qubit unitary

            # build two-qubit unitary
            twob_u = expm(-tstep * 1.0j * sds * J)
            # add SWAP
            full_gate = twob_u * mSWAP

            # make gate
            g01 = cirq.MatrixGate(full_gate, name="U_" + str(in0) + str(in1))

            tstep_fo.append(g01.on(iq0, iq1))
            # save index info
            layer_o.append([in0, in1])

            labels[2 * nn + 1] = in1
            labels[2 * nn + 2] = in0
        sequence.append(layer_o)

    if verbose:
        print(tstep_fo)

    return tstep_fo, sequence


# setup 2nd order Trotter step with swap_network
def get_tstep_circ_2nd(
    tstep, neutrino_number, J, int_terms, labels, qubit_list, verbose=False, cut=None
):
    nhalf = neutrino_number >> 1
    sequence = []
    tstep_so = cirq.Circuit()

    # FORWARD step
    for nl in range(nhalf):
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
                (item for item in int_terms if ((item["nui"] == in0) and (item["nuj"] == in1))),
                None,
            )

            if int_01 is None:
                if verbose:
                    print("WARNING: ordered pair not found, trying reverse")
                int_01 = next(
                    (item for item in int_terms if ((item["nui"] == in1) and (item["nuj"] == in0))),
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
                (item for item in int_terms if ((item["nui"] == in0) and (item["nuj"] == in1))),
                None,
            )
            if int_01 is None:
                if verbose:
                    print("WARNING: ordered pair not found, trying reverse")
                int_01 = next(
                    (item for item in int_terms if ((item["nui"] == in1) and (item["nuj"] == in0))),
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
                (item for item in int_terms if ((item["nui"] == in0) and (item["nuj"] == in1))),
                None,
            )
            if int_01 is None:
                if verbose:
                    print("WARNING: ordered pair not found, trying reverse")
                int_01 = next(
                    (item for item in int_terms if ((item["nui"] == in1) and (item["nuj"] == in0))),
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
                (item for item in int_terms if ((item["nui"] == in0) and (item["nuj"] == in1))),
                None,
            )
            if int_01 is None:
                if verbose:
                    print("WARNING: ordered pair not found, trying reverse")
                int_01 = next(
                    (item for item in int_terms if ((item["nui"] == in1) and (item["nuj"] == in0))),
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
