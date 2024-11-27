import pennylane as qml
from scipy.linalg import expm
import numpy as np


px = np.array([
    [0., 1.],
    [1.,0.]])
py = np.array([
    [0., -1.j],
    [1.j,0.]])
pz = np.array([
    [1., 0.],
    [0.,-1.]])

pxpx = np.kron(px,px)
pypy = np.kron(py,py)
pzpz = np.kron(pz,pz)


def spin_lattice_circuit(coupling, field, wf, dt, num_steps, order,paulis_coupling ):


    H = coupling[0] * pxpx + coupling[1] * pypy + coupling[2]* pzpz
    two_u = expm(-1.j*dt*H)

    field = field.reshape(-1)
    n = len(field)

    dev = qml.device("lightning.qubit", wires=n)
    @qml.qnode(dev)
    def my_circ(wf,dt,n,two_u,paulis_coupling,field):
        # Prepare some state
        # qml.MottonenStatePreparation(wf, wires = range(n))
        qml.StatePrep(wf,wires=range(n))
        # Evolve according to H
        for _ in range(num_steps):
            apply_field(range(n),field,dt)
            apply_coupling(range(n), two_u, paulis_coupling)
        # Measure some quantity
        return qml.state()

    wf_final = my_circ(wf,dt,n,two_u,paulis_coupling,field)

    return wf_final

def apply_field(qubits, field, dt):

    for i in range(len(field)):
        U = expm(-1.j*dt*field[i]*pz)
        qml.QubitUnitary(U, wires=qubits[i], id="Z")


def apply_coupling(qubits, g2, paulis_coupling):

    for p in paulis_coupling:
        qml.QubitUnitary(g2, wires=[p[0],p[1]], id="U")
