import numpy as np
import matplotlib.pyplot as plt
import tqdm
import scipy

# from algorithms.tcircuits import spin_lattice_circuit
from pennylane_circuits import spin_lattice_circuit
# from algorithms.state_preparation import *
# import cirq
# import qsimcirq
import GPUtil
import os
import rustworkx as rx
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    HyperCubicLattice,
    Lattice,
    LatticeDrawStyle,
    LineLattice,
    SquareLattice,
    TriangularLattice,
)
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel, HeisenbergModel
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import SpinOp
from qiskit.quantum_info import SparsePauliOp
# import qiskit




# if len(GPUtil.getAvailable())==0:
#     print('CPU')
#     options = {'t': 8,'f':3}
#     qsim_simulator = qsimcirq.QSimSimulator(options)
#
# else:
#     print('GPU available :) !')
#     options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode = 0, max_fused_gate_size=3)
#     qsim_simulator = qsimcirq.QSimSimulator(options)

def Hamiltonian(coupling, field):
    rows, cols = np.shape(field)
    boundary_condition = BoundaryCondition.OPEN
    lattice = SquareLattice(rows=rows, cols=cols, boundary_condition=boundary_condition)

    H = HeisenbergModel(lattice, coupling_constants= coupling, ext_magnetic_field=(0.0, 0.0, 0.0))
#     H = FermiHubbardModel(lattice.uniform_parameters(
#         uniform_interaction=-1.0,
#         uniform_onsite_potential=0.0,
#     ),
#     onsite_interaction=5.0,
# )

    H = H.second_q_op().simplify()
    # qubit_mapper = JordanWignerMapper()
    # H = qubit_mapper.map(H)
    print(H)
    for i in range(rows):
        for j in range(cols):
            op =  SpinOp({"Z_{}".format(i*rows+j): field[i,j]})
            H += op

    # # sparse
    paulis = []
    coeffs = []
    for t in H.terms():
        if (t[0][0][0]) != 'X':
            continue

        if len(t[0])==2:
            p = (t[0][0][1],t[0][1][1])
        #     p = 'I'*(t[0][0][1]) + t[0][0][0] + 'I'*(t[0][1][1]-t[0][0][1]-1) + t[0][1][0]
        # p = p+ (rows*cols-len(p))*'I'
        # coeffs.append(float(t[1].real))
        paulis.append(p)


    return H, paulis



def exact_dynamics(state,ew,ev, times):

    state = np.dot(ev.conjugate().transpose(),state.copy())

    values = []
    for t in times:
        op = np.diag(np.exp(-1.j*ew*t))
        values.append(np.dot(state.copy().conjugate().transpose(),np.dot(op,state.copy())))
    return np.array(values)



def main(number,initial_state, num_steps):


    ######## HAMILTONIAN #######
    n,m = 4,4
    Jx= -1
    Jz = 1
    coupling = -np.array(([Jx,Jx,Jz]))
    # problem with coupling in Z
    if initial_state[:4]=='dmrg':
        initial_state += '_{}_{}_{}'.format(n*m,Jx,Jz)

    wf = np.load('data/{}.npy'.format(initial_state))
    # wf = np.random.uniform(-1,1,size = (2**(n*m)))
    # wf = wf/np.linalg.norm(wf)
    # wf = np.zeros_like(wf)
    # wf[0]=1
    np.random.seed(42)
    field = np.random.uniform(-1,1,(n,m))


    ###########################
    H,paulis_coupling = Hamiltonian(coupling,field)


    field = field/2
    coupling = coupling/(n*m)
    ###########################

    # ham = H.to_matrix()

    print(np.sum(np.absolute(field)))
    norm = np.sum(np.absolute(field)) + (2*abs(Jx)+abs(Jz))*len(paulis_coupling)
    norm = 20 
    # norm = scipy.sparse.linalg.norm(ham, ord = 1)
    # norm = np.linalg.norm(ham,ord=2)
    print('norm {}'.format(norm))
    # norm = 12
    tau = np.pi/(2*norm)

    print('tau:', tau)
    epsilon = 10**-2

    d = np.sqrt(2)/(tau*epsilon)*np.log(4*np.sqrt(2*np.pi/(tau*epsilon))*(2+tau/epsilon))
    print('d :',d)
    d = min(d,4000)
    time_interval = tau*np.arange(1,2*int(d)+1,2)

    order = 1

    # moments = [np.dot(state.conjugate().transpose(),np.dot(np.diag(np.exp(-1.j*tau*n*ew)),state)) for n in range(1,2*d+1,2)]

    if num_steps ==0:
        ### exact ###
        ham = H.to_matrix()
        ew, ev = np.linalg.eigh(ham)


        res = exact_dynamics(wf,ew,ev,time_interval)
        results = np.array([res.real,res.imag])

        np.save('results/{}/ew.npy'.format(number),ew)
    else:
        wf_initial = wf.copy()
        results = np.zeros((2,len(time_interval)))
        for _,tt in tqdm.tqdm(enumerate(time_interval)):
            dt = tt/num_steps# WTF is happening here
            dt = time_interval[1]/num_steps


            # dt = tau/num_steps

            wf = spin_lattice_circuit(coupling, field, wf.copy(), dt.copy(), num_steps, order, paulis_coupling )

            overlap = np.dot(wf.conjugate().transpose(),wf_initial)

            results[0,_] = float(overlap.real)
            results[1,_] = float(overlap.imag)


    np.save('results/{}/moments_{}_{}.npy'.format(number,initial_state,num_steps),results)
    np.save('results/{}/tau.npy'.format(number),tau)


if __name__ == '__main__':
    number = 5

    try:
        os.mkdir('results/{}'.format(number))
    except:
        pass
    name = 'dmrg_'
    for state in [2]:
        for num_steps in [1,5,10]:
            main(number, name+str(state),num_steps)
