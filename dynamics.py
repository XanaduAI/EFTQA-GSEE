import numpy as np
import matplotlib.pyplot as plt
import tqdm
import scipy

# from algorithms.tcircuits import spin_lattice_circuit
# from algorithms.state_preparation import *
import cirq
import qsimcirq
import GPUtil
import os

import cirq
# from algorithms.neutrino_hamiltonian import get_coupling_matrix, gen_int_list
from algorithms.tcircuits_LMG import get_tevol_circ
from qiskit.opflow import X,Z,Y,I



if len(GPUtil.getAvailable())==0:
    print('CPU')
    options = {'t': 8,'f':3}
    qsim_simulator = qsimcirq.QSimSimulator(options)

else:
    print('GPU available :) !')
    options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode = 0, max_fused_gate_size=3)
    qsim_simulator = qsimcirq.QSimSimulator(options)



def LMG_ham(L,J):
    H = 0*(I^L)

    qubits = cirq.LineQubit.range(L)
    H_cirq = []
    coefficients = []

    print(H_cirq)

    it = 0
    for i in range(L-1):
        for j in range(i+1,L):
            l = j-i
            op_x = (I^(i))^X^(I^(l-1))^X^(I^(L-1-i-l))
            op_y = (I^(i))^Y^(I^(l-1))^Y^(I^(L-1-i-l))
            op_z = (I^(i))^Z^(I^(l-1))^Z^(I^(L-1-i-l))

            H = H+ (J[0,it]*op_x + J[1,it]*op_y +J[2,it]*op_z)

            H_cirq.append(cirq.X(qubits[i])*cirq.X(qubits[j]))
            coefficients.append(J[0,it])
            H_cirq.append(cirq.Y(qubits[i])*cirq.Y(qubits[j]))
            coefficients.append(J[1,it])
            H_cirq.append(cirq.Z(qubits[i])*cirq.Z(qubits[j]))
            coefficients.append(J[2,it])
            it +=1

    # H = J*H
    # for i in range(L):
    #     op_z = (I^(i))^Z^(I^(L-i-1))
    #     H = H + B*op_z
    return H, H_cirq, coefficients





def exact_dynamics(state,ew,ev, times):

    state = np.dot(ev.conjugate().transpose(),state.copy())

    values = []
    for t in times:
        op = np.diag(np.exp(-1.j*ew*t))
        values.append(np.dot(state.copy().conjugate().transpose(),np.dot(op,state.copy())))
    return np.array(values)



def main(number,initial_state, num_steps):


    ######## HAMILTONIAN #######
    L = 10 # number of spins

    # rdm initial state
    np.random.seed(23)
    wf = np.random.uniform(-1,1,size = (2**(L)))
    wf = wf/np.linalg.norm(wf)


    if initial_state[:4]=='dmrg':
        #fix the name with your initial state
        # initial_state += '_{}_{}_{}'.format(L,Jx,Jz)
        # wf = np.load('data/{}.npy'.format(initial_state)).reshape(-1)
        pass


    #sparsification

    # eps = 10**-2
    # wf[np.abs(wf) < eps] = 0
    # index = np.nonzero(wf)
    # # print(index)s
    # print('sparse',len(index[0]))



    #build the Hamiltonian
    dim = int(L*(L-1)/2)
    np.random.seed(42)
    J = np.random.normal(0,1,size=3*dim).reshape(3,dim)/L

    coupling = np.zeros((dim,dim,3))
    it = 0
    for i in range(L-1):
        for j in range(i+1,L):
            coupling[i,j,:] = J[:,it]
            it+=1

    norm = np.sum(np.absolute(J))
    H,H_cirq, H_coeff = LMG_ham(L,J)



    # norm = scipy.sparse.linalg.norm(ham, ord = 1)
    # norm = np.linalg.norm(ham,ord=2)
    print('norm {}'.format(norm))
    #norm = 10 #### her use info from DMRG
    # norm = 20
    tau = np.pi/(2*norm)

    print('tau:', tau)
    epsilon = 10**-2

    d = np.sqrt(2)/(tau*epsilon)*np.log(4*np.sqrt(2*np.pi/(tau*epsilon))*(2+tau/epsilon))
    print('d :',d)
    d = min(d,20000)
    time_interval = tau*np.arange(1,2*int(d)+1,2)

    order = 2

    # moments = [np.dot(state.conjugate().transpose(),np.dot(np.diag(np.exp(-1.j*tau*n*ew)),state)) for n in range(1,2*d+1,2)]
    # compute moments
    if num_steps ==0:
        ### exact evolution ###
        # print(H)
        ham = H.to_matrix()
        # print(ham)
        # print(ham.shape)
        ew, ev = np.linalg.eigh(ham)
        print('EW: ', ew)
        p = [np.dot(wf.copy().transpose().conjugate(),ev[:,i])**2 for i in range(2**L)]

        # print(np.sum(p))
        np.save('results/{}/overlap.npy'.format(number),p)

        res = exact_dynamics(wf.copy(),ew,ev,time_interval)
        results = np.array([res.real,res.imag])

        np.save('results/{}/ew.npy'.format(number),ew)
    else:
        #Trotter
        qubits = cirq.LineQubit.range(L)
        labels = np.arange(L)
        # wf = np.zeros((2**(n*m)))
        # wf[0] = 1
        wf = np.array(wf,dtype='complex64')
        wf_initial = wf.copy()

        # vij = get_coupling_matrix(L,29)
        vij = coupling


        int_terms, int_1norm, int_prob = gen_int_list(L,vij)

        # print(int_terms)

        circ_params = {
        "dt": 0,
        "order":2,
        "num_steps": num_steps,
        "J":coupling,
        "neutrino_number": L,
        "int_terms": int_terms,
        "labels": labels,
        "qubit_list": qubits,
        "echo": False,
        "verbose": False,
                  }

        results = np.zeros((2,len(time_interval)))

        tt_time = []
        total = 0

        full_circuit = get_tevol_circ(circ_params)
        res = qsim_simulator.simulate_expectation_values(full_circuit,observables = H_cirq, initial_state=wf.copy())
        res = np.dot(np.array(H_coeff), res)
        print('initial_state energy: ', res)
        np.savetxt('results/{}/is_energy.txt'.format(number), np.array(res).reshape(-1))

        for _,tt in tqdm.tqdm(enumerate(time_interval)):
            # dt = tt/num_steps# WTF is happening here
            # dt = 12*tau/num_steps
            if _==0:
                dt = tau
            else:
                dt = 2*tau
            # total+= dt
            # tt_time.append(total)
            dt = dt/num_steps
            circ_params['dt'] = dt



            # full_circuit, observable, qubits = spin_lattice_circuit(coupling, field, dt, num_steps, order, paulis_coupling, qubits  )
            full_circuit = get_tevol_circ(circ_params)

            res = qsim_simulator.simulate(full_circuit,initial_state=wf.copy())
            # res = qsim_simulator.simulate(full_circuit)
            wf = res.final_state_vector
            overlap = np.dot(wf,wf_initial.conjugate().transpose())
            results[0,_] = overlap.real
            results[1,_] = overlap.imag
            # for part in range(2):
            #     if part ==1:
            #         full_circuit.append(Sdagger(qubits[-1]))
            #     full_circuit.append(cirq.H(qubits[-1]))
            #     res = qsim_simulator.simulate_expectation_values(full_circuit, observable)
            #
            #     results[part,_] = float(res[0].real)


    np.save('results/{}/moments_{}_{}.npy'.format(number,initial_state,num_steps),results)
    np.save('results/{}/tau.npy'.format(number),tau)
    print('results/{}/moments_{}_{}.npy'.format(number,initial_state,num_steps))


if __name__ == '__main__':
    number = 32

    try:
        os.mkdir('results/{}'.format(number))
    except:
        pass
    name = 'dmrg_'
    for state in [0]:
        for num_steps in [0,10]:
            main(number, name+str(state),num_steps)
