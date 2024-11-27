import numpy as np
from algorithms.Fk import get_F, get_beta
# from Hamiltonian import construct, time_evolve, get_state
# from qcircuit import circuits
import scipy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import itertools
from algorithms.utils import *
import ruptures as rpt

# from compressed_sensing import interpolate_signal, frequencies_filter, CS_filter
# from algorithms.trendfilter import trend_filter

def main(number, initial_state,model, step):
    initial_state += model

    energies_dmrg = np.load('data/energies_dmrg_qc{}.npy'.format(model))

    # energies_dmrg = np.load('results/11/ew.npy')
    # energies_dmrg = np.array([-1.6861406616345072,1.1861406616345072])
    print(energies_dmrg)

    ##################################
    ####  algorithm  configuration ###
    moments = np.load('results/{}/moments_{}_{}.npy'.format(number,initial_state,step),allow_pickle=True)


    tau = np.load('results/{}/tau.npy'.format(number))
    print(tau)
    # tau = np.pi/(20)
    # tau = 0.6283185307179586

    # d = 2000
    #
    # moments = np.array([np.sqrt(1/2)*np.exp(-1.j*0.3*tau*n) +np.sqrt(1/2)*np.exp(+1.j*0.5*tau*n) for n in range(1,2*d+1,2)])
    # moments = np.array([moments.real,moments.imag])
    # #
    # plt.plot(moments[0,:])
    # plt.show()




    # moments = np.array([np.load('results/2/prob_real_dmrg_200.npy',allow_pickle=True),np.load('results/2/prob_imag_dmrg_200.npy',allow_pickle=True)])



    delta = abs(1/10**2)
    epsilon = 0.01

    theta = 0.01 #1%

    # eta unknown
    eta = 10**-1

    beta = get_beta(epsilon,delta)
    print(beta)
    beta = 100000
    d = compute_d(tau,epsilon)

    d = len(moments[0])-1
    d = int(d)
    print(eta,tau,epsilon, delta, beta, d)

    ########
    D_space = [int(d/2),int(d/4), int(d/10),int(d/20)][:1]
    print(D_space)
    sample_space = [0,100,1000,10000]
    # sample_space = [0,1000]
    #######
    np.save('results/{}/Depths_{}_{}.npy'.format(number,initial_state,step),D_space)
    np.save('results/{}/Samples_{}_{}.npy'.format(number,initial_state,step),sample_space)

    energy_guess_rpt = []
    energy_guess_var = []

    for ddd,d in enumerate(D_space):

        if d>len(moments[0]):
            continue
        energy_guess_rpt.append([])
        energy_guess_var.append([])

        K = np.arange(0,d+1)
        Fk = [get_F(k,beta,d) for k in K]
        Fk = np.array(Fk)  #get the coefficients


        norm_Fk   = np.sum(np.absolute(Fk))


        M_optimal = optimal_sample(np.sum(np.absolute(Fk)), delta, eta, theta)
        print('d = {}, M optimal = {} '.format(d,M_optimal))



        bound = lambda d : 2.07/(2*np.pi)*(2*np.log2(2)+np.log(d+0.5)+0.577 + 0.5/(d+0.5))


        for _i,N in enumerate(tqdm(sample_space)):
            print(N)
            energy_guess_rpt[-1].append([])
            energy_guess_var[-1].append([])

            for run in range(10):  #do statistics over sampling

                norm_Fk   = np.sum(np.absolute(Fk))
                # v = norm_Fk**2/N

                probs_Fk  = (np.absolute(Fk)/norm_Fk).real




                if N == 0:
                    # exact signal
                    ks = np.arange(0,d+1)
                    coeffs = np.sqrt(Fk[ks].real**2 + Fk[ks].imag**2)
                    shots = [0]
                else:
                    #sampled signal
                    np.random.seed(12*run+48)

                    sample_k  = list(np.random.choice(np.arange(d+1),size = N, p=probs_Fk,replace=True)) #sample
                    sample_k.sort()

                    elements  = np.array([[g[0], len(list(g[1]))] for g in itertools.groupby(sample_k)])
                    ks = elements[:,0]
                    shots = elements[:,1]

                    coeffs    = list(norm_Fk/N *np.ones_like(ks))


                ### time series ####
                # e^{-iH tau k}

                prob_real = (moments[0,:]+1)/2
                prob_imag = (moments[1,:]+1)/2

                gk_real_exacts = moments[0,:]
                gk_imag_exacts = moments[1,:]
                window = np.ones((d+1))

                if np.sum(shots) == 0:
                    gk_real = gk_real_exacts[ks]
                    gk_imag = gk_imag_exacts[ks]
                else:

                    pseudo_prob = np.array([
                                            np.sum(2*np.random.binomial(1,p=prob_real[ks[i]],size = shots[i])-1)
                                            for i in range(len(shots))])

                    gk_real = pseudo_prob

                    pseudo_prob = np.array([
                                            np.sum(2*np.random.binomial(1,p=prob_imag[ks[i]],size = shots[i])-1)

                                            for i in range(len(shots))])
                    gk_imag = pseudo_prob



                ### ACDF ###

                gap=min(0.015,abs(energies_dmrg[-1]-energies_dmrg[0]))/ tau

                energies = np.linspace(tau*(energies_dmrg[-1]-gap),tau*(energies_dmrg[0]+1*gap),10000)

                energies = np.linspace(-np.pi/2,np.pi/2,10000)
                energies = np.linspace(-3*tau,1*tau,10000)

                J = 2*np.array(ks)+1

                if len(J)<10**7 or True:

                    out = np.outer(J,energies)

                    __ = time.time()
                    acdf = 0.5 + 2*np.array(np.einsum('i, ij', coeffs*gk_real, np.sin(out))
                                          + np.einsum('i, ij', coeffs*gk_imag, np.cos(out)) )
                else:

                    acdf = 0.5*np.ones_like(energies)
                    for e_ in range(len(energies)):
                        acdf[e_] += 2* np.dot(coeffs*gk_real,np.sin(J*energies[e_]))
                        acdf[e_] += 2* np.dot(coeffs*gk_imag,np.cos(J*energies[e_]))

                plt.figure()
                plt.plot(energies,acdf,label='ACDF')
                # plt.ylim(0,1)
                __ = 0
                bonds = [2,5,10,20,50,100,200]
                for bd,e_ in enumerate(energies_dmrg):
                    plt.vlines(e_*tau,0,1,color='k',label=r'DMRG$ (\chi={})$'.format(bonds[bd]))

                    __ +=1
                    # if __==1:
                    #     break
                # plt.xlim(energies_dmrg[-1]*tau,energies_dmrg[0]*tau)
                plt.ylim(-0.01,1)
                # plt.xlim(-1.6,-0.5)

                # plt.show()


                np.save('results/{}/acdf_{}_{}_{}_{}_{}.npy'.format(number,initial_state,step,d,N,run),acdf)
                np.save('results/{}/energies_{}_{}_{}_{}_{}.npy'.format(number,initial_state,step,d,N,run),energies)

                final_guess = []
                if False:
                    guess_arg_0_left = find_first_jump_left(acdf, alpha = 0.1)
                    guess_arg_0_right = find_first_jump_right(acdf, alpha = 0.1)
                    guess_arg_0 = guess_arg_0_left

                    v = np.std(acdf[:guess_arg_0])

                    guess_arg_1 = find_large_variance(acdf,v,s=3)
                    guess_arg_2 = filter_(acdf)

                    guess_arg = [guess_arg_0,guess_arg_1,guess_arg_0]
                    guess = []
                    final_guess = []
                    for g in range(len(guess_arg)):
                        if guess_arg[g]==len(energies):
                            guess_arg[g] -=1

                        guess = energies[guess_arg[g]]
                        # guess = -0.61020815

                        new_energies = np.linspace(guess-1*delta,guess+1*delta,1000)
                        new_energies = np.linspace(tau*(energies_dmrg[-1]-0.08*gap),tau*(energies_dmrg[-1]+0.4*gap),4000)
                        new_energies = np.linspace(-5*tau,-3.5*tau,4000)
                        out =  np.outer(J,new_energies)
                        grad_acdf = np.array(np.einsum('i, ij', J*gk_real, np.cos(out))
                                              - np.einsum('i, ij', J*gk_imag, np.sin(out)) )
                        grad_acdf = grad_acdf/abs(max(grad_acdf))

                        final_guess.append(new_energies[np.argmax(grad_acdf)])
                        break
                else:

                    trial = np.array([-1.935093509350935,-0.9057105710571058,-1.8055605560556056,-1.812121212121212]).reshape(1,-1)
                    # trial = np.array([[-4.729272927292729,-4.777577757775776,-4.784238423842384,-4.729272927292729],
                    # [-4.726272627262727,-4.753875387538754,-4.810781078107811,-4.783878387838785],
                    # [-4.557744663355226,-4.750675067506751,-4.794879487948796,-4.784878487848785],[
                    # -4.678167816781679,-4.775277527752776,-4.859485948594861,-4.797879787978799]
                    # ])
                    # trial = np.array([-4.78,-4.679979109022014,-4.78,-4.78]).reshape(1,-1)
                    # trial = np.array([-4.42,-4.79,-4.7832,-4.78]).reshape(1,-1)
                    # trial = np.array([[-4.368736873687369,-4.790879087908791,-4.783278327832784,-4.783878387838785],
                    # [-4.502050205020502,-4.711071107110712,-4.783878387838785,-4.783878387838785],
                    # [-4.4198419841984204,-4.707570757075709,-4.786478647864787,-4.784878487848785],[
                    # -4.5587558755875595,-4.747574757475748,-4.781578157815782,-4.781878187818782]])
                    # trial = [-4.292529252925293,-2.473597359735974,-3.0116011601160113,-3.6340934093409336]
                    # trial = np.array([[-3.795879587958796,-4.57965796579658,-4.751075107510752, -4.783878387838785],
                    # [-3.788778877887789,-4.604460446044605,-4.727172717271728,-4.783878387838785],
                    # [-4.06,-4.695569556955696,-4.735973597359737,-4.784973597359737],
                    # [-4.3286328632863285,-4.793879387938795,-4.9150915091509155,-4.796879687968798]])
                    new_energies = np.linspace(tau*(energies_dmrg[-1]-0.05*gap),tau*(energies_dmrg[-1]+0.4*gap),4000)
                    # new_energies = np.linspace(-4.9*tau,-4.2*tau,4000)
                    new_energies = np.linspace((trial[ddd,_i]-0.1)*tau,(trial[ddd,_i]+0.1)*tau,4000)
                    # if _i==0:
                    #     new_energies = np.linspace((trial[_i]-0.1)*tau,(trial[_i]+0.25)*tau,4000)
                    # print(new_energies/tau)


                    out =  np.outer(J,new_energies)
                    grad_acdf = np.array(np.einsum('i, ij', J*gk_real, np.cos(out))
                                          - np.einsum('i, ij', J*gk_imag, np.sin(out)) )
                    grad_acdf = grad_acdf/abs(max(grad_acdf))

                    final_guess.append(new_energies[np.argmax(grad_acdf)])

                plt.plot(new_energies,grad_acdf,'.',color='red',label='gradient',markersize=1)
                plt.legend()
                plt.savefig('results/{}/acdf_{}_{}_{}_{}.png'.format(number,initial_state,step,d,N))
                # plt.show()
                # energy_guess_rpt[-1][-1].append(final_guess[0]/tau)
                # energy_guess_var[-1][-1].append(final_guess[1]/tau)

                # print(d,N,_i,new_energies[0],'results/{}/grad_energies_{}_{}_{}_{}_{}.npy'.format(number,initial_state,step, d, N,_i))
                np.save('results/{}/grad_acdf_{}_{}_{}_{}_{}.npy'.format(number,initial_state, step,d,N,run),grad_acdf)
                np.save('results/{}/grad_energies_{}_{}_{}_{}_{}.npy'.format(number,initial_state,step, d, N,_i),new_energies)
    np.save('results/{}/guess_energies_rpt_{}_{}.npy'.format(number,initial_state, step),new_energies)
    # np.save('results/{}/grad_energies_{}_{}.npy'.format(number, initial_state, step),new_energies)




if __name__ == '__main__':
    model = '_6_-1_1'
    states = ['dmrg_0']
    number = 32
    nbr_steps = [0,10]
    for Is in states:
        for step in nbr_steps:
            main(number, Is,model ,  step)
