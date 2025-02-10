"""
Title: Early Fault-Tolerant Quantum Algorithms in Practice: Application to Ground-State Energy Est.
Authors: O. Kiss, U. Azad, B. Requena, A. Roggero, D. Wakeham, J. M. Arrazola
Paper: arXiv:2405.03754 
Year: 2024
Description: This module contains functions to perform the bulk of the presented LT-based algorithm.
"""

import time
import itertools

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from algorithms.utils import (
    compute_d,
    optimal_sample,
    find_first_jump_left,
    find_first_jump_right,
    find_large_variance,
    filter_,
)
from algorithms.Fk import get_F, get_beta


def main(number, initial_state, model, step):
    initial_state += model
    energies_dmrg = np.load(f"data/energies_dmrg_qc{model}.npy")
    print(energies_dmrg)

    ##################################
    ####  algorithm  configuration ###
    try:
        moments = np.load(f"results/{number}/moments_{initial_state}_{step}.npy", allow_pickle=True)
    except FileNotFoundError:
        print(f"File results/{number}/moments_{initial_state}_{step}.npy not found.")
        return

    tau = np.load(f"results/{number}/tau.npy")
    print(tau)

    delta = abs(1 / 10**2)
    epsilon = 0.01

    theta = 0.01  # 1%

    # eta unknown
    eta = 10**-1

    beta = get_beta(epsilon, delta)
    print(beta)
    # beta = max(beta, 100000)

    d = compute_d(tau, epsilon)
    d = len(moments[0]) - 1
    d = int(d)
    print(eta, tau, epsilon, delta, beta, d)

    ########
    D_space = [int(d / 2), int(d / 4), int(d / 10), int(d / 20)][:1]
    print(D_space)
    sample_space = [0, 100, 1000, 10000]

    #######
    np.save(f"results/{number}/Depths_{initial_state}_{step}.npy", D_space)
    np.save(f"results/{number}/Samples_{initial_state}_{step}.npy", sample_space)

    energy_guess_rpt = []
    energy_guess_var = []

    for ddd, d in enumerate(D_space):

        if d > len(moments[0]):
            continue
        energy_guess_rpt.append([])
        energy_guess_var.append([])

        K = np.arange(0, d + 1)
        Fk = [get_F(k, beta, d) for k in K]
        Fk = np.array(Fk)  # get the coefficients

        norm_Fk = np.sum(np.absolute(Fk))

        M_optimal = optimal_sample(np.sum(np.absolute(Fk)), delta, eta, theta)
        print(f"d = {d}, M optimal = {M_optimal}")

        bound = (
            lambda d: 2.07
            / (2 * np.pi)
            * (2 * np.log2(2) + np.log(d + 0.5) + 0.577 + 0.5 / (d + 0.5))
        )

        for _i, N in enumerate(tqdm(sample_space)):
            print(N)
            energy_guess_rpt[-1].append([])
            energy_guess_var[-1].append([])

            for run in range(10):  # do statistics over sampling

                norm_Fk = np.sum(np.absolute(Fk))
                # v = norm_Fk**2/N

                probs_Fk = (np.absolute(Fk) / norm_Fk).real

                if N == 0:
                    # exact signal
                    ks = np.arange(0, d + 1)
                    coeffs = np.sqrt(Fk[ks].real ** 2 + Fk[ks].imag ** 2)
                    shots = [0]
                else:
                    # sampled signal
                    np.random.seed(12 * run + 48)

                    sample_k = list(
                        np.random.choice(np.arange(d + 1), size=N, p=probs_Fk, replace=True)
                    )  # sample
                    sample_k.sort()

                    elements = np.array(
                        [[g[0], len(list(g[1]))] for g in itertools.groupby(sample_k)]
                    )
                    ks = elements[:, 0]
                    shots = elements[:, 1]

                    coeffs = list(norm_Fk / N * np.ones_like(ks))

                ### time series ####
                # e^{-iH tau k}

                prob_real = (moments[0, :] + 1) / 2
                prob_imag = (moments[1, :] + 1) / 2

                gk_real_exacts = moments[0, :]
                gk_imag_exacts = moments[1, :]
                window = np.ones((d + 1))

                if np.sum(shots) == 0:
                    gk_real = gk_real_exacts[ks]
                    gk_imag = gk_imag_exacts[ks]
                else:

                    pseudo_prob = np.array(
                        [
                            np.sum(2 * np.random.binomial(1, p=prob_real[ks[i]], size=shots[i]) - 1)
                            for i in range(len(shots))
                        ]
                    )

                    gk_real = pseudo_prob

                    pseudo_prob = np.array(
                        [
                            np.sum(2 * np.random.binomial(1, p=prob_imag[ks[i]], size=shots[i]) - 1)
                            for i in range(len(shots))
                        ]
                    )
                    gk_imag = pseudo_prob

                ### ACDF ###

                gap = min(0.015, abs(energies_dmrg[-1] - energies_dmrg[0])) / tau

                energies = np.linspace(
                    tau * (energies_dmrg[-1] - gap), tau * (energies_dmrg[0] + 1 * gap), 10000
                )

                energies = np.linspace(-np.pi / 2, np.pi / 2, 10000)
                energies = np.linspace(-3 * tau, 1 * tau, 10000)

                J = 2 * np.array(ks) + 1

                if len(J) < 10**7 or True:

                    out = np.outer(J, energies)

                    __ = time.time()
                    acdf = 0.5 + 2 * np.array(
                        np.einsum("i, ij", coeffs * gk_real, np.sin(out))
                        + np.einsum("i, ij", coeffs * gk_imag, np.cos(out))
                    )
                else:

                    acdf = 0.5 * np.ones_like(energies)
                    for eidx, energy in enumerate(energies):
                        acdf[eidx] += 2 * np.dot(coeffs * gk_real, np.sin(J * energy))
                        acdf[eidx] += 2 * np.dot(coeffs * gk_imag, np.cos(J * energy))

                plt.figure()
                plt.plot(energies, acdf, label="ACDF")

                bonds = [2, 5, 10, 20, 50, 100, 200]
                for bd, e_ in enumerate(energies_dmrg):
                    plt.vlines(
                        e_ * tau, 0, 1, color="k", label=r"DMRG$ (\chi={})$".format(bonds[bd])
                    )

                np.save(
                    f"results/{number}/acdf_{initial_state}_{step}_{d}_{N}_{run}.npy",
                    acdf,
                )
                np.save(
                    f"results/{number}/energies_{initial_state}_{step}_{d}_{N}_{run}.npy",
                    energies,
                )

                final_guess = []
                if False:
                    guess_arg_0_left = find_first_jump_left(acdf, alpha=0.1)
                    guess_arg_0_right = find_first_jump_right(acdf, alpha=0.1)
                    guess_arg_0 = guess_arg_0_left

                    v = np.std(acdf[:guess_arg_0])

                    guess_arg_1 = find_large_variance(acdf, v, s=3)
                    guess_arg_2 = filter_(acdf)

                    guess_arg = [guess_arg_0, guess_arg_1, guess_arg_0]
                    guess = []
                    final_guess = []
                    for guess_arg_value in guess_arg:
                        if guess_arg_value == len(energies):
                            guess_arg_value -= 1

                        guess = energies[guess_arg_value]

                        new_energies = np.linspace(guess - 1 * delta, guess + 1 * delta, 1000)
                        new_energies = np.linspace(
                            tau * (energies_dmrg[-1] - 0.08 * gap),
                            tau * (energies_dmrg[-1] + 0.4 * gap),
                            4000,
                        )
                        new_energies = np.linspace(-5 * tau, -3.5 * tau, 4000)
                        out = np.outer(J, new_energies)
                        grad_acdf = np.array(
                            np.einsum("i, ij", J * gk_real, np.cos(out))
                            - np.einsum("i, ij", J * gk_imag, np.sin(out))
                        )
                        grad_acdf = grad_acdf / abs(max(grad_acdf))

                        final_guess.append(new_energies[np.argmax(grad_acdf)])
                        break

                else:

                    trial = np.array(
                        [
                            -1.935093509350935,
                            -0.9057105710571058,
                            -1.8055605560556056,
                            -1.812121212121212,
                        ]
                    ).reshape(1, -1)

                    new_energies = np.linspace(
                        tau * (energies_dmrg[-1] - 0.05 * gap),
                        tau * (energies_dmrg[-1] + 0.4 * gap),
                        4000,
                    )

                    new_energies = np.linspace(
                        (trial[ddd, _i] - 0.1) * tau, (trial[ddd, _i] + 0.1) * tau, 4000
                    )

                    out = np.outer(J, new_energies)
                    grad_acdf = np.array(
                        np.einsum("i, ij", J * gk_real, np.cos(out))
                        - np.einsum("i, ij", J * gk_imag, np.sin(out))
                    )
                    grad_acdf = grad_acdf / abs(max(grad_acdf))

                    final_guess.append(new_energies[np.argmax(grad_acdf)])

                plt.plot(new_energies, grad_acdf, ".", color="red", label="gradient", markersize=1)
                plt.legend()
                plt.savefig(f"results/{number}/acdf_{initial_state}_{step}_{d}_{N}_{run}.png")

                np.save(
                    f"results/{number}/grad_acdf_{initial_state}_{step}_{d}_{N}_{run}.npy",
                    grad_acdf,
                )
                np.save(
                    f"results/{number}/grad_energies_{initial_state}_{step}_{d}_{N}_{_i}.npy",
                    new_energies,
                )
    np.save(f"results/{number}/guess_energies_rpt_{initial_state}_{step}.npy", new_energies)


if __name__ == "__main__":
    model = "_6_-1_1"
    states = ["dmrg_0"]
    number = 32
    nbr_steps = [0, 10]
    for Is in states:
        for step in nbr_steps:
            main(number, Is, model, step)
