"""
Title: Early Fault-Tolerant Quantum Algorithms in Practice: Application to Ground-State Energy Est.
Authors: O. Kiss, U. Azad, B. Requena, A. Roggero, D. Wakeham, J. M. Arrazola
Paper: arXiv:2405.03754 
Year: 2024
Description: This module provides functions to perform DMRG calculations for a spin-1/2 XXZ chain.
"""
import numpy as np
from tenpy.networks.site import SpinSite, SpinHalfSite
from tenpy.models.lattice import Chain, Square
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.models.spins import SpinModel


class XXZChain(SpinModel):
    """Spin-1/2 XXZ chain with Sz conservation.

    The Hamiltonian reads:

    .. math ::
        H = \sum_i \mathtt{Jxx}/2 (S^{+}_i S^{-}_{i+1} + S^{-}_i S^{+}_{i+1})
                 + \mathtt{Jz} S^z_i S^z_{i+1} \\
            - \sum_i \mathtt{hz} S^z_i

    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Attributes:
        lat: Lattice
            The lattice on which the model is defined.
        J: array
            Randomly generated coupling constants.
    """

    def __init__(self, params):
        """
        Initialize the DMRG algorithm with the given parameters.

        Parameters:
        -----------
        params (dict): Dictionary containing the parameters for the model.
            - 'n' (int): Number of sites in one dimension.
            - 'm' (int): Number of sites in the other dimension.
            - 'Jxx' (array(float)): Coupling constant for the XX interaction.
            - 'Jz' (array(float)): Coupling constant for the ZZ interaction.
            - 'hz' (array(float)): Magnetic field in the z direction.
            - 'bc_MPS' (str): Boundary conditions for MPS, either 'finite' or 'infinite'.

        Notes:
        ------
        This method sets up the lattice and initializes the coupling model
        based on the provided parameters. It supports both square and chain
        lattices, with the latter being used if the square lattice condition
        is not met. Random coupling constants are generated for the interactions.
        """
        # 0) read out/set default parameters
        name = "XXZChain"
        n = params["n"]
        m = params["m"]
        Jxx = params["Jxx"]
        Jz = params["Jz"]
        hz = params["hz"]

        bc_MPS = "finite"

        site = SpinHalfSite()

        bc = "open"
        if False:
            lat = Square(n, m, site, bc=bc, bc_MPS=bc_MPS)

            CouplingModel.__init__(self, lat)
            nbr = lat.coupling_shape(np.array([1]))

            np.random.seed(42)
            J = np.random.normal(0, 1, size=3 * 5).reshape(3, 5, 1)

            for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
                self.add_coupling_term(8, 0, 1, "Sp", "Sm", plus_hc=True)

        else:
            L = n * m
            bc = "open"
            site = SpinHalfSite(conserve=None)
            site = SpinSite(S=0.5, conserve=None)  # You can adjust S (spin) and conserve as needed

            dim = int(L * (L - 1) / 2)
            np.random.seed(42)
            J = np.random.normal(0, 1, size=3 * dim).reshape(3, dim) / L


            lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)

            CouplingModel.__init__(self, lat)

            it = 0
            for i in range(L - 1):
                for j in range(i + 1, L):
                    Jp = J[0, it] + J[1, it]
                    Jm = J[0, it] - J[1, it]
                    Jz = J[2, it]

                    self.add_coupling_term(Jp, i, j, "Sp", "Sm", plus_hc=True)
                    self.add_coupling_term(Jm, i, j, "Sp", "Sp", plus_hc=True)
                    self.add_coupling_term(2 * Jz, i, j, "Sz", "Sz", plus_hc=True)
                    it += 1

        MPOModel.__init__(self, lat, self.calc_H_MPO())


if __name__ == "__main__":
    n, m = 1, 10
    L = n * m
    Jx = -1
    Jz = 1


    np.random.seed(42)
    field = -np.random.uniform(-1, 1, size=(n, m))
    # field = np.ones_like(field)


    params = {"n": n, "m": m, "Jxx": Jx, "Jz": Jz, "hz": field}
    model = XXZChain(params)

    sites = model.lat.mps_sites()

    # Create a SpinSite
    # Define the parameters of the model


    product_state = []
    for i in range(n * m):
        # product_state.append(np.random.choice(["up", "down"]))

        if i < L / 2:
            product_state.append("up")
        else:
            product_state.append("down")

    product_state = ["up", "down"] * int(((n * m) / 2))


    psi = MPS.from_product_state(sites, product_state, "finite")

    energies = []

    ### EXACT DIAG : For testing ###

    # ED = exact_diag.ExactDiag(model)
    # ED.build_full_H_from_mpo()
    # ED.full_diagonalization()
    #
    # print('exact gs energy', min(ED.E))
    # energies.append(min(ED.E))
    # # now ED.E is energies, ED.V has eigenvectors as columns
    # for i in range(ED.V.shape[1]):
    #    psi_ex = ED.V[:, i]

    ##### DMRG ######
    print(10 * "#", "DMRG")

    bond_dimension = [10]
    for bd in bond_dimension:
        del psi
        psi = MPS.from_product_state(sites, product_state, "finite")

        dmrg_params = {
            "trunc_params": {"chi_max": bd, "svd_min": 1.0e-10},
            "mixer": True,
            "max_sweeps": 500,
        }

        info = dmrg.run(psi, model, dmrg_params)

        print("E = ", info["E"])
        print("max. bond dimension = ", max(psi.chi))
        psi.test_sanity()

        wf = psi.get_theta(0, n * m).to_ndarray().reshape(-1)

        np.save("data/dmrg_{}_{}_{}_{}.npy".format(bd, n * m, Jx, Jz), wf)
        energies.append(info["E"])

        #
        # params['orthogonal_to'] = [psi]
        # results = dmrg.run(psi, model, params)
        # print(results['E'])


    np.save("data/energies_dmrg_qc_{}_{}_{}.npy".format(n * m, Jx, Jz), energies)
