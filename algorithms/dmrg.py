import tenpy
import numpy as np
import pennylane as qml
import qiskit
from tenpy.networks.site import SpinSite, SpinHalfSite
from tenpy.models.lattice import Chain, Square, TrivialLattice
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, exact_diag
from tenpy.algorithms import tebd
from tenpy.algorithms.dmrg import SubspaceExpansion, DensityMatrixMixer
from tenpy.models.spins import SpinChain, SpinModel



class XXZChain(SpinModel):
    r"""Spin-1/2 XXZ chain with Sz conservation.

    The Hamiltonian reads:

    .. math ::
        H = \sum_i \mathtt{Jxx}/2 (S^{+}_i S^{-}_{i+1} + S^{-}_i S^{+}_{i+1})
                 + \mathtt{Jz} S^z_i S^z_{i+1} \\
            - \sum_i \mathtt{hz} S^z_i

    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    L : int
        Length of the chain.
    Jxx, Jz, hz : float | array
        Couplings as defined for the Hamiltonian above.
    bc_MPS : {'finite' | 'infinte'}
        MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    """
    def __init__(self, params):
        # 0) read out/set default parameters
        name = "XXZChain"
        n = params['n']
        m =params['m']
        Jxx = params['Jxx']
        Jz = params['Jz']
        hz = params['hz']

        bc_MPS = 'finite'



        # 1-3):
        site = SpinHalfSite()
        # 4) lattice
        bc = 'open'
        if False:
            lat = Square(n,m, site, bc=bc, bc_MPS=bc_MPS)


            print(lat)
            # 5) initialize CouplingModel
            CouplingModel.__init__(self, lat)
            nbr = lat.coupling_shape(np.array([1]))
            print(nbr)
            np.random.seed(42)
            J = np.random.normal(0,1,size=3*5).reshape(3,5,1)
            # h  =np.random.uniform(-1,1,n*m).reshape(-1)
            # 6) add terms of the Hamiltonian
            #
            # for u in range(len(self.lat.unit_cell)):
            #
            #     self.add_onsite(hz, u, 'Sz')
            print(J[0,:]+J[1,:])
            for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
                print(u1,u2,dx)

                self.add_coupling_term(8, 0,1, 'Sp',  'Sm', plus_hc=True)
                # self.add_coupling(J[0,:]-J[1,:], u1, 'Sp', u2, 'Sp', dx,plus_hc=True)

                # self.add_coupling(0.5*Jxx, u1, 'Sp', u2, 'Sm', dx)
                # self.add_coupling(0.5*np.conj(Jxx), u2, 'Sp', u1, 'Sm', -dx)  # h.c.
                # self.add_coupling(J[2,:], u1, 'Sz', u2, 'Sz', dx)
        else:
            L = n*m
            bc = 'open'
            site = SpinHalfSite(conserve=None)
            site = SpinSite(S=0.5, conserve=None)  # You can adjust S (spin) and conserve as needed

            dim = int(L*(L-1)/2)
            np.random.seed(42)
            J = np.random.normal(0,1,size=3*dim).reshape(3,dim)/L



            print(np.sum(np.absolute(J)))

            # lat = Chain(L, spin_site, bc=bc, bc_MPS=bc_MPS)
            lat = Chain(L, site, bc=bc, bc_MPS=bc_MPS)

            CouplingModel.__init__(self, lat)


            # self.add_exponentially_decaying_coupling((Jx+Jy), 1,'Sp','Sm',plus_hc=True)
            # self.add_exponentially_decaying_coupling((Jx-Jy), 1,'Sp','Sp',plus_hc=True)
            # self.add_exponentially_decaying_coupling(Jy/2, 1,'Sm','Sp',plus_hc=True)
            it = 0
            for i in range(L-1):
                for j in range(i+1,L):
                    Jp = J[0,it]+J[1,it]
                    Jm = J[0,it]-J[1,it]
                    Jz = J[2,it]

                    self.add_coupling_term(Jp, i,j, 'Sp',  'Sm', plus_hc=True)
                    self.add_coupling_term(Jm, i,j, 'Sp',  'Sp', plus_hc=True)
                    self.add_coupling_term(2*Jz, i,j, 'Sz',  'Sz', plus_hc=True)
                    it+=1


            # for u in range(len(self.lat.unit_cell)):
            #     self.add_onsite(-2*B, u, 'Sz')



            # for i in range(L-1):
            #     for j in range(i+1,L):
            #         dx = np.array([[1]])
            #
            #         self.add_coupling(J, i, 'Sp', j, 'Sm', dx=dx)
            #         self.add_coupling(J, j, 'Sp', i, 'Sm', dx=-dx)
            #         self.add_coupling(J, i, 'Sm', j, 'Sp', dx=dx)
            #         self.add_coupling(J, j, 'Sm', i, 'Sp', dx=-dx)
            #

        # 7) initialize H_MPO

        # for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
        #     self.add_coupling(t, u1, 'Cd', u2, 'C', dx, plus_hc=True)

        MPOModel.__init__(self, lat, self.calc_H_MPO())
        # 8) initialize H_bond (the order of 7/8 doesn't matter)
        # NearestNeighborModel.__init__(self, lat, self.calc_H_bond())


n,m = 1,10
L = n*m
Jx = -1
Jz  = 1



np.random.seed(42)
field = -np.random.uniform(-1,1,size=(n,m))
# field = np.ones_like(field)


params = {'n':n,'m':m,'Jxx':Jx, 'Jz':Jz, 'hz':field}
model = XXZChain(params)

sites = model.lat.mps_sites()

# Create a SpinSite
# Define the parameters of the model




product_state = []
for i in range(n*m):
    # product_state.append(np.random.choice(["up", "down"]))

    if i<L/2:
        product_state.append("up")
    else:
        product_state.append("down")

product_state = ['up','down']*int(((n*m)/2))


psi = MPS.from_product_state(sites,product_state,"finite")

energies = []
### EXACT DIAG ###

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
print(10*'#', 'DMRG')

bond_dimension = [10]
for bd in bond_dimension:
    del psi
    psi = MPS.from_product_state(sites,product_state,"finite")

    dmrg_params = {"trunc_params": {"chi_max": bd, "svd_min": 1.e-10}, "mixer": True,'max_sweeps': 500}



    info = dmrg.run(psi, model, dmrg_params)

    print("E = ",info['E'])
    print("max. bond dimension = ",max(psi.chi))
    psi.test_sanity()


    wf = psi.get_theta(0,n*m).to_ndarray().reshape(-1)

    np.save('data/dmrg_{}_{}_{}_{}.npy'.format(bd,n*m,Jx,Jz),wf)
    energies.append(info['E'])

    #
    # params['orthogonal_to'] = [psi]
    # results = dmrg.run(psi, model, params)
    # print(results['E'])




np.save('data/energies_dmrg_qc_{}_{}_{}.npy'.format(n*m,Jx,Jz),energies)
