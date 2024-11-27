import numpy as np
import scipy

def get_F(j,beta,d):
    """
    compute the fourrier coefficients of the Heaviside function following arXiv:2110.1207

    Arg:
        j: time index, j \in {0,...,d}
        d: maximal time
        beta: parameter

    Out:
        F_(2j+1)(beta)

   """


    if abs(j)<d:
        coeff  = -1.j*np.sqrt(beta/(2*np.pi))
        bessel = (scipy.special.ive(j,beta) + scipy.special.ive(j+1,beta))/(2*j+1)
        return coeff * bessel

    else:
        coeff  = -1.j*np.sqrt(beta/(2*np.pi))
        bessel = scipy.special.ive(j,beta)/(2*j+1)
        return coeff * bessel


def get_beta(epsilon, delta):
    W = scipy.special.lambertw(3/(np.pi*epsilon**2)).real
    a = 1/(4*np.sin(delta)**2)*W
    if a>1:
        return a
    else:
        return 1
