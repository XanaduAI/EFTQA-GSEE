"""
Title: Early Fault-Tolerant Quantum Algorithms in Practice: Application to Ground-State Energy Est.
Authors: O. Kiss, U. Azad, B. Requena, A. Roggero, D. Wakeham, J. M. Arrazola
Paper: arXiv:2405.03754 
Year: 2024
Description: This module provides functions to compute the Fourier coefficients of the
Heaviside function and to calculate the beta value based on given epsilon and delta.
"""
import numpy as np
import scipy as sp


def get_F(j, beta, d):
    """Compute the Fourier coefficients of the Heaviside function following arXiv:2110.1207.

    Args:
        j (int): Time index, j ∈ {0, ..., d}.
        beta (float): Parameter.
        d (int): Maximal time.

    Returns:
        complex: The Fourier coefficient F_(2j+1)(beta).
    """

    if abs(j) < d:
        coeff = -1.0j * np.sqrt(beta / (2 * np.pi))
        bessel = (sp.special.ive(j, beta) + sp.special.ive(j + 1, beta)) / (2 * j + 1)
        return coeff * bessel

    else:
        coeff = -1.0j * np.sqrt(beta / (2 * np.pi))
        bessel = sp.special.ive(j, beta) / (2 * j + 1)
        return coeff * bessel


def get_beta(epsilon, delta):
    """Calculate the beta value based on the given epsilon and delta.

    This function computes the beta value using the Lambert W function.

    Args:
        epsilon (float): target precision.
        delta (float): window width.

    Returns:
        float: The computed beta value.
    """
    W = scipy.special.lambertw(3 / (np.pi * epsilon**2)).real
    a = 1 / (4 * np.sin(delta) ** 2) * W
    return max(a, 1)
