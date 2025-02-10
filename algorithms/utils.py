"""
Title: Early Fault-Tolerant Quantum Algorithms in Practice: Application to Ground-State Energy Est.
Authors: O. Kiss, U. Azad, B. Requena, A. Roggero, D. Wakeham, J. M. Arrazola
Paper: arXiv:2405.03754 
Year: 2024
Description: This module contains utility functions for filering and detecting change points in time series data.
"""
import numpy as np
import ruptures as rpt
import scipy
from trendfilter.trendfilter import trend_filter


def filter_(acdf):
    """Applies a trend filter to the ACDF data and detects change points.

    Args:
        acdf (numpy.ndarray): The approximate cumulative distribution data as a time series.

    Returns:
        int: The first detected change point index.

    Developer Notes:
        1. Applies a trend filter to the real part of the input array.
        2. Uses the Kernel Change Point Detection (CPD) algorithm to
        detect change points in the filtered data.
        3. Returns the index of the first detected change point.
    """
    acdf_filter_ = trend_filter(
        np.arange(len(acdf)),
        acdf.real,
        l_norm=1,
        alpha_1=10,
        alpha_2=0,
        sigma=100,
        constrain_zero=True,
        monotonic=True,
        positive=True,
    )

    algo = rpt.KernelCPD().fit(acdf_filter_["y_fit"])
    cps = [0] + algo.predict(1)
    cps[-1] += 1

    return cps[1]


def find_first_jump_left(acdf, alpha=0.01):
    """Finds the first significant change point in the given ACDF data.

    Parameters:
        acdf (numpy.ndarray): The approximate cumulative distribution function data to analyze.
        alpha (float, optional): The significance level for the statistical test. Default is 0.01.

    Returns:
        int: The index of the first significant change point. If no significant change point is found,
        returns the last index of the input data and prints "not converged".

    Developer Notes:
        1. The function uses the Kernel Change Point Detection (KernelCPD) algorithm
        from the `ruptures` library.
        2. The function iteratively increases the size of the data subset being analyzed
        until a significant change point is found or the end of the data is reached.
        3. The statistical relevance of the detected change point is evaluated using an F-test.
    """
    cut = 50
    bkp = 0
    while cut < len(acdf):
        acdf_trial = acdf[:cut]

        algo = rpt.KernelCPD(min_size=10).fit(acdf_trial)
        trial = algo.predict(1)[0]

        F = check_stat_relevant(acdf_trial, trial)

        df = len(acdf_trial) - 2
        p_value = scipy.stats.f.cdf(F, 1, df)
        if p_value > alpha:
            mean1 = np.mean(acdf_trial[:trial])
            mean2 = np.mean(acdf_trial[trial:])
            std1 = np.std(acdf_trial[:trial])
            perc = np.percentile(acdf_trial[trial:], 25)
            if mean2 > mean1:
                if perc > 0 + 2 * std1:
                    bkp = trial
                    break
        cut += 25
    if bkp == 0:
        bkp = len(acdf) - 1
        print("not converged")
    return bkp


def find_first_jump_right(acdf, alpha=0.01):
    """Identifies the first significant change point in the given ACDF data.

    Args:
        acdf (array-like): The approximate cumulative distribution function data.
        alpha (float, optional): The significance level for the statistical test. Default is 0.01.

    Returns:
        int: The index of the first significant change point in the ACDF data.
    """
    cps = [len(acdf)]
    for i in range(100):

        acdf_trial = acdf[: cps[-1]]
        #
        algo = rpt.KernelCPD().fit(acdf_trial)
        trial = algo.predict(1)[0]

        F = check_stat_relevant(acdf_trial, trial)
        df = len(acdf_trial) - 2
        p_value = scipy.stats.f.cdf(F, 1, df)
        #     print(p_value)
        if p_value < 1 - alpha:
            break
        if acdf_trial[:trial].mean() > acdf_trial[trial:].mean():
            break

        if np.mean(acdf_trial[:trial]) < 0:
            break
        cps.append(trial)
    return cps[-1]


def check_stat_relevant(acdf, bkp):
    """Calculate the F-statistic for a given breakpoint in a time series.

    Args:
        acdf (numpy.ndarray): The approximate cumulative distribution data as a time series.
        bkp (int): The index of the breakpoint in the time series.

    Returns:
        float: The F-statistic value.

    Developer Notes:
        The function computes the F-statistic to test the significance of the 
        difference between the means of two segments of the time series data 
        divided at the breakpoint. The F-statistic is calculated as the ratio 
        of the mean square between the groups to the mean square within the groups.
    """

    SSbet = len(acdf[:bkp]) * (np.mean(acdf[:bkp]) - acdf.mean()) ** 2
    SSbet += len(acdf[bkp:]) * (np.mean(acdf[bkp:]) - acdf.mean()) ** 2

    MSbetween = SSbet / 1

    SSwithin = np.sum((acdf[:bkp] - acdf[:bkp].mean()) ** 2)
    SSwithin += np.sum((acdf[bkp:] - acdf[bkp:].mean()) ** 2)

    MSwithin = SSwithin / (len(acdf) - 2)
    F = MSbetween / MSwithin

    return float(F)


def find_large_variance(acdf, v, s=3):
    """
    Finds the 25th percentile of large variance values based on the given parameters.

    Args:
        acdf (DataFrame): The input data frame containing the data.
        v (float): A value used in the large_variance function.
        s (int, optional): A scaling factor for the size parameter
                           in the large_variance function. Default is 3.

    Returns:
        int: The 25th percentile of the calculated large variance values.
    """
    guess = []
    for sigma in range(3, 4):
        for s in range(30, 35, 5):
            for r in range(8, 9):
                a = large_variance(acdf, v, sigma, size=(r * s / 10, s))
                guess.append(int(a))
    return int(np.percentile(guess, 25))


def large_variance(acdf, v, s=3, size=(6, 10)):
    """
    Identifies the first index in the array where the variance exceeds a threshold.

    Args:
        acdf (numpy.ndarray): The approximate cumulative distribution data as a time series.
        v (float): The variance threshold.
        s (int, optional): The scaling factor for the variance threshold. Default is 3.
        size (tuple, optional): A tuple containing two integers. The first integer is the minimum number of elements 
                                that must exceed the threshold, and the second integer is the window size to check. 
                                Default is (6, 10).

    Returns:
        int: The index of the first element where the variance exceeds the threshold. If no such element is found, 
            returns the last index of the array.
    """
    for a in range(len(acdf)):
        if np.sum([acdf[a : a + size[1]] > s * v]) > size[0]:
            return a
    return len(acdf) - 1


def optimal_sample(F, delta, eta, theta):
    """Calculate the optimal sample size for a given set of parameters.

    Args:
        F (float): Normalization constant related to the Fourier coefficients.
        delta (float): window width.
        eta (float): overlap between actual and prepared ground-state.
        theta (float): A small positive number representing the confidence level.

    Returns:
        float: The calculated optimal sample size.
    """
    gamma = 0.57721567

    M = 2 * F / (eta / 2 - delta)
    M = M**2
    M = M * (np.log(np.log(1 / (delta))) + np.log(1 / theta))

    return M


def compute_d(tau, epsilon):
    """Compute the value of 'd' based on the given 'tau' and 'epsilon' parameters.

    The function calculates 'd' using the formula:
    
    .. math::
        d = sqrt(2) / (tau * epsilon) * log(2 * sqrt(2 * pi) * (1 + 2 / epsilon))

    Args:
        tau (float): normalization constant.
        epsilon (float): target precision.

    Returns:
        int: The computed integer value of 'd'.
    """
    d = np.sqrt(2) / (tau * epsilon)
    d = d * np.log(2 * np.sqrt(2 * np.pi) * (1 + 2 / epsilon))
    d = int(d)
    return d


def toBinary(n):
    """Convert a decimal fraction between 0 and 1 to its binary representation.

    Args:
        n (float): A decimal number between 0 and 1 (exclusive).

    Returns:
        str: The binary representation of the decimal fraction, or "ERROR" if the input is not between 0 and 1,
             or if the binary representation exceeds 32 characters in length.
    """

    # Check if the number is Between 0 to 1 or Not
    if n >= 1 or n <= 0:
        return "ERROR"

    answer = ""
    frac = 0.5
    answer = answer + "."

    # Setting a limit on length: 32 characters.
    while n > 0:

        # Setting a limit on length: 32 characters
        if len(answer) >= 64:
            return "ERROR"

        # Multiply n by 2 to check it 1 or 0
        b = n * 2
        if b >= 1:

            answer = answer + "1"
            n = b - 1

        else:
            answer = answer + "0"
            n = b

    return answer


def toDecimal(binary):
    """Convert a binary string to its decimal representation.

    Args:
        binary (str): A string representing a binary number, where each character is '0' or '1'.

    Returns:
        float: The decimal representation of the binary string.
    """
    number = 0
    for b in range(len(binary)):
        number += 2 ** (-b - 1) * int(binary[b])

    return number


def truncate(ew, n):
    """
    Truncates the given value `ew` to a specified number of binary digits `n` after a transformation.

    Args:
        ew (float): The input value to be truncated.
        n (int): The number of binary digits to retain after truncation.

    Returns:
        float: The truncated value after the specified transformations.
    """

    binary = toBinary(3 * (ew + np.pi / 3) / (2 * np.pi))
    binary = binary[1 : n + 1]
    #     print(binary)
    binary = binary[:-1] + "1"
    decimal = toDecimal(binary)
    value = 2 * np.pi * decimal / 3 - np.pi / 3
    #     print(binary,value)
    return value

def gen_int_list(num_nu, sij_mat):
    """Generate a list of interaction terms, their one-norm, and their relative weights.

    Args:
        num_nu (int): The number of elements to consider for generating interaction terms.
        sij_mat (numpy.ndarray): A 2-D matrix containing interaction values between elements.

    Returns:
        tuple: A tuple containing:
            - int_terms (List[dict]): A list of dictionaries where each item represents an
                interaction term with keys:
                - "sui" (int): The index of the first element in the interaction.
                - "suj" (int): The index of the second element in the interaction.
                - "hij" (float): The interaction value between the two elements.
            - int_1norm (float): The one-norm of the interaction values.
            - int_prob (numpy.ndarray): An array of relative weights of the interaction terms.
    """
    # populate interaction terms
    int_terms = []
    for ii in range(num_nu):
        for jj in range(num_nu):
            if jj<=ii:
                continue
            pair_ij = {"sui": ii, "suj": jj, "hij": vij_mat[ii,jj]}
            int_terms.append(pair_ij)

    # calculate one-norm and relative weigth
    int_1norm = np.sum([pp["hij"] for pp in int_terms])
    int_prob = np.array([pp["hij"]/int_1norm for pp in int_terms])

    return int_terms, int_1norm, int_prob
