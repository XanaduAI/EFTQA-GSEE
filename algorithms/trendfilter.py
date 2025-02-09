"""
Title: Early Fault-Tolerant Quantum Algorithms in Practice: Application to Ground-State Energy Est.
Authors: O. Kiss, U. Azad, B. Requena, A. Roggero, D. Wakeham, J. M. Arrazola
Paper: arXiv:2405.03754 
Year: 2024
Description: This module contains core functions for filering time series data.
"""

import numpy as np
import cvxpy
from trendfilter.extrapolate import get_interp_extrapolate_functions
from trendfilter.derivatives import second_derivative_matrix_nes, first_derv_nes_cvxpy
from trendfilter.linear_deviations import complete_linear_deviations


def trend_filter(
    x,
    y,
    y_err=None,
    alpha_1=0.0,
    alpha_2=0.0,
    sigma=0,
    l_norm=2,
    constrain_zero=False,
    monotonic=False,
    positive=False,
    linear_deviations=None,
    solver="ECOS",
):
    """
    Applies trend filtering to the given data.

    Args:
        x (numpy.ndarray): The x-values.
        y (numpy.ndarray): The y-values.
        y_err (numpy.ndarray, optional): The errors in y-values. Defaults to an array of ones.
        alpha_1 (float, optional): Regularization against non-zero slope (first derivative). Defaults to 0.0.
        alpha_2 (float, optional): Regularization against changing slope (second derivative). Defaults to 0.0.
        sigma (float, optional): Parameter for exponential weighting. Defaults to 0.
        l_norm (int, optional): Norm to use for regularization (1 or 2). Defaults to 2.
        constrain_zero (bool, optional): If True, constrains the model to be zero at the origin. Defaults to False.
        monotonic (bool, optional): If True, constrains the model to be monotonically increasing. Defaults to False.
        positive (bool, optional): If True, constrains the model to be positive. Defaults to False.
        linear_deviations (list, optional): List of linear deviation objects. Defaults to None.
        solver (str, optional): Solver to use for optimization. Defaults to "ECOS".

    Returns:
        dict: A dictionary containing the fit model information, including:
            - "x": The x-values.
            - "y": The y-values.
            - "y_err": The errors in y-values.
            - "function": The fitted function.
            - "function_base": The base function.
            - "function_deviates": The function deviations.
            - "model": The model.
            - "base_model": The base model.
            - "objective_model": The objective function of the model.
            - "regularization_total": The total regularization.
            - "regularizations": The individual regularizations.
            - "objective_total": The total objective function.
            - "y_fit": The fitted y-values.
            - "constraints": The constraints applied to the model.
            - "linear_deviations": The linear deviations.
    """

    if linear_deviations is None:
        linear_deviations = []

    linear_deviations = complete_linear_deviations(linear_deviations, x)

    assert l_norm in [1, 2]
    n = len(x)

    # get the y_err is not supplied
    assert len(y) == n
    if y_err is None:
        y_err = np.ones(n)
    else:
        assert len(y_err) == n

    # the objective function
    result = get_obj_func_model(
        y, y_err=y_err, positive=positive, linear_deviations=linear_deviations
    )

    # TODO: this seems wrong
    # y_var = result['objective_function'].variables()[0]

    derv_1 = first_derv_nes_cvxpy(x, result["base_model"])

    if sigma > 0:
        diff_y = abs(y[1:] - y[:-1])
        exp_y = np.exp(-diff_y / (sigma * np.mean(diff_y)))

        derv_1 = cvxpy.multiply(derv_1, exp_y)

    # the regularization
    reg_sum, regs = get_reg(
        x,
        y,
        result["base_model"],
        derv_1,
        l_norm,
        alpha_1,
        alpha_2,
        linear_deviations=linear_deviations,
    )

    # the total objective function with regularization
    obj_total = result["objective_function"] + reg_sum

    # The objective
    obj = cvxpy.Minimize(obj_total)

    # Get the constraints if any
    constraints = []
    if constrain_zero:
        constraints.append(result["model"][0] == 0)

    if monotonic:
        # TODO:
        # should this be derv of base or model?
        constraints.append(derv_1 >= 0)

    # define and solve the problem
    problem = cvxpy.Problem(obj, constraints=constraints)
    problem.solve(solver=solver)

    func_base, func_deviates, func = get_interp_extrapolate_functions(
        x, result["base_model"], linear_deviations
    )

    tf_result = {
        "x": x,
        "y": y,
        "y_err": y_err,
        "function": func,
        "function_base": func_base,
        "function_deviates": func_deviates,
        "model": result["model"],
        "base_model": result["base_model"],
        "objective_model": result["objective_function"],
        "regularization_total": reg_sum,
        "regularizations": regs,
        "objective_total": obj,
        "y_fit": result["model"].value,
        "constraints": constraints,
        "linear_deviations": linear_deviations,
    }

    return tf_result


def get_reg(x, y, base_model, derv_1, l_norm, alpha_1, alpha_2, linear_deviations=None):
    """Get the regularization term.

    Args:
        x (numpy.ndarray): The x-value.
        y (cvxpy.Variable): The y variable.
        base_model (cvxpy.Variable): The base model variable.
        derv_1 (cvxpy.Expression): The first derivative cvxpy expression from first_derv_nes_cvxpy.
        l_norm (int): 1 or 2 to use either L1 or L2 norm.
        alpha_1 (float): Regularization against non-zero slope (first derivative).
                        Setting this very high will result in stair step model (if L1).
        alpha_2 (float): Regularization against (second derivative or changing slope).
                        Setting this very high will result in piecewise linear model (if L1).
        linear_deviations (list, optional): List of linear deviation objects.

    Returns:
        Tuple[cvxpy.Expression, list]: A tuple containing sum of regularization terms and
            a list of individual regularization terms.
    """
    d2 = second_derivative_matrix_nes(x, scale_free=True)

    if l_norm == 2:
        norm = cvxpy.sum_squares
    else:
        norm = cvxpy.norm1

    reg_1 = alpha_1 * norm(derv_1)
    reg_2 = alpha_2 * norm(d2 @ base_model)
    regs = [reg_1, reg_2]

    for lin_dev in linear_deviations:
        reg = lin_dev["alpha"] * norm(lin_dev["variable"])
        regs.append(reg)

    reg_sum = sum(regs)

    return reg_sum, regs


def get_obj_func_model(y, y_err=None, positive=False, linear_deviations=None):
    """Get the objective function and the model as cvxpy expressions.

    Args:
        y (numpy.ndarray): The y variable, a numpy array.
        y_err (numpy.ndarray, optional): The y_err variable, a numpy array. 
            Defaults to None, which sets it to an array of ones.
        positive (bool, optional): If set to True, will result in an always positive base model.
            Defaults to False.
        linear_deviations (list, optional): List of completed linear deviation objects.
            Defaults to an empty list.

    Returns:
        dict: A dictionary containing the base model, the model,
            and the objective function as cvxpy expressions.
    """

    if linear_deviations is None:
        linear_deviations = []

    n = len(y)
    if y_err is None:
        y_err = np.ones(n)
    else:
        assert len(y_err) == n

    buff = 0.01 * np.median(abs(y))
    buff_2 = buff**2
    isig = 1 / np.sqrt(buff_2 + y_err**2)

    base_model = cvxpy.Variable(n, pos=positive)

    model = base_model

    for lin_dev in linear_deviations:
        model += lin_dev["model_contribution"]

    diff = cvxpy.multiply(isig, model - y)
    obj_func = cvxpy.sum(cvxpy.huber(diff))

    result = {"base_model": base_model, "model": model, "objective_function": obj_func}

    return result
