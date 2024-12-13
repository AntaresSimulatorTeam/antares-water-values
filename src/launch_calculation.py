from calculate_reward_and_bellman_values import ReservoirManagement
from functions_iterative import itr_control
from read_antares_data import TimeScenarioParameter
from simple_bellman_value_calculation import (
    calculate_bellman_value_directly,
    calculate_bellman_value_with_precalculated_reward,
)
from type_definition import Array1D, Array2D


def calculate_bellman_values(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    method: str,
    solver: str = "CLP",
    N: int = 1,
    tol_gap: float = 1e-4,
    len_controls: int = 10,
) -> Array2D:
    """Algorithm to evaluate Bellman values with different methods.

    Args:
        param (TimeScenarioParameter): Time-related parameters for the Antares study
        reservoir_management (ReservoirManagement): Reservoir considered for Bellman values
        output_path (str): Path to mps files describing optimization problems
        X (Array1D): Discretization of sotck levels for Bellman values
        method (str): Method to evaluate Bellman values (either direct, precalculated or iterative)
        solver (str, optional): Solver to use with ortools. Defaults to "CLP".
        N (int, optional): Maximum number of iterations to do in iterative method. Defaults to 1.
        tol_gap (float, optional): Relative tolerance gap for the termination of the iterative algorithm. Defaults to 1e-4.
        len_controls (int, optional): Number of controls to evaluate to build reward approximation in precalculated reward. Defaults to 10.

    Returns:
        Array2D: Bellman values
    """

    assert method in ["direct", "precalculated", "iterative"]

    if method == "direct":
        # Compute Bellman values directly
        vb = calculate_bellman_value_directly(
            param=param,
            reservoir_management=reservoir_management,
            output_path=output_path,
            X=X,
            solver=solver,
        )

    elif method == "precalculated":
        # or with precalulated reward
        vb, _ = calculate_bellman_value_with_precalculated_reward(
            len_controls=len_controls,
            param=param,
            reservoir_management=reservoir_management,
            output_path=output_path,
            X=X,
            solver=solver,
        )

    elif method == "iterative":
        # or with iterative algorithm
        vb, _, _, _, _, _, _, _ = itr_control(
            param=param,
            reservoir_management=reservoir_management,
            output_path=output_path,
            X=X,
            N=N,
            tol_gap=tol_gap,
            solver=solver,
        )

    return vb
