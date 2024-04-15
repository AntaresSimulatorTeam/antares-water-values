from calculate_reward_and_bellman_values import (
    RewardApproximation,
    ReservoirManagement,
    BellmanValueCalculation,
)
from read_antares_data import TimeScenarioParameter, TimeScenarioIndex
from optimization import AntaresProblem
from optimization import Basis
import numpy as np
from scipy.interpolate import interp1d
from functions_iterative import (
    compute_upper_bound_without_stored_models,
    create_model,
    compute_upper_bound_with_stored_models,
)
from time import time
from multiprocessing import Process, Pool
from functools import partial
from type_definition import Array1D, Array2D, Dict, Array4D


def calculate_complete_reward(
    len_controls: int,
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    solver: str = "CLP",
) -> Dict[TimeScenarioIndex, RewardApproximation]:
    reward: Dict[TimeScenarioIndex, RewardApproximation] = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            r = RewardApproximation(
                lb_control=-reservoir_management.reservoir.max_pumping[week],
                ub_control=reservoir_management.reservoir.max_generating[week],
                ub_reward=float("inf"),
            )
            reward[TimeScenarioIndex(week, scenario)] = r

    controls = np.array(
        [
            np.linspace(
                -reservoir_management.reservoir.max_pumping[week],
                reservoir_management.reservoir.max_generating[week],
                num=len_controls,
            )
            for week in range(param.len_week)
        ]
    )

    with Pool() as pool:
        print(pool)
        scenario_reward = pool.map(
            partial(
                calculate_reward_for_one_scenario,
                param=param,
                reservoir_management=reservoir_management,
                output_path=output_path,
                controls=controls,
                solver=solver,
            ),
            range(param.len_scenario),
        )

    for scenario in range(param.len_scenario):
        for week in range(param.len_week):
            reward[TimeScenarioIndex(week, scenario)] = scenario_reward[scenario][
                TimeScenarioIndex(week, scenario)
            ]

    return reward


def calculate_reward_for_one_scenario(
    scenario: int,
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    controls: Array2D,
    solver: str,
) -> Dict[TimeScenarioIndex, RewardApproximation]:

    reward: Dict[TimeScenarioIndex, RewardApproximation] = {}
    for week in range(param.len_week):
        r = RewardApproximation(
            lb_control=-reservoir_management.reservoir.max_pumping[week],
            ub_control=reservoir_management.reservoir.max_generating[week],
            ub_reward=float("inf"),
        )
        reward[TimeScenarioIndex(week, scenario)] = r

    basis_0 = Basis([], [])
    for week in range(param.len_week):
        print(f"{scenario} {week}")
        m = AntaresProblem(
            scenario=scenario,
            week=week,
            path=output_path,
            itr=1,
            name_scenario=(
                param.name_scenario[scenario] if len(param.name_scenario) > 1 else -1
            ),
            name_solver=solver,
        )
        m.create_weekly_problem_itr(
            param=param,
            reservoir_management=reservoir_management,
        )

        for u in controls[week]:
            beta, lamb, itr, computation_time = m.solve_with_predefined_controls(
                control=u, prev_basis=basis_0
            )
            if m.store_basis:
                basis_0 = m.basis[-1]
            else:
                basis_0 = Basis([], [])

            reward[TimeScenarioIndex(week, scenario)].update_reward_approximation(
                slope_new_cut=-lamb,
                intercept_new_cut=-beta + lamb * u,
            )
    return reward


def calculate_bellman_value_with_precalculated_reward(
    len_controls: int,
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    solver: str = "CLP",
) -> tuple[Array2D, Dict[TimeScenarioIndex, RewardApproximation], Array2D]:
    """
    Algorithm to evaluate Bellman values. First reward is approximated thanks to multiple simulations. Then, Bellman values are computed with the reward approximation.

    Parameters
    ----------
    len_controls:int :
        Number of controls to evaluate to build reward approximation
    param:TimeScenarioParameter :
        Time-related parameters for the Antares study
    reservoir_management:ReservoirManagement :
        Reservoir considered for Bellman values
    output_path:str :
        Path to mps files describing optimization problems
    X:Array1D :
        Discretization of sotck levels for Bellman values
    solver:str :
        Solver to use (default is CLP) with ortools

    Returns
    -------
    V:np.array :
        Bellman values
    G:Dict[TimeScenarioIndex, RewardApproximation] :
        Reward approximation
    controls:Array2D :
        Controls found in upper bound computation
    """

    reward = calculate_complete_reward(
        len_controls=len_controls,
        param=param,
        reservoir_management=reservoir_management,
        output_path=output_path,
    )

    bellman_value_calculation = BellmanValueCalculation(
        param=param,
        reward=reward,
        reservoir_management=reservoir_management,
        stock_discretization=X,
    )

    V = bellman_value_calculation.calculate_VU()

    V_fut = interp1d(X, V[:, 0])
    V0 = V_fut(reservoir_management.reservoir.initial_level)

    upper_bound, controls, current_itr = compute_upper_bound_without_stored_models(
        bellman_value_calculation=bellman_value_calculation,
        V=V,
        output_path=output_path,
        solver=solver,
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return (V, reward, controls)


def calculate_bellman_value_directly(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    solver: str = "CLP",
) -> tuple[Array2D, Array2D, Array4D]:
    """
    Algorithm to evaluate Bellman values directly.

    Parameters
    ----------
    param:TimeScenarioParameter :
        Time-related parameters for the Antares study
    reservoir_management:ReservoirManagement :
        Reservoir considered for Bellman values
    output_path:str :
        Path to mps files describing optimization problems
    X:Array1D :
        Discretization of sotck levels for Bellman values
    N:int :
        Maximum number of iteration to do
    tol_gap:float :
        Relative tolerance gap for the termination of the algorithm
    solver:str :
        Solver to use (default is CLP) with ortools

    Returns
    -------
    V:np.array :
        Bellman values
    tot_t: np.array :
        Computing time for each week
    pert : np.array :
        Number of simplex iterations for each week and each scenario
    """

    bellman_value_calculation = BellmanValueCalculation(
        param=param,
        reservoir_management=reservoir_management,
        stock_discretization=X,
    )

    dict_basis: Dict[int, Basis] = {}
    for scenario in range(param.len_scenario):
        dict_basis[scenario] = Basis()

    V = np.zeros((len(X), param.len_week + 1), dtype=np.float32)
    tot_t = np.zeros((param.len_week, param.len_scenario), dtype=np.float32)
    perf = np.zeros((param.len_week, param.len_scenario, len(X), 2), dtype=np.float32)

    for week in range(param.len_week - 1, -1, -1):
        for scenario in range(param.len_scenario):
            print(f"{week} {scenario}", end="\r")
            debut = time()
            m = create_model(
                param, reservoir_management, output_path, week, scenario, solver
            )

            for i in range(len(X)):
                if i == 1:
                    m.solver_parameters.SetIntegerParam(
                        m.solver_parameters.PRESOLVE, m.solver_parameters.PRESOLVE_OFF
                    )
                t, itr, Vu, _, _ = m.solve_problem_with_bellman_values(
                    bellman_value_calculation=bellman_value_calculation,
                    V=V,
                    level_i=X[i],
                    take_into_account_z_and_y=True,
                    basis=dict_basis[scenario] if m.store_basis else Basis(),
                )

                if m.store_basis:
                    basis = m.basis[-1]
                    dict_basis[scenario] = basis

                V[i, week] += -Vu / param.len_scenario
                perf[week, scenario, i] = (t, itr)
            fin = time()
            tot_t[week, scenario] = fin - debut

    V_fut = interp1d(X, V[:, 0])
    V0 = V_fut(reservoir_management.reservoir.initial_level)

    upper_bound, controls, current_itr = compute_upper_bound_without_stored_models(
        bellman_value_calculation=bellman_value_calculation,
        V=V,
        output_path=output_path,
        store_basis=m.store_basis,
        solver=solver,
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return V, tot_t, perf
