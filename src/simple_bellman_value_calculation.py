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
from type_definition import Array1D, Array2D, Array3D, Dict, Array4D, Optional, Any


def calculate_complete_reward(
    len_controls: int,
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    dict_basis: Dict[TimeScenarioIndex, Basis],
    solver: str,
    processes: Optional[int] = None,
) -> tuple[
    Dict[TimeScenarioIndex, RewardApproximation],
    Array2D,
    Array4D,
    Dict[TimeScenarioIndex, Basis],
]:

    controls = {
        TimeScenarioIndex(week=week, scenario=scenario): np.linspace(
            -reservoir_management.reservoir.max_pumping[week]
            * reservoir_management.reservoir.efficiency,
            reservoir_management.reservoir.max_generating[week],
            num=len_controls,
        )
        for week in range(param.len_week)
        for scenario in range(param.len_scenario)
    }

    with Pool(processes=processes) as pool:
        scenario_reward = pool.map(
            partial(
                calculate_reward_for_one_scenario,
                param=param,
                reservoir_management=reservoir_management,
                output_path=output_path,
                controls=controls,
                solver=solver,
                dict_basis=dict_basis,
            ),
            range(param.len_scenario),
        )

    reward: Dict[TimeScenarioIndex, RewardApproximation] = {}
    tot_t = np.zeros((param.len_week, param.len_scenario), dtype=np.float32)
    perf = np.zeros(
        (param.len_week, param.len_scenario, len_controls, 2), dtype=np.float32
    )

    for scenario in range(param.len_scenario):
        for week in range(param.len_week):
            reward[TimeScenarioIndex(week, scenario)] = scenario_reward[scenario][0][
                TimeScenarioIndex(week, scenario)
            ]
            tot_t[week][scenario] = scenario_reward[scenario][1][week]
            perf[week][scenario] = scenario_reward[scenario][2][week]
            dict_basis[TimeScenarioIndex(week=week, scenario=scenario)] = (
                scenario_reward[scenario][3][TimeScenarioIndex(week, scenario)]
            )

    return reward, tot_t, perf, dict_basis


def calculate_reward_for_one_scenario(
    scenario: int,
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    controls: Dict[TimeScenarioIndex, Array1D],
    solver: str,
    dict_basis: Dict[TimeScenarioIndex, Basis],
) -> tuple[
    Dict[TimeScenarioIndex, RewardApproximation],
    Array1D,
    Array3D,
    Dict[TimeScenarioIndex, Basis],
]:

    reward: Dict[TimeScenarioIndex, RewardApproximation] = {}
    for week in range(param.len_week):
        r = RewardApproximation(
            lb_control=-reservoir_management.reservoir.max_pumping[week]
            * reservoir_management.reservoir.efficiency,
            ub_control=reservoir_management.reservoir.max_generating[week],
            ub_reward=float("inf"),
        )
        reward[TimeScenarioIndex(week, scenario)] = r

    tot_t = np.zeros((param.len_week), dtype=np.float32)
    perf = np.zeros(
        (param.len_week, len(controls[TimeScenarioIndex(0, 0)]), 2), dtype=np.float32
    )

    for week in range(param.len_week):
        start = time()
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

        basis_0 = Basis([], [])
        if m.store_basis:
            if dict_basis[TimeScenarioIndex(week, scenario=scenario)].not_empty():
                basis_0 = dict_basis[TimeScenarioIndex(week=week, scenario=scenario)]
            elif (
                week > 0
                and dict_basis[
                    TimeScenarioIndex(week - 1, scenario=scenario)
                ].not_empty()
            ):
                basis_0 = dict_basis[TimeScenarioIndex(week - 1, scenario=scenario)]

        for j, u in enumerate(controls[TimeScenarioIndex(week, scenario)]):
            beta, lamb, itr, computation_time = m.solve_with_predefined_controls(
                control=float(u), prev_basis=basis_0
            )
            if m.store_basis:
                basis_0 = m.basis[-1]
                if j == 0:
                    dict_basis[TimeScenarioIndex(week, scenario=scenario)] = basis_0

            reward[TimeScenarioIndex(week, scenario)].update_reward_approximation(
                slope_new_cut=-lamb,
                intercept_new_cut=-beta + lamb * u,
            )
            perf[week, j] = (computation_time, itr)
        end = time()
        tot_t[week] = end - start

    return reward, tot_t, perf, dict_basis


def calculate_bellman_value_with_precalculated_reward(
    len_controls: int,
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    solver: str = "CLP",
    processes: Optional[int] = None,
) -> tuple[
    Array2D,
    Dict[TimeScenarioIndex, RewardApproximation],
    Array2D,
    Array4D,
    Array3D,
    Array2D,
]:
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
    tot_t: np.array :
        Computing time for each week
    perf : np.array :
        Number of simplex iterations for each week and each scenario
    controls:Array2D :
        Controls found in upper bound computation
    """

    dict_basis = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            dict_basis[TimeScenarioIndex(week, scenario)] = Basis([], [])

    reward, tot_t, perf, dict_basis = calculate_complete_reward(
        len_controls=len_controls,
        param=param,
        reservoir_management=reservoir_management,
        output_path=output_path,
        processes=processes,
        solver=solver,
        dict_basis=dict_basis,
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

    upper_bound, controls, current_itr, dict_basis = (
        compute_upper_bound_without_stored_models(
            param=param,
            stock_discretization=X,
            reservoir_management=reservoir_management,
            V=V,
            output_path=output_path,
            solver=solver,
            store_basis=True if solver == "XPRESS_LP" else False,
            processes=processes,
            dict_basis=dict_basis,
        )
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return (V, reward, tot_t, perf, current_itr, controls)


def calculate_bellman_value_directly(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    solver: str = "CLP",
    processes: Optional[int] = None,
) -> tuple[Array2D, Array2D, Array4D, Array3D, Array2D]:
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
    perf : np.array :
        Number of simplex iterations for each week and each scenario
    controls :np.array:
        Optimal trajectories found with Bellman values
    """
    dict_basis: Dict[TimeScenarioIndex, Basis] = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            dict_basis[TimeScenarioIndex(week, scenario)] = Basis([], [])

    V = np.zeros((len(X), param.len_week + 1), dtype=np.float32)
    tot_t = np.zeros((param.len_week, param.len_scenario), dtype=np.float32)
    perf = np.zeros((param.len_week, param.len_scenario, len(X), 2), dtype=np.float32)

    for week in range(param.len_week - 1, -1, -1):
        print(week, end="\r")
        with Pool(processes=processes) as pool:

            intermediate_weekly_results = pool.map(
                partial(
                    calculate_bellman_values_for_one_week_and_one_scenario,
                    param=param,
                    reservoir_management=reservoir_management,
                    output_path=output_path,
                    X=X,
                    solver=solver,
                    dict_basis=dict_basis,
                    V=V,
                    week=week,
                ),
                range(param.len_scenario),
            )

        V, perf, tot_t, dict_basis = update_weekly_results(
            param, X, dict_basis, V, tot_t, perf, week, intermediate_weekly_results
        )

    V_fut = interp1d(X, V[:, 0])
    V0 = V_fut(reservoir_management.reservoir.initial_level)

    upper_bound, controls, current_itr, dict_basis = (
        compute_upper_bound_without_stored_models(
            param=param,
            stock_discretization=X,
            reservoir_management=reservoir_management,
            V=V,
            output_path=output_path,
            store_basis=True if solver == "XPRESS_LP" else False,
            solver=solver,
            processes=processes,
            dict_basis=dict_basis,
        )
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return V, tot_t, perf, current_itr, controls


def update_weekly_results(
    param: TimeScenarioParameter,
    X: Array1D,
    dict_basis: Dict[TimeScenarioIndex, Basis],
    V: Array2D,
    tot_t: Array2D,
    perf: Array4D,
    week: int,
    intermediate_weekly_results: list[Dict[str, Any]],
) -> tuple[Array2D, Array4D, Array2D, Dict[TimeScenarioIndex, Basis]]:
    for i in range(len(X)):
        V[i, week] = np.sum(
            [
                intermediate_weekly_results[s]["partial_vu"][i]
                for s in range(param.len_scenario)
            ]
        )

    for scenario in range(param.len_scenario):
        perf[week, scenario] = intermediate_weekly_results[scenario]["partial_perf"]
        tot_t[week, scenario] = intermediate_weekly_results[scenario]["partial_time"]
        dict_basis[TimeScenarioIndex(week, scenario)] = intermediate_weekly_results[
            scenario
        ]["last_basis"]

    return V, perf, tot_t, dict_basis


def calculate_bellman_values_for_one_week_and_one_scenario(
    scenario: int,
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    solver: str,
    dict_basis: Dict[TimeScenarioIndex, Basis],
    V: Array2D,
    week: int,
) -> Dict[str, Any]:
    print(f"{scenario} {week}", end="\r")
    debut = time()
    m = create_model(param, reservoir_management, output_path, week, scenario, solver)

    basis = Basis([], [])
    if m.store_basis:
        if dict_basis[TimeScenarioIndex(week, scenario)].not_empty():
            basis = dict_basis[TimeScenarioIndex(week, scenario)]
        elif (
            week < param.len_week - 1
            and dict_basis[TimeScenarioIndex(week + 1, scenario)].not_empty()
        ):
            basis = dict_basis[TimeScenarioIndex(week + 1, scenario)]

    partial_vu = np.zeros(len(X), dtype=np.float32)
    partial_perf = np.zeros((len(X), 2), dtype=np.float32)
    for i in range(len(X)):
        if i == 1:
            m.solver_parameters.SetIntegerParam(
                m.solver_parameters.PRESOLVE, m.solver_parameters.PRESOLVE_OFF
            )

        t, itr, Vu, _, _ = m.solve_problem_with_bellman_values(
            stock_discretization=X,
            reservoir_management=reservoir_management,
            V=V,
            level_i=X[i],
            take_into_account_z_and_y=True,
            basis=basis,
        )

        if m.store_basis:
            basis = m.basis[-1]
            if i == 0:
                dict_basis[TimeScenarioIndex(week, scenario)] = basis

        partial_vu[i] += -Vu / param.len_scenario
        partial_perf[i] = (t, itr)
    fin = time()

    partial_results = {
        "partial_vu": partial_vu,
        "partial_perf": partial_perf,
        "partial_time": fin - debut,
        "last_basis": (
            dict_basis[TimeScenarioIndex(week, scenario)]
            if m.store_basis
            else Basis([], [])
        ),
    }

    return partial_results
