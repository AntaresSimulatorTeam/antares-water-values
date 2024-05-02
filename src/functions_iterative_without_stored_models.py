import numpy as np
from scipy.interpolate import interp1d
from time import time
from read_antares_data import TimeScenarioParameter, TimeScenarioIndex
from calculate_reward_and_bellman_values import (
    RewardApproximation,
    ReservoirManagement,
    BellmanValueCalculation,
)
from functions_iterative import (
    compute_x_multi_scenario,
    compute_upper_bound_without_stored_models,
)
from simple_bellman_value_calculation import calculate_reward_for_one_scenario
from type_definition import Array1D, Array2D, Array3D, Array4D, Dict, List, Optional
from multiprocessing import Pool
from functools import partial
from optimization import Basis


def calculate_reward(
    reward: Dict[TimeScenarioIndex, RewardApproximation],
    controls: Array2D,
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    solver: str,
    dict_basis: Dict[TimeScenarioIndex, Basis],
    processes: Optional[int] = None,
) -> tuple[
    Dict[TimeScenarioIndex, RewardApproximation],
    Array2D,
    Array4D,
    Dict[TimeScenarioIndex, Basis],
]:
    """
    Evaluate reward for a set of given controls for each week and each scenario to update reward approximation.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    controls:Array2D :
        Set of controls to evaluate
    list_models:Dict[TimeScenarioIndex, AntaresProblem] :
        Optimization problems for every week and every scenario
    G:Dict[TimeScenarioIndex, RewardApproximation] :
        Reward approximation to update for every week and every scenario
    i:int :
        Iteration of iterative algorithm

    Returns
    -------
    current_itr:Array3D :
        Time and simplex iterations used to solve the problem
    G:Dict[TimeScenarioIndex, RewardApproximation] :
        Updated reward approximation
    """

    dict_controls = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            dict_controls[TimeScenarioIndex(week, scenario)] = np.reshape(
                np.array(controls[week, scenario], dtype=np.float32), (1)
            )

    with Pool(processes=processes) as pool:

        scenario_reward = pool.map(
            partial(
                calculate_reward_for_one_scenario,
                param=param,
                reservoir_management=reservoir_management,
                output_path=output_path,
                controls=dict_controls,
                solver=solver,
                dict_basis=dict_basis,
            ),
            range(param.len_scenario),
        )

    tot_t = np.zeros((param.len_week, param.len_scenario), dtype=np.float32)
    perf = np.zeros((param.len_week, param.len_scenario, 1, 2), dtype=np.float32)

    for scenario in range(param.len_scenario):
        for week in range(param.len_week):
            reward[TimeScenarioIndex(week, scenario)].update_reward_approximation(
                slope_new_cut=scenario_reward[scenario][0][
                    TimeScenarioIndex(week, scenario)
                ].list_cut[0][0],
                intercept_new_cut=scenario_reward[scenario][0][
                    TimeScenarioIndex(week, scenario)
                ].list_cut[0][1],
            )
            tot_t[week][scenario] = scenario_reward[scenario][1][week]
            perf[week][scenario] = scenario_reward[scenario][2][week]
            dict_basis[TimeScenarioIndex(week=week, scenario=scenario)] = (
                scenario_reward[scenario][3][TimeScenarioIndex(week, scenario)]
            )

    return reward, tot_t, perf, dict_basis


def calculate_bellman_values_with_iterative_method_without_stored_models(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    N: int,
    tol_gap: float,
    solver: str = "GLOP",
    processes: Optional[int] = None,
) -> tuple[
    Array2D,
    Dict[TimeScenarioIndex, RewardApproximation],
    Array4D,
    list[float],
    list[Array2D],
    list[Array2D],
]:
    """
    Algorithm to evaluate Bellman values. Each iteration of the algorithm consists in computing optimal trajectories based on reward approximation then evaluating rewards for those trajectories and finally updating reward approximation and calculating Bellman values. The algorithm stops when a certain number of iterations is done or when the gap between the lower bound and the upper bound is small enough.

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
    G:Dict[TimeScenarioIndex, RewardApproximation] :
        Reward approximation
    itr:np.array :
        Time and simplex iterations used to solve optimization problems at each iteration
    tot_t:list[float] :
        Time spent at each iteration
    controls_upper:list[np.array] :
        Optimal controls found at each iteration during the evaluation of the upper bound
    traj:list[np.array] :
        Trajectories computed at each iteration
    """

    (
        tot_t,
        V,
        itr_tot,
        controls_upper,
        traj,
        bellman_value_calculation,
        gap,
        G,
    ) = init_iterative_calculation(param, reservoir_management, output_path, X, solver)
    i = 0

    dict_basis = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            dict_basis[TimeScenarioIndex(week, scenario)] = Basis([], [])

    while (gap >= tol_gap and gap >= 0) and i < N:  # and (i<3):
        debut = time()

        initial_x, controls = compute_x_multi_scenario(
            bellman_value_calculation=bellman_value_calculation,
            V=V,
            itr=i,
        )
        traj.append(np.array(initial_x))

        G, _, current_itr, dict_basis = calculate_reward(
            reward=G,
            param=param,
            controls=controls,
            reservoir_management=reservoir_management,
            output_path=output_path,
            solver=solver,
            processes=processes,
            dict_basis=dict_basis,
        )
        itr_tot.append(current_itr[:, :, 0])

        bellman_value_calculation = BellmanValueCalculation(
            param=param,
            reward=G,
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
        itr_tot.append(current_itr)
        controls_upper.append(controls)

        gap = upper_bound + V0
        print(gap, upper_bound, -V0)
        gap = gap / -V0
        i += 1
        fin = time()
        tot_t.append(fin - debut)
    return (V, G, np.array(itr_tot), tot_t, controls_upper, traj)


def init_iterative_calculation(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    solver: str,
) -> tuple[
    List,
    Array2D,
    List,
    List,
    List,
    BellmanValueCalculation,
    float,
    Dict[TimeScenarioIndex, RewardApproximation],
]:
    len_week = param.len_week
    len_scenario = param.len_scenario

    tot_t = []
    debut = time()

    V = np.zeros((len(X), len_week + 1), dtype=np.float32)

    G: Dict[TimeScenarioIndex, RewardApproximation] = {}
    for week in range(len_week):
        for scenario in range(len_scenario):
            r = RewardApproximation(
                lb_control=-reservoir_management.reservoir.max_pumping[week]
                * reservoir_management.reservoir.efficiency,
                ub_control=reservoir_management.reservoir.max_generating[week],
                ub_reward=0,
            )
            G[TimeScenarioIndex(week, scenario)] = r

    itr_tot: List = []
    controls_upper: List = []
    traj: List = []

    bellman_value_calculation = BellmanValueCalculation(
        param=param,
        reward=G,
        reservoir_management=reservoir_management,
        stock_discretization=X,
    )

    i = 0
    gap = 1e3
    fin = time()
    tot_t.append(fin - debut)
    return (
        tot_t,
        V,
        itr_tot,
        controls_upper,
        traj,
        bellman_value_calculation,
        gap,
        G,
    )
