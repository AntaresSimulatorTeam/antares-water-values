from time import time

import numpy as np

from calculate_reward_and_bellman_values import (
    compute_upper_bound,
    get_antares_costs,
    get_bellman_values_from_approximate_costs,
    get_optimal_trajectory_from_approximate_costs,
)
from estimation import Estimator, LinearCostEstimator, PieceWiseLinearInterpolator
from optimization import AntaresProblem
from reservoir_management import MultiStockManagement, ReservoirManagement
from type_definition import (
    AreaIndex,
    Array1D,
    Dict,
    List,
    TimeScenarioIndex,
    TimeScenarioParameter,
    WeekIndex,
    area_value_to_array,
    list_area_value_to_array,
)


def itr_control(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    N: int,
    tol_gap: float,
    solver: str = "GLOP",
) -> tuple[
    Dict[WeekIndex, List[float]],
    LinearCostEstimator,
    List[Dict[TimeScenarioIndex, int]],
    List[float],
    List[Dict[TimeScenarioIndex, Dict[AreaIndex, float]]],
    List[Dict[TimeScenarioIndex, Dict[AreaIndex, float]]],
    float,
    float,
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
    V:Dict[WeekIndex,List[float]] :
        Bellman values
    G:Dict[TimeScenarioIndex, LinearInterpolator] :
        Reward approximation
    itr:Dict[TimeScenarioIndex, int] :
        Time and simplex iterations used to solve optimization problems at each iteration
    tot_t:list[float] :
        Time spent at each iteration
    controls_upper:List[Dict[TimeScenarioIndex, Dict[AreaIndex, float]]] :
        Optimal controls found at each iteration during the evaluation of the upper bound
    traj:List[Dict[TimeScenarioIndex, float]] :
        Trajectories computed at each iteration
    """

    (
        tot_t,
        list_models,
        V,
        itr_tot,
        controls_upper,
        traj,
        gap,
        G,
    ) = init_iterative_calculation(param, reservoir_management, output_path, X, solver)
    i = 0

    while (gap >= tol_gap and gap >= 0) and i < N:
        debut = time()

        if i == 0:
            controls = {
                TimeScenarioIndex(w, s): {
                    reservoir_management.reservoir.area: reservoir_management.reservoir.inflow[
                        w, s
                    ]
                }
                for w in range(param.len_week)
                for s in range(param.len_scenario)
            }
            initial_x: Dict[TimeScenarioIndex, Dict[AreaIndex, float]] = {}
            for s in range(param.len_scenario):
                for w in range(param.len_week + 1):
                    initial_x[TimeScenarioIndex(w, s)] = {
                        reservoir_management.reservoir.area: reservoir_management.reservoir.initial_level
                    }
        else:
            initial_x, controls, _ = get_optimal_trajectory_from_approximate_costs(
                bellman_values=V,
                param=param,
                multi_stock_management=MultiStockManagement([reservoir_management]),
                costs_approx=G,
                random_seed=19 * i,
                random_scenario=True,
            )
        traj.append(initial_x)

        costs, duals, _, current_itr = get_antares_costs(
            param=param, controls=controls, list_models=list_models
        )
        for idx in G.estimators.keys():
            G[idx].update(
                controls=np.array([area_value_to_array(controls[idx])]),
                duals=list_area_value_to_array(duals[idx]),
                costs=np.array(costs[idx]),
            )
        itr_tot.append(current_itr)

        V = get_bellman_values_from_approximate_costs(
            levels={
                WeekIndex(w): [{reservoir_management.reservoir.area: x} for x in X]
                for w in range(param.len_week + 1)
            },
            param=param,
            multi_stock_management=MultiStockManagement([reservoir_management]),
            costs_approx=G,
            piecewiselinear=True,
        )

        V0 = V[WeekIndex(0)](
            {
                reservoir_management.reservoir.area: reservoir_management.reservoir.initial_level
            }
        )

        upper_bound, ctr, current_itr, times = compute_upper_bound(
            param=param,
            multi_stock_management=MultiStockManagement([reservoir_management]),
            list_models=list_models,
            V=V,
            reward_approximation=G,
        )
        itr_tot.append(current_itr)
        controls_upper.append(ctr)

        print(upper_bound + V0, upper_bound, -V0)
        gap = (upper_bound + V0) / -V0
        i += 1
        fin = time()
        tot_t.append(fin - debut)
    return (
        {
            WeekIndex(week): list(V[WeekIndex(week)].get_costs())
            for week in range(param.len_week + 1)
        },
        G,
        itr_tot,
        tot_t,
        controls_upper,
        traj,
        V0,
        upper_bound,
    )


def init_iterative_calculation(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    solver: str,
) -> tuple[
    List[float],
    Dict[TimeScenarioIndex, AntaresProblem],
    Dict[WeekIndex, Estimator],
    List[Dict[TimeScenarioIndex, int]],
    List[Dict[TimeScenarioIndex, Dict[AreaIndex, float]]],
    List[Dict[TimeScenarioIndex, Dict[AreaIndex, float]]],
    float,
    LinearCostEstimator,
]:
    len_week = param.len_week
    len_scenario = param.len_scenario

    tot_t = []
    debut = time()

    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {}
    for week in range(len_week):
        for scenario in range(len_scenario):
            m = AntaresProblem(
                scenario=scenario,
                week=week,
                path=output_path,
                name_solver=solver,
                param=param,
                multi_stock_management=MultiStockManagement([reservoir_management]),
            )
            list_models[TimeScenarioIndex(week, scenario)] = m

    V: Dict[WeekIndex, Estimator] = {
        WeekIndex(week): PieceWiseLinearInterpolator(
            X, np.zeros((len(X)), dtype=np.float32)
        )
        for week in range(len_week + 1)
    }

    G = LinearCostEstimator(
        param=param,
        controls={
            TimeScenarioIndex(week, scenario): [
                {
                    reservoir_management.reservoir.area: reservoir_management.reservoir.max_generating[
                        week
                    ]
                }
            ]
            for week in range(len_week)
            for scenario in range(len_scenario)
        },
        costs={
            TimeScenarioIndex(week, scenario): [0]
            for week in range(len_week)
            for scenario in range(len_scenario)
        },
        duals={
            TimeScenarioIndex(week, scenario): [
                {reservoir_management.reservoir.area: 0}
            ]
            for week in range(len_week)
            for scenario in range(len_scenario)
        },
        type_estimator="LinearInterpolator",
    )

    itr_tot: List = []
    controls_upper: List = []
    traj: List = []

    gap = 1e3
    fin = time()
    tot_t.append(fin - debut)
    return (
        tot_t,
        list_models,
        V,
        itr_tot,
        controls_upper,
        traj,
        gap,
        G,
    )
