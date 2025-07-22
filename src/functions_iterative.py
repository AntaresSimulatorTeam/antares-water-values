from time import time

import numpy as np

from calculate_reward_and_bellman_values import (
    ReservoirManagement,
    calculate_VU,
    solve_weekly_problem_with_approximation,
)
from estimation import (
    Estimator,
    LinearCostEstimator,
    PieceWiseLinearInterpolator,
    UniVariateEstimator,
)
from optimization import AntaresProblem, Basis
from reservoir_management import MultiStockManagement
from type_definition import (
    AreaIndex,
    Array1D,
    Dict,
    List,
    Optional,
    TimeScenarioIndex,
    TimeScenarioParameter,
    WeekIndex,
)


def compute_x_multi_scenario(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    reward: LinearCostEstimator,
    V: Dict[WeekIndex, PieceWiseLinearInterpolator],
    itr: int,
) -> tuple[Dict[TimeScenarioIndex, float], Dict[TimeScenarioIndex, float]]:
    """
    Compute several optimal trajectories for the level of stock based on reward approximation and Bellman values. The number of trajectories is equal to the number of scenarios but trajectories doesn't depend on Monte Carlo years, ie for a given trajectory each week correspond to a random scenario.

    Parameters
    ----------
    bellman_value_calculation:BellmanValueCalculation:
        Parameters to use to calculate Bellman values
    V:Dict[WeekIndex, PieceWiseLinearInterpolator] :
        Bellman values
    itr:int :
        Iteration of iterative algorithm used to generate seed

    Returns
    -------
    initial_x:Dict[TimeScenarioIndex,float] :
        Trajectories
    controls:Dict[TimeScenarioIndex,float] :
        Controls associated to trajectories
    """
    initial_x: Dict[TimeScenarioIndex, float] = {}
    for s in range(param.len_scenario):
        initial_x[TimeScenarioIndex(0, s)] = (
            reservoir_management.reservoir.initial_level
        )
    np.random.seed(19 * itr)
    controls: Dict[TimeScenarioIndex, float] = {}

    for week in range(param.len_week):

        for trajectory, scenario in enumerate(
            np.random.permutation(range(param.len_scenario))
        ):

            _, xf, u, _ = solve_weekly_problem_with_approximation(
                week=week,
                scenario=scenario,
                level_i=initial_x[TimeScenarioIndex(week, trajectory)],
                V_fut=V[WeekIndex(week + 1)],
                reservoir_management=reservoir_management,
                param=param,
                reward=reward[TimeScenarioIndex(week, scenario)],
            )

            initial_x[TimeScenarioIndex(week + 1, trajectory)] = xf
            controls[TimeScenarioIndex(week, scenario)] = u

    return (initial_x, controls)


def compute_upper_bound(
    multi_stock_management: MultiStockManagement,
    param: TimeScenarioParameter,
    list_models: Dict[TimeScenarioIndex, AntaresProblem],
    V: Dict[WeekIndex, Estimator],
    reward_approximation: Optional[LinearCostEstimator] = None,
) -> tuple[
    float,
    Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
    Dict[TimeScenarioIndex, int],
    Dict[TimeScenarioIndex, float],
]:
    """
    Compute an approximate upper bound on the overall problem by solving the real complete Antares problem with Bellman values.

    Parameters
    ----------
    bellman_value_calculation: BellmanValueCalculation :
        Parameters to use to calculate Bellman values
    list_models:Dict[TimeScenarioIndex, AntaresProblem] :
        Optimization problems for every week and every scenario
    V:Dict[WeekIndex, Estimator] :
        Bellman values

    Returns
    -------
    upper_bound:float :
        Upper bound on the overall problem
    controls:Dict[TimeScenarioIndex, Dict[AreaIndex, float]] :
        Optimal controls for every week and every scenario
    current_itr:Dict[TimeScenarioIndex, int] :
        Simplex iterations used to solve the problem
    time:Dict[TimeScenarioIndex, float] :
        Time to solve the problem
    """

    current_itr = {}
    times = {}

    if reward_approximation is None:
        reward = LinearCostEstimator(
            param=param,
            controls={
                TimeScenarioIndex(week, scenario): [
                    {
                        area: res_management.reservoir.max_generating[week]
                        for area, res_management in multi_stock_management.dict_reservoirs.items()
                    }
                ]
                for week in range(param.len_week)
                for scenario in range(param.len_scenario)
            },
            costs={
                TimeScenarioIndex(week, scenario): [0]
                for week in range(param.len_week)
                for scenario in range(param.len_scenario)
            },
            duals={
                TimeScenarioIndex(week, scenario): [
                    {area: 0 for area in multi_stock_management.areas}
                ]
                for week in range(param.len_week)
                for scenario in range(param.len_scenario)
            },
            type_estimator="LinearInterpolator",
        )
    else:
        reward = reward_approximation

    cout = 0.0
    controls = {}
    for scenario in range(param.len_scenario):

        level_i = multi_stock_management.get_initial_level()
        for week in range(param.len_week):
            print(f"{scenario} {week}", end="\r")
            m = list_models[TimeScenarioIndex(week, scenario)]

            computational_time, itr, current_cost, _, control, level_i, _ = (
                m.solve_problem_with_bellman_values(
                    V=V[WeekIndex(week + 1)],
                    level_i=level_i,
                    take_into_account_z_and_y=(week == param.len_week - 1),
                    multi_stock_management=multi_stock_management,
                    param=param,
                    reward=reward,
                )
            )
            cout += current_cost
            controls[TimeScenarioIndex(week, scenario)] = control
            current_itr[TimeScenarioIndex(week, scenario)] = itr
            times[TimeScenarioIndex(week, scenario)] = computational_time

        upper_bound = cout / param.len_scenario
    return (upper_bound, controls, current_itr, times)


def calculate_reward(
    param: TimeScenarioParameter,
    controls: Dict[TimeScenarioIndex, float],
    list_models: Dict[TimeScenarioIndex, AntaresProblem],
    G: LinearCostEstimator,
    i: int,
    name_reservoir: AreaIndex,
) -> tuple[
    Dict[TimeScenarioIndex, int],
    Dict[TimeScenarioIndex, float],
    LinearCostEstimator,
]:
    """
    Evaluate reward for a set of given controls for each week and each scenario to update reward approximation.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    controls:Dict[TimeScenarioIndex, Dict[AreaIndex, float]] :
        Set of controls to evaluate
    list_models:Dict[TimeScenarioIndex, AntaresProblem] :
        Optimization problems for every week and every scenario
    G:Dict[TimeScenarioIndex, LinearInterpolator] :
        Reward approximation to update for every week and every scenario
    i:int :
        Iteration of iterative algorithm

    Returns
    -------
    current_itr:Dict[TimeScenarioIndex, int] :
        Simplex iterations used to solve the problem
    time:Dict[TimeScenarioIndex, float] :
        Time to solve the problem
    G:Dict[TimeScenarioIndex, LinearInterpolator] :
        Updated reward approximation
    """

    current_itr = {}
    times = {}

    for scenario in range(param.len_scenario):
        basis_0 = Basis([], [])
        for week in range(param.len_week):
            print(f"{scenario} {week}", end="\r")

            beta, lamb, itr, computation_time = list_models[
                TimeScenarioIndex(week, scenario)
            ].solve_with_predefined_controls(
                control={name_reservoir: controls[TimeScenarioIndex(week, scenario)]},
                prev_basis=basis_0 if i == 0 else Basis([], []),
            )
            if list_models[TimeScenarioIndex(week, scenario)].store_basis:
                basis_0 = list_models[TimeScenarioIndex(week, scenario)].basis[-1]
            else:
                basis_0 = Basis([], [])

            G[TimeScenarioIndex(week, scenario)].update(
                controls=np.array([[controls[TimeScenarioIndex(week, scenario)]]]),
                duals=np.array([[lamb[name_reservoir]]]),
                costs=np.array([beta]),
            )

            current_itr[TimeScenarioIndex(week, scenario)] = itr
            times[TimeScenarioIndex(week, scenario)] = computation_time

    return (current_itr, times, G)


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
    List[Dict[TimeScenarioIndex, float]],
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

        initial_x, controls = compute_x_multi_scenario(
            V=V, itr=i, param=param, reservoir_management=reservoir_management, reward=G
        )
        traj.append(initial_x)

        current_itr, times, G = calculate_reward(
            param=param,
            controls=controls,
            list_models=list_models,
            G=G,
            i=i,
            name_reservoir=reservoir_management.reservoir.area,
        )
        itr_tot.append(current_itr)

        V = calculate_VU(
            stock_discretization=X,
            time_scenario_param=param,
            reservoir_management=reservoir_management,
            reward=G,
        )

        V0 = V[WeekIndex(0)](reservoir_management.reservoir.initial_level)

        upper_bound, ctr, current_itr, times = compute_upper_bound(
            param=param,
            multi_stock_management=MultiStockManagement([reservoir_management]),
            list_models=list_models,
            V={
                WeekIndex(week): UniVariateEstimator(
                    {reservoir_management.reservoir.area.area: V[WeekIndex(week)]}
                )
                for week in range(param.len_week + 1)
            },
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
            WeekIndex(week): list(V[WeekIndex(week)].costs)
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
    Dict[WeekIndex, PieceWiseLinearInterpolator],
    List[Dict[TimeScenarioIndex, int]],
    List[Dict[TimeScenarioIndex, Dict[AreaIndex, float]]],
    List[Dict[TimeScenarioIndex, float]],
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

    V = {
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
