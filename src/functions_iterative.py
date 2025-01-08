from time import time

import numpy as np

from calculate_reward_and_bellman_values import (
    ReservoirManagement,
    RewardApproximation,
    calculate_VU,
    solve_weekly_problem_with_approximation,
)
from estimation import Estimator, PieceWiseLinearInterpolator, UniVariateEstimator
from optimization import AntaresProblem, Basis
from read_antares_data import TimeScenarioIndex, TimeScenarioParameter
from reservoir_management import MultiStockManagement
from stock_discretization import StockDiscretization
from type_definition import (
    AreaIndex,
    Array1D,
    Array2D,
    Array3D,
    Array4D,
    Dict,
    List,
    Optional,
)


def compute_x_multi_scenario(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    reward: Dict[TimeScenarioIndex, RewardApproximation],
    V: Dict[int, PieceWiseLinearInterpolator],
    itr: int,
) -> tuple[Array2D, Array2D]:
    """
    Compute several optimal trajectories for the level of stock based on reward approximation and Bellman values. The number of trajectories is equal to the number of scenarios but trajectories doesn't depend on Monte Carlo years, ie for a given trajectory each week correspond to a random scenario.

    Parameters
    ----------
    bellman_value_calculation:BellmanValueCalculation:
        Parameters to use to calculate Bellman values
    V:np.array :
        Bellman values
    itr:int :
        Iteration of iterative algorithm used to generate seed

    Returns
    -------
    initial_x:np.array :
        Trajectories
    controls:np.array :
        Controls associated to trajectories
    """
    initial_x = np.zeros(
        (
            param.len_week + 1,
            param.len_scenario,
        ),
        dtype=np.float32,
    )
    initial_x[0] = reservoir_management.reservoir.initial_level
    np.random.seed(19 * itr)
    controls = np.zeros(
        (
            param.len_week,
            param.len_scenario,
        ),
        dtype=np.float32,
    )

    for week in range(param.len_week):

        for trajectory, scenario in enumerate(
            np.random.permutation(range(param.len_scenario))
        ):

            _, xf, u = solve_weekly_problem_with_approximation(
                week=week,
                scenario=scenario,
                level_i=initial_x[week, trajectory],
                V_fut=V[week + 1],
                reservoir_management=reservoir_management,
                param=param,
                reward=reward[TimeScenarioIndex(week, scenario)],
            )

            initial_x[week + 1, trajectory] = xf
            controls[week, scenario] = u

    return (initial_x, controls)


def compute_upper_bound(
    multi_stock_management: MultiStockManagement,
    stock_discretization: StockDiscretization,
    param: TimeScenarioParameter,
    list_models: Dict[TimeScenarioIndex, AntaresProblem],
    V: Dict[int, Estimator],
    reward_approximation: Optional[
        Dict[str, Dict[TimeScenarioIndex, RewardApproximation]]
    ] = None,
) -> tuple[float, Dict[TimeScenarioIndex, Dict[str, float]], Array3D]:
    """
    Compute an approximate upper bound on the overall problem by solving the real complete Antares problem with Bellman values.

    Parameters
    ----------
    bellman_value_calculation: BellmanValueCalculation :
        Parameters to use to calculate Bellman values
    list_models:Dict[TimeScenarioIndex, AntaresProblem] :
        Optimization problems for every week and every scenario
    V:Array2D :
        Bellman values

    Returns
    -------
    upper_bound:float :
        Upper bound on the overall problem
    controls:Array2D :
        Optimal controls for every week and every scenario
    current_itr:Array2D :
        Time and simplex iterations used to solve the problem
    """

    current_itr = np.zeros((param.len_week, param.len_scenario, 2), dtype=np.float32)

    if reward_approximation is None:
        reward: Dict[str, Dict[TimeScenarioIndex, RewardApproximation]] = {}
        for (
            area,
            reservoir_management,
        ) in multi_stock_management.dict_reservoirs.items():
            reward[area.area] = {}
            for week in range(param.len_week):
                for scenario in range(param.len_scenario):
                    r = RewardApproximation(
                        lb_control=-reservoir_management.reservoir.max_pumping[week],
                        ub_control=reservoir_management.reservoir.max_generating[week],
                        ub_reward=0,
                    )
                    reward[area.area][TimeScenarioIndex(week, scenario)] = r
    else:
        reward = reward_approximation

    cout = 0.0
    controls = {}
    for scenario in range(param.len_scenario):

        level_i = {
            area.area: multi_stock_management.dict_reservoirs[
                area
            ].reservoir.initial_level
            for area in multi_stock_management.areas
        }
        for week in range(param.len_week):
            print(f"{scenario} {week}", end="\r")
            m = list_models[TimeScenarioIndex(week, scenario)]

            computational_time, itr, current_cost, _, control, level_i, _ = (
                m.solve_problem_with_bellman_values(
                    V=V[week + 1],
                    level_i=level_i,
                    take_into_account_z_and_y=(week == param.len_week - 1),
                    multi_stock_management=multi_stock_management,
                    stock_discretization=stock_discretization,
                    param=param,
                    reward=reward,
                )
            )
            cout += current_cost
            controls[TimeScenarioIndex(week, scenario)] = control
            current_itr[week, scenario] = (itr, computational_time)

        upper_bound = cout / param.len_scenario
    return (upper_bound, controls, current_itr)


def calculate_reward(
    param: TimeScenarioParameter,
    controls: Array2D,
    list_models: Dict[TimeScenarioIndex, AntaresProblem],
    G: Dict[TimeScenarioIndex, RewardApproximation],
    i: int,
    name_reservoir: str,
) -> tuple[Array3D, Dict[TimeScenarioIndex, RewardApproximation]]:
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

    current_itr = np.zeros((param.len_week, param.len_scenario, 2), dtype=np.float32)

    for scenario in range(param.len_scenario):
        basis_0 = Basis([], [])
        for week in range(param.len_week):
            print(f"{scenario} {week}", end="\r")

            beta, lamb, itr, computation_time = list_models[
                TimeScenarioIndex(week, scenario)
            ].solve_with_predefined_controls(
                control={name_reservoir: float(controls[week][scenario])},
                prev_basis=basis_0 if i == 0 else Basis([], []),
            )
            if list_models[TimeScenarioIndex(week, scenario)].store_basis:
                basis_0 = list_models[TimeScenarioIndex(week, scenario)].basis[-1]
            else:
                basis_0 = Basis([], [])

            G[TimeScenarioIndex(week, scenario)].update(
                duals=-lamb[name_reservoir],
                costs=-beta + lamb[name_reservoir] * controls[week][scenario],
            )

            current_itr[week, scenario] = (itr, computation_time)

    return (current_itr, G)


def itr_control(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    N: int,
    tol_gap: float,
    solver: str = "GLOP",
) -> tuple[
    Array2D,
    Dict[TimeScenarioIndex, RewardApproximation],
    Array4D,
    list[float],
    list[Dict[TimeScenarioIndex, Dict[str, float]]],
    list[Array2D],
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
        traj.append(np.array(initial_x))

        current_itr, G = calculate_reward(
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

        V0 = V[0](reservoir_management.reservoir.initial_level)

        upper_bound, ctr, current_itr = compute_upper_bound(
            param=param,
            multi_stock_management=MultiStockManagement([reservoir_management]),
            list_models=list_models,
            V={
                week: UniVariateEstimator(
                    {reservoir_management.reservoir.area: V[week]}
                )
                for week in range(param.len_week + 1)
            },
            stock_discretization=StockDiscretization(
                {AreaIndex(reservoir_management.reservoir.area): X}
            ),
            reward_approximation={reservoir_management.reservoir.area: G},
        )
        itr_tot.append(current_itr)
        controls_upper.append(ctr)

        gap = upper_bound + V0
        print(gap, upper_bound, -V0)
        gap = gap / -V0
        i += 1
        fin = time()
        tot_t.append(fin - debut)
    return (
        np.transpose([V[week].costs for week in range(param.len_week + 1)]),
        G,
        np.array(itr_tot),
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
    List,
    Dict[TimeScenarioIndex, AntaresProblem],
    Dict[int, PieceWiseLinearInterpolator],
    List,
    List,
    List,
    float,
    Dict[TimeScenarioIndex, RewardApproximation],
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
                itr=1,
                name_solver=solver,
                name_scenario=(
                    param.name_scenario[scenario]
                    if len(param.name_scenario) > 1
                    else -1
                ),
            )
            m.create_weekly_problem_itr(
                param=param,
                multi_stock_management=MultiStockManagement([reservoir_management]),
            )
            list_models[TimeScenarioIndex(week, scenario)] = m

    V = {
        week: PieceWiseLinearInterpolator(X, np.zeros((len(X)), dtype=np.float32))
        for week in range(len_week + 1)
    }

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
