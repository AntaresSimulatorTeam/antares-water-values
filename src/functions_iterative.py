import pickle as pkl
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from time import time

import numpy as np
from ortools.linear_solver.python import model_builder
from scipy.interpolate import interp1d

from calculate_reward_and_bellman_values import (
    BellmanValueCalculation,
    ReservoirManagement,
    RewardApproximation,
)
from optimization import AntaresProblem, Basis
from read_antares_data import TimeScenarioIndex, TimeScenarioParameter
from type_definition import Array1D, Array2D, Array3D, Array4D, Dict, List, Optional


def compute_x_multi_scenario(
    bellman_value_calculation: BellmanValueCalculation,
    V: Array2D,
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
    param = bellman_value_calculation.time_scenario_param
    initial_x = np.zeros(
        (
            param.len_week + 1,
            param.len_scenario,
        ),
        dtype=np.float32,
    )
    initial_x[0] = (
        bellman_value_calculation.reservoir_management.reservoir.initial_level
    )
    np.random.seed(19 * itr)
    controls = np.zeros(
        (
            param.len_week,
            param.len_scenario,
        ),
        dtype=np.float32,
    )

    for week in range(param.len_week):

        V_fut = interp1d(bellman_value_calculation.stock_discretization, V[:, week + 1])
        for trajectory, scenario in enumerate(
            np.random.permutation(range(param.len_scenario))
        ):

            _, xf, u = (
                bellman_value_calculation.solve_weekly_problem_with_approximation(
                    week=week,
                    scenario=scenario,
                    level_i=initial_x[week, trajectory],
                    V_fut=V_fut,
                )
            )

            initial_x[week + 1, trajectory] = xf
            controls[week, scenario] = u

    return (initial_x, controls)


def compute_upper_bound_with_stored_models(
    bellman_value_calculation: BellmanValueCalculation,
    list_models: Dict[TimeScenarioIndex, AntaresProblem],
    V: Array2D,
) -> tuple[float, Array2D, Array3D]:
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
    param = bellman_value_calculation.time_scenario_param

    current_itr = np.zeros((param.len_week, param.len_scenario, 2), dtype=np.float32)

    cout = 0.0
    controls = np.zeros((param.len_week, param.len_scenario), dtype=np.float32)
    for scenario in range(param.len_scenario):

        level_i = bellman_value_calculation.reservoir_management.reservoir.initial_level
        for week in range(param.len_week):
            print(f"{scenario} {week}", end="\r")
            m = list_models[TimeScenarioIndex(week, scenario)]
            basis = find_basis(
                bellman_value_calculation,
                V[:, week + 1],
                scenario,
                level_i,
                week,
                m,
            )

            computational_time, itr, current_cost, control, level_i = (
                m.solve_problem_with_bellman_values(
                    stock_discretization=bellman_value_calculation.stock_discretization,
                    reservoir_management=bellman_value_calculation.reservoir_management,
                    V=V,
                    level_i=level_i,
                    take_into_account_z_and_y=(week == param.len_week - 1),
                    basis=basis,
                )
            )
            cout += current_cost
            controls[week, scenario] = control
            current_itr[week, scenario] = (itr, computational_time)

        upper_bound = cout / param.len_scenario
    return (upper_bound, controls, current_itr)


def calculate_reward(
    param: TimeScenarioParameter,
    controls: Array2D,
    list_models: Dict[TimeScenarioIndex, AntaresProblem],
    G: Dict[TimeScenarioIndex, RewardApproximation],
    i: int,
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
                control=float(controls[week][scenario]),
                prev_basis=basis_0 if i == 0 else Basis([], []),
            )
            if list_models[TimeScenarioIndex(week, scenario)].store_basis:
                basis_0 = list_models[TimeScenarioIndex(week, scenario)].basis[-1]
            else:
                basis_0 = Basis([], [])

            G[TimeScenarioIndex(week, scenario)].update_reward_approximation(
                slope_new_cut=-lamb,
                intercept_new_cut=-beta + lamb * controls[week][scenario],
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
        list_models,
        V,
        itr_tot,
        controls_upper,
        traj,
        bellman_value_calculation,
        gap,
        G,
    ) = init_iterative_calculation(param, reservoir_management, output_path, X, solver)
    i = 0

    while (gap >= tol_gap and gap >= 0) and i < N:  # and (i<3):
        debut = time()

        initial_x, controls = compute_x_multi_scenario(
            bellman_value_calculation=bellman_value_calculation,
            V=V,
            itr=i,
        )
        traj.append(np.array(initial_x))

        current_itr, G = calculate_reward(
            param=param, controls=controls, list_models=list_models, G=G, i=i
        )
        itr_tot.append(current_itr)

        bellman_value_calculation = BellmanValueCalculation(
            param=param,
            reward=G,
            reservoir_management=reservoir_management,
            stock_discretization=X,
        )

        V = bellman_value_calculation.calculate_VU()

        V_fut = interp1d(X, V[:, 0])
        V0 = V_fut(reservoir_management.reservoir.initial_level)

        upper_bound, controls, current_itr = compute_upper_bound_with_stored_models(
            bellman_value_calculation=bellman_value_calculation,
            list_models=list_models,
            V=V,
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
    Dict[TimeScenarioIndex, AntaresProblem],
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
                reservoir_management=reservoir_management,
            )
            list_models[TimeScenarioIndex(week, scenario)] = m

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
        list_models,
        V,
        itr_tot,
        controls_upper,
        traj,
        bellman_value_calculation,
        gap,
        G,
    )


def create_model(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    week: int,
    scenario: int,
    solver: str,
    saving_dir: str,
) -> AntaresProblem:
    if len(param.name_scenario) == param.len_scenario:
        proto_path = (
            saving_dir + f"/problem-{param.name_scenario[scenario]}-{week+1}.pkl"
        )
    else:
        proto_path = saving_dir + f"/problem-{scenario}-{week+1}.pkl"
    if Path(proto_path).is_file():
        m = AntaresProblem(
            scenario=scenario,
            week=week,
            path=output_path,
            itr=1,
            name_solver=solver,
            name_scenario=(
                param.name_scenario[scenario] if len(param.name_scenario) > 1 else -1
            ),
            load_from_proto=True,
            proto_path=proto_path,
        )
        m.reset(reservoir_management)
    else:
        m = AntaresProblem(
            scenario=scenario,
            week=week,
            path=output_path,
            itr=1,
            name_solver=solver,
            name_scenario=(
                param.name_scenario[scenario] if len(param.name_scenario) > 1 else -1
            ),
        )
        m.create_weekly_problem_itr(
            param=param,
            reservoir_management=reservoir_management,
        )

        proto = model_builder.ModelBuilder().export_to_proto()  # type: ignore[no-untyped-call]
        m.solver.ExportModelToProto(output_model=proto)
        with open(proto_path, "wb") as file:
            pkl.dump(proto, file)

    return m


def find_basis(
    bellman_value_calculation: BellmanValueCalculation,
    V: Array1D,
    scenario: int,
    level_i: float,
    week: int,
    m: AntaresProblem,
) -> Basis:

    basis = Basis([], [])

    if len(m.control_basis) >= 1:
        if len(m.control_basis) >= 2:
            V_fut = interp1d(
                bellman_value_calculation.stock_discretization,
                V,
            )

            _, _, likely_control = (
                bellman_value_calculation.solve_weekly_problem_with_approximation(
                    level_i=level_i,
                    V_fut=V_fut,
                    week=week,
                    scenario=scenario,
                )
            )
        else:
            likely_control = 0
        basis = m.find_closest_basis(likely_control)
    return basis
