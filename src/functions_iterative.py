import numpy as np
from scipy.interpolate import interp1d
from time import time
from typing import Annotated, Literal, Dict
import numpy.typing as npt
from dataclasses import dataclass
from optimization import (
    AntaresProblem,
    solve_problem_with_Bellman_values,
    Basis,
)
from read_antares_data import TimeScenarioParameter
from calculate_reward_and_bellman_values import (
    RewardApproximation,
    ReservoirManagement,
    solve_weekly_problem_with_approximation,
    calculate_VU,
)

Array2D = Annotated[npt.NDArray[np.float32], Literal["N", "N"]]
Array1D = Annotated[npt.NDArray[np.float32], Literal["N"]]
Array3D = Annotated[npt.NDArray[np.float32], Literal["N", "N", "N"]]
Array4D = Annotated[npt.NDArray[np.float32], Literal["N", "N", "N", "N"]]


@dataclass(frozen=True)
class TimeScenarioIndex:
    time: int
    scenario: int


def compute_x_multi_scenario(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    X: Array1D,
    V: Array2D,
    reward: list[list[RewardApproximation]],
    itr: int,
) -> tuple[Array2D, Array2D]:
    """
    Compute several optimal trajectories for the level of stock based on reward approximation and Bellman values. The number of trajectories is equal to the number of scenarios but trajectories doesn't depend on Monte Carlo years, ie for a given trajectory each week correspond to a random scenario.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    reservoir:Reservoir :
        Reservoir considered
    X:np.array :
        Discretization of stock levels
    V:np.array :
        Bellman values
    reward:list[list[RewardApproximation]] :
        Reward approximation for every week and every scenario
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve
    pen_final:float :
        Penalty for violating final rule curves
    itr:int :
        Iteration of iterative algorithm

    Returns
    -------
    initial_x:np.array :
        Trajectories
    controls:np.array :
        Controls associated to trajectories
    """

    initial_x = np.zeros((param.len_week + 1, param.len_scenario), dtype=np.float32)
    initial_x[0] = reservoir_management.reservoir.initial_level
    np.random.seed(19 * itr)
    controls = np.zeros((param.len_week, param.len_scenario), dtype=np.float32)

    for week in range(param.len_week):

        V_fut = interp1d(X, V[:, week + 1])
        for trajectory, scenario in enumerate(
            np.random.permutation(range(param.len_scenario))
        ):

            pen = reservoir_management.get_penalty(week=week, len_week=param.len_week)
            Gs = reward[week][scenario].reward_function()

            _, xf, u = solve_weekly_problem_with_approximation(
                points=reward[week][scenario].breaking_point,
                X=X,
                inflow=reservoir_management.reservoir.inflow[week, scenario],
                lb=-reservoir_management.reservoir.max_pumping[week],
                ub=reservoir_management.reservoir.max_generating[week],
                level_i=initial_x[week, trajectory],
                xmax=reservoir_management.reservoir.upper_rule_curve[week],
                xmin=reservoir_management.reservoir.bottom_rule_curve[week],
                cap=reservoir_management.reservoir.capacity,
                pen=pen,
                V_fut=V_fut,
                Gs=Gs,
            )

            initial_x[week + 1, trajectory] = xf
            controls[week, scenario] = u

    return (initial_x, controls)


def compute_upper_bound(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    list_models: Dict[TimeScenarioIndex, AntaresProblem],
    X: Array1D,
    V: Array2D,
    G: list[list[RewardApproximation]],
) -> tuple[float, Array2D, Array3D]:
    """
    Compute an approximate upper bound on the overall problem by solving the real complete Antares problem with Bellman values.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    reservoir:Reservoir :
        Reservoir considered
    list_models:list[list[AntaresProblem]] :
        Optimization problems for every week and every scenario
    X:np.array :
        Discretization of Bellman values
    V:np.array :
        Bellman values
    G:list[list[RewardApproximation]] :
        Reward approximation for every week and every scenario
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve
    pen_final:float :
        Penalty for violating final rule curves

    Returns
    -------
    upper_bound:float :
        Upper bound on the overall problem
    controls:np.array :
        Optimal controls for every week and every scenario
    current_itr:np.array :
        Time and simplex iterations used to solve the problem
    """

    current_itr = np.zeros((param.len_week, param.len_scenario, 2), dtype=np.float32)

    cout = 0.0
    controls = np.zeros((param.len_week, param.len_scenario), dtype=np.float32)
    for scenario in range(param.len_scenario):

        level_i = reservoir_management.reservoir.initial_level
        for week in range(param.len_week):
            print(f"{scenario} {week}", end="\r")
            m = list_models[TimeScenarioIndex(week, scenario)]

            computational_time, itr, current_cost, control, level_i = (
                solve_problem_with_Bellman_values(
                    param=param,
                    reservoir_management=reservoir_management,
                    X=X,
                    V=V,
                    G=G,
                    scenario=scenario,
                    level_i=level_i,
                    week=week,
                    m=m,
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
    G: list[list[RewardApproximation]],
    i: int,
) -> tuple[Array3D, list[list[RewardApproximation]]]:
    """
    Evaluate reward for a set of given controls for each week and each scenario to update reward approximation.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    controls:list[list[float]] :
        Set of controls to evaluate
    list_models:list[list[AntaresProblem]] :
        Optimization problems for every week and every scenario
    G:list[list[RewardApproximation]] :
        Reward approximation to update for every week and every scenario
    i:int :
        Iteration of iterative algorithm

    Returns
    -------
    current_itr:np.array :
        Time and simplex iterations used to solve the problem
    G:list[list[RewardApproximation]] :
        Updated reward approximation
    """

    current_itr = np.zeros((param.len_week, param.len_scenario, 2), dtype=np.float32)

    for scenario in range(param.len_scenario):
        basis_0 = Basis([], [])
        for week in range(param.len_week):
            print(f"{scenario} {week}", end="\r")

            beta, lamb, itr, basis_0, computation_time = list_models[
                TimeScenarioIndex(week, scenario)
            ].modify_weekly_problem_itr(
                control=controls[week][scenario], i=i, prev_basis=basis_0
            )

            G[week][scenario].update_reward_approximation(
                lamb, beta, controls[week][scenario]
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
) -> tuple[
    Array2D,
    list[list[RewardApproximation]],
    Array4D,
    list[float],
    list[Array2D],
    list[Array2D],
]:
    """
    Algorithm to evaluate Bellman values. Each iteration of the algorithm consists in computing optimal trajectories based on reward approximation then evaluating rewards for those trajectories and finally updating reward approximation and calculating Bellman values. The algorithm stops when a certain number of iteratin is done or when the gap between the lower bound and the upper bound is small enough.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters for the Antares study
    reservoir:Reservoir :
        Reservoir considered for Bellman values
    output_path:str :
        Path to mps files describing optimization problems
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve
    X:np.array :
        Discretization of sotck levels for Bellman values
    N:int :
        Maximum number of iteration to do
    pen_final:float :
        Penalty for violating final rule curves
    tol_gap:float :
        Relative tolerance gap for the termination of the algorithm

    Returns
    -------
    V:np.array :
        Bellman values
    G:list[list[RewardApproximation]] :
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

    len_week = param.len_week
    len_scenario = param.len_scenario

    tot_t = []
    debut = time()

    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {}
    for week in range(len_week):
        for scenario in range(len_scenario):
            m = AntaresProblem(scenario=scenario, week=week, path=output_path, itr=1)
            m.create_weekly_problem_itr(
                param=param,
                reservoir_management=reservoir_management,
            )
            list_models[TimeScenarioIndex(week, scenario)] = m

    V = np.zeros((len(X), len_week + 1), dtype=np.float32)
    G = [
        [
            RewardApproximation(
                lb_control=-reservoir_management.reservoir.max_pumping[week],
                ub_control=reservoir_management.reservoir.max_generating[week],
                ub_reward=0,
            )
            for scenario in range(len_scenario)
        ]
        for week in range(len_week)
    ]

    itr_tot = []
    controls_upper = []
    traj = []

    i = 0
    gap = 1e3
    fin = time()
    tot_t.append(fin - debut)
    while (gap >= tol_gap and gap >= 0) and i < N:  # and (i<3):
        debut = time()

        initial_x, controls = compute_x_multi_scenario(
            param=param,
            reservoir_management=reservoir_management,
            X=X,
            V=V,
            reward=G,
            itr=i,
        )
        traj.append(np.array(initial_x))

        current_itr, G = calculate_reward(
            param=param, controls=controls, list_models=list_models, G=G, i=i
        )
        itr_tot.append(current_itr)

        V = calculate_VU(
            param=param,
            reward=G,
            reservoir_management=reservoir_management,
            X=X,
        )
        V_fut = interp1d(X, V[:, 0])
        V0 = V_fut(reservoir_management.reservoir.initial_level)

        upper_bound, controls, current_itr = compute_upper_bound(
            param=param,
            reservoir_management=reservoir_management,
            list_models=list_models,
            X=X,
            V=V,
            G=G,
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
