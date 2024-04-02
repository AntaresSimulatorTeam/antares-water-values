from calculate_reward_and_bellman_values import (
    RewardApproximation,
    ReservoirManagement,
    BellmanValueCalculation,
)
from read_antares_data import TimeScenarioParameter, TimeScenarioIndex
from optimization import AntaresProblem
from typing import Annotated, Literal, Dict
from optimization import Basis, solve_problem_with_Bellman_values
import numpy.typing as npt
import numpy as np
from scipy.interpolate import interp1d
from functions_iterative import compute_upper_bound

Array1D = Annotated[npt.NDArray[np.float32], Literal["N"]]
Array2D = Annotated[npt.NDArray[np.float32], Literal["N", "N"]]


def calculate_complete_reward(
    len_controls: int,
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
) -> Dict[TimeScenarioIndex, RewardApproximation]:
    reward: Dict[TimeScenarioIndex, RewardApproximation] = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            r = RewardApproximation(
                lb_control=-reservoir_management.reservoir.max_pumping[week],
                ub_control=reservoir_management.reservoir.max_generating[week],
                ub_reward=0,
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

    for scenario in range(param.len_scenario):
        basis_0 = Basis([], [])
        for week in range(param.len_week):
            print(f"{scenario} {week}", end="\r")
            m = AntaresProblem(scenario=scenario, week=week, path=output_path, itr=1)
            m.create_weekly_problem_itr(
                param=param,
                reservoir_management=reservoir_management,
            )

            for u in controls[week]:
                beta, lamb, itr, basis_0, computation_time = (
                    m.modify_weekly_problem_itr(control=u, i=0, prev_basis=basis_0)
                )

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
) -> tuple[
    Array2D,
    Dict[TimeScenarioIndex, RewardApproximation],
]:

    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            m = AntaresProblem(scenario=scenario, week=week, path=output_path, itr=1)
            m.create_weekly_problem_itr(
                param=param,
                reservoir_management=reservoir_management,
            )
            list_models[TimeScenarioIndex(week, scenario)] = m

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

    upper_bound, controls, current_itr = compute_upper_bound(
        bellman_value_calculation=bellman_value_calculation,
        list_models=list_models,
        V=V,
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return (V, reward)


def calculate_bellman_value_directly(
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
) -> Array2D:

    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            m = AntaresProblem(scenario=scenario, week=week, path=output_path, itr=1)
            m.create_weekly_problem_itr(
                param=param,
                reservoir_management=reservoir_management,
            )
            list_models[TimeScenarioIndex(week, scenario)] = m

    reward: Dict[TimeScenarioIndex, RewardApproximation] = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            r = RewardApproximation(
                lb_control=-reservoir_management.reservoir.max_pumping[week],
                ub_control=reservoir_management.reservoir.max_generating[week],
                ub_reward=0,
            )
            reward[TimeScenarioIndex(week, scenario)] = r

    bellman_value_calculation = BellmanValueCalculation(
        param=param,
        reward=reward,
        reservoir_management=reservoir_management,
        stock_discretization=X,
    )

    V = np.zeros((len(X), param.len_week + 1), dtype=np.float32)

    for week in range(param.len_week - 1, -1, -1):
        for scenario in range(param.len_scenario):
            print(f"{week} {scenario}", end="\r")
            m = list_models[TimeScenarioIndex(week, scenario)]

            for i in range(len(X)):
                _, _, Vu, _, _ = solve_problem_with_Bellman_values(
                    bellman_value_calculation=bellman_value_calculation,
                    V=V,
                    scenario=scenario,
                    level_i=X[i],
                    week=week,
                    m=m,
                    find_optimal_basis=False,
                    take_into_account_z_and_y=True,
                )
                V[i, week] += -Vu / param.len_scenario

    V_fut = interp1d(X, V[:, 0])
    V0 = V_fut(reservoir_management.reservoir.initial_level)

    upper_bound, controls, current_itr = compute_upper_bound(
        bellman_value_calculation=bellman_value_calculation,
        list_models=list_models,
        V=V,
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return V