from calculate_reward_and_bellman_values import (
    RewardApproximation,
    ReservoirManagement,
    BellmanValueCalculation,
    MultiStockManagement,
    MultiStockBellmanValueCalculation,
)
from read_antares_data import TimeScenarioParameter, TimeScenarioIndex
from optimization import AntaresProblem
from typing import Annotated, Literal, Dict
from optimization import Basis, solve_problem_with_multivariate_bellman_values
import numpy.typing as npt
import numpy as np
from scipy.interpolate import interp1d
from functions_iterative import compute_upper_bound

Array1D = Annotated[npt.NDArray[np.float32], Literal["N"]]
Array2D = Annotated[npt.NDArray[np.float32], Literal["N", "N"]]


def calculate_bellman_value_multi_stock(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    X: Dict[str, Array1D],
    name_solver: str = "CLP",
) -> Dict[int, Dict[str, npt.NDArray[np.float32]]]:
    """Function to calculate multivariate Bellman values

    Args:
        param (TimeScenarioParameter): Time-related parameters for the Antares study
        multi_stock_management (MultiStockManagement): Stocks considered for Bellman values
        output_path (str): Path to mps files describing optimization problems
        X (Dict[str, Array1D]): Discretization of sotck levels for Bellman values for each reservoir
        name_solver (str, optional): Solver to use with ortools. Defaults to "CLP".

    Returns:
        Dict[int, Dict[str, npt.NDArray[np.float32]]]: Bellman values for each week. Bellman values are described by a slope for each area and a intercept
    """

    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            m = AntaresProblem(
                scenario=scenario,
                week=week,
                path=output_path,
                itr=1,
                name_solver=name_solver,
            )
            m.create_weekly_problem_itr(
                param=param,
                multi_stock_management=multi_stock_management,
            )
            list_models[TimeScenarioIndex(week, scenario)] = m

    reward: Dict[str, Dict[TimeScenarioIndex, RewardApproximation]] = {}
    for area, reservoir_management in multi_stock_management.dict_reservoirs.items():
        reward[area] = {}
        for week in range(param.len_week):
            for scenario in range(param.len_scenario):
                r = RewardApproximation(
                    lb_control=-reservoir_management.reservoir.max_pumping[week],
                    ub_control=reservoir_management.reservoir.max_generating[week],
                    ub_reward=0,
                )
                reward[area][TimeScenarioIndex(week, scenario)] = r

    bellman_value_calculation = []
    for area, reservoir_management in multi_stock_management.dict_reservoirs.items():
        bellman_value_calculation.append(
            BellmanValueCalculation(
                param=param,
                reward=reward[area],
                reservoir_management=reservoir_management,
                stock_discretization=X[area],
            )
        )
    multi_bellman_value_calculation = MultiStockBellmanValueCalculation(
        bellman_value_calculation
    )

    dim_bellman_value = tuple(len(x) for x in X.values())
    V = {
        week: {
            name: np.zeros((dim_bellman_value), dtype=np.float32)
            for name in ["intercept"]
            + [
                f"slope_{area}"
                for area in multi_bellman_value_calculation.dict_reservoirs.keys()
            ]
        }
        for week in range(param.len_week + 1)
    }

    for week in range(param.len_week - 1, -1, -1):
        for scenario in range(param.len_scenario):
            print(f"{week} {scenario}", end="\r")
            m = list_models[TimeScenarioIndex(week, scenario)]

            for (
                idx
            ) in multi_bellman_value_calculation.get_product_stock_discretization():
                _, _, Vu, slope, _ = solve_problem_with_multivariate_bellman_values(
                    multi_bellman_value_calculation=multi_bellman_value_calculation,
                    V=V[week + 1],
                    scenario=scenario,
                    level_i={
                        area: multi_bellman_value_calculation.dict_reservoirs[
                            area
                        ].stock_discretization[idx[i]]
                        for i, area in enumerate(m.range_reservoir)
                    },
                    week=week,
                    m=m,
                    take_into_account_z_and_y=True,
                )
                V[week]["intercept"][idx] += Vu / param.len_scenario
                for area in m.range_reservoir:
                    V[week][f"slope_{area}"][idx] += slope[area] / param.len_scenario

    # V_fut = interp1d(X, V[:, 0])
    # V0 = V_fut(reservoir_management.reservoir.initial_level)

    # upper_bound, controls, current_itr = compute_upper_bound(
    #     bellman_value_calculation=bellman_value_calculation,
    #     list_models=list_models,
    #     V=V,
    # )

    # gap = upper_bound + V0
    # print(gap, upper_bound, -V0)

    return V
