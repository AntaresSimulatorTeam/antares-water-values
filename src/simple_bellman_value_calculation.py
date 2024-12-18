import numpy as np

from calculate_reward_and_bellman_values import (
    BellmanValueCalculation,
    MultiStockBellmanValueCalculation,
    ReservoirManagement,
    RewardApproximation,
)
from estimation import (
    BellmanValueEstimation,
    Estimator,
    PieceWiseLinearInterpolator,
    UniVariateEstimator,
)
from functions_iterative import compute_upper_bound
from optimization import AntaresProblem, Basis
from read_antares_data import TimeScenarioIndex, TimeScenarioParameter
from reservoir_management import MultiStockManagement
from stock_discretization import StockDiscretization
from type_definition import Array1D, Array2D, Dict, Union


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
                lb_control=-reservoir_management.reservoir.max_pumping[week]
                * reservoir_management.reservoir.efficiency,
                ub_control=reservoir_management.reservoir.max_generating[week],
                ub_reward=float("inf"),
            )
            reward[TimeScenarioIndex(week, scenario)] = r

    controls = np.array(
        [
            np.linspace(
                -reservoir_management.reservoir.max_pumping[week]
                * reservoir_management.reservoir.efficiency,
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
                multi_stock_management=MultiStockManagement([reservoir_management]),
            )

            for u in controls[week]:
                beta, lamb, itr, computation_time = m.solve_with_predefined_controls(
                    control={reservoir_management.reservoir.area: u},
                    prev_basis=basis_0,
                )
                if m.store_basis:
                    basis_0 = m.basis[-1]
                else:
                    basis_0 = Basis([], [])

                reward[TimeScenarioIndex(week, scenario)].update(
                    duals=-lamb[reservoir_management.reservoir.area],
                    costs=-beta + lamb[reservoir_management.reservoir.area] * u,
                )

    return reward


def calculate_bellman_value_with_precalculated_reward(
    len_controls: int,
    param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    output_path: str,
    X: Array1D,
    solver: str = "CLP",
) -> tuple[
    Array2D,
    Dict[TimeScenarioIndex, RewardApproximation],
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
    """

    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            m = AntaresProblem(
                scenario=scenario,
                week=week,
                path=output_path,
                itr=1,
                name_solver=solver,
            )
            m.create_weekly_problem_itr(
                param=param,
                multi_stock_management=MultiStockManagement([reservoir_management]),
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

    V0 = V[0](reservoir_management.reservoir.initial_level)

    upper_bound, controls, current_itr = compute_upper_bound(
        param=param,
        multi_bellman_value_calculation=MultiStockBellmanValueCalculation(
            [bellman_value_calculation]
        ),
        list_models=list_models,
        V={
            week: UniVariateEstimator({reservoir_management.reservoir.area: V[week]})
            for week in range(param.len_week + 1)
        },
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return (np.transpose([V[week].costs for week in range(param.len_week + 1)]), reward)


def calculate_bellman_value_directly(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    X: Dict[str, Array1D],
    univariate: bool,
    solver: str = "CLP",
) -> tuple[Dict[int, Estimator], float, float]:
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
    """

    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {}
    for week in range(param.len_week):
        for scenario in range(param.len_scenario):
            m = AntaresProblem(
                scenario=scenario,
                week=week,
                path=output_path,
                itr=1,
                name_solver=solver,
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
    V: Dict[int, Estimator] = {}
    if univariate:
        assert len(X) == 1
        dim_bellman_value = tuple(len(x) for x in X.values())
        V = {
            week: UniVariateEstimator(
                {
                    area: PieceWiseLinearInterpolator(
                        X[area], np.zeros(len(X[area]), dtype=np.float32)
                    )
                    for area in multi_stock_management.dict_reservoirs.keys()
                }
            )
            for week in range(param.len_week + 1)
        }
    else:
        dim_bellman_value = tuple(len(x) for x in X.values())
        V = {
            week: BellmanValueEstimation(
                {
                    name: np.zeros((dim_bellman_value), dtype=np.float32)
                    for name in ["intercept"]
                    + [
                        f"slope_{area}"
                        for area in multi_bellman_value_calculation.dict_reservoirs.keys()
                    ]
                },
                StockDiscretization(X),
            )
            for week in range(param.len_week + 1)
        }

    for week in range(param.len_week - 1, -1, -1):
        for scenario in range(param.len_scenario):
            print(f"{week} {scenario}", end="\r")
            m = list_models[TimeScenarioIndex(week, scenario)]

            for (
                idx
            ) in multi_bellman_value_calculation.get_product_stock_discretization():
                _, _, Vu, slope, _, _, _ = m.solve_problem_with_bellman_values(
                    multi_bellman_value_calculation=multi_bellman_value_calculation,
                    V=V[week + 1],
                    level_i={
                        area: multi_bellman_value_calculation.dict_reservoirs[
                            area
                        ].stock_discretization[idx[i]]
                        for i, area in enumerate(m.range_reservoir)
                    },
                    find_optimal_basis=False,
                    take_into_account_z_and_y=True,
                )
                V[week].update(
                    Vu,
                    slope,
                    param.len_scenario,
                    idx,
                    list(multi_stock_management.dict_reservoirs.keys()),
                )

    V0 = V[0].get_value(
        {
            area: multi_stock_management.dict_reservoirs[area].reservoir.initial_level
            for area in multi_stock_management.dict_reservoirs.keys()
        }
    )

    upper_bound, controls, current_itr = compute_upper_bound(
        param=param,
        multi_bellman_value_calculation=multi_bellman_value_calculation,
        list_models=list_models,
        V={week: V[week] for week in range(param.len_week + 1)},
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return {week: V[week] for week in range(param.len_week + 1)}, V0, upper_bound
