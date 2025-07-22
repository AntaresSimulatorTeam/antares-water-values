import numpy as np

from calculate_reward_and_bellman_values import calculate_VU
from estimation import (
    BellmanValueEstimation,
    Estimator,
    LinearCostEstimator,
    PieceWiseLinearInterpolator,
    UniVariateEstimator,
)
from functions_iterative import compute_upper_bound
from multi_stock_bellman_value_calculation import generate_controls, get_all_costs
from optimization import initialize_antares_problems
from reservoir_management import MultiStockManagement
from stock_discretization import StockDiscretization
from type_definition import (
    AreaIndex,
    Array1D,
    Array2D,
    Dict,
    TimeScenarioIndex,
    TimeScenarioParameter,
    WeekIndex,
)


def calculate_bellman_value_with_precalculated_reward(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    len_controls: int,
    len_bellman: int,
    name_solver: str = "CLP",
) -> tuple[
    Array2D,
    LinearCostEstimator,
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
    G:Dict[TimeScenarioIndex, LinearInterpolator] :
        Reward approximation
    """

    list_models = initialize_antares_problems(
        param=param,
        multi_stock_management=multi_stock_management,
        output_path=output_path,
        name_solver=name_solver,
    )

    controls = generate_controls(
        param=param,
        multi_stock_management=multi_stock_management,
        controls_looked_up="grid",
        xNsteps=len_controls,
    )

    costs, slopes, _ = get_all_costs(
        param=param, list_models=list_models, controls_list=controls
    )

    reward = LinearCostEstimator(
        param=param,
        controls=controls,
        costs=costs,
        duals=slopes,
        type_estimator="LinearInterpolator",
    )

    for area, reservoir_management in multi_stock_management.dict_reservoirs.items():
        X = np.linspace(0, reservoir_management.reservoir.capacity, num=len_bellman)

        V = calculate_VU(
            stock_discretization=X,
            time_scenario_param=param,
            reservoir_management=reservoir_management,
            reward=reward,
        )

    V0 = V[WeekIndex(0)](reservoir_management.reservoir.initial_level)

    upper_bound, control_ub, current_itr, times = compute_upper_bound(
        multi_stock_management=multi_stock_management,
        param=param,
        list_models=list_models,
        V={
            WeekIndex(week): UniVariateEstimator(
                {reservoir_management.reservoir.area.area: V[WeekIndex(week)]}
            )
            for week in range(param.len_week + 1)
        },
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return (
        np.transpose([V[WeekIndex(week)].costs for week in range(param.len_week + 1)]),
        reward,
    )


def calculate_bellman_value_directly(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    X: Dict[AreaIndex, Array1D],
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

    list_models = initialize_antares_problems(
        output_path=output_path,
        name_solver=solver,
        param=param,
        multi_stock_management=multi_stock_management,
    )

    stock_discretization = StockDiscretization(X)

    V: Dict[int, Estimator] = {}
    if univariate:
        assert len(X) == 1
        dim_bellman_value = tuple(len(x) for x in X.values())
        V = {
            week: UniVariateEstimator(
                {
                    area.area: PieceWiseLinearInterpolator(
                        X[area], np.zeros(len(X[area]), dtype=np.float32)
                    )
                    for area in multi_stock_management.areas
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
                        for area in multi_stock_management.dict_reservoirs.keys()
                    ]
                },
                stock_discretization,
            )
            for week in range(param.len_week + 1)
        }

    for week in range(param.len_week - 1, -1, -1):
        for scenario in range(param.len_scenario):
            print(f"{week} {scenario}", end="\r")
            m = list_models[TimeScenarioIndex(week, scenario)]

            for idx in stock_discretization.get_product_stock_discretization():
                _, _, Vu, slope, _, _, _ = m.solve_problem_with_bellman_values(
                    V=V[week + 1],
                    level_i={
                        area: stock_discretization.list_discretization[area][idx[i]]
                        for i, area in enumerate(m.range_reservoir)
                    },
                    find_optimal_basis=False,
                    take_into_account_z_and_y=True,
                    multi_stock_management=multi_stock_management,
                )
                V[week].update(
                    Vu,
                    {a.area: x for a, x in slope.items()},
                    param.len_scenario,
                    idx,
                    list([a.area for a in multi_stock_management.areas]),
                )

    V0 = V[0].get_value(
        {
            area.area: multi_stock_management.dict_reservoirs[
                area
            ].reservoir.initial_level
            for area in multi_stock_management.areas
        }
    )

    upper_bound, controls, current_itr, times = compute_upper_bound(
        multi_stock_management=multi_stock_management,
        param=param,
        list_models=list_models,
        V={WeekIndex(week): V[week] for week in range(param.len_week + 1)},
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return {week: V[week] for week in range(param.len_week + 1)}, V0, upper_bound
