import numpy as np

from calculate_reward_and_bellman_values import RewardApproximation, calculate_VU
from estimation import (
    BellmanValueEstimation,
    Estimator,
    PieceWiseLinearInterpolator,
    UniVariateEstimator,
)
from functions_iterative import compute_upper_bound
from multi_stock_bellman_value_calculation import (
    generate_controls,
    get_all_costs,
    initialize_antares_problems,
)
from optimization import AntaresProblem
from read_antares_data import TimeScenarioIndex, TimeScenarioParameter
from reservoir_management import MultiStockManagement
from stock_discretization import StockDiscretization
from type_definition import AreaIndex, Array1D, Array2D, Dict, List


def calculate_complete_reward(
    controls: Dict[int, List[Dict[str, float]]],
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    costs: Dict[TimeScenarioIndex, List[float]],
    slopes: Dict[TimeScenarioIndex, List[Dict[str, float]]],
) -> Dict[str, Dict[TimeScenarioIndex, RewardApproximation]]:
    reward: Dict[str, Dict[TimeScenarioIndex, RewardApproximation]] = {}

    for area, reservoir_management in multi_stock_management.dict_reservoirs.items():
        reward[area.area] = {}
        for week in range(param.len_week):
            for scenario in range(param.len_scenario):
                r = RewardApproximation(
                    lb_control=-reservoir_management.reservoir.max_pumping[week],
                    ub_control=reservoir_management.reservoir.max_generating[week],
                    ub_reward=0,
                )

                for i, u in enumerate(controls[week]):
                    r.update(
                        duals=-slopes[TimeScenarioIndex(week, scenario)][i][area.area],
                        costs=-costs[TimeScenarioIndex(week, scenario)][i]
                        + slopes[TimeScenarioIndex(week, scenario)][i][area.area]
                        * u[area.area],
                    )

                reward[area.area][TimeScenarioIndex(week, scenario)] = r

    return reward


def calculate_bellman_value_with_precalculated_reward(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    len_controls: int,
    len_bellman: int,
    name_solver: str = "CLP",
) -> tuple[
    Array2D,
    Dict[str, Dict[TimeScenarioIndex, RewardApproximation]],
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

    list_models = initialize_antares_problems(
        param=param,
        multi_stock_management=multi_stock_management,
        output_path=output_path,
        name_solver=name_solver,
        direct_bellman_calc=True,
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

    reward = calculate_complete_reward(
        controls=controls,
        param=param,
        multi_stock_management=multi_stock_management,
        costs=costs,
        slopes=slopes,
    )

    for area, reservoir_management in multi_stock_management.dict_reservoirs.items():
        X = np.linspace(0, reservoir_management.reservoir.capacity, num=len_bellman)

        V = calculate_VU(
            stock_discretization=X,
            time_scenario_param=param,
            reservoir_management=reservoir_management,
            reward=reward[area.area],
        )

    V0 = V[0](reservoir_management.reservoir.initial_level)

    upper_bound, control_ub, current_itr = compute_upper_bound(
        multi_stock_management=multi_stock_management,
        stock_discretization=StockDiscretization(
            {AreaIndex(reservoir_management.reservoir.area): X}
        ),
        param=param,
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

    stock_discretization = StockDiscretization({AreaIndex(a): x for a, x in X.items()})

    V: Dict[int, Estimator] = {}
    if univariate:
        assert len(X) == 1
        dim_bellman_value = tuple(len(x) for x in X.values())
        V = {
            week: UniVariateEstimator(
                {
                    area.area: PieceWiseLinearInterpolator(
                        X[area.area], np.zeros(len(X[area.area]), dtype=np.float32)
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
                        area.area: stock_discretization.list_discretization[area][
                            idx[i]
                        ]
                        for i, area in enumerate(m.range_reservoir)
                    },
                    find_optimal_basis=False,
                    take_into_account_z_and_y=True,
                    multi_stock_management=multi_stock_management,
                    stock_discretization=stock_discretization,
                )
                V[week].update(
                    Vu,
                    slope,
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

    upper_bound, controls, current_itr = compute_upper_bound(
        multi_stock_management=multi_stock_management,
        stock_discretization=stock_discretization,
        param=param,
        list_models=list_models,
        V={week: V[week] for week in range(param.len_week + 1)},
    )

    gap = upper_bound + V0
    print(gap, upper_bound, -V0)

    return {week: V[week] for week in range(param.len_week + 1)}, V0, upper_bound
