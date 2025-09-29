import os as os
import pickle as pkl
from pathlib import Path

import numpy as np
from tqdm import tqdm

from estimation import (
    Estimator,
    LinearCostEstimator,
    LinearInterpolator,
    PieceWiseLinearInterpolator,
)
from optimization import AntaresProblem, WeeklyBellmanProblem
from reservoir_management import MultiStockManagement
from type_definition import (
    AreaIndex,
    Dict,
    List,
    Optional,
    ScenarioIndex,
    TimeScenarioIndex,
    TimeScenarioParameter,
    Union,
    WeekIndex,
    area_value_to_area_scenario_value,
    area_value_to_array,
    list_area_value_to_array,
    mean_scenario_value,
)


def get_bellman_values_from_approximate_costs(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    costs_approx: LinearCostEstimator,
    levels: Dict[WeekIndex, List[Dict[AreaIndex, float]]],
    piecewiselinear: bool,
    final_bellman_values: Optional[Estimator] = None,
    name_solver: str = "CLP",
    divisor: dict[str, float] = {"euro": 1e8, "energy": 1e4},
    verbose: bool = False,
    n_cycle: int = 1,
) -> Dict[WeekIndex, Estimator]:
    """
    Calculate Bellman values for every week based on reward approximation

    Parameters
    ----------

    Returns
    -------

    """
    if final_bellman_values is None:
        if piecewiselinear:
            X = list_area_value_to_array(levels[WeekIndex(param.len_week)])[:, 0]
            final_bellman_values = PieceWiseLinearInterpolator(
                X, np.zeros(len(X), dtype=np.float32)
            )
        else:
            final_bellman_values = get_default_linear_interpolator(
                multi_stock_management
            )

    for i in range(n_cycle):
        bellman_values = {WeekIndex(param.len_week): final_bellman_values}

        week_range = range(param.len_week - 1, -1, -1)
        if verbose:
            week_range = tqdm(week_range, colour="Green", desc="Dynamic Solving")
        for week in week_range:
            costs_w: List[float] = []
            duals_w: List[Dict[AreaIndex, float]] = []

            for lvl_init in levels[WeekIndex(week)]:

                problem = WeeklyBellmanProblem(
                    param=param,
                    multi_stock_management=multi_stock_management,
                    week_costs_estimation=costs_approx.get_week_estimators(week),
                    name_solver=name_solver,
                    divisor=divisor,
                    week=week,
                )

                _, cost_wl, duals_wl, _ = problem.solve(
                    level_init=area_value_to_area_scenario_value(
                        lvl_init, param.len_scenario
                    ),
                    future_costs_estimation=bellman_values[WeekIndex(week + 1)],
                )

                costs_w.append(cost_wl)
                duals_w.append(
                    {
                        a: mean_scenario_value(duals_wl[a])
                        for a in multi_stock_management.areas
                    }
                )

            if piecewiselinear:
                bellman_values[WeekIndex(week)] = PieceWiseLinearInterpolator(
                    list_area_value_to_array(levels[WeekIndex(week)])[:, 0],
                    -np.array(costs_w),
                )
            else:
                bellman_values[WeekIndex(week)] = LinearInterpolator(
                    controls=list_area_value_to_array(levels[WeekIndex(week)]),
                    costs=np.array(costs_w),
                    duals=list_area_value_to_array(duals_w),
                )
        final_bellman_values = bellman_values[WeekIndex(0)]
    return bellman_values


def get_default_linear_interpolator(
    multi_stock_management: MultiStockManagement,
) -> LinearInterpolator:
    return LinearInterpolator(
        controls=np.array(
            [
                area_value_to_array(
                    {
                        a: res.reservoir.capacity
                        for a, res in multi_stock_management.dict_reservoirs.items()
                    }
                )
            ]
        ),
        costs=np.array([0]),
        duals=np.array(
            [area_value_to_array({a: 0 for a in multi_stock_management.areas})]
        ),
    )


def get_optimal_trajectory_from_approximate_costs(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    costs_approx: LinearCostEstimator,
    bellman_values: Dict[WeekIndex, Estimator],
    level_init: Optional[Dict[AreaIndex, float]] = None,
    random_scenario: bool = False,
    random_seed: int = 0,
    name_solver: str = "CLP",
    divisor: dict[str, float] = {"euro": 1e8, "energy": 1e4},
) -> tuple[
    Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
    Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
    float,
]:
    """Finds the optimal trajectory starting from starting_pts

    Args:
        param (TimeScenarioParameter): Number of weeks and scenarios
        multi_stock_management (MultiStockManagement): _description_
        costs_approx (Estimator): _description_
        future_estimators_l (Dict[WeekIndex,LinearInterpolator]): _description_
        starting_pt (Dict[AreaIndex, float]): _description_
        name_solver (str): _description_
        verbose (bool): _derscription_

    Returns:
        tuple[Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
              Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
              Dict[WeekIndex, float],]:
        Optimal trajectory, optimal controls, corresponding costs
    """
    trajectory = {}
    for scenario in range(param.len_scenario):
        if level_init is None:
            trajectory[TimeScenarioIndex(-1, scenario)] = (
                multi_stock_management.get_initial_level()
            )
        else:
            trajectory[TimeScenarioIndex(-1, scenario)] = level_init
    controls: Dict[TimeScenarioIndex, Dict[AreaIndex, float]] = {}
    costs = 0.0
    np.random.seed(random_seed)
    for week in range(param.len_week):
        if week >= 0:
            # Write problem
            problem = WeeklyBellmanProblem(
                param=param,
                multi_stock_management=multi_stock_management,
                week_costs_estimation=costs_approx.get_week_estimators(week),
                divisor=divisor,
                name_solver=name_solver,
                week=week,
            )
            if random_scenario:
                scenarios = np.random.permutation(range(param.len_scenario))
            else:
                scenarios = np.array([s for s in range(param.len_scenario)])
            level_i = {
                a: {
                    ScenarioIndex(s): trajectory[TimeScenarioIndex(week - 1, s)][a]
                    for s in scenarios
                }
                for a in multi_stock_management.areas
            }

            # Solve, might be cool to reuse bases
            controls_w, cost_w, _, levels = problem.solve(
                level_init=level_i,
                future_costs_estimation=bellman_values[WeekIndex(week + 1)],
                remove_future_costs=True,
            )
            trajectory[TimeScenarioIndex(week, scenario)] = {
                a: l[ScenarioIndex(scenario)] for a, l in levels.items()
            }
            controls[TimeScenarioIndex(week, scenario)] = {
                a: c[ScenarioIndex(scenario)] for a, c in controls_w.items()
            }
            costs += cost_w

    return trajectory, controls, costs


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


def get_week_scenario_costs(
    m: AntaresProblem,
    controls_list: Union[List[Dict[AreaIndex, float]], Dict[AreaIndex, float]],
) -> tuple[List[float], List[Dict[AreaIndex, float]], int, float]:
    """
    Takes a control and an initialized Antares problem setup and returns the objective and duals
    for every week and every Scenario

    Parameters
    ----------
        m:AntaresProblem: Instance of Antares problem describing the problem currently solved
            (at specified week and scenario),
        multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
        controls_list:List[Dict[AreaIndex, float]], list of all controls to be checked

    Returns
    -------
        costs:List[float]: cost of each control,
        slopes:List[Dict[AreaIndex, float]]: dual values for each control,
        tot_iter:int: number of simplex pivots,
        times:List[float]: list of solving times
    """

    tot_iter = 0
    times = 0.0

    # Initialize costs
    costs: List[float] = []
    slopes: List[Dict[AreaIndex, float]] = []
    if type(controls_list) is list:
        controls = controls_list
    else:
        controls = [controls_list]  # type:ignore
    for u in controls:

        # Solving the problem
        control_cost, control_slopes, itr, time_taken = (
            m.solve_with_predefined_controls(control=u)
        )
        tot_iter += itr
        times += time_taken

        # Save results
        # print(f"Imposing control {control} costs {costs}, with duals {control_slopes}")
        costs.append(control_cost)
        slopes.append(control_slopes)

    return costs, slopes, tot_iter, times


def get_antares_costs(
    param: TimeScenarioParameter,
    controls: Union[
        Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]],
        Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
    ],
    multi_stock_management: Optional[MultiStockManagement] = None,
    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {},
    output_path: str = "",
    name_solver: str = "CLP",
    saving_dir: Optional[str] = None,
    save_protos: bool = False,
    verbose: bool = False,
    keep_intermed_res: bool = False,
    prefix: str = "",
) -> tuple[
    Dict[TimeScenarioIndex, List[float]],
    Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]],
    Dict[TimeScenarioIndex, float],
    Dict[TimeScenarioIndex, int],
]:
    """
    Takes a problem and a discretization level and solves the Antares pb for every combination of stock for every week

        Parameters
        ----------
            param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
            list_models:Dict[TimeScenarioIndex, AntaresProblem]: List of models
            multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
            controls_list:Dict[WeekIndex, List[Dict[AreaIndex, float]]]: Controls to be evaluated, per week, per scenario
            verbose:bool:Control the level of outputs show by the function

        Returns
        -------
            Bellman Values:Dict[TimeScenarioIndex, List[float]]: Bellman values
    """
    current_itr = {}
    times: Dict[TimeScenarioIndex, float] = {}
    costs: Dict[TimeScenarioIndex, List[float]] = {}
    slopes: Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]] = {}
    week_start = 0
    if keep_intermed_res:
        assert saving_dir is not None
        filename = f"{saving_dir}/{prefix}get_all_costs_run.pkl"
        if Path(filename).is_file():
            with open(filename, "rb") as file:
                week_start, pre_costs, pre_slopes = pkl.load(file)
            costs = pre_costs
            slopes = pre_slopes
    week_range = range(param.len_week)
    if verbose:
        week_range = tqdm(range(param.len_week), colour="blue", desc="Simulation")
    for week in week_range:
        if week >= week_start:
            for scenario in range(param.len_scenario):
                ts_id = TimeScenarioIndex(week=week, scenario=scenario)
                try:
                    m = list_models[ts_id]
                except KeyError:
                    assert output_path != ""
                    assert multi_stock_management is not None
                    m = AntaresProblem(
                        scenario=scenario,
                        week=week,
                        path=output_path,
                        saving_directory=saving_dir,
                        name_solver=name_solver,
                        save_protos=save_protos,
                        param=param,
                        multi_stock_management=multi_stock_management,
                    )
                costs_ws, slopes_ws, iters, times_ws = get_week_scenario_costs(
                    m=m,
                    controls_list=controls[ts_id],
                )
                times[ts_id] = times_ws
                costs[ts_id] = costs_ws
                slopes[ts_id] = slopes_ws
                current_itr[ts_id] = iters
            if keep_intermed_res:
                assert saving_dir is not None
                if not (os.path.exists(saving_dir)):
                    os.makedirs(saving_dir)
                with open(filename, "wb") as file:
                    pkl.dump((week, costs, slopes), file)
    return costs, slopes, times, current_itr
