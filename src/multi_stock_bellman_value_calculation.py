import os as os
import pickle as pkl
from itertools import product
from pathlib import Path

import juliacall
import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
from ortools.linear_solver.python import model_builder
from scipy.stats import random_correlation
from tqdm import tqdm

from display import ConvergenceProgressBar, draw_usage_values, draw_uvs_sddp
from estimation import LinearCostEstimator, LinearInterpolator
from optimization import (AntaresProblem, Basis, WeeklyBellmanProblem,
                          solve_for_optimal_trajectory)
from reservoir_management import MultiStockManagement
from type_definition import (
    AreaIndex, Dict, List, Optional, ScenarioIndex, TimeScenarioIndex,
    TimeScenarioParameter, WeekIndex, area_value_to_array, array_to_area_value,
    list_area_value_to_array, list_to_week_value, mean_scenario_value,
    time_list_area_value_to_array, time_list_to_array,
    timescenario_area_value_to_array,
    timescenario_area_value_to_weekly_mean_area_values,
    timescenario_list_area_value_to_array, timescenario_list_value_to_array)

jl = juliacall.Main


def initialize_antares_problems(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    name_solver: str,
    direct_bellman_calc: bool = True,
    verbose: bool = False,
    save_protos: bool = False,
    load_from_protos: bool = False,
    saving_dir: Optional[str] = None,
) -> Dict[TimeScenarioIndex, AntaresProblem]:
    """
    Creates Instances of the Antares problem for every week / scenario

    Parameters
    ----------
    param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
    multi_stock_management:MultiStockManagement: Management information for every stock ,
    output_path:str: Folder containing the mps files generated for our study,
    name_solver:str: Name of the solver used,
    direct_bellman_calc:bool: Integrate future costs in the LP formulation
    verbose:bool: Control displays at runnning time

    Returns
    -------
    list_models: Dict[TimeScenarioIndex, AntaresProblem]: Dictionnary with all problems
    """
    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {}
    week_range = range(param.len_week)
    if verbose:
        week_range = tqdm(week_range, desc="Problem initialization", colour="Yellow")
    for week in week_range:
        for scenario in range(param.len_scenario):
            if saving_dir is not None:
                proto_path = (
                    saving_dir
                    + f"/problem-{param.name_scenario[scenario]}-{week+1}.pkl"
                )
                already_processed = Path(proto_path).is_file() and load_from_protos
            else:
                already_processed = False
            m = AntaresProblem(
                scenario=scenario,
                week=week,
                path=output_path,
                saving_directory=saving_dir,
                itr=1,
                name_solver=name_solver,
                name_scenario=param.name_scenario[scenario],
                already_processed=already_processed,
            )
            if already_processed:
                m.reset_from_loaded_version(
                    multi_stock_management=multi_stock_management
                )
            else:
                m.create_weekly_problem_itr(
                    param=param,
                    multi_stock_management=multi_stock_management,
                    direct_bellman_calc=direct_bellman_calc,
                )
                if save_protos:
                    proto = model_builder.ModelBuilder().export_to_proto()  # type: ignore[no-untyped-call]
                    m.solver.ExportModelToProto(output_model=proto)
                    with open(proto_path, "wb") as file:
                        pkl.dump((proto, m.stored_variables_and_constraints_ids), file)

            list_models[TimeScenarioIndex(week, scenario)] = m
    return list_models


def get_bellman_values_from_costs(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    costs_approx: LinearCostEstimator,
    future_costs_approx: LinearInterpolator,
    nSteps_bellman: int,
    name_solver: str,
    method: str,
    trajectory: Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
    correlations: np.ndarray,
    divisor: dict[str, float] = {"euro": 1e8, "energy": 1e4},
    verbose: bool = False,
) -> tuple[
    Dict[WeekIndex, List[Dict[AreaIndex, float]]],
    Dict[WeekIndex, List[float]],
    Dict[WeekIndex, List[Dict[AreaIndex, float]]],
    Dict[WeekIndex, List[Dict[AreaIndex, Dict[ScenarioIndex, float]]]],
    Dict[WeekIndex,LinearInterpolator],
]:
    """
    Dynamically solves the problem of minimizing the optimal control problem for every discretized level,
    balancing between saving costs this week or saving stock to avoid future costs

    Parameters
    ----------
        param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
        multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
        costs_approx:LinearInterpolator: All precalculated lower bounding hyperplains of weekly cost estimation,
        future_costs_approx:LinearInterpolator: All previously obtained lower bounding hyperplains of future cost estimation,
        nSteps_bellman:int: Discretization level used
        name_solver:str: Solver chosen for the optimization problem, default -> CLP
        method:str: Method used to select levels to check
        verbose:bool: Controls the displays at running time

    Returns
    -------
        levels: Dict[WeekIndex, List[Dict[AreaIndex, float]]]: Level discretization used,
        costs:Dict[WeekIndex, List[float]]: Objective value at every week (and for every level combination),
        duals:Dict[WeekIndex, List[Dict[AreaIndex, float]]]: Dual values of the initial level constraint at every week (and for every level combination),
        future_costs_approx:LinearInterpolator: Final estimation of total system prices
    """
    # Initializing the Weekly Bellman Problem instance
    problem = WeeklyBellmanProblem(
        param=param,
        multi_stock_management=multi_stock_management,
        week_costs_estimation=costs_approx,
        name_solver=name_solver,
        divisor=divisor,
    )

    # Keeping in memory all future costs approximations
    future_costs_approx_l = {WeekIndex(param.len_week):future_costs_approx}

    # Parameters
    n_weeks = param.len_week

    # Initializing controls, costs and duals
    controls: Dict[WeekIndex, List[Dict[AreaIndex, Dict[ScenarioIndex, float]]]] = {}
    costs: Dict[WeekIndex, List[float]] = {}
    duals: Dict[WeekIndex, List[Dict[AreaIndex, float]]] = {}
    all_levels: Dict[WeekIndex, List[Dict[AreaIndex, float]]] = {}

    # Starting from last week dynamically solving the optimal control problem (from every starting level)
    week_range = range(n_weeks - 1, -1, -1)
    if verbose:
        week_range = tqdm(week_range, colour="Green", desc="Dynamic Solving")
    for week in week_range:
        # Getting levels along which we heck
        levels = multi_stock_management.get_disc(
            week=week,
            xNsteps=nSteps_bellman,
            reference_pt=timescenario_area_value_to_weekly_mean_area_values(
                trajectory, week, param, multi_stock_management.areas
            ),
            correlation_matrix=correlations,
            method=method,
        )
        all_levels[WeekIndex(week)] = levels

        controls_w: List[Dict[AreaIndex, Dict[ScenarioIndex, float]]] = []
        costs_w: List[float] = []
        duals_w: List[Dict[AreaIndex, float]] = []
        for lvl_init in levels:

            # Remove previous constraints / vars
            problem.solver = pywraplp.Solver.CreateSolver(name_solver)
            # problem.reset_solver()

            # Rewrite problem
            problem.write_problem(
                week=week,
                level_init=lvl_init,
                future_costs_estimation=future_costs_approx,
            )

            # Solve, should we make use of previous Bases ? Not yet, computations still tractable for n<=2
            try:
                controls_wls, cost_wl, duals_wl, _ = problem.solve()
            except ValueError:
                print(
                    f"""Failed to solve at week {week} with initial levels {lvl_init}"""
                )
                print(
                    f"We were using the following future costs estimation: {[f'Cost(lvl) >= {cost} +  (lvl_0 - {input[0]})*{duals[0]} + (lvl_1 - {input[1]})*{duals[1]}' for input, cost, duals in zip(future_costs_approx.inputs, future_costs_approx.costs, future_costs_approx.duals)]}"
                )
                raise ValueError

            # Writing down results
            controls_w.append(controls_wls)
            costs_w.append(cost_wl)
            duals_w.append(duals_wl)

        controls[WeekIndex(week)] = controls_w
        costs[WeekIndex(week)] = costs_w
        duals[WeekIndex(week)] = duals_w

        # Updating the future estimator
        # costs_w - np.min(costs_w)
        future_costs_approx = LinearInterpolator(
            controls=list_area_value_to_array(levels),
            costs=costs_w - np.min(costs_w),
            duals=list_area_value_to_array(duals_w),
        )
        future_costs_approx_l[WeekIndex(week)] = future_costs_approx
    return (
        all_levels,
        costs,
        duals,
        controls,
        future_costs_approx_l,
    )


def initialize_future_costs(
    starting_pt: Dict[AreaIndex, float],
    multi_stock_management: MultiStockManagement,
    mult: float = 0.0,
) -> LinearInterpolator:
    """
    Proposes an estimation of yearly costs based on the precalculated weekly costs, to prevent the model from
    emptying the stock on the last week as if it was the last

    Parameters
    ----------
        param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
        multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
        costs_approx:Estimator: Precalculated weekly costs based on control

    Returns
    -------
        LinearInterpolator: for any level associates a corresponding price
    """
    n_reservoirs = len(multi_stock_management.areas)
    inputs = np.array(
        [area_value_to_array(starting_pt) for _ in range(2 * n_reservoirs)]
    )
    duals = np.array([np.zeros((n_reservoirs)) for _ in range(2 * n_reservoirs)])
    for i in range(n_reservoirs):
        duals[i][i] = mult
        duals[i + 1][i] = -mult
    return LinearInterpolator(
        controls=inputs,
        costs=np.zeros(2 * n_reservoirs),
        duals=duals,
    )


def get_week_scenario_costs(
    m: AntaresProblem,
    controls_list: List[Dict[AreaIndex, float]],
) -> tuple[List[float], List[Dict[AreaIndex, float]], int, List[float]]:
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
    times = []

    # Initialize Basis
    basis = Basis([], [])

    # Initialize costs
    costs: List[float] = []
    slopes: List[Dict[AreaIndex, float]] = []
    for u in controls_list:

        # Solving the problem
        control_cost, control_slopes, itr, time_taken = (
            m.solve_with_predefined_controls(control=u, prev_basis=basis)
        )
        tot_iter += itr
        times.append(time_taken)

        # Save results
        # print(f"Imposing control {control} costs {costs}, with duals {control_slopes}")
        costs.append(control_cost)
        slopes.append(control_slopes)

        # Save basis
        rstatus, cstatus = m.get_basis()
        basis = Basis(rstatus=rstatus, cstatus=cstatus)
    return costs, slopes, tot_iter, times


def get_all_costs(
    param: TimeScenarioParameter,
    list_models: Dict[TimeScenarioIndex, AntaresProblem],
    controls_list: Dict[WeekIndex, List[Dict[AreaIndex, float]]],
    saving_dir: Optional[str] = None,
    verbose: bool = False,
    already_init: bool = False,
    keep_intermed_res: bool = False,
) -> tuple[
    Dict[TimeScenarioIndex, List[float]],
    Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]],
    Dict[TimeScenarioIndex, List[float]],
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
    tot_iter = 0
    times: Dict[TimeScenarioIndex, List[float]] = {}
    if keep_intermed_res or already_init:
        assert saving_dir is not None
        filename = saving_dir + "/get_all_costs_run.pkl"

    # Initializing the n_weeks*n_scenarios*n_controls*n_stocks values to fill
    costs: Dict[TimeScenarioIndex, List[float]] = {}
    slopes: Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]] = {}
    week_start = 0
    if already_init:
        with open(filename, "rb") as file:
            pre_costs, pre_slopes = pkl.load(file)
        week_start = len(pre_costs) // param.len_scenario
        costs = pre_costs
        slopes = pre_slopes
    week_range = range(week_start, param.len_week)
    if verbose:
        week_range = tqdm(
            range(week_start, param.len_week), colour="blue", desc="Simulation"
        )
    for week in week_range:
        for scenario in range(param.len_scenario):
            ts_id = TimeScenarioIndex(week=week, scenario=scenario)
            # Antares problem
            m = list_models[ts_id]
            try:
                costs_ws, slopes_ws, iters, times_ws = get_week_scenario_costs(
                    m=m,
                    controls_list=controls_list[WeekIndex(week)],
                )
            except ValueError:
                print(
                    f"Failed at week {week}, the conditions on control were: {controls_list[WeekIndex(week)]}"
                )
                raise ValueError
            tot_iter += iters
            times[TimeScenarioIndex(week, scenario)] = times_ws
            costs[TimeScenarioIndex(week, scenario)] = costs_ws
            slopes[TimeScenarioIndex(week, scenario)] = slopes_ws
        if keep_intermed_res:
            with open(filename, "wb") as file:
                pkl.dump((costs, slopes), file)
    # print(f"Number of simplex pivot {tot_iter}")
    return costs, slopes, times


def Lget_costs(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    name_solver: str,
    controls_list: Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]],
    saving_directory: str,
    verbose: bool = False,
    direct_bellman_calc: bool = True,
    load_from_protos: bool = False,
    prefix: str = "",
) -> tuple[
    Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]],
    Dict[TimeScenarioIndex, List[float]],
    Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]],
]:
    filename = f"{saving_directory}/{prefix}get_all_costs_run_{output_path.replace('/','_')[-27:]}.pkl"

    costs: Dict[TimeScenarioIndex, List[float]] = {}
    slopes: Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]] = {}
    week_start = 0
    if Path(filename).is_file():
        with open(filename, "rb") as file:
            week_start, pre_controls, pre_costs, pre_slopes = pkl.load(file)
        controls_list = pre_controls
        costs = pre_costs
        slopes = pre_slopes
    week_range = range(param.len_week)
    if verbose:
        if week_start > 0:
            print(f"Starting again at week {week_start}")
        week_range = tqdm(range(param.len_week), colour="blue", desc="Simulation")
    for week in week_range:
        if week >= week_start:
            for scenario in range(param.len_scenario):
                if not (os.path.exists(saving_directory)):
                    os.makedirs(saving_directory)
                proto_path = (
                    saving_directory
                    + f"/problem-{param.name_scenario[scenario]}-{week+1}.pkl"
                )
                already_processed = load_from_protos and Path(proto_path).is_file()
                m = AntaresProblem(
                    scenario=scenario,
                    week=week,
                    path=output_path,
                    saving_directory=saving_directory,
                    itr=1,
                    name_solver=name_solver,
                    name_scenario=param.name_scenario[scenario],
                    already_processed=already_processed,
                )
                if already_processed:
                    m.reset_from_loaded_version(
                        multi_stock_management=multi_stock_management
                    )
                else:
                    m.create_weekly_problem_itr(
                        param=param,
                        multi_stock_management=multi_stock_management,
                        direct_bellman_calc=direct_bellman_calc,
                    )
                    proto = model_builder.ModelBuilder().export_to_proto()  # type: ignore[no-untyped-call]
                    m.solver.ExportModelToProto(output_model=proto)
                    with open(proto_path, "wb") as file:
                        pkl.dump((proto, m.stored_variables_and_constraints_ids), file)
                costs_ws, slopes_ws, _, _ = get_week_scenario_costs(
                    m=m,
                    controls_list=controls_list[TimeScenarioIndex(week, scenario)],
                )
                costs[TimeScenarioIndex(week, scenario)] = costs_ws
                slopes[TimeScenarioIndex(week, scenario)] = slopes_ws
            with open(filename, "wb") as file:
                pkl.dump(
                    (week, controls_list, costs, slopes),
                    file,
                )
    return controls_list, costs, slopes


def generate_controls(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    controls_looked_up: str,
    xNsteps: int,
) -> Dict[WeekIndex, List[Dict[AreaIndex, float]]]:
    """
    Generates controls that will be precalculated for every week / scenario

    Parameters
    ----------
        param:TimeScenarioParameter: Contains the informations relative to the number of scenario and weeks
        multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
        controls_looked_up:str: 'grid', 'line', 'random', 'oval' description of the controls to generate
        xNsteps:int: Discretization level used
        name_solver:str: Solver chosen for the optimization problem, default -> CLP

    Returns
    -------
        Bellman Values:Dict[WeekIndex, List[Dict[AreaIndex, float]]]: Bellman values
    """
    # General parameters
    n_res = len(multi_stock_management.dict_reservoirs)

    # Useful data on reservoirs
    max_res = np.array(
        [
            mng.reservoir.max_generating
            for mng in multi_stock_management.dict_reservoirs.values()
        ]
    )
    min_res = np.array(
        [
            -mng.reservoir.max_pumping * mng.reservoir.efficiency
            for mng in multi_stock_management.dict_reservoirs.values()
        ]
    )

    # Points to start lines at
    start_pts = [
        min_res
        + (1 / 4)
        * ((i + 1) / (n_res + 1))
        * (max_res - min_res)
        * (np.array([j != i for j in range(n_res)])[:, None])
        for i in range(n_res)
    ]
    end_points = [
        start_pt
        + (max_res - min_res) * (np.array([j == i for j in range(n_res)])[:, None])
        for i, start_pt in enumerate(start_pts)
    ]

    if controls_looked_up == "line":
        controls = np.concatenate(
            [
                np.linspace(strt_pt, end_pt, xNsteps)
                for strt_pt, end_pt in zip(start_pts, end_points)
            ]
        )

    elif controls_looked_up == "line+diagonal":
        controls = np.concatenate(
            [
                np.linspace(strt_pt, end_pt, xNsteps)
                for strt_pt, end_pt in zip(start_pts, end_points)
            ]
            + [np.linspace(max_res, min_res, xNsteps)]
        )

    elif controls_looked_up == "zero+diagonal":

        controls = np.linspace(max_res, min_res, xNsteps)
        for i in range(n_res):
            c = np.zeros((xNsteps, n_res, 52))
            c[:, i] = np.linspace(max_res[i], min_res[i], xNsteps)
            controls = np.concatenate([controls, c])

    elif controls_looked_up == "diagonal":
        controls = np.linspace(max_res, min_res, xNsteps)

    else:  # Applying default: grid
        controls = np.array(
            [
                i
                for i in product(
                    *[
                        np.linspace(
                            -manag.reservoir.max_pumping * manag.reservoir.efficiency,
                            manag.reservoir.max_generating,
                            xNsteps,
                        )
                        for _, manag in multi_stock_management.dict_reservoirs.items()
                    ]
                )
            ]
        )

    controls = np.moveaxis(controls, -1, 0)
    dict_control = {
        WeekIndex(week): [
            {area: cont for area, cont in zip(multi_stock_management.areas, u)}
            for u in controls[week]
        ]
        for week in range(param.len_week)
    }
    return dict_control


def precalculated_method(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    len_controls: int = 12,
    len_bellman: int = 12,
    name_solver: str = "CLP",
    controls_looked_up: str = "grid",
    verbose: bool = False,
) -> tuple[
    Dict[WeekIndex, List[Dict[AreaIndex, float]]],
    LinearCostEstimator,
    Dict[WeekIndex, List[float]],
    Dict[WeekIndex, List[Dict[AreaIndex, Dict[ScenarioIndex, float]]]],
    Dict[WeekIndex, List[Dict[AreaIndex, float]]],
    Dict[TimeScenarioIndex, List[float]],
]:
    """
    Takes a control and an initialized Antares problem setup and returns the objective and duals
    for every week and every Scenario

    Parameters
    ----------
        param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
        multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
        output_path:str: file containing the previouvsly generated mps corresponding to the study,
        xNsteps:int: Discretization level used
        name_solver:str: Solver chosen for the optimization problem, default -> CLP
        controls_looked_up:str: 'grid', 'line', 'random', 'oval' description of the controls to generate

    Returns
    -------
        Bellman Values:Dict[WeekIndex, List[Dict[AreaIndex, float]]]: Bellman values
    """

    # Initialize the problems
    list_models = initialize_antares_problems(
        param=param,
        multi_stock_management=multi_stock_management,
        output_path=output_path,
        name_solver=name_solver,
        direct_bellman_calc=False,
        verbose=verbose,
    )

    controls_list = generate_controls(
        param=param,
        multi_stock_management=multi_stock_management,
        controls_looked_up=controls_looked_up,
        xNsteps=len_controls,
    )

    costs, slopes, times = get_all_costs(
        param=param,
        list_models=list_models,
        controls_list=controls_list,
        verbose=verbose,
    )

    # Initialize cost functions
    costs_approx = LinearCostEstimator(
        param=param,
        controls=np.array(
            [
                np.broadcast_to(
                    [[x for x in u.values()] for u in controls_list[WeekIndex(w)]],
                    [
                        param.len_scenario,
                        len(controls_list[WeekIndex(w)]),
                        len(multi_stock_management.dict_reservoirs),
                    ],
                )
                for w in range(param.len_week)
            ]
        ),
        costs=np.array(
            [
                [
                    np.array(costs[TimeScenarioIndex(w, s)])
                    for s in range(param.len_scenario)
                ]
                for w in range(param.len_week)
            ]
        ),
        duals=np.array(
            [
                [
                    np.array(
                        [
                            [y for y in x.values()]
                            for x in slopes[TimeScenarioIndex(w, s)]
                        ]
                    )
                    for s in range(param.len_scenario)
                ]
                for w in range(param.len_week)
            ]
        ),
    )

    starting_pt = {
        a: mng.reservoir.bottom_rule_curve[0] * 0.7
        + mng.reservoir.upper_rule_curve[0] * 0.3
        for a, mng in multi_stock_management.dict_reservoirs.items()
    }

    future_costs_approx = initialize_future_costs(
        multi_stock_management=multi_stock_management,
        starting_pt=starting_pt,
    )

    trajectory = {
        TimeScenarioIndex(w, s): {a: x for a, x in starting_pt.items()}
        for w in range(param.len_week)
        for s in range(param.len_scenario)
    }

    correlations = get_correlation_matrix(
        multi_stock_management=multi_stock_management,
        corr_type="no_corrs",
    )

    method = "lines"

    for i in range(2):
        (
            levels,
            bellman_costs,
            bellman_duals,
            bellman_controls,
            future_costs_approx_l,
        ) = get_bellman_values_from_costs(
            param=param,
            multi_stock_management=multi_stock_management,
            costs_approx=costs_approx,
            future_costs_approx=future_costs_approx,
            nSteps_bellman=len_bellman,
            name_solver=name_solver,
            verbose=verbose,
            method=method,
            trajectory=trajectory,
            correlations=correlations,
        )
        future_costs_approx = future_costs_approx_l[WeekIndex(0)]

    return (
        levels,
        costs_approx,
        bellman_costs,
        bellman_controls,
        bellman_duals,
        times,
    )


def initialize_controls(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    n_controls_init: int,
    rand: bool = False,
) -> Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]]:
    """
    Selects a group of controls to initialize our cost estimation and get a convex hull within which the optimal hopefully resides

    Parameters
    ----------
    param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
    multi_stock_management:MultiStockManagement: Management information for every stock ,
    n_controls: number of controls we estimate are needed to initialize our iterative method,
    rand: have the groups of reservoirs chosen to be turned on together be random, default=False

    Returns
    -------
    controls_list:Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]]
    """
    n_stocks = len(multi_stock_management.dict_reservoirs)
    n_weeks = param.len_week
    n_scenarios = param.len_scenario
    max_cont = np.array(
        [
            mng.reservoir.max_generating
            for mng in multi_stock_management.dict_reservoirs.values()
        ]
    )[:, :n_weeks]
    min_cont = np.array(
        [
            -mng.reservoir.max_pumping * mng.reservoir.efficiency
            for mng in multi_stock_management.dict_reservoirs.values()
        ]
    )[:, :n_weeks]

    # Choose how many reservoirs will fully spend
    n_stocks_full_spend = np.round(np.linspace(0, n_stocks, num=n_controls_init))
    controls = (
        np.zeros((n_weeks, n_scenarios, n_controls_init, n_stocks))
        + min_cont.T[:, None, None, :]
    )
    if rand:
        full_throttle = (
            np.zeros((n_weeks, n_scenarios, n_controls_init, n_stocks))
            + (n_stocks_full_spend[:, None] > np.arange(n_stocks)[None, :])[
                None, None, :, :
            ]
        )
        full_throttle = np.random.default_rng().permuted(full_throttle, axis=3)
    else:
        full_throttle = np.array(
            [
                [
                    [
                        np.roll([l < n_full_spend for l in range(n_stocks)], i + j + k)
                        for k, n_full_spend in enumerate(n_stocks_full_spend)
                    ]
                    for j in range(n_scenarios)
                ]
                for i in range(n_weeks)
            ]
        )
    controls += full_throttle * (max_cont - min_cont).T[:, None, None, :]
    return {
        TimeScenarioIndex(w, s): [
            {
                a: controls[w, s, j, i]
                for i, a in enumerate(multi_stock_management.areas)
            }
            for j in range(n_controls_init)
        ]
        for w in range(param.len_week)
        for s in range(param.len_scenario)
    }


def select_controls_to_explore(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    pseudo_opt_controls: Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
    costs_approx: LinearCostEstimator,
    rng: Optional[np.random.Generator] = None,
) -> Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]]:
    """Takes in some optimal (for now) controls and returns a list of controls to explore,
    in this version: only returns the optimal control

    Args:
        param (TimeScenarioParameter): _description_
        multi_stock_management (MultiStockManagement): _description_
        pseudo_opt_controls (Dict[TimeScenarioIndex, Dict[AreaIndex, float]]): _description_
        costs_approx (Estimator): _description_

    Returns:
        Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]]: Controls for a week / scenario / control_id / reservoir given
    """
    array_pseudo_opt_controls = timescenario_area_value_to_array(
        pseudo_opt_controls, param, multi_stock_management.areas
    )
    if not rng:
        rng = np.random.default_rng(seed=12345)
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    n_weeks, n_scenarios = param.len_week, param.len_scenario
    controls_to_explore = np.zeros((n_weeks, n_scenarios, 1, n_reservoirs))
    max_gen = np.array(
        [
            mng.reservoir.max_generating[:n_weeks]
            for mng in multi_stock_management.dict_reservoirs.values()
        ]
    ).T
    max_pump = np.array(
        [
            -mng.reservoir.max_pumping[:n_weeks] * mng.reservoir.efficiency
            for mng in multi_stock_management.dict_reservoirs.values()
        ]
    ).T
    max_var = max_gen - max_pump
    proximity_to_other_control = np.zeros((n_weeks, n_scenarios, n_reservoirs))
    n_inputs = costs_approx[TimeScenarioIndex(0, 0)].true_costs.shape[0]
    for week in range(n_weeks):
        for scen in range(n_scenarios):
            proximity_to_other_control[week, scen] = np.min(
                costs_approx[TimeScenarioIndex(week, scen)].true_inputs
                - array_pseudo_opt_controls[week, scen][None, :],
                axis=0,
            )
    relative_proximity = 100 * proximity_to_other_control / max_var[:, None, :]
    array_pseudo_opt_controls += rng.normal(
        scale=np.exp(-relative_proximity)
        * max_var[:, None, :]
        / 50
        * (2 * n_inputs + 1),
        size=(n_weeks, n_scenarios, n_reservoirs),
    )
    controls_to_explore[:, :, 0, :] = np.maximum(
        np.minimum(array_pseudo_opt_controls, max_gen[:, None, :]), max_pump[:, None, :]
    )
    return {
        TimeScenarioIndex(w, s): [
            {
                a: controls_to_explore[w, s, 0, i]
                for i, a in enumerate(multi_stock_management.areas)
            }
        ]
        for w in range(param.len_week)
        for s in range(param.len_scenario)
    }


def get_correlation_matrix(
    multi_stock_management: MultiStockManagement,
    adjacency_mat: Optional[np.ndarray] = None,
    corr_type: str = "no_corrs",
) -> np.ndarray:
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    if adjacency_mat is not None:
        dist_mat = adjacency_mat
        # Turning adjacency mat into full distance matrix by raising to power R in min + algebra
        for i in range(n_reservoirs):
            dist_mat = (dist_mat[None, :] + dist_mat.T[:, None]).min(axis=2).T
        correlations = np.exp(-dist_mat)
    elif corr_type == "no_corrs":
        correlations = np.eye(N=n_reservoirs)
    elif corr_type == "unif_corrs":
        t = 0.5
        correlations = np.ones((n_reservoirs, n_reservoirs)) * t + (1 - t) * np.eye(
            N=n_reservoirs
        )
    else:
        rng = np.random.default_rng()
        correlations = random_correlation.rvs(
            np.ones(n_reservoirs),
            random_state=rng,
        )
        correlations = np.abs(correlations)
    return correlations


def compute_usage_values_from_costs(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    name_solver: str,
    costs_approx: LinearCostEstimator,
    future_costs_approx_l: Dict[WeekIndex,LinearInterpolator],
    optimal_trajectory: Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
    nSteps_bellman: int,
    correlations: np.ndarray,
    divisor: dict[str, float] = {"euro": 1e8, "energy": 1e4},
    verbose: bool = False,
) -> tuple[
    dict[AreaIndex, Dict[WeekIndex, List[float]]],
    Dict[WeekIndex, List[Dict[AreaIndex, float]]],
]:
    """For now we want this function to solve a lot of optimization problems for different initial values of stock
    and get the dual values for the initial stock constraint to get the corresponding value

    Args:
        param (TimeScenarioParameter): _description_
        multi_stock_management (MultiStockManagement): _description_
        name_solver (str): _description_
        costs_approx (LinearCostEstimator): _description_
        future_costs_approx_l (list[LinearInterpolator]): _description_
        optimal_trajectory (Dict[TimeScenarioIndex, Dict[AreaIndex, float]]): _description_
        nSteps_bellman (int): _description_
        correlations (np.ndarray): _description_,
        verbose (bool)(Default to False): _description_,

    Returns:
        dict[AreaIndex, Dict[WeekIndex,List[float]]]: _description_
    """
    # Initialize problem
    problem = WeeklyBellmanProblem(
        param=param,
        multi_stock_management=multi_stock_management,
        week_costs_estimation=costs_approx,
        name_solver=name_solver,
        divisor=divisor,
    )

    # Parameter
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    n_weeks = param.len_week
    discretization = nSteps_bellman

    # Initialize usage values
    usage_values: Dict[AreaIndex, Dict[WeekIndex, List[float]]] = {
        a: {WeekIndex(w): [] for w in range(param.len_week)}
        for a in multi_stock_management.areas
    }
    base_levels = np.mean(
        timescenario_area_value_to_array(
            optimal_trajectory, param, multi_stock_management.areas
        ),
        axis=1,
    )[:n_weeks]

    # Displaying options
    week_range = range(n_weeks)
    if verbose:
        week_range = tqdm(week_range, colour="red", desc="Usage values deduction")

    # N
    capacities = np.array(
        [
            mng.reservoir.capacity
            for mng in multi_stock_management.dict_reservoirs.values()
        ]
    )
    # W x Disc x N
    alpha = 0
    bottoms = (
        np.array(
            [
                mng.reservoir.bottom_rule_curve[:n_weeks]
                for mng in multi_stock_management.dict_reservoirs.values()
            ]
        )
        * alpha
    )
    tops = np.array(
        [
            mng.reservoir.upper_rule_curve[:n_weeks]
            for mng in multi_stock_management.dict_reservoirs.values()
        ]
    ) * alpha + np.array(
        [
            [mng.reservoir.capacity] * (n_weeks)
            for mng in multi_stock_management.dict_reservoirs.values()
        ]
    ) * (
        1 - alpha
    )
    levels_imposed = np.moveaxis(np.linspace(bottoms, tops, discretization), -1, 0)[
        :n_weeks
    ]

    # W x Disc x N
    lvl_relat_gaps = (levels_imposed - base_levels[:, None, :]) / capacities[
        None, None, :
    ]
    relat_diffs = np.dot(
        lvl_relat_gaps[:, :, :, None] * np.eye(n_reservoirs)[None, None, :, :],
        correlations,
    )
    levels_to_test = (
        base_levels[:, None, None, :] + relat_diffs * capacities[None, None, None, :]
    )
    # Correct impossible levels
    levels_to_test *= levels_to_test > 0
    levels_to_test = np.minimum(levels_to_test, capacities[None, None, :])

    destination_levels = np.zeros((n_weeks, discretization, n_reservoirs))
    for week in week_range:
        for i, (area, _) in enumerate(multi_stock_management.dict_reservoirs.items()):
            for j, levels in enumerate(levels_to_test[week, :, i]):
                # Remove previous constraints / vars
                problem.reset_solver()
                try:
                    # Rewrite problem
                    problem.write_problem(
                        week=week,
                        level_init=array_to_area_value(
                            levels, multi_stock_management.areas
                        ),
                        future_costs_estimation=future_costs_approx_l[WeekIndex(week + 1)],
                    )

                    # Solve
                    _, _, dual_vals, level = problem.solve()
                except ValueError:
                    levels_dict = {
                        area: levels[i]
                        for i, area in enumerate(multi_stock_management.areas)
                    }
                    future_costs_approx = future_costs_approx_l[WeekIndex(week)]
                    print(
                        f"We were using the following future costs estimation: {[f'Cost(lvl) >= {cost} +  (lvl_0 - {input[0]})*{duals[0]} + (lvl_1 - {input[1]})*{duals[1]}' for input, cost, duals in zip(future_costs_approx.inputs, future_costs_approx.costs, future_costs_approx.duals)]}"
                    )
                    print(
                        f"""Failed to solve at week {week} with initial levels {levels_dict} \n
                        (when imposing level of {area} to {levels_imposed[j, i]}) \n
                            thus having levels_diff as {relat_diffs[week, j,i]} \n
                            when base_level was {base_levels[week,i]}   """
                    )
                    raise ValueError
                destination_levels[week, j, i] = mean_scenario_value(level[area])
                usage_values[area][WeekIndex(week)].append(-dual_vals[area])
    dict_levels_imposed = {
        WeekIndex(w): [
            {area: x[i] for i, area in enumerate(multi_stock_management.areas)}
            for x in levels_imposed[w]
        ]
        for w in range(param.len_week)
    }
    return usage_values, dict_levels_imposed


def get_opt_gap(
    param: TimeScenarioParameter,
    costs: Dict[TimeScenarioIndex, List[float]],
    costs_approx: LinearCostEstimator,
    controls_list: Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]],
    opt_gap: float,
    max_gap: Dict[WeekIndex, float],
) -> float:
    costs_approx.remove_interpolations()
    pre_update_costs = np.array(
        [
            [
                costs_approx[TimeScenarioIndex(week, scenario)](
                    list_area_value_to_array(
                        controls_list[TimeScenarioIndex(week, scenario)]
                    )
                )
                for scenario in range(param.len_scenario)
            ]
            for week in range(param.len_week)
        ]
    )
    array_max_gap = np.array([x for x in max_gap.values()])
    # Computing the optimality gap
    array_cost = timescenario_list_value_to_array(costs, param)
    array_cost = np.mean(array_cost, axis=-1)
    opt_gap = max(
        1e-10,
        min(
            1, np.max(np.mean((array_cost - pre_update_costs), axis=1) / array_max_gap)
        ),
    )
    return opt_gap


def cutting_plane_method(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    name_solver: str,
    starting_pt: Dict[AreaIndex, float],
    costs_approx: LinearCostEstimator,
    costs: Dict[TimeScenarioIndex, List[float]],
    future_costs_approx: LinearInterpolator,
    nSteps_bellman: int,
    method: str,
    correlations: np.ndarray,
    saving_dir: str,
    maxiter: Optional[int] = None,
    precision: float = 5e-2,
    divisor: dict[str, float] = {"euro": 1e8, "energy": 1e4},
    output_path: str = "",
    verbose: bool = False,
) -> tuple[
    Dict[WeekIndex, List[float]],
    Dict[WeekIndex, List[Dict[AreaIndex, float]]],
    Dict[WeekIndex,LinearInterpolator],
    LinearCostEstimator,
    Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
]:
    """
    Iteratively: computes an optimal control from costs_approx -> Refines approximation for this control
    Stops: When too many iterations occured or when optimality gap is small enough

    Parameters
    ----------
        param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
        multi_stock_management: MultiStockManagement: Description of stocks and their global policies,
        name_solver:str: Solver chosen for the optimization problem, default -> CLP
        list_models:list: List of antares models
        starting_pt:Dict[AreaIndex, float]: Point our simulations will start from
        costs_approx:Estimator: Linear approximation of weekly costs depending on control
        future_costs_approx:LinearInterpolator: Linear approximation of future costs when starting from certain level
        nSteps_bellman:int: Discretization level used
        maxiter:Optional[int]: Maximum number of iterations used
        precision:Optional[float]: Convergence criterion on optimality gap

    Returns
    -------
        Bellman Values:Dict[WeekIndex, List[float]]: Bellman values
    """
    iter = 0
    opt_gap = 1.0
    maxiter = int(1e2) if maxiter is None else maxiter
    # Init trajectory
    trajectory = {
        TimeScenarioIndex(w, s): starting_pt
        for w in range(param.len_week)
        for s in range(param.len_scenario)
    }
    # Max gap
    array_cost = timescenario_list_value_to_array(costs, param)
    max_gap = np.mean(np.max(array_cost, axis=2) - np.min(array_cost, axis=2), axis=1)
    rng = np.random.default_rng(1234115)

    # To display convergence:
    if verbose:
        pbar = ConvergenceProgressBar(
            convergence_goal=precision, maxiter=maxiter, degrowth=4
        )

    while iter < maxiter and (iter < 4 or (opt_gap > precision)):
        iter += 1
        if verbose:
            pbar.describe("Dynamic Programming")

        (
            levels,
            bellman_costs,
            _,
            _,
            future_costs_approx_l,
        ) = get_bellman_values_from_costs(
            param=param,
            multi_stock_management=multi_stock_management,
            costs_approx=costs_approx,
            future_costs_approx=future_costs_approx,
            nSteps_bellman=nSteps_bellman,
            name_solver=name_solver,
            method=method,
            trajectory=trajectory,
            correlations=correlations,
            divisor=divisor,
            verbose=verbose,
        )
        future_costs_approx = future_costs_approx_l[WeekIndex(0)]

        # Evaluate optimal
        trajectory, pseudo_opt_controls, _ = solve_for_optimal_trajectory(
            param=param,
            multi_stock_management=multi_stock_management,
            costs_approx=costs_approx,
            future_costs_approx_l=future_costs_approx_l,
            starting_pt=starting_pt,
            name_solver=name_solver,
            divisor=divisor,
        )
        # Beware, some trajectories seem to be overstep the bounds with values such as -2e-12

        controls_list = select_controls_to_explore(
            param=param,
            multi_stock_management=multi_stock_management,
            pseudo_opt_controls=pseudo_opt_controls,
            costs_approx=costs_approx,
        )

        if verbose:
            pbar.describe("Simulation")

        # Evaluating this "optimal" trajectory
        controls, costs, slopes = Lget_costs(
            param=param,
            multi_stock_management=multi_stock_management,
            controls_list=controls_list,
            saving_directory=saving_dir,
            output_path=output_path,
            name_solver=name_solver,
            verbose=verbose,
            load_from_protos=True,
            prefix=f"cut_plan_iter_{iter}_",
        )

        opt_gap = get_opt_gap(
            param=param,
            costs=costs,
            costs_approx=costs_approx,
            controls_list=controls,
            opt_gap=opt_gap,
            max_gap={WeekIndex(w): max_gap[w] for w in range(param.len_week)},
        )

        costs_approx.update(
            controls=timescenario_list_area_value_to_array(
                controls_list, param, multi_stock_management.areas
            ),
            costs=timescenario_list_value_to_array(costs, param),
            duals=timescenario_list_area_value_to_array(
                slopes, param, multi_stock_management.areas
            ),
        )
        costs_approx.remove_redundants(tolerance=1e-2)

        # If we want to look at the usage values evolution
        if verbose:
            pbar.describe("Drawing")
            usage_values, levels_uv = compute_usage_values_from_costs(
                param=param,
                multi_stock_management=multi_stock_management,
                name_solver=name_solver,
                costs_approx=costs_approx,
                future_costs_approx_l=future_costs_approx_l,
                optimal_trajectory=trajectory,
                nSteps_bellman=nSteps_bellman,
                correlations=correlations,
                divisor=divisor,
            )
            draw_usage_values(
                usage_values={
                    a.area: time_list_to_array(x) for a, x in usage_values.items()
                },
                levels_uv=time_list_area_value_to_array(
                    levels_uv, param, multi_stock_management.areas
                ),
                n_weeks=param.len_week,
                nSteps_bellman=nSteps_bellman,
                multi_stock_management=multi_stock_management,
                trajectory=timescenario_area_value_to_array(
                    trajectory, param, multi_stock_management.areas
                ),
                ub=500,
            )
            pbar.update(precision=opt_gap)

    if verbose:
        pbar.close()
    return bellman_costs, levels, future_costs_approx_l, costs_approx, trajectory


def iter_bell_vals(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    n_controls_init: int,
    starting_pt: Dict[AreaIndex, float],
    nSteps_bellman: int,
    method: str,
    saving_dir: str,
    name_solver: str = "CLP",
    precision: float = 1e-2,
    correlations: Optional[np.ndarray] = None,
    divisor: dict[str, float] = {"euro": 1e8, "energy": 1e4},
    verbose: bool = False,
) -> tuple[
    Dict[WeekIndex, List[float]],
    LinearCostEstimator,
    Dict[WeekIndex,LinearInterpolator],
    Dict[WeekIndex, List[Dict[AreaIndex, float]]],
    Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
    Dict[AreaIndex, Dict[WeekIndex, List[float]]],
]:
    """
    In a similar fashion to Kelley's algorithm (1960), the idea is to approximate the (convex) cost function
    of the antares solver using cutting planes to minimize the system cost over a year by choosing the best control for stocks
    The solver is considered as a black blox oracle providing the function value and derivatives at any given point.

    We start by initializing the cut approximation of the system cost for every week as a function of the control
    Several optimizations points have been raised:
        - What points to use for first approximation
        - Should we "enrich" the data by trying to 'interpolate between our cutting planes' to get a more curvy / stable solution
        - Should we stabilize with a quadratic term i.e proximal bundle method
    We also initialize a future cost function. Hypothesises:
        - We might want to overpenalize being under the initial level since the flexibility will be learned once we have a good approx for system prices


    After that every iteration should consist in:
        - Use a Dynamic Programing algorithm (SDDP ?) to get the optimal yearly cost / control from each weekly cost
            - NB: Yearly costs will be used at next iteration as a final level cost,
        - Compute δ = Bellman_costs[...] - (end_of_year_costs) - old_costs_without_end_of_year_costs
        - Get controls to add: a list of all controls we want to test: the optimal and others if wanted
            - If the SDDP algorithm is costly we'll want to do more test between calls to SDDP
        - For all these controls get blackbox's values
        - Update our cost approx (if we use cuts interpolations add new ones and remove blatantly false ones)

    Args:
        param (TimeScenarioParameter): Defines number of weeks and scenarios we optimize on
        multi_stock_management (MultiStockManagement): Description of the reservoirs
        output_path (str): Link to the pre generated mps files of the study
        n_controls_init (int): Number of controls heuristically chosen to initialize our search
        starting_pt (Dict[AreaIndex, float]): Reservoir state before week 0
        nSteps_bellman (int): Precision of the bellman solving part
        name_solver (str, optional): name of the solver used. Defaults to "CLP".
        precision (float, optional): Gap to minimum tolerated. Defaults to 1e-2.
        divisor (dict, optional): https://www.gurobi.com/documentation/8.1/refman/numerics_advanced_user_sca.html
        maxiter (int, optional): Maximum number of iterations. Defaults to 2.
        verbose (bool, optional): Defines whether or not the function displlays informations while running. Defaults to False.

    Returns:
        tuple[Dict[WeekIndex, List[float]], Estimator, list[LinearInterpolator], Dict[WeekIndex, List[Dict[AreaIndex, float]]]]: Bellman costs, costs approximation, future costs approximation, levels
    """
    # Choose first controls to test
    controls_list = initialize_controls(
        param=param,
        multi_stock_management=multi_stock_management,
        n_controls_init=n_controls_init,
    )

    # Get hyperplanes resulting from initial controls
    controls, costs, duals = Lget_costs(
        param=param,
        multi_stock_management=multi_stock_management,
        output_path=output_path,
        saving_directory=saving_dir,
        name_solver=name_solver,
        controls_list=controls_list,
        load_from_protos=True,
        verbose=verbose,
    )

    costs_approx = LinearCostEstimator(
        param=param,
        controls=timescenario_list_area_value_to_array(
            controls, param, multi_stock_management.areas
        ),
        costs=timescenario_list_value_to_array(costs, param),
        duals=timescenario_list_area_value_to_array(
            duals, param, multi_stock_management.areas
        ),
    )

    # Initialize our approximation on future costs
    future_costs_approx = initialize_future_costs(
        starting_pt=starting_pt,
        multi_stock_management=multi_stock_management,
    )

    # Correlations matrix
    correlations = get_correlation_matrix(
        multi_stock_management=multi_stock_management,
        corr_type="no_corrs",
    )

    # Iterative part
    bellman_costs, levels, future_costs_approx_l, costs_approx, optimal_trajectory = (
        cutting_plane_method(
            param=param,
            multi_stock_management=multi_stock_management,
            name_solver=name_solver,
            starting_pt=starting_pt,
            costs_approx=costs_approx,
            saving_dir=saving_dir,
            costs=costs,
            future_costs_approx=future_costs_approx,
            nSteps_bellman=nSteps_bellman,
            method=method,
            correlations=correlations,
            precision=precision,
            divisor=divisor,
            output_path=output_path,
            verbose=verbose,
        )
    )

    # Deducing usage values
    usage_values, _ = compute_usage_values_from_costs(
        param=param,
        multi_stock_management=multi_stock_management,
        name_solver=name_solver,
        costs_approx=costs_approx,
        future_costs_approx_l=future_costs_approx_l,
        optimal_trajectory=optimal_trajectory,
        nSteps_bellman=101,
        correlations=correlations,
        divisor=divisor,
    )

    return (
        bellman_costs,
        costs_approx,
        future_costs_approx_l,
        levels,
        optimal_trajectory,
        usage_values,
    )


def sddp_cutting_planes(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    name_solver: str,
    costs_approx: LinearCostEstimator,
    costs: Dict[TimeScenarioIndex, List[float]],
    level_init: Dict[AreaIndex, float],
    saving_dir: str,
    normalization: Dict[str, float],
    maxiter: Optional[int] = None,
    precision: float = 1e-2,
    verbose: bool = False,
) -> tuple[
    np.ndarray,
    Dict[TimeScenarioIndex, float],
    LinearCostEstimator,
    List[Dict[WeekIndex, Dict[AreaIndex, List[float]]]],
]:

    # Initialize julia
    jl.include("src/sddp.jl")
    # Import the ReservoirManagement module
    jl_sddp = jl.Jl_SDDP
    julia_reservoirs = np.array(
        [
            dict(
                capacity=mng.reservoir.capacity,
                efficiency=mng.reservoir.efficiency,
                max_pumping=mng.reservoir.max_pumping,
                max_generating=mng.reservoir.max_generating,
                upper_level=mng.reservoir.upper_rule_curve,
                lower_level=mng.reservoir.bottom_rule_curve,
                upper_curve_penalty=mng.penalty_upper_rule_curve,
                lower_curve_penalty=mng.penalty_bottom_rule_curve,
                spillage_penalty=2 * mng.penalty_upper_rule_curve + 10,
                level_init=level_init[area],
                inflows=mng.reservoir.inflow,
            )
            for area, mng in multi_stock_management.dict_reservoirs.items()
        ]
    )

    # Initialize loop
    iter = 0
    opt_gap = 1.0
    maxiter = 12 if maxiter is None else maxiter
    array_costs = timescenario_list_value_to_array(costs, param)
    max_gap = np.mean(np.max(array_costs, axis=2) - np.min(array_costs, axis=2), axis=1)
    all_uvs = []

    # Display
    if verbose:
        pbar = ConvergenceProgressBar(
            convergence_goal=precision, maxiter=maxiter, degrowth=4
        )
        pbar.describe("SDDP-ing")
    julia_capp = costs_approx.to_julia_compatible_structure()
    formatted_data = jl_sddp.formater(
        param.len_week,
        param.len_scenario,
        julia_reservoirs,
        julia_capp,
        saving_dir,
        normalization["euro"],
        normalization["energy"],
    )
    jl_sddp.reinit_cuts(*formatted_data)
    # Body
    while iter < maxiter and (iter < 4 or (opt_gap > precision)):
        iter += 1
        julia_capp = costs_approx.to_julia_compatible_structure()
        formatted_data = jl_sddp.formater(
            param.len_week,
            param.len_scenario,
            julia_reservoirs,
            julia_capp,
            saving_dir,
            normalization["euro"],
            normalization["energy"],
        )
        jl_reservoirs, norms = formatted_data[2], formatted_data[4]
        # SDDP to train on our cost approx
        sim_res, model = jl_sddp.manage_reservoirs(*formatted_data)

        # Resulting controls per scenario
        array_controls = np.array(
            [[week["control"] for week in res][: param.len_week] for res in sim_res]
        ).swapaxes(
            0, 1
        )  # n_weeks x n_scenarios x 1 x N_reservoirs
        input_controls = {
            TimeScenarioIndex(w, s): {
                a: array_controls[w, s, i]
                for i, a in enumerate(multi_stock_management.areas)
            }
            for w in range(param.len_week)
            for s in range(param.len_scenario)
        }
        controls = select_controls_to_explore(
            param=param,
            multi_stock_management=multi_stock_management,
            pseudo_opt_controls=input_controls,
            costs_approx=costs_approx,
        )

        if verbose:
            pbar.describe("Simulating")

        controls, costs, slopes = Lget_costs(
            param=param,
            multi_stock_management=multi_stock_management,
            controls_list=controls,
            output_path=output_path,
            saving_directory=saving_dir,
            name_solver=name_solver,
            verbose=False,
            load_from_protos=True,
            prefix=f"SDDP_iter_{iter}_",
        )

        # Computing opt gap
        opt_gap = get_opt_gap(
            param=param,
            costs=costs,
            costs_approx=costs_approx,
            controls_list=controls,
            opt_gap=opt_gap,
            max_gap={WeekIndex(w): max_gap[w] for w in range(param.len_week)},
        )

        if verbose:
            pbar.describe("Drawing")
            usage_values, bellman_costs = jl_sddp.get_usage_values(
                param.len_week, param.len_scenario, jl_reservoirs, model, norms, 101
            )
            all_uvs.append(usage_values)
            draw_uvs_sddp(
                param=param,
                multi_stock_management=multi_stock_management,
                usage_values=usage_values,
                simulation_results=sim_res,
                div=1,
            )
            pbar.update(precision=opt_gap)

        costs_approx.update(
            controls=timescenario_list_area_value_to_array(
                controls, param, multi_stock_management.areas
            ),
            costs=timescenario_list_value_to_array(costs, param),
            duals=timescenario_list_area_value_to_array(
                slopes, param, multi_stock_management.areas
            ),
        )
        costs_approx.remove_redundants(tolerance=1e-2)
    usage_values, bellman_costs = jl_sddp.get_usage_values(
        param.len_week, param.len_scenario, jl_reservoirs, model, norms, 101
    )
    return usage_values, bellman_costs, costs_approx, all_uvs


def iter_bell_vals_v2(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    n_controls_init: int,
    starting_pt: Dict[AreaIndex, float],
    saving_dir: str,
    normalization: Dict[str, float],
    name_solver: str = "CLP",
    precision: float = 1e-2,
    maxiter: int = 2,
    verbose: bool = False,
) -> tuple[
    np.ndarray,
    Dict[TimeScenarioIndex, float],
    LinearCostEstimator,
    List[Dict[WeekIndex, Dict[AreaIndex, List[float]]]],
]:

    # Choose first controls to test
    controls_list = initialize_controls(
        param=param,
        multi_stock_management=multi_stock_management,
        n_controls_init=n_controls_init,
    )

    controls_list, costs, duals = Lget_costs(
        param=param,
        multi_stock_management=multi_stock_management,
        output_path=output_path,
        saving_directory=saving_dir,
        name_solver=name_solver,
        controls_list=controls_list,
        load_from_protos=False,
        verbose=verbose,
    )

    costs_approx = LinearCostEstimator(
        param=param,
        controls=timescenario_list_area_value_to_array(
            controls_list, param, multi_stock_management.areas
        ),
        costs=timescenario_list_value_to_array(costs, param),
        duals=timescenario_list_area_value_to_array(
            duals, param, multi_stock_management.areas
        ),
    )
    # Iterative part
    usage_values, bellman_costs, costs_approx, all_uvs = sddp_cutting_planes(
        param=param,
        multi_stock_management=multi_stock_management,
        output_path=output_path,
        name_solver=name_solver,
        costs_approx=costs_approx,
        saving_dir=saving_dir,
        costs=costs,
        level_init=starting_pt,
        precision=precision,
        normalization=normalization,
        maxiter=maxiter,
        verbose=verbose,
    )

    return usage_values, bellman_costs, costs_approx, all_uvs


def generate_fast_uvs_v2(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    mrg_prices: Dict[AreaIndex, Dict[str, float]],
) -> Dict[AreaIndex, Dict[WeekIndex, List[float]]]:
    disc = 101
    uvs: Dict[AreaIndex, Dict[WeekIndex, List[float]]] = {
        a: {WeekIndex(w): [] for w in range(param.len_week)}
        for a in multi_stock_management.areas
    }
    for area, mng in multi_stock_management.dict_reservoirs.items():
        res = mng.reservoir
        alpha = 1 + 4 * (area == "es")
        print(
            mrg_prices[area]["mean"] - alpha * mrg_prices[area]["std"],
            mrg_prices[area]["mean"] + alpha * mrg_prices[area]["std"],
        )
        low_curve = res.bottom_rule_curve[1:] / res.capacity
        high_curve = res.upper_rule_curve[1:] / res.capacity
        percent_cap = np.arange(101) / 100
        below_low = percent_cap[None, :] <= low_curve[:, None]
        over_high = percent_cap[None, :] >= high_curve[:, None]
        max_below = np.sum(below_low, axis=1)
        min_over = np.sum(1 - over_high, axis=1)
        dists_intercurve = min_over - max_below
        for week in range(param.len_week):
            weekly_uv = np.zeros(disc)
            low, high, dist = max_below[week], min_over[week], dists_intercurve[week]
            weekly_uv[:low] = 100
            weekly_uv[high:] = 0
            weekly_uv[low:high] = np.linspace(
                mrg_prices[area]["mean"] - alpha * mrg_prices[area]["std"],
                mrg_prices[area]["mean"] + alpha * mrg_prices[area]["std"],
                dist,
            )[::-1]
            uvs[area][WeekIndex(week)] = list(weekly_uv)
    return uvs
