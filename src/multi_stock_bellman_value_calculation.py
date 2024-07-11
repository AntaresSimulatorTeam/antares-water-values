from itertools import product
from calculate_reward_and_bellman_values import (
    RewardApproximation,
    ReservoirManagement,
    BellmanValueCalculation,
    MultiStockManagement,
    MultiStockBellmanValueCalculation,
)
from ortools.linear_solver.python import model_builder
import ortools.linear_solver.pywraplp as pywraplp
from estimation import Estimator, LinearCostEstimator, LinearInterpolator
from read_antares_data import TimeScenarioParameter, TimeScenarioIndex
from optimization import (AntaresProblem, 
                          WeeklyBellmanProblem, 
                          Basis, 
                          solve_problem_with_multivariate_bellman_values, 
                          solve_for_optimal_trajectory)
from typing import Annotated, List, Literal, Dict, Optional, Any
import numpy.typing as npt
import numpy as np
from scipy.interpolate import interp1d
from functions_iterative import compute_upper_bound
from display import ConvergenceProgressBar, draw_usage_values
from scipy.stats import random_correlation
import copy
from tools import Caller
from tqdm import tqdm

Array1D = Annotated[npt.NDArray[np.float32], Literal["N"]]
Array2D = Annotated[npt.NDArray[np.float32], Literal["N", "N"]]


def calculate_bellman_value_multi_stock(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    X: Dict[str, Array1D],
    name_solver: str = "CLP",
) -> tuple[Dict[int, Dict[str, npt.NDArray[np.float32]]], float, float]:
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
                _, _, Vu, slope, _, _ = solve_problem_with_multivariate_bellman_values(
                    multi_bellman_value_calculation=multi_bellman_value_calculation,
                    V=V[week + 1],
                    level_i={
                        area: multi_bellman_value_calculation.dict_reservoirs[
                            area
                        ].stock_discretization[idx[i]]
                        for i, area in enumerate(m.range_reservoir)
                    },
                    m=m,
                    take_into_account_z_and_y=True,
                )
                V[week]["intercept"][idx] += Vu / param.len_scenario
                for area in m.range_reservoir:
                    V[week][f"slope_{area}"][idx] += slope[area] / param.len_scenario

    lower_bound = max(
        [
            sum(
                [
                    V[0][f"slope_{area}"][idx]
                    * (
                        multi_bellman_value_calculation.dict_reservoirs[
                            area
                        ].reservoir_management.reservoir.initial_level
                        - multi_bellman_value_calculation.dict_reservoirs[
                            area
                        ].stock_discretization[idx[i]]
                    )
                    for i, area in enumerate(
                        multi_bellman_value_calculation.dict_reservoirs.keys()
                    )
                ]
            )
            + V[0]["intercept"][idx]
            for idx in multi_bellman_value_calculation.get_product_stock_discretization()
        ]
    )

    upper_bound = compute_upper_bound_multi_stock(
        param=param,
        multi_bellman_value_calculation=multi_bellman_value_calculation,
        list_models=list_models,
        V=V,
    )

    return V, lower_bound, upper_bound

def compute_upper_bound_multi_stock(
    param: TimeScenarioParameter,
    multi_bellman_value_calculation: MultiStockBellmanValueCalculation,
    list_models: Dict[TimeScenarioIndex, AntaresProblem],
    V: Dict[int, Dict[str, npt.NDArray[np.float32]]],
) -> float:

    cout = 0.0
    for scenario in range(param.len_scenario):

        level_i = {
            area: multi_bellman_value_calculation.dict_reservoirs[
                area
            ].reservoir_management.reservoir.initial_level
            for area in multi_bellman_value_calculation.dict_reservoirs.keys()
        }

        for week in range(param.len_week):
            print(f"{scenario} {week}", end="\r")
            m = list_models[TimeScenarioIndex(week, scenario)]

            _, _, current_cost, _, level_i, _ = (
                solve_problem_with_multivariate_bellman_values(
                    multi_bellman_value_calculation=multi_bellman_value_calculation,
                    V=V[week + 1],
                    level_i=level_i,
                    m=m,
                    take_into_account_z_and_y=(week == param.len_week - 1),
                )
            )
            cout += current_cost

    upper_bound = cout / param.len_scenario

    return upper_bound

def initialize_antares_problems(
    param:TimeScenarioParameter,
    multi_stock_management:MultiStockManagement,
    output_path:str,
    name_solver:str,
    direct_bellman_calc:bool=True,
    verbose:bool=False,
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
        week_range = tqdm(week_range, desc='Problem initialization', colour='Yellow')
    for week in week_range:
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
                direct_bellman_calc=direct_bellman_calc,
            )
            
            list_models[TimeScenarioIndex(week, scenario)] = m
    return list_models

def get_bellman_values_from_costs(
    param:TimeScenarioParameter,
    multi_stock_management:MultiStockManagement,
    costs_approx:Estimator,
    future_costs_approx:LinearInterpolator,
    nSteps_bellman:int,
    name_solver:str,
    method:str,
    trajectory:np.ndarray,
    correlations:np.ndarray,
    verbose:bool=False,
    )->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[LinearInterpolator]]:
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
        levels:np.ndarray: Level discretization used,
        costs:np.ndarray: Objective value at every week (and for every level combination), 
        duals:np.ndarray: Dual values of the initial level constraint at every week (and for every level combination), 
        future_costs_approx:LinearInterpolator: Final estimation of total system prices
    """
    #Initializing the Weekly Bellman Problem instance
    problem = WeeklyBellmanProblem(
        param=param,
        multi_stock_management=multi_stock_management,
        week_costs_estimation=costs_approx,
        name_solver=name_solver
    )
    # problem.parameters.SetDoubleParam(problem.parameters.PRESOLVE, problem.parameters.PRESOLVE_ON)
    
    #Keeping in memory all future costs approximations
    future_costs_approx_l = [future_costs_approx]
    
    # Parameters
    n_weeks = param.len_week
    n_scenarios = param.len_scenario
    
    # Initializing controls, costs and duals
    controls = []
    costs = []
    duals = []
    all_levels = []

    # Starting from last week dynamically solving the optimal control problem (from every starting level)
    week_range = range(n_weeks-1, -1, -1)
    if verbose:
        week_range = tqdm(week_range, colour="Green", desc="Dynamic Solving")
    for week in week_range:
        # Getting levels along which we heck 
        levels = multi_stock_management.get_disc(
            week=week,
            xNsteps=nSteps_bellman,
            trajectory=trajectory,
            correlation_matrix=correlations,
            method=method)
        all_levels.append(levels)
        
        #Preparing receiving arrays
        controls_w = np.zeros(list(levels.shape) + [n_scenarios])
        costs_w = np.zeros(list(levels.shape)[:-1])
        duals_w = np.zeros(list(levels.shape))
        for lvl_id, lvl_init in enumerate(levels):
            
            #Remove previous constraints / vars
            problem.solver = pywraplp.Solver.CreateSolver(name_solver)
            # problem.reset_solver()
            
            #Rewrite problem
            problem.write_problem(
                week=week,
                level_init=lvl_init,
                future_costs_estimation=future_costs_approx,
            )
            
            #Solve, should we make use of previous Bases ? Not yet, computations still tractable for n<=2
            try:
                controls_wls, cost_wl, duals_wl, _ = problem.solve()
            except ValueError:
                print(f"""Failed to solve at week {week} with initial levels {lvl_init}""")
                raise ValueError
            #Writing down results
            controls_w[lvl_id] = controls_wls
            costs_w[lvl_id] = cost_wl
            duals_w[lvl_id] = duals_wl

        controls.append(controls_w)
        costs.append(costs_w)
        duals.append(duals_w)

        # Updating the future estimator
        future_costs_approx = LinearInterpolator(inputs=levels, costs=costs_w-np.min(costs_w), duals=duals_w)
        future_costs_approx_l.insert(0,future_costs_approx)
        n_rdds, _ = future_costs_approx.count_redundant(tolerance=0, remove=True)
    return np.array(all_levels), np.array(costs), np.array(duals), np.array(controls), future_costs_approx_l

def initialize_future_costs(
    param:TimeScenarioParameter,
    starting_pt:np.ndarray,
    multi_stock_management:MultiStockManagement,
    costs_approx:Estimator,
    
) -> LinearInterpolator: #NOT ESTIMATOR ??
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
    mult = 100
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    inputs = [starting_pt for _ in range(2*n_reservoirs)]
    duals = [np.zeros((n_reservoirs)) for _ in range(2*n_reservoirs)]
    for i in range(n_reservoirs):
        duals[i][i] = mult
        duals[i+1][i] = -mult
    return LinearInterpolator(
        inputs=inputs,
        costs=np.zeros(2*n_reservoirs),
        duals=duals,)

def get_week_scenario_costs(m:AntaresProblem,
                            multi_stock_management:MultiStockManagement,
                            controls_list:np.ndarray,
                            )-> tuple[np.ndarray, np.ndarray, int, list[float]]:
    """
    Takes a control and an initialized Antares problem setup and returns the objective and duals 
    for every week and every Scenario
    
    Parameters
    ----------
        m:AntaresProblem: Instance of Antares problem describing the problem currently solved
            (at specified week and scenario),
        multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
        controls_list:np.ndarray, array of all controls to be checked
        
    Returns
    -------
        costs:np.ndarray: cost of each control,
        slopes:np.ndarray: dual values for each control,
        tot_iter:int: number of simplex pivots,
        times:list[float]: list of solving times
    """
    
    tot_iter=0
    times = []
    
    #Initialize Basis
    basis = Basis([], [])
    
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    
    #Initialize costs
    n_controls = controls_list.shape[0]
    costs = np.zeros(n_controls)
    slopes = np.zeros([n_controls, n_reservoirs])
    for i, stock_management in enumerate(controls_list):
        
        #Formatting control
        control = {area:cont for area, cont in zip(multi_stock_management.dict_reservoirs, stock_management)}
        
        #Solving the problem
        control_cost, control_slopes, itr, time_taken = m.solve_with_predefined_controls(control=control,
                                                                                         prev_basis=basis)
        tot_iter += itr
        times.append(time_taken)
        
        #Save results
        costs[i] = control_cost
        for j, area in enumerate(multi_stock_management.dict_reservoirs):
            slopes[i][j] = control_slopes[area]
        
        #Save basis
        rstatus, cstatus = m.get_basis()
        basis = Basis(rstatus=rstatus, cstatus=cstatus)
    return costs, slopes, tot_iter, times

def get_all_costs(param:TimeScenarioParameter,
                    list_models:Dict[TimeScenarioIndex, AntaresProblem],
                    multi_stock_management:MultiStockManagement,
                    controls_list:np.ndarray,
                    verbose:bool=False,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes a problem and a discretization level and solves the Antares pb for every combination of stock for every week
    
        Parameters
        ----------
            param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
            list_models:Dict[TimeScenarioIndex, AntaresProblem]: List of models
            multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
            controls_list:np.ndarray: Controls to be evaluated, per week, per scenario
            verbose:bool:Control the level of outputs show by the function
            
        Returns
        -------
            Bellman Values:Dict[int, Dict[str, npt.NDArray[np.float32]]]: Bellman values
    """
    
    tot_iter = 0
    times = []
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    
    # Initializing the n_weeks*n_scenarios*n_controls*n_stocks values to fill
    shape_controls = list(controls_list.shape)
    costs = np.zeros(shape_controls[:-1])
    slopes = np.zeros(shape_controls)
    week_range = range(param.len_week)
    if verbose:
        week_range = tqdm(range(param.len_week), colour='blue', desc="Simulation")
    for week in week_range:
        for scenario in range(param.len_scenario):
            ts_id = TimeScenarioIndex(week=week, scenario=scenario)
            #Antares problem
            m = list_models[ts_id]
            try:
                costs_ws, slopes_ws, iters, times_ws = get_week_scenario_costs(
                    m=m,
                    multi_stock_management=multi_stock_management,
                    controls_list=controls_list[week, scenario],
                )
            except ValueError:
                reservoirs=[mng.reservoir for mng in multi_stock_management.dict_reservoirs.values()]
                print(f"Failed at week {week}, the conditions on control were: {controls_list[week, scenario]}")
                raise ValueError
            tot_iter += iters
            times.append(times_ws)
            costs[week, scenario] = costs_ws
            slopes[week, scenario] = slopes_ws
    # print(f"Number of simplex pivot {tot_iter}")
    return costs, slopes

def generate_controls(
    param:TimeScenarioParameter,
    multi_stock_management:MultiStockManagement,
    controls_looked_up:str,
    xNsteps:int,
) -> np.ndarray:
    """ 
    Generates an array of controls that will be precalculated for every week / scenario
    
    Parameters
    ----------
        param:TimeScenarioParameter: Contains the informations relative to the number of scenario and weeks
        multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
        controls_looked_up:str: 'grid', 'line', 'random', 'oval' description of the controls to generate
        xNsteps:int: Discretization level used
        name_solver:str: Solver chosen for the optimization problem, default -> CLP
        
    Returns
    -------
        Bellman Values:Dict[int, Dict[str, npt.NDArray[np.float32]]]: Bellman values
    """
    # General parameters
    weeks = param.len_week
    n_scenarios = param.len_scenario
    n_res = len(multi_stock_management.dict_reservoirs)
    
    # Useful data on reservoirs
    max_res = np.array([mng.reservoir.max_generating for mng in multi_stock_management.dict_reservoirs.values()])
    min_res = np.array([-mng.reservoir.max_pumping*mng.reservoir.efficiency 
                        for mng in multi_stock_management.dict_reservoirs.values()])
    mean_res = (max_res + min_res) /2
    
    # Points to start lines at
    start_pts = [min_res + (1/4)*((i+1)/(n_res+1))*(max_res - min_res)*(np.array([j != i for j in range(n_res)])[:,None]) for i in range(n_res)]
    end_points = [start_pt + (max_res - min_res)*(np.array([j == i for j in range(n_res)])[:,None]) for i,start_pt in enumerate(start_pts)]
    
    
    if controls_looked_up=="line":
        controls = np.concatenate([np.linspace(strt_pt, end_pt, xNsteps) for strt_pt, end_pt in zip(start_pts, end_points)])
        
    elif controls_looked_up=="random":
        rand_placements = np.random.uniform(0, 1, size=(xNsteps, n_res, weeks))
        rand_placements = min_res[None,:,:] + (max_res - min_res)[None,:,:]*rand_placements
        
    elif controls_looked_up=="oval":
        ...
    elif controls_looked_up=="line+diagonal":
        controls = np.concatenate([np.linspace(strt_pt, end_pt, xNsteps) for strt_pt, end_pt in zip(start_pts, end_points)]+[np.linspace(max_res, min_res, xNsteps)])
        
    elif controls_looked_up=="diagonal":
        controls = np.linspace(max_res, min_res, xNsteps)
        
    else: #Applying default: grid
        controls = np.array([i for i in product(
                *[np.linspace(-manag.reservoir.max_pumping*manag.reservoir.efficiency,
                            manag.reservoir.max_generating, xNsteps)
                    for area, manag in multi_stock_management.dict_reservoirs.items()])])

    controls = np.broadcast_to(controls, shape=[n_scenarios]+list(controls.shape))
    controls = np.moveaxis(controls, -1, 0) #Should be of shape [n_weeks, n_scenarios, n_controls, n_reservoirs]
    return controls

def precalculated_method(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    xNsteps: int = 12,
    Nsteps_bellman: int = 12,
    name_solver: str = "CLP",
    controls_looked_up: str = 'grid',
    verbose:bool = False,
) -> tuple[np.ndarray, LinearCostEstimator, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        Bellman Values:Dict[int, Dict[str, npt.NDArray[np.float32]]]: Bellman values
    """
    # __________________________________________________________________________________________
    #                           PRECALCULATION PART
    # __________________________________________________________________________________________
    
    #Initialize the problems
    list_models = initialize_antares_problems(
        param=param,
        multi_stock_management=multi_stock_management,
        output_path=output_path,
        name_solver=name_solver,
        direct_bellman_calc=False,
        verbose=verbose,)
    
    controls_list = generate_controls(
        param=param,
        multi_stock_management=multi_stock_management,
        controls_looked_up=controls_looked_up,
        xNsteps=xNsteps)
    
    print("=======================[      Precalculation of costs     ]=======================")
    costs, slopes, times = get_all_costs(
        param=param,
        list_models=list_models,
        multi_stock_management=multi_stock_management,
        controls_list=controls_list,
        verbose=verbose,
        )
    
    # print("=======================[       Reward approximation       ]=======================")
    #Initialize cost functions
    costs_approx = LinearCostEstimator(
        param=param,
        controls=controls_list,
        costs=costs,
        duals=slopes)
    
    # __________________________________________________________________________________________
    #                           BELLMAN CALCULATION PART
    # __________________________________________________________________________________________
    print("=======================[       Dynamic  Programming       ]=======================")
    future_costs_approx = initialize_future_costs(
        param=param,
        multi_stock_management=multi_stock_management,
        lvl_init=...,
        costs_approx=costs_approx,
    )
    for i in range(2):
        levels, bellman_costs, bellman_duals, bellman_controls, future_costs_approx_l = get_bellman_values_from_costs(
            param=param,
            multi_stock_management=multi_stock_management,
            cost_estimation=costs_approx,
            future_estimator=future_costs_approx,
            xNsteps=Nsteps_bellman,
            name_solver=name_solver,
            verbose=verbose,
        )
        future_costs_approx = future_costs_approx_l[-1]
    
    return levels, costs_approx, bellman_costs, bellman_controls, bellman_duals, np.array(times)

def initialize_controls(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    n_controls_init: int,
    rand:bool=False,
) -> np.ndarray:
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
    controls_list:np.ndarray:Shape: (n_controls, n_reservoirs)
    """
    n_stocks = len(multi_stock_management.dict_reservoirs)
    n_weeks = param.len_week
    n_scenarios = param.len_scenario
    max_cont = np.array([mng.reservoir.max_generating for mng in multi_stock_management.dict_reservoirs.values()])[:, :n_weeks]
    min_cont = np.array([-mng.reservoir.max_pumping*mng.reservoir.efficiency 
                        for mng in multi_stock_management.dict_reservoirs.values()])[:, :n_weeks]
    
    #Choose how many reservoirs will fully spend
    n_stocks_full_spend = np.round(np.linspace(0, n_stocks, num=n_controls_init))
    controls = np.zeros((n_weeks, n_scenarios, n_controls_init, n_stocks)) + min_cont.T[:, None, None, :]
    if rand:
        full_throttle = np.zeros((n_weeks, n_scenarios, n_controls_init, n_stocks)) + (n_stocks_full_spend[:,None] > np.arange(n_stocks)[None,:])[None, None, :, :]
        full_throttle = np.random.default_rng().permuted(full_throttle, axis=3)
    else:
        full_throttle = np.array([
            [
                [
                    np.roll([l < n_full_spend for l in range(n_stocks)], i+j+k)
                for k, n_full_spend in enumerate(n_stocks_full_spend)]
            for j in range(n_scenarios)]
        for i in range(n_weeks)])
    controls += full_throttle * (max_cont - min_cont).T[:, None, None, :]
    return controls
    
def select_controls_to_explore(
    param:TimeScenarioParameter,
    multi_stock_management:MultiStockManagement,
    pseudo_opt_controls:np.array,
    costs_approx:Estimator,
) -> np.ndarray:
    """Takes in some optimal (for now) controls and returns a list of controls to explore,
    in this version: only returns the optimal control

    Args:
        param (TimeScenarioParameter): _description_
        multi_stock_management (MultiStockManagement): _description_
        pseudo_opt_controls (np.array): _description_
        costs_approx (Estimator): _description_

    Returns:
        np.ndarray: Controls for a week / scenario / control_id / reservoir given
    """
    #This only a first version:
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    controls_to_explore = np.zeros((param.len_week, param.len_scenario, 1, n_reservoirs))
    controls_to_explore[:,:,0,:] = pseudo_opt_controls
    
    return controls_to_explore

def get_correlation_matrix(param:TimeScenarioParameter,
                           multi_stock_management:MultiStockManagement,
                           adjacency_mat:Optional[np.ndarray]=None,
                           method:str="no_corrs") -> np.ndarray:
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    if adjacency_mat is not None:
        dist_mat =  adjacency_mat
        for i in range(n_reservoirs):
            dist_mat = (dist_mat[None,:] + dist_mat.T[:,None]).min(axis=2).T
        correlations = 1 / dist_mat
    elif method=="random_corrs":
        rng = np.random.default_rng()
        correlations = random_correlation.rvs(
            np.ones(n_reservoirs),
            random_state=rng,)
        correlations = np.abs(correlations)
    elif method=="unif_corrs":
        t = 0.5
        correlations = np.ones((n_reservoirs, n_reservoirs))*t + (1-t)*np.eye(N=n_reservoirs)
    else:
        correlations = np.eye(N=n_reservoirs)
    return correlations

def compute_usage_values_from_costs(
    param:TimeScenarioParameter,
    multi_stock_management:MultiStockManagement,
    name_solver:str,
    costs_approx:LinearCostEstimator,
    future_costs_approx_l:list[LinearInterpolator],
    optimal_trajectory:np.ndarray,
    discretization:int,
    correlations:np.ndarray,
    verbose:bool=False,
)-> dict[str, np.ndarray]:
    """ For now we want this function to solve a lot of optimization problems for different initial values of stock
    and get the dual values for the initial stock constraint to get the corresponding value

    Args:
        param (TimeScenarioParameter): _description_
        multi_stock_management (MultiStockManagement): _description_
        name_solver (str): _description_
        costs_approx (LinearCostEstimator): _description_
        future_costs_approx_l (list[LinearInterpolator]): _description_
        optimal_trajectory (np.ndarray): _description_
        discretization (int): _description_
        correlations (np.ndarray): _description_,
        verbose (bool)(Default to False): _description_,

    Returns:
        dict[str, np.ndarray]: _description_
    """
    #Initialize problem
    problem = WeeklyBellmanProblem(
        param=param,
        multi_stock_management=multi_stock_management,
        week_costs_estimation=costs_approx,
        name_solver=name_solver
    )

    #Parameter
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    n_weeks = param.len_week
    
    #Initialize usage values
    usage_values = {area:np.zeros((n_weeks ,discretization)) for area in multi_stock_management.dict_reservoirs.keys()}
    base_levels = np.mean(optimal_trajectory, axis=2)

    #Displaying options
    week_range = range(n_weeks)
    if verbose:
        week_range = tqdm(week_range, colour='red', desc="Usage values deduction")
    
    # N
    capacities = np.array([mng.reservoir.capacity
                           for mng in multi_stock_management.dict_reservoirs.values()])
    # W x Disc x N
    alpha = 0
    bottoms = np.array([mng.reservoir.bottom_rule_curve[:n_weeks+1] for mng in multi_stock_management.dict_reservoirs.values()])*alpha
    tops = np.array([mng.reservoir.upper_rule_curve[:n_weeks+1] for mng in multi_stock_management.dict_reservoirs.values()])*alpha\
         + np.array([[mng.reservoir.capacity]*(n_weeks+1) for mng in multi_stock_management.dict_reservoirs.values()]) * (1 - alpha)
    levels_imposed = np.moveaxis(np.linspace(bottoms, tops, discretization), -1, 0)

    # W x Disc x N
    lvl_relat_gaps = (levels_imposed - base_levels[:, None, :]) / capacities[None, None, :]
    relat_diffs = np.dot(lvl_relat_gaps[:, :, :, None] * np.eye(n_reservoirs)[None, None, :, :],
                        correlations)
    levels_to_test = base_levels[:, None, None, :] + relat_diffs * capacities[None, None, None, :]
    # Correct impossible levels
    levels_to_test *= levels_to_test > 0
    levels_to_test = np.minimum(levels_to_test, capacities[None, None, :])
    for week in week_range:
        for i, (area, management) in enumerate(multi_stock_management.dict_reservoirs.items()):
            for j, levels in enumerate(levels_to_test[week, :, i]):
                #Remove previous constraints / vars
                problem.reset_solver()
                
                #Rewrite problem
                problem.write_problem(
                    week=week,
                    level_init=levels,
                    future_costs_estimation=future_costs_approx_l[week],
                )

                # Solve
                try:
                    _, _, dual_vals, _ = problem.solve()
                except ValueError:
                    levels_dict = {area:levels[i] for i, area in enumerate(multi_stock_management.dict_reservoirs.keys())}
                    print(f"""Failed to solve at week {week} with initial levels {levels_dict} \n
                        (when imposing level of {area} to {levels_imposed[j, i]}) \n
                            thus having levels_diff as {relat_diffs[week, j,i]} \n
                            when base_level was {base_levels[week,i]}   """)
                    raise ValueError
                usage_values[area][week, j] = -dual_vals[i]
    return usage_values, levels_imposed

def get_opt_gap(
    param:TimeScenarioParameter,
    costs:np.ndarray, 
    costs_approx:Estimator, 
    controls_to_check:np.ndarray,
    max_gap:float,
    ):
    pre_update_costs = np.array([[costs_approx[week,scenario](controls_to_check[week, scenario]) 
                for scenario in range(param.len_scenario)] 
               for week in range(param.len_week)])
    #Computing the optimality gap
    pre_update_tot_costs = np.sum(pre_update_costs, axis=0)
    real_tot_costs = np.sum(costs, axis=0)
    opt_gap = max(1e-10, min(opt_gap, np.mean((real_tot_costs - pre_update_tot_costs)/max_gap)))
    return opt_gap

def cutting_plane_method(
    param:TimeScenarioParameter,
    multi_stock_management:MultiStockManagement,
    name_solver:str,
    list_models:list,
    starting_pt:np.ndarray,
    costs_approx:Estimator,
    true_controls:np.ndarray,
    true_costs:np.ndarray,
    true_duals:np.ndarray,
    future_costs_approx:LinearInterpolator,
    nSteps_bellman:int,
    method:str,
    correlations:np.ndarray,
    maxiter:Optional[int]=None,
    precision:Optional[float]=5e-2,
    interp_mode:bool=False,
    verbose:bool=False,
):
    """
    Iteratively: computes an optimal control from costs_approx -> Refines approximation for this control
    Stops: When too many iterations occured or when optimality gap is small enough
    
    Parameters
    ----------
        param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
        multi_stock_management: MultiStockManagement: Description of stocks and their global policies,
        name_solver:str: Solver chosen for the optimization problem, default -> CLP
        list_models:list: List of antares models
        starting_pt:np.ndarray: Point our simulations will start from
        costs_approx:Estimator: Linear approximation of weekly costs depending on control
        future_costs_approx:LinearInterpolator: Linear approximation of future costs when starting from certain level
        nSteps_bellman:int: Discretization level used
        maxiter:Optional[int]: Maximum number of iterations used
        precision:Optional[float]: Convergence criterion on optimality gap
        
    Returns
    -------
        Bellman Values:Dict[int, Dict[str, npt.NDArray[np.float32]]]: Bellman values
    """
    
    iter=0
    opt_gap=1.
    maxiter = int(1e2) if maxiter is None else maxiter
    inflows = np.array([mng.reservoir.inflow[:param.len_week, :param.len_scenario]\
                    for mng in multi_stock_management.dict_reservoirs.values()]).T
    # Init trajectory
    trajectory = np.array([starting_pt]*param.len_week)
    
    # Max gap
    max_gap = np.sum(np.mean(np.max(true_costs, axis=2) - np.min(true_costs, axis=2), axis=1), axis=0)

    hlpr = Caller(
        param=param,
        multi_stock_management=multi_stock_management,
        name_solver=name_solver,
        list_models=list_models,
        starting_pt=starting_pt,
        inflows=inflows,
        trajectory=trajectory,
        max_gap=max_gap,
        costs_approx=costs_approx,
        true_controls=true_controls,
        true_costs=true_costs,
        true_duals=true_duals,
        future_costs_approx=future_costs_approx,
        nSteps_bellman=nSteps_bellman,
        method=method,
        correlations=correlations,
        maxiter=maxiter,
        precision=precision,
        interp_mode=interp_mode,
        verbose=False,)

    # To display convergence:
    if verbose:
        pbar = ConvergenceProgressBar(convergence_goal=precision, maxiter=maxiter, degrowth=4)
    
    while iter < maxiter and (iter < 3 or (opt_gap > precision)):
        iter+=1
        if verbose:
            pbar.describe("Dynamic Programming")
        levels, bellman_costs, _, _, future_costs_approx_l = hlpr(get_bellman_values_from_costs,\
                   ("levels", "bellman_costs", "bellman_duals", "bellman_controls", "future_costs_approx_l"))
        # levels, bellman_costs, _, _, future_costs_approx_l = get_bellman_values_from_costs(
        #     param=param,
        #     multi_stock_management=multi_stock_management,
        #     costs_approx=costs_approx,
        #     future_costs_approx=future_costs_approx,
        #     nSteps_bellman=nSteps_bellman,
        #     name_solver=name_solver,
        #     method=method,
        #     trajectory=trajectory,
        #     correlations=correlations,
        #     verbose=False
        # )
        hlpr.update(dict(future_costs_approx = future_costs_approx_l[0]))
        # future_costs_approx.count_redundant(tolerance=0, remove=True)
        
        #Evaluate optimal 
        trajectory, pseudo_opt_controls, _ = hlpr(solve_for_optimal_trajectory,\
                                            ("trajectory", "pseudo_opt_controls", "pseudo_opt_costs"))
        # trajectory, pseudo_opt_controls, _ = solve_for_optimal_trajectory(
        #     param=param,
        #     multi_stock_management=multi_stock_management,
        #     costs_approx=costs_approx,
        #     future_costs_approx_l=future_costs_approx_l,
        #     inflows=inflows,
        #     starting_pt=starting_pt,
        #     name_solver=name_solver,
        #     verbose=False
        # ) #Beware, some trajectories seem to be overstep the bounds with values such as -2e-12
        
        controls_to_check = hlpr(select_controls_to_explore, ("controls_to_check"))
        # controls_to_check = select_controls_to_explore(
        #     param=param,
        #     multi_stock_management=multi_stock_management,
        #     pseudo_opt_controls=pseudo_opt_controls,
        #     costs_approx=costs_approx
        #     )
        #Getting the costs from the costs_approx before updating it
        
        if verbose:
            pbar.describe("Simulation")

        #Evaluating this "optimal" trajectory
        costs, slopes = hlpr(get_all_costs, ("costs, slopes"))
        # costs, slopes = get_all_costs(
        #     param=param,
        #     list_models=list_models,
        #     multi_stock_management=multi_stock_management,
        #     controls_list=controls_to_check,
        #     verbose=False)
        
        opt_gap = get_opt_gap(param=param, costs=costs, costs_approx=costs_approx, controls_to_check=controls_to_check, max_gap=max_gap)
        #Update costs approx accordingly
        # Reinterpolating / cleaning approximations
        # if interp_mode:
        #     # true_controls = np.append(true_controls, controls_to_check, axis=2)
        #     # true_costs = np.append(true_costs, costs, axis=2) #I think costs has right shape
        #     # true_duals = np.append(true_duals, slopes, axis=2)
        #     costs_approx = LinearCostEstimator(
        #                         param=param,
        #                         controls=true_controls,
        #                         costs=true_costs,
        #                         duals=true_duals)
        #     hlpr.call(costs_approx.enrich_estimator)
        #     # costs_approx.enrich_estimator(param=param)
        #     hlpr.call(costs_approx.cleanup_approximations)
        #     # costs_approx.cleanup_approximations(param=param,
        #     #                                       true_controls=true_controls,
        #     #                                       true_costs=true_costs)
        # else:
        costs_approx.update(inputs=controls_to_check, costs=costs, duals=slopes, interp_mode=interp_mode)
        costs_approx.remove_redundants(tolerance=1e-3, param=param)
        
        #If we want to look at the usage values evolution
        if verbose:
            usage_value, levels_imposed = compute_usage_values_from_costs(
                param=param,
                multi_stock_management=multi_stock_management,
                name_solver=name_solver,
                costs_approx=costs_approx,
                future_costs_approx_l=future_costs_approx_l,
                optimal_trajectory=trajectory,
                discretization=10*nSteps_bellman,
                correlations=correlations,
                verbose=False,
            )
            draw_usage_values(usage_values=usage_value, levels=levels_imposed, n_weeks=param.len_week, nSteps_bellman=nSteps_bellman,
                              multi_stock_management=multi_stock_management, trajectory=trajectory)

        if verbose:
            pbar.update(precision=opt_gap)

    if verbose:    
        pbar.close()
    if interp_mode:
        costs_approx.remove_interpolations()
        # costs_approx = LinearCostEstimator(
        #                         param=param,
        #                         controls=true_controls,
        #                         costs=true_costs,
        #                         duals=true_duals)
    return bellman_costs, levels, future_costs_approx_l, costs_approx, trajectory





# def reuse_models_and_sims(reuse:str) -> LinearCostEstimator:
#     with open(f"{reuse}/costs_approx.pkl", "rb") as cost_approx_file:
#         costs_approx = pkl.load(cost_approx_file)
#     return costs_approx

# def save_models_and_sims(save:str, costs_approx:LinearCostEstimator):
#     with open(f"{save}/costs_approx.pkl", "wb") as cost_approx_file:
#         pkl.dump(costs_approx, cost_approx_file)


def iter_bell_vals(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    n_controls_init:int,
    starting_pt: np.ndarray,
    nSteps_bellman:int,
    method:str,
    name_solver: str = "CLP",
    precision: float = 1e-2,
    maxiter: int = 2,
    interp_mode: bool = False,
    verbose: bool = False,
) -> tuple[Any, Any, Any, Any, Any]:
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
        starting_pt (np.ndarray): Reservoir state before week 0
        nSteps_bellman (int): Precision of the bellman solving part
        name_solver (str, optional): name of the solver used. Defaults to "CLP".
        precision (float, optional): Gap to minimum tolerated. Defaults to 1e-2.
        maxiter (int, optional): Maximum number of iterations. Defaults to 2.
        verbose (bool, optional): Defines whether or not the function displlays informations while running. Defaults to False.

    Returns:
        tuple[np.ndarray, Estimator, list[LinearInterpolator], np.ndarray]: Bellman costs, costs approximation, future costs approximation, levels
    """
    #Initialize the caller
    hlpr = Caller(
        param=param,
        multi_stock_management=multi_stock_management,
        output_path=output_path,
        n_controls_init=n_controls_init,
        starting_pt=starting_pt,
        nSteps_bellman=nSteps_bellman,
        method=method,
        name_solver=name_solver,
        precision=precision,
        maxiter=maxiter,
        interp_mode=interp_mode,
        verbose=verbose,)
    
    #Choose first controls to test
    controls = hlpr(initialize_controls, 'controls_list')
    # controls_init = initialize_controls(param=param, 
    #                                     multi_stock_management=multi_stock_management,
    #                                     n_controls=n_controls_init)
    
    #Initialize the Antares problems
    hlpr(initialize_antares_problems, 'list_models')
    # list_models = initialize_antares_problems(param=param,
    #                                         multi_stock_management=multi_stock_management,
    #                                         output_path=output_path,
    #                                         name_solver=name_solver,
    #                                         direct_bellman_calc=False,
    #                                         verbose=verbose)
        
    #Get hyperplanes resulting from initial controls
    _, duals = hlpr(get_all_costs, ('costs', 'slopes'))
    # costs, slopes, _ = get_all_costs(
    #     param=param,
    #     list_models=list_models,
    #     multi_stock_management=multi_stock_management,
    #     controls_list=controls_list,
    #     verbose=verbose)
    
    #Use them to initiate our costs approximation
    hlpr.update(dict(duals=duals, controls=controls))
    costs_approx = hlpr(LinearCostEstimator, "costs_approx")
    # costs_approx = LinearCostEstimator(
    #     param=param,
    #     controls=controls_init,
    #     costs=costs,
    #     duals=slopes)
    hlpr.update(dict(costs_approx=costs_approx))

    # Enrichment to accelerate convergence:
    # if interp_mode:
    #     print("Beware of interpolation")
    #     print(costs_approx[0,0].costs, costs_approx[0,0].inputs, costs_approx[0,0].duals)
    #     # hlpr.call(costs_approx.enrich_estimator)
    #     # costs_approx.enrich_estimator(param=param)
    #     # hlpr.call(costs_approx.cleanup_approximations)
    #     # costs_approx.cleanup_approximations(param=param,
    #     #                                     true_controls=controls_init,
    #     #                                     true_costs=costs)
    #     print(costs_approx[0,0].costs, costs_approx[0,0].inputs, costs_approx[0,0].duals)
    
    #Initialize our approximation on future costs
    hlpr(initialize_future_costs, "future_costs_approx")
    # future_costs_approx = initialize_future_costs(
    #     param=param,
    #     multi_stock_management=multi_stock_management,
    #     lvl_init=starting_pt,
    #     costs_approx=costs_approx,
    # )

    # Correlations matrix
    hlpr(get_correlation_matrix)
    # correlations = get_correlation_matrix(param=param,
    #                                       multi_stock_management=multi_stock_management,
    #                                       method="random_corrs")
    
    # Iterative part
    hlpr(cutting_plane_method, ("bellman_costs", "levels", "future_costs_approx_l",
                                      "costs_approx", "opt_trajectory", "all_uvs"))
    # bellman_costs, levels, future_costs_approx_l, costs_approx,\
    #       opt_trajectory, all_uvs = cutting_plane_method(
    #     param=param,
    #     multi_stock_management=multi_stock_management,
    #     list_models=list_models,
    #     name_solver=name_solver,
    #     starting_pt=starting_pt,
    #     costs_approx=costs_approx,
    #     true_controls=controls_init,
    #     true_costs=costs,
    #     true_duals=slopes,
    #     future_costs_approx=future_costs_approx,
    #     nSteps_bellman=nSteps_bellman,
    #     precision=precision,
    #     method=method,
    #     correlations=correlations,
    #     maxiter=maxiter,
    #     interp_mode=interp_mode,
    #     verbose=verbose,
    # )

    #Deducing usage values
    hlpr(compute_usage_values_from_costs, ('usage_values', 'levels_imposed'))
    # usage_values, _ = compute_usage_values_from_costs(
    #     param=param,
    #     multi_stock_management=multi_stock_management,
    #     name_solver=name_solver,
    #     costs_approx=costs_approx,
    #     future_costs_approx_l=future_costs_approx_l,
    #     optimal_trajectory=opt_trajectory,
    #     discretization=nSteps_bellman*10,
    #     correlations=correlations,
    #     verbose=verbose) 
    returns = ["bellman_costs", "costs_approx", "future_costs_approx_l", "levels", "opt_trajectory",\
          "usage_values", "all_uvs"]
    bellman_costs, costs_approx, future_costs_approx_l, levels, opt_trajectory,\
          usage_values, all_uvs = tuple([hlpr.args[ret] for ret in returns])
    return bellman_costs, costs_approx, future_costs_approx_l, levels, opt_trajectory, usage_values, all_uvs