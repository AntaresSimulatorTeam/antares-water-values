from itertools import product
from calculate_reward_and_bellman_values import (
    RewardApproximation,
    ReservoirManagement,
    BellmanValueCalculation,
    MultiStockManagement,
    MultiStockBellmanValueCalculation,
)
from estimation import Estimator, LinearCostEstimator, LinearInterpolator
from read_antares_data import TimeScenarioParameter, TimeScenarioIndex
from optimization import AntaresProblem, WeeklyBellmanProblem
from typing import Annotated, List, Literal, Dict, Optional
from optimization import Basis, solve_problem_with_multivariate_bellman_values
import numpy.typing as npt
import numpy as np
from scipy.interpolate import interp1d
from functions_iterative import compute_upper_bound
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
    cost_estimation:Estimator,
    future_estimator:LinearInterpolator,
    xNsteps:int,
    name_solver:str,
    verbose:bool=False,
    )->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[LinearInterpolator]]:
    """
    Dynamically solves the problem of minimizing the optimal control problem for every discretized level, 
    balancing between saving costs this week or saving stock to avoid future costs
    
    Parameters
    ----------
        param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
        multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
        cost_estimation:LinearInterpolator: All precalculated lower bounding hyperplains of weekly cost estimation,
        future_estimator:LinearInterpolator: All previously obtained lower bounding hyperplains of future cost estimation,
        xNsteps:int: Discretization level used
        name_solver:str: Solver chosen for the optimization problem, default -> CLP
        verbose:bool: Controls the displays at running time
        
    Returns
    -------
        levels:np.ndarray: Level discretization used,
        costs:np.ndarray: Objective value at every week (and for every level combination), 
        duals:np.ndarray: Dual values of the initial level constraint at every week (and for every level combination), 
        future_estimator:LinearInterpolator: Final estimation of total system prices
    """
    #Initializing the Weekly Bellman Problem instance
    problem = WeeklyBellmanProblem(
        param=param,
        multi_stock_management=multi_stock_management,
        week_costs_estimation=cost_estimation,
        name_solver=name_solver
    )
    
    #Keeping in memory all future costs approximations
    future_estimators_l = [future_estimator]
    
    # Parameters
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    n_weeks = param.len_week
    
    # Listing all levels
    levels_discretization = product(*[np.linspace(0, manag.reservoir.capacity, xNsteps) for manag in multi_stock_management.dict_reservoirs.values()])
    levels = np.array([level for level in levels_discretization])
    
    # Initializing controls, costs and duals
    controls = np.zeros([n_weeks] + [xNsteps**n_reservoirs] + [n_reservoirs])
    costs = np.zeros([n_weeks] + [xNsteps**n_reservoirs])
    duals = np.zeros([n_weeks] + [xNsteps**n_reservoirs] + [n_reservoirs])
    
    # Starting from last week dynamically solving the optimal control problem (from every starting level)
    week_range = range(param.len_week-1, -1, -1)
    if verbose:
        week_range = tqdm(week_range, colour="Green", desc="Dynamic Solving")
    for week in week_range:
        for lvl_id, lvl_init in enumerate(levels):
            
            #Remove previous constraints / vars
            problem.reset_solver()
            
            #Rewrite problem
            problem.write_problem(
                week=week,
                level_init=lvl_init,
                future_costs_estimation=future_estimator,
            )
            
            #Solve, should we make use of previous Bases ? Not yet, computations still tractable for n<=2
            controls_wl, cost_wl, duals_wl = problem.solve(remove_future_costs=False, verbose=False)
            
            #Writing down results
            controls[week, lvl_id] = controls_wl
            costs[week, lvl_id] = cost_wl
            duals[week, lvl_id] = duals_wl

        # Updating the future estimator
        future_estimator = LinearInterpolator(inputs=levels, costs=costs[week], duals=duals[week])
        future_estimators_l.insert(0,future_estimator)
        # n_rddts, rddants = future_estimator.count_redundant(tolerance=1e7)
        # future_estimator.remove(1-np.array(rddants))
    return levels, costs, duals, controls, future_estimators_l

def initialize_future_costs(
    param:TimeScenarioParameter,
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
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    return LinearInterpolator(
        inputs=np.zeros((1, n_reservoirs)),
        costs=np.zeros(1),
        duals=np.zeros((1, n_reservoirs)),
        )

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
                print(f"Failed at week {week}, the conditions on control were:")
                # raise ValueError
            tot_iter += iters
            times.append(times_ws)
            costs[week, scenario] = costs_ws
            slopes[week, scenario] = slopes_ws
    # print(f"Number of simplex pivot {tot_iter}")
    return costs, slopes, np.array(times)

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
        costs_approx=costs_approx
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
    