from itertools import product
from calculate_reward_and_bellman_values import (
    RewardApproximation,
    ReservoirManagement,
    BellmanValueCalculation,
    MultiStockManagement,
    MultiStockBellmanValueCalculation,
)
from estimation import Estimator, LinearInterpolator
from read_antares_data import TimeScenarioParameter, TimeScenarioIndex
from optimization import AntaresProblem, WeeklyBellmanProblem
from typing import Annotated, Literal, Dict
from optimization import Basis, solve_problem_with_multivariate_bellman_values
import numpy.typing as npt
import numpy as np
from scipy.interpolate import interp1d
from functions_iterative import compute_upper_bound
import tqdm

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
    ) -> Dict[TimeScenarioIndex, AntaresProblem]:
    """
        Creates Instances of the Antares problem for every week / scenario
        
        Parameters
        ----------
        param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
        multi_stock_management:MultiStockManagement: Management information for every stock ,
        output_path:str: Folder containing the mps files generated for our study,
        name_solver:str: Name of the solver used,
        
        Returns
        -------
        list_models: Dict[TimeScenarioIndex, AntaresProblem]: Dictionnary with all problems
    """
    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {}
    print(f"=======================[Initialization of Antares Problems]=======================")
    for week in tqdm(range(param.len_week), desc='Week', colour='Yellow'):
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
    )->tuple[np.ndarray, np.ndarray, np.ndarray, LinearInterpolator]:
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
    
    # Parameters
    n_reservoirs = len(multi_stock_management.dict_reservoirs)
    n_weeks = param.len_week
    
    # Listing all levels
    levels_discretization = product(*[np.linspace(0, manag.reservoir.capacity, xNsteps) for manag in multi_stock_management.dict_reservoirs.values()])
    levels = np.array([level for level in levels_discretization])
    
    # Initializing controls, costs and duals
    controls = np.zeros([n_weeks] + [xNsteps**n_reservoirs] + [n_reservoirs]) #Not used
    costs = np.zeros([n_weeks] + [xNsteps**n_reservoirs])
    duals = np.zeros([n_weeks] + [xNsteps**n_reservoirs] +[n_reservoirs])
    
    # Starting from last week dynamically solving the optimal control problem (from every starting level)
    for week in tqdm(range(param.len_week-1, -1, -1), colour="Green", desc="Week"):
        for lvl_id, lvl_init in enumerate(levels):
            
            #Remove previous constraints / vars
            problem.reset_solver()
            
            #Rewrite problem
            problem.write_problem(
                week=week,
                level_init=lvl_init,
                future_costs_estimation=future_estimator,
            )
            
            #Solve, should we make use of previous Bases ?
            controls_wl, cost_wl, duals_wl = problem.solve(verbose=False)
            
            #Wriing down results
            controls[week, lvl_id] = controls_wl
            costs[week, lvl_id] = cost_wl
            duals[week, lvl_id] = duals_wl

        # Updating the future estimator
        future_estimator = LinearInterpolator(inputs=levels, costs=costs[week], duals=np.moveaxis(duals[week],-1,0))
    return levels, costs, duals, future_estimator