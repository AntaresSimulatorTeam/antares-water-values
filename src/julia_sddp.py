import julia
from julia import Main
from estimation import LinearCostEstimator
from multi_stock_bellman_value_calculation import MultiStockManagement
import numpy as np

#Test
def python_to_julia_data(n_weeks, n_scenarios, multi_stock_management:MultiStockManagement, costs_approx:LinearCostEstimator, level_init:np.ndarray):
    # Convert Python objects to Julia-compatible structures
    julia_reservoirs = [dict(
        capacity = mng.res.capacity,
        efficiency = mng.res.efficiency,
        max_pumping = mng.res.max_pumping,
        max_generating = mng.res.max_generating,
        upper_level = mng.res.upper_rule_curve,
        lower_level = mng.res.bottom_rule_curve,
        upper_curve_penalty = mng.upper_curve_penalty,
        lower_curve_penalty = mng.bottom_curve_penalty,
        spillage_penalty = 2*mng.upper_curve_penaly,
        level_init = level_init[i],
        inflows = mng.res.inflows,)
        for i, mng in enumerate(multi_stock_management.dict_reservoirs.values())]

    julia_costs_approx = Main.eval(costs_approx.to_julia_compatible_structure())
    return n_weeks, n_scenarios, julia_reservoirs, julia_costs_approx

# Your Python function to call the Julia script
def manage_reservoirs_py(n_weeks, n_scenarios, reservoirs, costs_approx, level_init, julia_path="sddp.jl"):
    n_weeks, n_scenarios, julia_reservoirs, julia_costs_approx = python_to_julia_data(n_weeks, 
                                                                                      n_scenarios, 
                                                                                      reservoirs, 
                                                                                      costs_approx,
                                                                                      level_init)
    Main.include(julia_path)
    Main.reservoirs = julia_reservoirs
    Main.costs_approx = julia_costs_approx
    Main.n_weeks = n_weeks
    Main.n_scenarios = n_scenarios
    results = Main.manage_reservoirs(n_weeks=n_weeks,
                                     n_scenarios=n_scenarios,
                                     reservoirs=julia_reservoirs,
                                     costs_approx=julia_costs_approx)
    return results
