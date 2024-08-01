import julia
from julia import Main
from estimation import LinearCostEstimator
from multi_stock_bellman_value_calculation import MultiStockManagement

def python_to_julia_data(n_weeks, n_scenarios, multi_stock_management:MultiStockManagement, costs_approx):
    # Convert Python objects to Julia-compatible structures
    julia_reservoirs = []
    for area, mng in multi_stock_management.dict_reservoirs.item():
        julia_res = Main.eval(f"""
        (capacity = {mng.res.capacity},
         efficiency = {mng.res.efficiency},
         max_pumping = {mng.res.max_pumping},
         max_generating = {mng.res.max_generating},
         upper_level = {mng.res.upper_rule_curve},
         lower_level = {mng.res.bottom_rule_curve},
         upper_curve_penalty = {mng.upper_curve_penalty},
         lower_curve_penalty = {mng.bottom_curve_penalty},
         spillage_penalty = {2*mng.upper_curve_penaly},
         inflows = {mng.res.inflows})
        """)
        julia_reservoirs.append(julia_res)

    julia_costs_approx = Main.eval(costs_approx.to_julia_compatible_structure())

    return n_weeks, n_scenarios, julia_reservoirs, julia_costs_approx

# Your Python function to call the Julia script
def manage_reservoirs_py(n_weeks, n_scenarios, reservoirs, costs_approx):
    julia_data = python_to_julia_data(n_weeks, n_scenarios, reservoirs, costs_approx)
    results = Main.manage_reservoirs(*julia_data)
    return results
