from proxy_stage_cost_function import ProxyStageCostFunction
import numpy as np
import os
from type_definition import Callable
from scipy.interpolate import interp1d
from tqdm import tqdm

STOCK_DISCR=2

class BellmanValuesProxy:
    def __init__(self, proxy: ProxyStageCostFunction, export_dir: str, pbar : tqdm,TS_selection:list[int]|None=None)->None:
        """
        Initialize BellmanValuesProxy with given Proxy and export directory.
        Sets up cost functions, storage arrays, then computes Bellman and usage values.
        """
        self.proxy = proxy
        self.nb_weeks = proxy.nb_weeks
        self.scenarios = proxy.scenarios
        self.TS_selection=TS_selection if TS_selection is not None else self.scenarios
        self.export_dir = export_dir
        self.pbar = pbar

        self.stage_cost_functions = self.proxy.stage_cost_functions

        self.cost_functions = self.stage_cost_functions[:, :, 0]
        self.turb_functions = self.stage_cost_functions[:, :, 1]
        self.pump_functions = self.stage_cost_functions[:, :, 2]

        self.mean_bv = np.zeros((self.nb_weeks, 100//STOCK_DISCR+1))

        self.compute_bellman_values()
    
    def penalty_final_stock(self) -> Callable:
        """
        Returns a penalty function based on deviation from initial reservoir level at final week.
        The penalty scales with the upper bound cost and relative negative deviation percentage (1%).
        """
        penalty = lambda x:abs(x-self.proxy.reservoir.initial_level)/self.proxy.reservoir.initial_level * 100 * self.proxy.upper_bound_cost(self.nb_weeks-1)

        return penalty
    
    def penalty_rule_curves(self, week_idx: int) -> Callable:
        """
        Returns a piecewise penalty function penalizing deviations outside the weekly lower and upper rule curves.
        Penalties grow linearly beyond ±1% of reservoir capacity from the rule curves.
        """
        ub_cost = self.proxy.upper_bound_cost(week_idx)
        penalty = interp1d(
            [
                self.proxy.reservoir.weekly_lower_rule_curve[week_idx] - 0.01 * self.proxy.reservoir.capacity,
                self.proxy.reservoir.weekly_lower_rule_curve[week_idx],
                self.proxy.reservoir.weekly_upper_rule_curve[week_idx],
                self.proxy.reservoir.weekly_upper_rule_curve[week_idx] + 0.01 * self.proxy.reservoir.capacity,
            ],
            [
                ub_cost,
                0,
                0,
                ub_cost,
            ],
            fill_value='extrapolate',
        )
        return penalty

    def bellman_function(self, week: int) -> interp1d:
        """
        Returns an interpolated Bellman value function for given week over reservoir stock levels.
        """
        return interp1d(
            np.linspace(0, self.proxy.reservoir.capacity, 100//STOCK_DISCR+1),
            self.mean_bv[week],
            kind="linear",
            fill_value="extrapolate",
        )

    def iterate_over_controls_vec(self,controls:np.ndarray, current_stock:float, weekly_inflow:float,
                              stage_cost_function:interp1d, future_bellman_function:interp1d, 
                              penalty_function:interp1d)-> tuple:
        next_stock = current_stock - controls + weekly_inflow
        total_value = (stage_cost_function(controls)
                    + future_bellman_function(next_stock)
                    + penalty_function(next_stock))
        j = int(np.argmin(total_value))
        return float(total_value[j]), float(next_stock[j]), float(controls[j])

    def iterate_over_stock_levels_vec(
        self,
        best_value: float,
        best_stock: float | None,
        best_control: float | None,
        current_stock: float,
        weekly_inflow: float,
        max_week_pump: float,
        max_week_turb: float,
        stage_cost_function,
        future_bellman_function,
        penalty_function,
        max_control: float,
    ) -> tuple[float, float | None, float | None]:
        """
        Enumerates next stock levels on the 0..100% grid,
        computes implied control, filters infeasible controls, evaluates total value,
        and keeps the best.
        """
        capacity = float(self.proxy.reservoir.capacity)

        next_stock_grid = (np.arange(0, 101, STOCK_DISCR, dtype=float) / 100.0) * capacity

        controls = current_stock - next_stock_grid + weekly_inflow
        feasible = (
            (controls >= -max_week_pump * self.proxy.reservoir.efficiency) &
            (controls <=  max_week_turb * self.proxy.turb_efficiency) &
            (controls <=  max_control)
        )

        if not np.any(feasible):
            return best_value, best_stock, best_control

        ns = next_stock_grid[feasible]
        c = controls[feasible]

        total_value = (
            stage_cost_function(c)
            + future_bellman_function(ns)
            + penalty_function(ns)
        )

        j = int(np.argmin(total_value))
        cand_value = float(total_value[j])

        if cand_value < best_value:
            best_value = cand_value
            best_stock = float(ns[j])
            best_control = float(c[j])

        return best_value, best_stock, best_control

    def compute_bellman_values(self) -> None:
        """
        Computes Bellman values at end of each week by backward induction over weeks and scenarios.
        Applies penalties and selects optimal controls to minimize cost-to-go.
        """

        penalty_final_stock = self.penalty_final_stock()
        self.mean_bv[self.nb_weeks - 1] = np.array([
            penalty_final_stock((c / 100) * self.proxy.reservoir.capacity) for c in range(0, 101, STOCK_DISCR)
        ])

        self.pbar.set_postfix_str("Bellman values computing") 
        for w in reversed(range(self.nb_weeks - 1)):

            penalty_function = self.penalty_rule_curves(w + 1)
            future_bellman_function = self.bellman_function(w + 1)

            max_week_turb = self.proxy.reservoir.max_weekly_turb[w+1]
            max_week_pump = self.proxy.reservoir.max_weekly_pump[w+1]

            for c in range(0, 101, STOCK_DISCR):
                current_stock = (c / 100) * self.proxy.reservoir.capacity
                bv=np.zeros((len(self.TS_selection)))
                for i,s in enumerate(self.TS_selection):

                    self.pbar.update(1)
                    weekly_inflow = self.proxy.reservoir.weekly_inflow[w + 1, s]
                    cost_function = self.cost_functions[w + 1, s]
                    controls = cost_function.x

                    best_value, best_stock, best_control = self.iterate_over_controls_vec(
                        controls=controls,
                        current_stock=current_stock,
                        weekly_inflow=weekly_inflow,
                        stage_cost_function=cost_function,
                        future_bellman_function=future_bellman_function,
                        penalty_function=penalty_function
                    )


                    final_best_value, _, _ = self.iterate_over_stock_levels_vec(
                        best_value=best_value,
                        best_stock=best_stock,
                        best_control=best_control,
                        current_stock=current_stock,
                        weekly_inflow=weekly_inflow,
                        max_week_pump=max_week_pump,
                        max_week_turb=max_week_turb,
                        stage_cost_function=cost_function,
                        future_bellman_function=future_bellman_function,
                        penalty_function=penalty_function,
                        max_control=controls[-1])

                    bv[i] = final_best_value

                self.mean_bv[w, c // STOCK_DISCR] = np.mean(bv)

class OptimalTrajectories:
    def __init__(self,
                 bellman_values : BellmanValuesProxy,
                 pbar : tqdm):
        """
        Initialize OptimalTrajectories with a BellmanValuesProxy instance.
        Prepares data and computes optimal trajectories.
        """
        self.bellman_values=bellman_values
        self.nb_weeks = bellman_values.nb_weeks
        self.scenarios = bellman_values.scenarios
        self.export_dir = bellman_values.export_dir
        self.pbar = pbar
        
        self.mean_bv=bellman_values.mean_bv
        self.compute_trajectories()

    def compute_trajectories(self) -> None:
        """
        Compute optimal reservoir trajectories, controls, turbining and pumping schedules
        for all scenarios and weeks using Bellman values, penalties and inflows.
        Adjusts hourly inflows to avoid overflow or negative stock with external method.
        """
        self.pbar.set_postfix_str("Optimal trajectories computing") 
        self.trajectories = np.zeros((len(self.scenarios), self.nb_weeks))
        self.optimal_controls = np.zeros_like(self.trajectories)
        self.optimal_turb = np.zeros_like(self.trajectories)
        self.optimal_pump = np.zeros_like(self.trajectories)
        self.inflow_adjust_overflow = np.zeros((self.nb_weeks, len(self.scenarios), 168))

        penalty_by_week = [self.bellman_values.penalty_rule_curves(w) for w in range(self.nb_weeks)]
        bellman_by_week = [self.bellman_values.bellman_function(w) for w in range(self.nb_weeks)]   
        
        for s in self.scenarios:
            current_stock = self.bellman_values.proxy.reservoir.initial_level
            
            for w in range(self.nb_weeks):
                self.pbar.update(1)

                weekly_inflow = self.bellman_values.proxy.reservoir.weekly_inflow[w, s]
                hourly_inflow = self.bellman_values.proxy.reservoir.hourly_inflow[w * 168:(w + 1) * 168, s]


                cost_function = self.bellman_values.cost_functions[w, s]
                penalty_function = penalty_by_week[w]

                controls = cost_function.x

                future_bellman_function = bellman_by_week[w]

                max_week_turb = self.bellman_values.proxy.reservoir.max_weekly_turb[w]
                max_week_pump = self.bellman_values.proxy.reservoir.max_weekly_pump[w]

                max_control = self.adjust_hourly_inflow_overflow(scenario=s, 
                                                                 week=w, 
                                                                 stock_init=current_stock, 
                                                                 inflow=hourly_inflow)                
                if max_control < controls[-1]  :
                    controls = controls[controls < max_control]
                    controls = np.concatenate([controls, [max_control]])

                weekly_inflow -= np.sum(self.inflow_adjust_overflow[w, s])

                best_value, best_stock, best_control = self.bellman_values.iterate_over_controls_vec(
                        controls=controls,
                        current_stock=current_stock,
                        weekly_inflow=weekly_inflow,
                        stage_cost_function=cost_function,
                        future_bellman_function=future_bellman_function,
                        penalty_function=penalty_function
                    )

                __, final_best_stock, final_best_control = self.bellman_values.iterate_over_stock_levels_vec(
                    best_value=best_value,
                    best_stock=best_stock,
                    best_control=best_control,
                    current_stock=current_stock,
                    weekly_inflow=weekly_inflow,
                    max_week_pump=max_week_pump,
                    max_week_turb=max_week_turb,
                    stage_cost_function=cost_function,
                    future_bellman_function=future_bellman_function,
                    penalty_function=penalty_function,
                    max_control=controls[-1])

                self.trajectories[s, w] = final_best_stock
                self.optimal_controls[s, w] = final_best_control
                self.optimal_turb[s, w] = self.bellman_values.turb_functions[w, s](final_best_control)
                self.optimal_pump[s, w] = self.bellman_values.pump_functions[w, s](final_best_control)
                current_stock = final_best_stock

    def adjust_hourly_inflow_overflow(
        self,
        scenario: int,
        week: int,
        stock_init: float,
        inflow: np.ndarray
    ) -> float:
        """
        Adjust hourly inflow to mitigate overflow or negative stock by detecting violations and recording adjustments.
        Returns the maximum feasible control (turbining) for the week after adjustments and taking in account potential overflow or negative stock during the week.
        """
        cap = float(self.bellman_values.proxy.reservoir.capacity)

        turb = self.bellman_values.proxy.reservoir.max_hourly_turb[week * 168:(week + 1) * 168]
        max_control = turb * self.bellman_values.proxy.turb_efficiency

        net_hourly_turb = inflow - max_control

        stock = float(stock_init)

        for h in range(168):
            stock += float(net_hourly_turb[h])

            if stock > cap:
                overflow = stock - cap

                self.inflow_adjust_overflow[week, scenario, h] = overflow

                net_hourly_turb[h] -= overflow
                stock = cap


            if stock < 0.0:
                neg = stock  

                max_control[h] += neg  


                net_hourly_turb[h] -= neg 
                stock = 0.0

        return float(np.sum(max_control))
