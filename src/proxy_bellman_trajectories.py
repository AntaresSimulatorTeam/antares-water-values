from proxy_stage_cost_function import Proxy
import numpy as np
from proxy_logger import LoggerSetup
import os
from type_definition import Callable
from scipy.interpolate import interp1d
from tqdm import tqdm


class BellmanValuesProxy:
    def __init__(self, proxy: Proxy, enable_logging: bool, export_dir: str, pbar : tqdm, h:int):
        """
        Initialize BellmanValuesProxy with given Proxy, logging flag, export directory and margin parameter h.
        Sets up cost functions, storage arrays, and logger, then computes Bellman and usage values.
        """
        self.proxy = proxy
        self.nb_weeks = proxy.nb_weeks
        self.scenarios = proxy.scenarios
        self.export_dir = export_dir
        self.pbar = pbar
        self.h = h

        self.stage_cost_functions = self.proxy.stage_cost_functions

        self.cost_functions = self.stage_cost_functions[:, :, 0]
        self.turb_functions = self.stage_cost_functions[:, :, 1]
        self.pump_functions = self.stage_cost_functions[:, :, 2]

        self.bv = np.zeros((self.nb_weeks, 51, len(self.scenarios)))
        self.mean_bv = np.zeros((self.nb_weeks, 51))

        if not isinstance(export_dir, str) or not export_dir:
            raise ValueError("export_dir must be provided as a non-empty string to BellmanValuesProxy.")
        self.export_dir = export_dir
        logger_setup = LoggerSetup(self.export_dir)
        self.logger = logger_setup.get_logger() if enable_logging else logger_setup.get_null_logger()

        self.compute_rule_curve_margins()
        self.compute_bellman_values()
        self.compute_usage_values()
    
    def penalty_final_stock(self) -> Callable:
        """
        Returns a penalty function based on deviation from initial reservoir level at final week.
        The penalty scales with the upper bound cost and relative deviation percentage (1%).
        """
        penalty = lambda x: abs(x - self.proxy.reservoir.initial_level) / self.proxy.reservoir.initial_level * 100 * self.proxy.upper_bound_cost(self.nb_weeks - 1)
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

    def init_log_bellman(self) -> None:
        """
        Logs the start of Bellman values computation.
        """
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{'COMPUTING BELLMAN VALUES'.center(70)}")
        self.logger.debug("=" * 70 + "\n")
        self.logger.debug(">>> Initializing final Bellman values")

    def init_log_bellman_week(self, week: int) -> None:
        """
        Logs the processing of a specific week during Bellman value computation.
        """
        self.logger.debug("\n" + "-" * 60)
        self.logger.debug(f"---- Processing week {week + 1} ----")
        self.logger.debug("-" * 60 + "\n")

    def bellman_function(self, week: int) -> interp1d:
        """
        Returns an interpolated Bellman value function for given week over reservoir stock levels.
        """
        return interp1d(
            np.linspace(0, self.proxy.reservoir.capacity, 51),
            self.mean_bv[week],
            kind="linear",
            fill_value="extrapolate",
        )

    def iterate_over_controls(self,
                              best_value: float | None,
                              best_stock: float | None,
                              best_control: float | None,
                              controls: np.ndarray,
                              current_stock: float,
                              weekly_inflow: float,
                              stage_cost_function: interp1d,
                              future_bellman_function: interp1d,
                              penalty_function: interp1d) -> tuple:
        """
        Iterates through all candidate controls (decisions), calculates total cost + future value + penalties,
        and updates best control if a lower total value is found.
        """
        for control in controls:
            if control > current_stock + weekly_inflow:
                continue
            next_stock = current_stock - control + weekly_inflow
            
            cost = stage_cost_function(control)
            future_value = future_bellman_function(next_stock)
            penalty = penalty_function(next_stock)
            total_value = cost + future_value + penalty

            self.logger.debug(
                f"Test control (free): {control:.2f}, next stock: {next_stock:.2f}, "
                f"Cost: {cost:.2f}, future BV: {future_value:.2f}, penalty: {penalty:.2f}, total: {total_value:.2f}"
            )

            if total_value < best_value:
                best_value = total_value
                best_stock = next_stock
                best_control = control
                self.logger.debug(
                    f"→ New best control selected (free): {control:.2f}, total: {total_value:.2f}"
                )
        
        return best_value, best_stock, best_control

    def iterate_over_stock_levels(self,
                                  best_value: float | None,
                                  best_stock: float | None,
                                  best_control: float | None,
                                  current_stock: float,
                                  weekly_inflow: float,
                                  max_week_pump: float,
                                  max_week_turb: float,
                                  stage_cost_function: interp1d,
                                  future_bellman_function: interp1d,
                                  penalty_function: interp1d,
                                  max_control: float) -> tuple:
        """
        Iterates over discretized stock levels, filtering infeasible controls that violate max pump/turb constraints and hourly reservoir bounds,
        computes total value for feasible controls and updates best control if improvement is found.
        """
        for c_new in range(0, 101, 2):
            new_level = (c_new / 100) * self.proxy.reservoir.capacity
            control = current_stock - new_level + weekly_inflow

            if control < -max_week_pump * self.proxy.reservoir.efficiency or \
               control > max_week_turb * self.proxy.turb_efficiency or control > max_control:
                continue
            next_stock = current_stock - control + weekly_inflow
            cost = stage_cost_function(control)
            future_value = future_bellman_function(next_stock)
            penalty = penalty_function(next_stock)
            total_value = cost + future_value + penalty

            self.logger.debug(
                f"Test control (forced): {control:.2f}, next stock: {next_stock:.2f}, "
                f"Cost: {cost:.2f}, future BV: {future_value:.2f}, penalty: {penalty:.2f}, total: {total_value:.2f}"
            )

            if total_value < best_value:
                best_value = total_value
                best_stock = next_stock
                best_control = control
                self.logger.debug(
                    f"→ New best control selected (forced): {control:.2f}, total: {total_value:.2f}"
                )
        return best_value, best_stock, best_control

    def compute_bellman_values(self) -> None:
        """
        Computes Bellman values by backward induction over weeks and scenarios.
        Applies penalties and selects optimal controls to minimize cost-to-go.
        """
        self.init_log_bellman()

        penalty_final_stock = self.penalty_final_stock()
        self.mean_bv[self.nb_weeks - 1] = np.array([
            penalty_final_stock((c / 100) * self.proxy.reservoir.capacity) for c in range(0, 101, 2)
        ])

        self.logger.debug(f"Final penalty values (week {self.nb_weeks}): {self.mean_bv[self.nb_weeks - 1]}")
        self.pbar.set_postfix_str("Bellman values computing") 
        for w in reversed(range(self.nb_weeks - 1)):
            self.init_log_bellman_week(w)

            penalty_function = self.penalty_rule_curves(w + 1)
            future_bellman_function = self.bellman_function(w + 1)

            max_week_turb = self.proxy.reservoir.max_weekly_turb[w]
            max_week_pump = self.proxy.reservoir.max_weekly_pump[w]

            for c in range(0, 101, 2):
                current_stock = (c / 100) * self.proxy.reservoir.capacity

                for s in self.scenarios:
                    self.pbar.update(1)
                    weekly_inflow = self.proxy.reservoir.weekly_inflow[w + 1, s]
                    cost_function = self.cost_functions[w + 1, s]
                    controls = cost_function.x

                    best_value_init = np.inf
                    best_stock = None
                    best_control = None
                    self.logger.debug(f"\n[Week {w + 1} | Scenario {s + 1} | Stock {current_stock:.2f} MWh]")

                    best_value, best_stock, best_control = self.iterate_over_controls(
                        best_value=best_value_init,
                        best_stock=best_stock,
                        best_control=best_control,
                        controls=controls,
                        current_stock=current_stock,
                        weekly_inflow=weekly_inflow,
                        stage_cost_function=cost_function,
                        future_bellman_function=future_bellman_function,
                        penalty_function=penalty_function)

                    final_best_value, final_best_stock, final_best_control = self.iterate_over_stock_levels(
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

                    self.bv[w, c // 2, s] = final_best_value
                    self.logger.debug(f"Bellman value stored for stock {current_stock:.2f} MWh : {final_best_value:.2f}")

                self.mean_bv[w, c // 2] = np.mean(self.bv[w, c // 2, self.scenarios])
            self.logger.debug(f"Average Bellman values for week {w + 1} : {self.mean_bv[w]}")

    def compute_usage_values(self) -> None:
        """
        Computes usage values as discrete derivatives of mean Bellman values across stock levels for each week.
        """
        self.usage_values = np.zeros((self.nb_weeks, 50))
        for w in range(self.nb_weeks):
            for c in range(2, 102, 2):
                self.usage_values[w, (c // 2) - 1] = self.mean_bv[w, c // 2] - self.mean_bv[w, (c // 2) - 1]

    def compute_rule_curve_margins(self) -> None:
        """
        Modifies daily and weekly rule curves by applying margins based on max hourly turbining and margin parameter h.
        """
        hourly_turb_day = self.proxy.reservoir.max_hourly_turb.reshape(-1, 24)[:, 0]
        hourly_turb_day = np.concatenate([hourly_turb_day, [hourly_turb_day[-1]]])
        self.proxy.reservoir.daily_upper_rule_curve=np.minimum(
            self.proxy.reservoir.daily_upper_rule_curve,
            self.proxy.reservoir.capacity-self.h*hourly_turb_day
        )
        self.proxy.reservoir.weekly_upper_rule_curve=self.proxy.reservoir.daily_upper_rule_curve[::7]
        self.proxy.reservoir.daily_lower_rule_curve=np.maximum(
            self.proxy.reservoir.daily_lower_rule_curve,
            self.h*hourly_turb_day
        )
        self.proxy.reservoir.weekly_lower_rule_curve=self.proxy.reservoir.daily_lower_rule_curve[::7]

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
        self.logger = bellman_values.logger
        self.export_dir = bellman_values.export_dir
        self.pbar = pbar
        
        self.mean_bv=bellman_values.mean_bv
        self.compute_trajectories()
        
        # self.new_lower_rule_curve()
        # self.new_upper_rule_curve()

    def init_log_trajectories(self) -> None:
        """
        Log the start of the optimal trajectories calculation.
        """
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{'COMPUTING OPTIMAL TRAJECTORIES'.center(70)}")
        self.logger.debug("=" * 70 + "\n")

    def init_log_trajectories_week(self, week:int, scenario:int, previous_stock :float) -> None:
        """
        Log detailed info for the start of a specific week and scenario during trajectories computation.
        """
        self.logger.debug("\n" + "-" * 60)
        self.logger.debug(f"---- Week {week+1}, Scenario {scenario+1} ----")
        self.logger.debug("-" * 60)
        self.logger.debug(f"Previous stock: {previous_stock:.2f} MWh")

    def init_log_lower_rule_curves(self) -> None:
        """
        Log the start of the adjusted lower rule curve calculation.
        """
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{'COMPUTING ADJUSTED HOURLY LOWER RULE CURVE'.center(70)}")
        self.logger.debug("=" * 70 + "\n")

    def init_log_upper_rule_curves(self) -> None:
        """
        Log the start of the adjusted upper rule curve calculation.
        """
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{'COMPUTING ADJUSTED HOURLY UPPER RULE CURVE'.center(70)}")
        self.logger.debug("=" * 70 + "\n")

    def write_warnings(self) -> None:
        """
        Write accumulated warning messages to a 'warnings.txt' file if export directory and warnings exist.
        """
        if hasattr(self, 'export_dir') and self.warning_lines:
            warning_path = os.path.join(self.export_dir, "warnings.txt")
            with open(warning_path, "w", encoding="utf-8") as f:
                for line in self.warning_lines:
                    f.write(line + "\n")

    def compute_trajectories(self) -> None:
        """
        Compute optimal reservoir trajectories, controls, turbining and pumping schedules
        for all scenarios and weeks using Bellman values and penalties.
        Adjusts hourly inflows to avoid overflow or negative stock.
        """
        self.init_log_trajectories()
        self.pbar.set_postfix_str("Optimal trajectories computing") 
        self.trajectories = np.zeros((len(self.scenarios), self.nb_weeks))
        self.optimal_controls = np.zeros_like(self.trajectories)
        self.optimal_turb = np.zeros_like(self.trajectories)
        self.optimal_pump = np.zeros_like(self.trajectories)
        self.inflow_adjust_overflow = np.zeros((self.nb_weeks, len(self.scenarios), 168))
        self.warning_lines: list = []

        for s in self.scenarios:
            current_stock = self.bellman_values.proxy.reservoir.initial_level
            
            for w in range(self.nb_weeks):
                self.pbar.update(1)
                self.logger.debug(f"\n[Week {w+1} | Scenario {s+1} | Previous stock: {current_stock:.2f} MWh]")
                weekly_inflow = self.bellman_values.proxy.reservoir.weekly_inflow[w, s]
                hourly_inflow = self.bellman_values.proxy.reservoir.hourly_inflow[w * 168:(w + 1) * 168, s]
                self.logger.debug(f"Inflow: {weekly_inflow:.2f} MWh")
                cost_function = self.bellman_values.cost_functions[w, s]
                penalty_function = self.bellman_values.penalty_rule_curves(w)
                controls = cost_function.x

                future_bellman_function = self.bellman_values.bellman_function(w)

                max_week_turb = self.bellman_values.proxy.reservoir.max_weekly_turb[w]
                max_week_pump = self.bellman_values.proxy.reservoir.max_weekly_pump[w]

                best_value = np.inf
                best_stock = None
                best_control = None

                max_control = self.adjust_hourly_inflow_overflow(scenario=s, week=w, stock_init=current_stock, inflow=hourly_inflow)                
                if max_control != controls[-1]  :
                    controls = controls[controls <= max_control]
                    controls = np.concatenate([controls, [max_control]])

                weekly_inflow -= np.sum(self.inflow_adjust_overflow[w, s])


                best_value, best_stock, best_control = self.bellman_values.iterate_over_controls(
                    best_value=best_value,
                    best_stock=best_stock,
                    best_control=best_control,
                    controls=controls,
                    current_stock=current_stock,
                    weekly_inflow=weekly_inflow,
                    stage_cost_function=cost_function,
                    future_bellman_function=future_bellman_function,
                    penalty_function=penalty_function)

                final_best_value, final_best_stock, final_best_control = self.bellman_values.iterate_over_stock_levels(
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
                    max_control=max_control)


                self.logger.debug(f"=> Selected stock for week {w+1}: {final_best_stock:.2f} MWh")

                self.trajectories[s, w] = final_best_stock
                self.optimal_controls[s, w] = final_best_control
                self.optimal_turb[s, w] = self.bellman_values.turb_functions[w, s](final_best_control)
                self.optimal_pump[s, w] = self.bellman_values.pump_functions[w, s](final_best_control)
                current_stock = final_best_stock
                
        self.write_warnings()

    def adjust_hourly_inflow_overflow(self,
                                      scenario: int,
                                      week: int,
                                      stock_init: float,
                                      inflow: np.ndarray) -> float:
        """
        Adjust hourly inflow to mitigate overflow or negative stock by detecting violations and recording adjustments.
        """
        turb = self.bellman_values.proxy.reservoir.max_hourly_turb[week * 168: (week + 1) * 168]
        net_hourly_turb = inflow - turb * self.bellman_values.proxy.turb_efficiency
        max_control = turb * self.bellman_values.proxy.turb_efficiency

        for h in range(168):
            hourly_overflow = self.detect_hourly_overflow(
                scenario=scenario,
                week=week,
                hour=h,
                stock_init=stock_init,
                net_hourly=net_hourly_turb
            )
            if hourly_overflow is not None:
                self.inflow_adjust_overflow[week, scenario, h] = hourly_overflow
                net_hourly_turb[h] -= self.inflow_adjust_overflow[week, scenario, h]
            
            hourly_negative_stock = self.detect_hourly_negative_stock(
                scenario=scenario,
                week=week,
                hour=h,
                stock_init=stock_init,
                net_hourly=net_hourly_turb
            )
            if hourly_negative_stock is not None:
                max_control[h] += hourly_negative_stock
                net_hourly_turb[h] -= hourly_negative_stock
        
        return np.sum(max_control)
            
    def detect_hourly_overflow(self,
                               scenario: int,
                               week: int,
                               hour: int,
                               stock_init: float,
                               net_hourly: np.ndarray) -> float | None:
        """
        Detect if reservoir stock exceeds capacity at given hour, returning overflow amount if any.
        """
        stock = stock_init + np.cumsum(net_hourly[:hour+1])
        if stock[hour] > self.bellman_values.proxy.reservoir.capacity:
            self.logger.debug(
                f"⚠️ Overflow detected for scenario {scenario+1}, week {week+1}, hour {hour+1}: "
                f"{stock[hour]:.2f} MWh > capacity {self.bellman_values.proxy.reservoir.capacity:.2f} MWh"
            )
            return stock[hour] - self.bellman_values.proxy.reservoir.capacity
        return None
    
    def detect_hourly_negative_stock(self,
                                    scenario: int,
                                    week: int,
                                    hour: int,
                                    stock_init: float,
                                    net_hourly: np.ndarray) -> float | None:
        """
        Detect if reservoir stock falls below zero at given hour, returning negative amount if any.
        """
        stock = stock_init + np.cumsum(net_hourly[:hour+1])
        if stock[hour] < 0:
            self.logger.debug(
                f"⚠️ Negative stock detected for scenario {scenario+1}, week {week+1}, hour {hour+1}: "
                f"{stock[hour]:.2f} MWh < 0 MWh"
            )
            return stock[hour]
        return None
    
    def daily_to_hourly_curve(self, daily_curve: np.ndarray) -> np.ndarray:
        """
        Interpolate daily curve to hourly resolution and extend with linear interpolation to initial reservoir level.
        """
        n_days = len(daily_curve)
        n_hours = (n_days - 1) * 24 + 1
        hourly_curve = np.interp(
            np.arange(n_hours),
            np.arange(0, n_days) * 24,
            daily_curve
        )
        last_val = daily_curve[-1]
        final_interp = np.linspace(last_val, self.bellman_values.proxy.reservoir.initial_level, 25)[1:-1]
        hourly_curve = np.concatenate([hourly_curve, final_interp])
        return hourly_curve

    def compute_upper_enveloppe(self) -> np.ndarray:
        """
        Compute upper envelope curve as the minimum across scenarios of hourly upper feasible stock levels.
        """
        upper_curves = np.zeros((len(self.scenarios), self.nb_weeks, 168))
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                stock_init = self.trajectories[s, w-1] if w > 0 else self.bellman_values.proxy.reservoir.initial_level
                stock_final = self.trajectories[s, w]

                max_hour_pump = self.bellman_values.proxy.reservoir.max_hourly_pump[w*168:(w+1)*168]
                max_hour_turb = self.bellman_values.proxy.reservoir.max_hourly_turb[w*168:(w+1)*168]

                hourly_inflow = self.bellman_values.proxy.reservoir.hourly_inflow[w*168:(w+1)*168, s]

                cumsum_pump = np.concatenate([[0],
                                             np.cumsum(
                                                 max_hour_pump * self.bellman_values.proxy.reservoir.efficiency + hourly_inflow - self.inflow_adjust_overflow[w, s])
                                            ])[:-1] + stock_init
                cumsum_turb = stock_final - np.cumsum(
                    -self.bellman_values.proxy.turb_efficiency * max_hour_turb[::-1] + (hourly_inflow - self.inflow_adjust_overflow[w, s])[::-1])[::-1]

                hourly_curve = np.minimum(cumsum_pump, cumsum_turb)
                upper_curves[s, w] = hourly_curve

        weekly_envelope = np.min(upper_curves, axis=0)
        hourly_upper_envelope = weekly_envelope.flatten()
        hourly_upper_envelope = np.concatenate([hourly_upper_envelope, hourly_upper_envelope[-24:]])
        return hourly_upper_envelope

    def compute_lower_enveloppe(self) -> np.ndarray:
        """
        Compute lower envelope curve as the maximum across scenarios of hourly lower feasible stock levels.
        """
        lower_curves = np.zeros((len(self.scenarios), self.nb_weeks, 168))
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                stock_init = self.trajectories[s, w-1] if w > 0 else self.bellman_values.proxy.reservoir.initial_level
                stock_final = self.trajectories[s, w]

                max_hour_pump = self.bellman_values.proxy.reservoir.max_hourly_pump[w*168:(w+1)*168]
                max_hour_turb = self.bellman_values.proxy.reservoir.max_hourly_turb[w*168:(w+1)*168]

                hourly_inflow = self.bellman_values.proxy.reservoir.hourly_inflow[w*168:(w+1)*168, s]
                
                cumsum_turb = np.concatenate(
                    [[0],
                     np.cumsum(-self.bellman_values.proxy.turb_efficiency * max_hour_turb + hourly_inflow - self.inflow_adjust_overflow[w, s])])[:-1] + stock_init
                cumsum_pump = stock_final - np.cumsum(
                    self.bellman_values.proxy.reservoir.efficiency * max_hour_pump[::-1] + (hourly_inflow + self.inflow_adjust_overflow[w, s])[::-1])[::-1]

                hourly_curve = np.maximum(cumsum_pump, cumsum_turb)
                lower_curves[s, w] = hourly_curve

        weekly_envelope = np.max(lower_curves, axis=0)
        hourly_lower_envelope = weekly_envelope.flatten()
        hourly_lower_envelope = np.concatenate([hourly_lower_envelope, hourly_lower_envelope[-24:]])
        return hourly_lower_envelope

    def new_lower_rule_curve(self) -> None:
        """
        Compute adjusted lower hourly rule curve as minimum between interpolated daily lower rule curve and upper envelope.
        Log hours where adjusted curve deviates significantly from interpolated curve.
        """
        self.init_log_lower_rule_curves()
        hourly_upper_envelope = self.compute_upper_enveloppe()
        self.hourly_lower_rule_curve = self.daily_to_hourly_curve(self.bellman_values.proxy.reservoir.daily_lower_rule_curve)
        final_lower_rule_curve = np.minimum(hourly_upper_envelope, self.hourly_lower_rule_curve)
        self.final_lower_rule_curve = np.concatenate([final_lower_rule_curve[1:], [self.bellman_values.proxy.reservoir.initial_level]])

        difference = np.abs(self.final_lower_rule_curve - self.hourly_lower_rule_curve)
        threshold = 1e-3
        hours_with_diff = np.where(difference > threshold)[0]

        if hours_with_diff.size > 0:
            self.logger.debug(
                f"{len(hours_with_diff)} hour(s) with difference > {threshold} between adjusted and interpolated lower rule curve."
            )
            for hour in hours_with_diff:
                diff_value = difference[hour]
                self.logger.debug(f"Hour {hour}: difference = {diff_value:.6f}")

    def new_upper_rule_curve(self) -> None:
        """
        Compute adjusted upper hourly rule curve as maximum between interpolated daily upper rule curve and lower envelope.
        Log hours where adjusted curve deviates significantly from interpolated curve.
        """
        self.init_log_upper_rule_curves()
        hourly_lower_envelope = self.compute_lower_enveloppe()
        self.hourly_upper_rule_curve = self.daily_to_hourly_curve(self.bellman_values.proxy.reservoir.daily_upper_rule_curve)
        final_upper_rule_curve = np.maximum(hourly_lower_envelope, self.hourly_upper_rule_curve)
        self.final_upper_rule_curve = np.concatenate([final_upper_rule_curve[1:], [self.bellman_values.proxy.reservoir.initial_level]])

        difference = np.abs(self.final_upper_rule_curve - self.hourly_upper_rule_curve)
        threshold = 1e-3
        hours_with_diff = np.where(difference > threshold)[0]

        if hours_with_diff.size > 0:
            self.logger.debug(
                f"{len(hours_with_diff)} hour(s) with difference > {threshold} between adjusted and interpolated upper rule curve."
            )
            for hour in hours_with_diff:
                diff_value = difference[hour]
                self.logger.debug(f"Hour {hour}: difference = {diff_value:.6f}")
