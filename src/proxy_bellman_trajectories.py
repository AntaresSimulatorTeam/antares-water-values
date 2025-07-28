from proxy_stage_cost_function import Proxy
import numpy as np
from proxy_logger import LoggerSetup
import os
from type_definition import Callable
from scipy.interpolate import interp1d




class BellmanValuesProxy:
    def __init__(self, proxy: Proxy, enable_logging: bool, export_dir: str):
        self.proxy = proxy
        self.nb_weeks = proxy.nb_weeks
        self.scenarios = proxy.scenarios
        self.export_dir = export_dir

        self.stage_cost_functions = self.proxy.compute_stage_cost_functions()

        self.cost_functions = self.stage_cost_functions[:, :, 0]
        self.turb_functions = self.stage_cost_functions[:, :, 1]
        self.pump_functions = self.stage_cost_functions[:, :, 2]

        self.bv = np.zeros((self.nb_weeks, 51, 200))
        self.mean_bv = np.zeros((self.nb_weeks, 51))

        if not isinstance(export_dir, str) or not export_dir:
            raise ValueError("export_dir must be provided as a non-empty string to BellmanValuesProxy.")
        self.export_dir = export_dir
        logger_setup = LoggerSetup(self.export_dir)
        self.logger = logger_setup.get_logger() if enable_logging else logger_setup.get_null_logger()

        self.compute_bellman_values()
        self.compute_usage_values()
    
    def penalty_final_stock(self)->Callable:
        penalty = lambda x: abs(x-self.proxy.reservoir.initial_level)/self.proxy.reservoir.initial_level *100* self.proxy.upper_bound_cost(self.nb_weeks - 1)
        return penalty
    
    def penalty_rule_curves(self,week_idx:int)->Callable:
        ub_cost = self.proxy.upper_bound_cost(week_idx)
        penalty=interp1d(
                [
                    self.proxy.reservoir.weekly_lower_rule_curve[week_idx]-0.1*self.proxy.reservoir.capacity,
                    self.proxy.reservoir.weekly_lower_rule_curve[week_idx],
                    self.proxy.reservoir.weekly_upper_rule_curve[week_idx],
                    self.proxy.reservoir.weekly_upper_rule_curve[week_idx]+0.1*self.proxy.reservoir.capacity,
                ],
                [
                    10*ub_cost,
                    0,
                    0,
                    10*ub_cost,
                ],fill_value='extrapolate',
            )
        return penalty

    def init_log_bellman(self) -> None:
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{"CALCUL DES VALEURS DE BELLMAN".center(70)}")
        self.logger.debug("=" * 70 + "\n")
        self.logger.debug(">>> Initialisation des valeurs de Bellman finales")

    def init_log_bellman_week(self,week : int)-> None:
        self.logger.debug("\n" + "-" * 60)
        self.logger.debug(f"---- Traitement de la semaine {week+1} ----")
        self.logger.debug("-" * 60 + "\n")

    def bellman_function(self,week : int) -> interp1d:
        return interp1d(
                        np.linspace(0, self.proxy.reservoir.capacity, 51),
                        self.mean_bv[week],
                        kind="linear",
                        fill_value="extrapolate",
                    )

    def iterate_over_controls(self,
                              best_value : float|None,
                              best_stock :float|None,
                              best_control : float|None,
                              controls : np.ndarray,
                              current_stock : float,
                              weekly_inflow : float,
                              stage_cost_function : interp1d,
                              future_bellman_function : interp1d,
                              penalty_function : interp1d) -> tuple:
        

        for control in controls:
            next_stock = current_stock - control + weekly_inflow
            
            cost = stage_cost_function(control)
            future_value = future_bellman_function(next_stock)
            penalty = penalty_function(next_stock)
            total_value = cost + future_value + penalty

            self.logger.debug(
                f"Test contrôle (libre): {control:.2f}, stock suivant: {next_stock:.2f}, "
                f"Coût: {cost:.2f}, BV futur: {future_value:.2f}, pénalité: {penalty:.2f}, total: {total_value:.2f}"
            )

            if total_value < best_value:
                best_value = total_value
                best_stock = next_stock
                best_control = control
                self.logger.debug(
                    f"→ Nouveau meilleur contrôle retenu (libre): {control:.2f}, total: {total_value:.2f}"
                )
        
        return best_value, best_stock, best_control

    def iterate_over_stock_levels(self,
                            best_value : float|None,
                            best_stock : float|None,
                            best_control : float|None,
                            current_stock : float,
                            weekly_inflow : float,
                            max_week_pump : float,
                            max_week_turb : float,
                            stage_cost_function : interp1d,
                            future_bellman_function : interp1d,
                            penalty_function : interp1d) -> tuple:

        for c_new in range(0, 101, 2):
            
            new_level = (c_new / 100) * self.proxy.reservoir.capacity
            week_energy_var = current_stock - new_level

            if week_energy_var < -max_week_pump * self.proxy.reservoir.efficiency or \
            week_energy_var > max_week_turb * self.proxy.turb_efficiency:
                continue

            control = current_stock - new_level
            next_stock = current_stock - control + weekly_inflow
            cost = stage_cost_function(control)
            future_value = future_bellman_function(next_stock)
            penalty = penalty_function(next_stock)
            total_value = cost + future_value + penalty

            self.logger.debug(
                f"Test contrôle (forcé): {control:.2f}, stock suivant: {next_stock:.2f}, "
                f"Coût: {cost:.2f}, BV futur: {future_value:.2f}, pénalité: {penalty:.2f}, total: {total_value:.2f}"
            )

            if total_value < best_value:
                best_value = total_value
                best_stock = next_stock
                best_control = control
                self.logger.debug(
                    f"→ Nouveau meilleur contrôle retenu (forcé): {control:.2f}, total: {total_value:.2f}"
                )
        return best_value, best_stock, best_control

    def compute_bellman_values(self) -> None:
        self.init_log_bellman()

        penalty_final_stock = self.penalty_final_stock()
        self.mean_bv[self.nb_weeks - 1] = np.array([
            penalty_final_stock((c / 100) * self.proxy.reservoir.capacity) for c in range(0, 101, 2)
        ])

        self.logger.debug(f"Valeurs de pénalité finales (semaine {self.nb_weeks}): {self.mean_bv[self.nb_weeks - 1]}")

        for w in reversed(range(self.nb_weeks - 1)):
            
            self.init_log_bellman_week(w)

            penalty_function = self.penalty_rule_curves(w + 1)
            future_bellman_function = self.bellman_function(w+1)

            max_week_turb = self.proxy.reservoir.max_weekly_turb[w]
            max_week_pump = self.proxy.reservoir.max_weekly_pump[w]

            for c in range(0, 101, 2):
                current_stock = (c / 100) * self.proxy.reservoir.capacity

                for s in self.scenarios:
                    
                    weekly_inflow = self.proxy.reservoir.weekly_inflow[w + 1, s]
                    cost_function = self.cost_functions[w + 1, s]
                    controls = cost_function.x

                    best_value_init = np.inf
                    best_stock = None
                    best_control = None
                    self.logger.debug(f"\n[Semaine {w+1} | Scénario {s+1} | Stock {current_stock:.2f} MWh]")

                    best_value, best_stock, best_control = self.iterate_over_controls(best_value=best_value_init,
                                                            best_stock=best_stock,
                                                            best_control=best_control,
                                                            controls=controls,
                                                            current_stock=current_stock,
                                                            weekly_inflow=weekly_inflow,
                                                            stage_cost_function=cost_function,
                                                            future_bellman_function=future_bellman_function,
                                                            penalty_function=penalty_function)

                    final_best_value, final_best_stock, final_best_control = self.iterate_over_stock_levels(best_value=best_value,
                                                                    best_stock = best_stock,
                                                                    best_control = best_control,
                                                                    current_stock=current_stock,
                                                                    weekly_inflow=weekly_inflow,
                                                                    max_week_pump=max_week_pump,
                                                                    max_week_turb=max_week_turb,
                                                                    stage_cost_function=cost_function,
                                                                    future_bellman_function=future_bellman_function,
                                                                    penalty_function=penalty_function,
                                                                    )

                    self.bv[w, c // 2, s] = final_best_value
                    self.logger.debug(f"Valeur de Bellman enregistrée pour stock {current_stock:.2f} MWh : {best_value:.2f}")

                self.mean_bv[w, c // 2] = np.mean(self.bv[w, c // 2,self.scenarios])
            self.logger.debug(f"Valeurs de Bellman moyennes pour la semaine {w+1} : {self.mean_bv[w]}")

    def compute_usage_values(self) -> None:
        self.usage_values = np.zeros((self.nb_weeks, 50))
        for w in range(self.nb_weeks):
            for c in range(2, 102, 2):
                self.usage_values[w, (c // 2) - 1] = self.mean_bv[w, c // 2] - self.mean_bv[w, (c // 2) - 1]


class OptimalTrajectories:
    def __init__(self,
                 bellman_values : BellmanValuesProxy):
        self.bellman_values=bellman_values
        self.nb_weeks = bellman_values.nb_weeks
        self.scenarios = bellman_values.scenarios
        self.logger = bellman_values.logger
        self.export_dir = bellman_values.export_dir
        
        self.mean_bv=bellman_values.mean_bv
        self.compute_trajectories()
        
        # self.new_lower_rule_curve()
        # self.new_upper_rule_curve()

    def init_log_trajectories(self)-> None:
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{"CALCUL DES TRAJECTOIRES".center(70)}")
        self.logger.debug("=" * 70 + "\n")

    def init_log_trajectories_week(self, week:int, scenario:int, previous_stock :float)->None:
        self.logger.debug("\n" + "-" * 60)
        self.logger.debug(f"---- Semaine {week+1}, scénario {scenario+1} ----")
        self.logger.debug("-" * 60)
        self.logger.debug(f"Stock précédent : {previous_stock:.2f} MWh")

    def init_log_lower_rule_curves(self,)->None:
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{"CALCUL DE LA COURBE GUIDE INFERIEURE HORAIRE AJUSTEE".center(70)}")
        self.logger.debug("=" * 70 + "\n")

    def init_log_upper_rule_curves(self,)->None:
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{"CALCUL DE LA COURBE GUIDE SUPERIEURE HORAIRE AJUSTEE".center(70)}")
        self.logger.debug("=" * 70 + "\n")

    def write_warnings(self)->None:
        if hasattr(self, 'export_dir') and self.warning_lines:
            warning_path = os.path.join(self.export_dir, "warnings.txt")
            with open(warning_path, "w", encoding="utf-8") as f:
                for line in self.warning_lines:
                    f.write(line + "\n")

    def compute_trajectories(self) -> None:
        self.init_log_trajectories()

        self.trajectories = np.zeros((200, self.nb_weeks))
        self.optimal_controls = np.zeros_like(self.trajectories)
        self.optimal_turb = np.zeros_like(self.trajectories)
        self.optimal_pump = np.zeros_like(self.trajectories)
        # self.inflow_adjust_rule_curves = np.zeros((self.nb_weeks, len(self.scenarios), 168))
        self.inflow_adjust_overflow = np.zeros((self.nb_weeks,200, 168))
        self.warning_lines:list = []

        for s in self.scenarios:
            current_stock = self.bellman_values.proxy.reservoir.initial_level
            
            for w in range(self.nb_weeks):
                self.logger.debug(f"\n[Semaine {w+1} | Scénario {s+1} | Stock préceédent : {current_stock:.2f} MWh]")
                weekly_inflow = self.bellman_values.proxy.reservoir.weekly_inflow[w, s]
                hourly_inflow = self.bellman_values.proxy.reservoir.hourly_inflow[w * 168:(w + 1) * 168, s]
                self.logger.debug(f"Inflow : {weekly_inflow:.2f} MWh")
                cost_function = self.bellman_values.cost_functions[w, s]
                penalty_function = self.bellman_values.penalty_rule_curves(w)
                controls = cost_function.x

                future_bellman_function = self.bellman_values.bellman_function(w)

                max_week_turb = self.bellman_values.proxy.reservoir.max_weekly_turb[w]
                max_week_pump = self.bellman_values.proxy.reservoir.max_weekly_pump[w]

                best_value=np.inf
                best_stock=None
                best_control=None

                self.adjust_hourly_inflow_overflow(scenario=s, week=w, stock_init=current_stock, inflow=hourly_inflow)
                weekly_inflow -= np.sum(self.inflow_adjust_overflow[w, s])

                best_value, best_stock, best_control = self.bellman_values.iterate_over_controls(best_value=best_value,
                                                            best_stock=best_stock,
                                                            best_control=best_control,
                                                            controls=controls,
                                                            current_stock=current_stock,
                                                            weekly_inflow=weekly_inflow,
                                                            stage_cost_function=cost_function,
                                                            future_bellman_function=future_bellman_function,
                                                            penalty_function=penalty_function)

                final_best_value, final_best_stock, final_best_control = self.bellman_values.iterate_over_stock_levels(best_value=best_value,
                                                                    best_stock=best_stock,
                                                                    best_control=best_control,
                                                                    current_stock=current_stock,
                                                                    weekly_inflow=weekly_inflow,
                                                                    max_week_pump=max_week_pump,
                                                                    max_week_turb=max_week_turb,
                                                                    stage_cost_function=cost_function,
                                                                    future_bellman_function=future_bellman_function,
                                                                    penalty_function=penalty_function,
                                                                    )

                self.logger.debug(f"==> Stock retenu pour la semaine {w+1} : {final_best_stock:.2f} MWh")

                self.trajectories[s, w] = final_best_stock
                self.optimal_controls[s, w] = final_best_control
                self.optimal_turb[s, w] = self.bellman_values.turb_functions[w, s](final_best_control)
                self.optimal_pump[s, w] = self.bellman_values.pump_functions[w, s](final_best_control)
                current_stock = final_best_stock
                
        self.write_warnings()


    def adjust_hourly_inflow_overflow(self,
                                      scenario : int,
                                      week : int,
                                      stock_init : float,
                                      inflow : np.ndarray) -> None:
        
        turb = self.bellman_values.proxy.reservoir.max_hourly_turb[week * 168 : (week + 1) * 168]
        pump = self.bellman_values.proxy.reservoir.max_hourly_pump[week * 168 : (week + 1) * 168]
        net_hourly_turb = inflow - turb*self.bellman_values.proxy.turb_efficiency
        net_hourly_pump = inflow + pump*self.bellman_values.proxy.reservoir.efficiency
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
                net_hourly=net_hourly_pump
            )
            if hourly_negative_stock is not None:
                self.inflow_adjust_overflow[week, scenario, h] = hourly_negative_stock
                net_hourly_pump[h] -= self.inflow_adjust_overflow[week, scenario, h]
            
    def detect_hourly_overflow(self,
                               scenario : int,
                               week : int,
                               hour : int,
                               stock_init : float,
                               net_hourly : np.ndarray) -> float|None:
        
        stock = stock_init + np.cumsum(net_hourly[:hour+1])
        if stock[hour] > self.bellman_values.proxy.reservoir.capacity :
            self.logger.debug(
                f"⚠️ Déversement détecté pour scénario {scenario+1}, semaine {week+1}, heure {hour+1} : "
                f"{stock[hour]:.2f} MWh > capacité {self.bellman_values.proxy.reservoir.capacity:.2f} MWh"
            )
            return stock[hour] - self.bellman_values.proxy.reservoir.capacity
        return None
    
    def detect_hourly_negative_stock(self,
                                    scenario : int,
                                    week : int,
                                    hour : int,
                                    stock_init : float,
                                    net_hourly : np.ndarray) -> float|None:
        stock = stock_init + np.cumsum(net_hourly[:hour+1])
        if stock[hour] < 0:
            self.logger.debug(
                f"⚠️ Stock négatif détecté pour scénario {scenario+1}, semaine {week+1}, heure {hour+1} : "
                f"{stock[hour]:.2f} MWh < 0 MWh"
            )
            return stock[hour]
        return None
    
    def daily_to_hourly_curve(self,daily_curve: np.ndarray) -> np.ndarray:
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
        upper_curves=np.zeros((len(self.scenarios),self.nb_weeks,168))
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                stock_init=self.trajectories[s,w-1] if w>0 else self.bellman_values.proxy.reservoir.initial_level
                stock_final = self.trajectories[s,w]

                max_hour_pump = self.bellman_values.proxy.reservoir.max_hourly_pump[w*168:(w+1)*168]
                max_hour_turb = self.bellman_values.proxy.reservoir.max_hourly_turb[w*168:(w+1)*168]

                hourly_inflow = self.bellman_values.proxy.reservoir.hourly_inflow[w*168:(w+1)*168,s]

                cumsum_pump = np.concatenate([[0],
                                              np.cumsum(
                                                  max_hour_pump * self.bellman_values.proxy.reservoir.efficiency+hourly_inflow-self.inflow_adjust_overflow[w,s])])[:-1] + stock_init
                cumsum_turb = stock_final - np.cumsum(
                    -self.bellman_values.proxy.turb_efficiency*max_hour_turb[::-1] + (hourly_inflow-self.inflow_adjust_overflow[w,s])[::-1])[::-1]

                hourly_curve = np.minimum(cumsum_pump, cumsum_turb)
                upper_curves[s,w]=hourly_curve

        weekly_envelope=np.min(upper_curves,axis=0)
        hourly_upper_envelope=weekly_envelope.flatten()
        hourly_upper_envelope=np.concatenate([hourly_upper_envelope, hourly_upper_envelope[-24:]])
        return hourly_upper_envelope

    def compute_lower_enveloppe(self) -> np.ndarray:
        lower_curves=np.zeros((len(self.scenarios),self.nb_weeks,168))
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                stock_init=self.trajectories[s,w-1] if w>0 else self.bellman_values.proxy.reservoir.initial_level
                stock_final = self.trajectories[s,w]

                max_hour_pump = self.bellman_values.proxy.reservoir.max_hourly_pump[w*168:(w+1)*168]
                max_hour_turb = self.bellman_values.proxy.reservoir.max_hourly_turb[w*168:(w+1)*168]

                hourly_inflow = self.bellman_values.proxy.reservoir.hourly_inflow[w*168:(w+1)*168,s]
                
                cumsum_turb = np.concatenate(
                    [[0],
                     np.cumsum(-self.bellman_values.proxy.turb_efficiency * max_hour_turb + hourly_inflow-self.inflow_adjust_overflow[w,s])])[:-1] +stock_init
                cumsum_pump = stock_final - np.cumsum(
                    self.bellman_values.proxy.reservoir.efficiency*max_hour_pump[::-1]  + (hourly_inflow+self.inflow_adjust_overflow[w,s])[::-1])[::-1]

                hourly_curve = np.maximum(cumsum_pump, cumsum_turb)
                lower_curves[s,w]=hourly_curve

        weekly_envelope=np.max(lower_curves,axis=0)
        hourly_lower_envelope=weekly_envelope.flatten()
        hourly_lower_envelope=np.concatenate([hourly_lower_envelope, hourly_lower_envelope[-24:]])
        return hourly_lower_envelope

    def new_lower_rule_curve(self)->None:
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
                f"{len(hours_with_diff)} heure(s) avec un écart > {threshold} entre la courbe guide inférieure ajustée et interpolée."
            )
            for hour in hours_with_diff:
                diff_value = difference[hour]
                self.logger.debug(f"Heure {hour} : écart = {diff_value:.6f}")

    def new_upper_rule_curve(self) -> None:
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
                f"{len(hours_with_diff)} heure(s) avec un écart > {threshold} entre la courbe guide supérieure ajustée et interpolée."
            )
            for hour in hours_with_diff:
                diff_value = difference[hour]
                self.logger.debug(f"Heure {hour} : écart = {diff_value:.6f}")
