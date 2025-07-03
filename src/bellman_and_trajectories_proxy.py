from proxy_stage_cost_function import Proxy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rcParams
from scipy.interpolate import interp1d
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from type_definition import Callable
import pandas as pd
import os
import logging
from logging.handlers import RotatingFileHandler
from configparser import ConfigParser
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from tqdm import tqdm

rcParams['font.family'] = 'Cambria'

class LoggerSetup:
    def __init__(self, export_dir: str, log_name: str = "logger", max_bytes: int = 10_000_000, backup_count: int = 5):
        self.export_dir = export_dir
        self.log_name = log_name
        self.max_bytes = max_bytes
        self.backup_count = backup_count

    def get_logger(self, logger_name: str = "logger") -> logging.Logger:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()

        log_path = os.path.join(self.export_dir, self.log_name)
        handler = RotatingFileHandler(log_path, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    @staticmethod
    def get_null_logger(name: str = "NullLogger") -> logging.Logger:
        null_logger = logging.getLogger(name)
        null_logger.setLevel(logging.CRITICAL + 1)
        if not null_logger.hasHandlers():
            null_logger.addHandler(logging.NullHandler())
        return null_logger

class BellmanValuesProxy:
    def __init__(self, cost_function: Proxy, alpha: float, coeff: float, enable_logging: bool, export_dir: str):
        self.cost_function = cost_function
        self.reservoir = cost_function.reservoir
        self.reservoir_capacity = self.reservoir.capacity
        self.initial_level = self.reservoir.initial_level
        self.alpha = alpha

        self.bottom_rule_curve = cost_function.reservoir.bottom_rule_curve
        self.upper_rule_curve = cost_function.reservoir.upper_rule_curve
        self.daily_bottom_rule_curve = self.cost_function.reservoir.daily_bottom_rule_curve
        self.daily_upper_rule_curve = self.cost_function.reservoir.daily_upper_rule_curve

        self.inflow = cost_function.reservoir.inflow
        self.daily_inflow = cost_function.reservoir.daily_inflow

        self.nb_weeks = self.cost_function.nb_weeks
        self.scenarios = self.cost_function.scenarios

        self.cost_functions_turb_and_pump = self.cost_function.compute_stage_cost_functions(alpha, coeff)
        self.cost_functions = self.cost_functions_turb_and_pump[:, :, 0]
        self.turb_functions = self.cost_functions_turb_and_pump[:, :, 1]
        self.pump_functions = self.cost_functions_turb_and_pump[:, :, 2]
        self.bv = np.zeros((self.nb_weeks, 51, len(self.scenarios)))
        self.mean_bv = np.zeros((self.nb_weeks, 51))

        if not isinstance(export_dir, str) or not export_dir:
            raise ValueError("export_dir must be provided as a non-empty string to BellmanValuesProxy.")
        self.export_dir = export_dir
        logger_setup = LoggerSetup(self.export_dir)
        self.logger = logger_setup.get_logger() if enable_logging else logger_setup.get_null_logger()

        self.compute_bellman_values()
        self.compute_usage_values()
        self.compute_trajectories()
        self.new_lower_rule_curve()
        self.new_upper_rule_curve()

    def penalty_final_stock(self)->Callable:
        penalty = lambda x: abs(x-self.initial_level)/1e3
        return penalty
    
    def penalty_rule_curves(self,week_idx:int)->Callable:
        # penalty=interp1d(np.array([-0.05,0,1,1.05])*self.reservoir_capacity,[1e3,0,0,1e3],kind='linear',fill_value='extrapolate')
        penalty=interp1d(
                [
                    0,
                    self.reservoir.bottom_rule_curve[week_idx],
                    self.reservoir.upper_rule_curve[week_idx],
                    self.reservoir.capacity,
                ],
                [
                    self.reservoir.upper_rule_curve[week_idx]/1e4,
                    0,
                    0,
                    self.reservoir.upper_rule_curve[week_idx]/1e4,
                ],fill_value='extrapolate',
            )
        # penalty=lambda x:0
        return penalty

    def compute_bellman_values(self) -> None:
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{"CALCUL DES VALEURS DE BELLMAN".center(70)}")
        self.logger.debug("=" * 70 + "\n")
        self.logger.debug(">>> Initialisation des valeurs de Bellman finales")
        penalty_final_stock = self.penalty_final_stock()
        self.mean_bv[self.nb_weeks - 1] = np.array([
            penalty_final_stock((c / 100) * self.reservoir_capacity) for c in range(0, 101, 2)
        ])
        self.logger.debug(f"Valeurs de pénalité finales (semaine {self.nb_weeks}): {self.mean_bv[self.nb_weeks - 1]}")

        for w in reversed(range(self.nb_weeks - 1)):
            self.logger.debug("\n" + "-" * 60)
            self.logger.debug(f"---- Traitement de la semaine {w+1} ----")
            self.logger.debug("-" * 60 + "\n")
            penalty_function = self.penalty_rule_curves(w + 1)

            for c in range(0, 101, 2):
                current_stock = (c / 100) * self.reservoir_capacity

                for s in self.scenarios:
                    
                    inflows_for_week = self.inflow[w + 1, s]
                    cost_function = self.cost_functions[w + 1, s]
                    controls = cost_function.x

                    future_bellman_function = interp1d(
                        np.linspace(0, self.reservoir_capacity, 51),
                        self.mean_bv[w + 1],
                        kind="linear",
                        fill_value="extrapolate",
                    )

                    best_value = np.inf
                    self.logger.debug(f"\n[Semaine {w+1} | Scénario {s+1} | Stock {current_stock:.2f} MWh]")

                    # Contrôles libres
                    for control in controls:
                        next_stock = current_stock - control + inflows_for_week
                        cost = cost_function(control)
                        future_value = future_bellman_function(next_stock)
                        penalty = penalty_function(next_stock)
                        total_value = cost + future_value + penalty

                        self.logger.debug(
                            f"Test contrôle (libre): {control:.2f}, stock suivant: {next_stock:.2f}, "
                            f"Coût: {cost:.2f}, BV futur: {future_value:.2f}, pénalité: {penalty:.2f}, total: {total_value:.2f}"
                        )

                        if total_value < best_value:
                            best_value = total_value
                            self.logger.debug(
                                f"→ Nouveau meilleur contrôle retenu (libre): {control:.2f}, total: {total_value:.2f}"
                            )

                    # Contrôles forcés
                    for c_new in range(0, 101, 2):
                        max_week_turb = np.sum(self.cost_function.max_daily_generating[(w + 1) * 7:(w + 2) * 7])
                        max_week_pump = np.sum(self.cost_function.max_daily_pumping[(w + 1) * 7:(w + 2) * 7])
                        new_level = (c_new / 100) * self.reservoir_capacity
                        week_energy_var = current_stock - new_level

                        if week_energy_var < -max_week_pump * self.cost_function.efficiency or \
                        week_energy_var > max_week_turb * self.cost_function.turb_efficiency:
                            continue

                        control = current_stock - new_level
                        next_stock = current_stock - control + inflows_for_week
                        cost = cost_function(control)
                        future_value = future_bellman_function(next_stock)
                        penalty = penalty_function(next_stock)
                        total_value = cost + future_value + penalty

                        self.logger.debug(
                            f"Test contrôle (forcé): {control:.2f}, stock suivant: {next_stock:.2f}, "
                            f"Coût: {cost:.2f}, BV futur: {future_value:.2f}, pénalité: {penalty:.2f}, total: {total_value:.2f}"
                        )

                        if total_value < best_value:
                            best_value = total_value
                            self.logger.debug(
                                f"→ Nouveau meilleur contrôle retenu (forcé): {control:.2f}, total: {total_value:.2f}"
                            )

                    self.bv[w, c // 2, s] = best_value
                    self.logger.debug(f"Valeur de Bellman enregistrée pour stock {current_stock:.2f} MWh : {best_value:.2f}")

                self.mean_bv[w, c // 2] = np.mean(self.bv[w, c // 2])
            self.logger.debug(f"→ Moyenne BV semaine {w+1} : {self.mean_bv[w]}")

    def compute_usage_values(self) -> None:
        self.usage_values=np.zeros((self.nb_weeks,50))
        for w in range(self.nb_weeks):
            for c in range(2,102,2):
                self.usage_values[w,(c//2)-1]=self.mean_bv[w,c//2]-self.mean_bv[w,(c//2)-1]

    def compute_trajectories(self) -> None:
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{"CALCUL DES TRAJECTOIRES".center(70)}")
        self.logger.debug("=" * 70 + "\n")
        self.trajectories = np.zeros((len(self.scenarios),self.nb_weeks))
        self.optimal_controls = np.zeros((len(self.scenarios),self.nb_weeks))
        self.optimal_turb = np.zeros((len(self.scenarios),self.nb_weeks))
        self.optimal_pump = np.zeros((len(self.scenarios),self.nb_weeks))
        self.warning_lines = []
        for s in self.scenarios:
            previous_stock = self.initial_level
            for w in range(self.nb_weeks):
                self.logger.debug("\n" + "-" * 60)
                self.logger.debug(f"---- Semaine {w+1}, scénario {s+1} ----")
                self.logger.debug("-" * 60)
                self.logger.debug(f"Stock précédent : {previous_stock:.2f} MWh")

                inflows_for_week = self.inflow[w, s]
                self.logger.debug(f"Inflow : {inflows_for_week:.2f} MWh")
                cost_function = self.cost_functions[w, s]
                penalty_function = self.penalty_rule_curves(w)
                controls = cost_function.x
                future_bellman_function = interp1d(
                    np.linspace(0, self.reservoir_capacity, 51),
                    self.mean_bv[w],
                    kind="linear",
                    fill_value="extrapolate",
                )

                best_value = np.inf
                best_new_stock = None

                for control in controls:
                    new_stock = previous_stock - control + inflows_for_week
                    cost = cost_function(control)
                    future_value = future_bellman_function(new_stock)
                    penalty = penalty_function(new_stock)
                    total_value = cost + future_value + penalty
                    self.logger.debug(f"Test contrôle (libre): {control:.2f}, stock suivant: {new_stock:.2f}, coût: {cost:.2f}, future_value: {future_value:.2f}, pénalité: {penalty:.2f}, total: {total_value:.2f}")

                    if total_value < best_value:
                        best_value = total_value
                        best_new_stock = new_stock
                        optimal_control = control
                        self.logger.debug(f"→ Nouveau meilleur contrôle retenu (libre) : {control:.2f}, stock suivant: {new_stock:.2f}, total: {total_value:.2f}")

                for c_new in range(0, 101, 2):
                    max_week_turb = np.sum(self.cost_function.max_daily_generating[w * 7:(w + 1) * 7])
                    max_week_pump = np.sum(self.cost_function.max_daily_pumping[w * 7:(w + 1) * 7])
                    week_energy_var = previous_stock - c_new / 100 * self.reservoir_capacity
                    if week_energy_var < -max_week_pump * self.cost_function.efficiency or week_energy_var > max_week_turb * self.cost_function.turb_efficiency:
                        continue
                    control = previous_stock - c_new / 100 * self.reservoir_capacity
                    new_stock = previous_stock - control + inflows_for_week
                    cost = cost_function(control)
                    future_value = future_bellman_function(new_stock)
                    penalty = penalty_function(new_stock)
                    total_value = cost + future_value + penalty
                    self.logger.debug(f"Test contrôle (forcé): {control:.2f}, stock suivant: {new_stock:.2f}, coût: {cost:.2f}, future_value: {future_value:.2f}, pénalité: {penalty:.2f}, total: {total_value:.2f}")

                    if total_value < best_value:
                        best_value = total_value
                        best_new_stock = new_stock
                        optimal_control = control
                        self.logger.debug(f"→ Nouveau meilleur contrôle retenu (forcé) : {control:.2f}, stock suivant: {new_stock:.2f}, total: {total_value:.2f}")

                self.logger.debug(f"==> Stock retenu pour la semaine {w+1} : {best_new_stock:.2f} MWh\n")

                if best_new_stock is not None:
                    self.trajectories[s, w] = best_new_stock
                    self.optimal_controls[s, w] = optimal_control
                    self.optimal_turb[s, w] = self.turb_functions[w, s](optimal_control)
                    self.optimal_pump[s, w] = self.pump_functions[w, s](optimal_control)
                    lower_bound = self.bottom_rule_curve[w]
                    upper_bound = self.upper_rule_curve[w]
                    if not (lower_bound <= best_new_stock <= upper_bound):
                        warning_msg = (
                            f"⚠️ Stock hors courbes guides - Semaine {w+1}, scénario {s+1} : "
                            f"{best_new_stock:.2f} ∉ [{lower_bound:.2f}, {upper_bound:.2f}]"
                        )
                        self.warning_lines.append(warning_msg)
                    previous_stock = best_new_stock
                else:
                    self.trajectories[s, w] = None
                    self.optimal_controls[s, w] = None
                    self.optimal_pump[s, w] = None
                    self.optimal_turb[s, w] = None
        # Write warnings to file if any
        if hasattr(self, 'export_dir') and self.warning_lines:
            warning_path = os.path.join(self.export_dir, "warnings.txt")
            with open(warning_path, "w", encoding="utf-8") as f:
                for line in self.warning_lines:
                    f.write(line + "\n")


    def daily_to_hourly_curve(self,daily_curve: np.ndarray) -> np.ndarray:
        n_days = len(daily_curve)
        n_hours = (n_days - 1) * 24 + 1
        hourly_curve = np.interp(
            np.arange(n_hours),
            np.arange(0, n_days) * 24,
            daily_curve
        )
        last_val = daily_curve[-1]
        final_interp = np.linspace(last_val, self.reservoir.initial_level, 25)[1:-1]
        hourly_curve = np.concatenate([hourly_curve, final_interp])
        return hourly_curve

    def new_lower_rule_curve(self)->None:
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{"CALCUL DE LA COURBE GUIDE INFERIEURE HORAIRE AJUSTEE".center(70)}")
        self.logger.debug("=" * 70 + "\n")
        upper_curves=np.zeros((len(self.scenarios),self.nb_weeks,168))
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                stock_init=self.trajectories[s,w-1] if w>0 else self.initial_level
                stock_final = self.trajectories[s,w]

                pump_max = np.repeat(self.reservoir.max_daily_pumping[w * 7:(w + 1) * 7], 24) / 24
                turb_max = np.repeat(self.reservoir.max_daily_generating[w * 7:(w + 1) * 7], 24) / 24
                inflows = np.repeat(self.daily_inflow[w * 7:(w + 1) * 7,s],24)/24

                cumsum_pump = np.concatenate([[0],np.cumsum(pump_max * self.cost_function.efficiency+inflows)])[:-1] + stock_init
                cumsum_turb = stock_final - np.cumsum(-self.cost_function.turb_efficiency*turb_max[::-1] + inflows[::-1])[::-1]

                hourly_curve = np.minimum(cumsum_pump, cumsum_turb)
                upper_curves[s,w]=hourly_curve

        weekly_envelope=np.min(upper_curves,axis=0)
        hourly_envelope=weekly_envelope.flatten()
        hourly_envelope=np.concatenate([hourly_envelope, hourly_envelope[-24:]])
        self.hourly_lower_rule_curve = self.daily_to_hourly_curve(self.daily_bottom_rule_curve)
        final_lower_rule_curve=np.minimum(hourly_envelope,self.hourly_lower_rule_curve)
        self.final_lower_rule_curve = np.concatenate([final_lower_rule_curve[1:], [self.initial_level]])

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
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{"CALCUL DE LA COURBE GUIDE SUPERIEURE HORAIRE AJUSTEE".center(70)}")
        self.logger.debug("=" * 70 + "\n")
        lower_curves = np.zeros((len(self.scenarios), self.nb_weeks, 168))
        
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                stock_init = self.trajectories[s, w - 1] if w > 0 else self.initial_level
                stock_final = self.trajectories[s, w]

                turb_max = np.repeat(self.reservoir.max_daily_generating[w * 7:(w + 1) * 7], 24) / 24
                pump_max = np.repeat(self.reservoir.max_daily_pumping[w * 7:(w + 1) * 7], 24) / 24
                inflows = np.repeat(self.daily_inflow[w * 7:(w + 1) * 7, s], 24) / 24

                cumsum_turb = np.concatenate([[0],np.cumsum(-turb_max * self.cost_function.turb_efficiency + inflows)])[:-1] +stock_init
                cumsum_pump = stock_final - np.cumsum(self.cost_function.efficiency*pump_max[::-1]  + inflows[::-1])[::-1]

                hourly_curve = np.maximum(cumsum_turb, cumsum_pump)
                lower_curves[s, w] = hourly_curve

        weekly_envelope = np.max(lower_curves, axis=0)
        hourly_envelope = weekly_envelope.flatten()
        hourly_envelope = np.concatenate([hourly_envelope, hourly_envelope[-24:]])
        self.hourly_upper_rule_curve = self.daily_to_hourly_curve(self.daily_upper_rule_curve)
        final_upper_rule_curve=np.maximum(hourly_envelope,self.hourly_upper_rule_curve)
        self.final_upper_rule_curve = np.concatenate([final_upper_rule_curve[1:], [self.initial_level]])

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


class Plotter:
    def __init__(self,bv:BellmanValuesProxy):
        self.bv = bv

    def plot_bellman_value(self, week_index: int) -> None:
        if week_index < 0 or week_index >= self.bv.nb_weeks:
            raise ValueError(f"Semaine invalide : {week_index}. Doit être entre 0 et {self.bv.nb_weeks - 1}.")

        stock_levels = np.linspace(0, 100, 51)
        bellman_values = self.bv.mean_bv[week_index,:]

        plt.figure(figsize=(10, 5))
        plt.plot(stock_levels, bellman_values, label=f"Semaine {week_index + 1}", color='tab:blue')

        plt.xlabel("Stock (%)")
        plt.ylabel("Valeur de Bellman")
        area = getattr(self.bv.cost_function, 'name_area', None)
        if area is None:
            area = getattr(self.bv, 'area', None)
        area_str = f" - Zone : {area}" if area else ""
        plt.title(f"Valeur de Bellman en fonction du stock - Semaine {week_index + 1}{area_str}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_usage_values(self) -> None:
        stock_levels = np.linspace(2, 100, 50) 
        plt.figure(figsize=(12, 6))

        for w in range(self.bv.nb_weeks):
            plt.plot(
                stock_levels, 
                self.bv.usage_values[w],
                label=f"S {w+1}"
            )

        plt.xlabel('Stock (%)')
        plt.ylabel('Valeur d\'usage (MWh)')
        area = getattr(self.bv.cost_function, 'name_area', None)
        if area is None:
            area = getattr(self.bv, 'area', None)
        area_str = f" - Zone : {area}" if area else ""
        plt.title(f"Valeurs d'usage en fonction du stock{area_str}")
        plt.legend(
            loc='lower right',
            bbox_to_anchor=(1, -0.15),
            ncol=6
        )
        plt.tight_layout(rect=(0, 0.1, 1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_usage_values_heatmap(self) -> None:
        fig, ax = plt.subplots(figsize=(14, 6))

        norm = colors.Normalize(np.min(self.bv.usage_values[:-1]), np.max(self.bv.usage_values[:-1]))
        # norm = colors.Normalize(vmin=-29, vmax=0)

        im = ax.imshow(
            self.bv.usage_values[:-1].T,
            aspect='auto',
            origin='lower',
            cmap='nipy_spectral',
            extent=(1, 52, 2, 100),
            norm=norm,
            interpolation='bilinear'  # lissage
        )

        cbar = fig.colorbar(im, ax=ax, ticks=np.linspace(np.min(self.bv.usage_values[:-1]), np.max(self.bv.usage_values[:-1]), 10))
        # cbar = fig.colorbar(im, ax=ax, ticks=np.linspace(-29, 0, 10))
        cbar.set_label("Valeur d’usage")

        ax.set_xlabel("Semaine")
        ax.set_ylabel("Stock (%)")
        area = getattr(self.bv.cost_function, 'name_area', None)
        if area is None:
            area = getattr(self.bv, 'area', None)
        area_str = f" - Zone : {area}" if area else ""
        ax.set_title(f"Nappes de valeurs d’usage (α={self.bv.alpha}){area_str}")

        plt.grid(False)
        plt.tight_layout()
        plt.show()


    def plot_trajectories(self) -> None:
        fig = go.Figure()
        weeks = list(range(1, self.bv.nb_weeks + 2))

        upper_percent = self.bv.upper_rule_curve / self.bv.reservoir_capacity * 100
        fig.add_trace(go.Scatter(
            x=weeks,
            y=upper_percent,
            mode='lines',
            name='Upper rule curve',
            line=dict(dash='dash', color='green'),
            visible=True
        ))

        lower_percent = self.bv.bottom_rule_curve / self.bv.reservoir_capacity * 100
        fig.add_trace(go.Scatter(
            x=weeks,
            y=lower_percent,
            mode='lines',
            name='Lower rule curve',
            line=dict(dash='dash', color='red'),
            visible=True
        ))

        colors = px.colors.qualitative.Plotly

        for s in self.bv.scenarios:
            visible = True if s == 0 else False
            color = colors[s % len(colors)]
            stock_percent = self.bv.trajectories[s] / self.bv.reservoir_capacity * 100
            fig.add_trace(go.Scatter(
                x=weeks,
                y=stock_percent,
                mode='lines',
                name=f'MC {s + 1}',
                line=dict(color=color),
                visible=visible
            ))

        n_scenarios = len(self.bv.scenarios)
        n_shared_guides = 2 
        buttons = []

        area = getattr(self.bv.cost_function, 'name_area', None)
        if area is None:
            area = getattr(self.bv, 'area', None)
        area_str = f" - Zone : {area}" if area else ""

        for s in self.bv.scenarios:
            visibility = [True] * n_shared_guides + [False] * n_scenarios
            visibility[n_shared_guides + s] = True
            buttons.append(dict(
                label=f"MC {s + 1}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title.text": f"Trajectoire du stock - MC {s + 1}{area_str}"}
                ]
            ))

        visibility_all = [True] * (n_shared_guides + n_scenarios)
        buttons.append(dict(
            label=f"all MC",
            method="update",
            args=[
                {"visible": visibility_all},
                {"title.text": f"Trajectoires du stock - All MC{area_str}"}
            ]
        ))

        fig.update_layout(
            font=dict(family="Cambria", size=14),
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=1.1,
                y=1.15,
                showactive=True
            )],
            title=dict(text=f"Trajectoire du stock - MC 1{area_str}", font=dict(family="Cambria", size=18)),
            xaxis=dict(
                title="Semaine",
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                dtick=1,
                zeroline=False
            ),
            yaxis=dict(
                title="Stock (%)",
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                tick0=0,
                dtick=5,
                zeroline=False
            ),
            legend=dict(x=0, y=-0.2, orientation="h"),
            showlegend=False
        )

        fig.show()
        if not isinstance(self.bv.export_dir, str) or not self.bv.export_dir:
            raise ValueError("export_dir must be a non-empty string before saving the plot.")
        html_path = os.path.join(self.bv.export_dir, "trajectories_plot.html")
        fig.write_html(html_path)
        print(f"Interactive plot saved at: {html_path}")

    def plot_all_trajectories_pyplot(self) -> None:
        weeks = np.arange(1, self.bv.nb_weeks + 1)
        n_scenarios = len(self.bv.scenarios)
        color_palette = plt.cm.get_cmap('tab20', n_scenarios)

        plt.figure(figsize=(14, 7))

        plt.plot(
            weeks,
            self.bv.upper_rule_curve[:self.bv.nb_weeks] / self.bv.reservoir_capacity * 100,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Courbe guide supérieure"
        )

        plt.plot(
            weeks,
            self.bv.bottom_rule_curve[:self.bv.nb_weeks] / self.bv.reservoir_capacity * 100,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Courbe guide inférieure"
        )

        for s in range(n_scenarios):
            stock = self.bv.trajectories[s]
            plt.plot(
                weeks,
                stock / self.bv.reservoir_capacity * 100,
                color=color_palette(s)
            )

        plt.xlabel("Semaine", fontsize=14)
        plt.ylabel("Stock (%)", fontsize=14)
        area = getattr(self.bv.cost_function, 'name_area', None)
        if area is None:
            area = getattr(self.bv, 'area', None)
        area_str = f" - Zone : {area}" if area else ""
        plt.title(f"Trajectoires de stock avec courbes guides - Tous scénarios (α={self.bv.alpha}){area_str}", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=12, loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_adjusted_rule_curves(self) -> None:
        plt.figure(figsize=(16, 6))

        plt.plot(self.bv.final_lower_rule_curve / self.bv.reservoir_capacity * 100, label="Inférieure ajustée", color="blue", linewidth=2)
        plt.plot(self.bv.hourly_lower_rule_curve / self.bv.reservoir_capacity * 100, label="Inférieure interpolée", color="cyan", linestyle="--", linewidth=1.5)

        plt.plot(self.bv.final_upper_rule_curve / self.bv.reservoir_capacity * 100, label="Supérieure ajustée", color="darkred", linewidth=2)
        plt.plot(self.bv.hourly_upper_rule_curve / self.bv.reservoir_capacity * 100, label="Supérieure interpolée", color="orange", linestyle="--", linewidth=1.5)

        plt.xlabel("Heure de l'année")
        plt.ylabel("Stock (%)")
        area = getattr(self.bv.cost_function, 'name_area', None)
        if area is None:
            area = getattr(self.bv, 'area', None)
        area_str = f" - Zone : {area}" if area else ""
        plt.title(f"Courbes guides horaires : ajustées vs interpolées{area_str}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class Exporter:
    def __init__(self, cost_function: Proxy, bv: BellmanValuesProxy):
        self.cost_function = cost_function
        self.bv = bv
        self.reservoir = cost_function.reservoir
        self.export_dir = self.bv.export_dir
        self.initial_level = self.reservoir.initial_level
        self.nb_weeks = cost_function.nb_weeks
        self.scenarios = cost_function.scenarios

    def export_controls(self,filename:str="controls.csv") -> None:
        data = []
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                u = self.bv.optimal_controls[s, w]
                t = self.bv.optimal_turb[s,w]
                p = self.bv.optimal_pump[s,w]
                data.append({
                    "area": self.cost_function.name_area,
                    "u": u,
                    "turb" : t,
                    "pump": p,
                    "week": w + 1,
                    "mcYear": s + 1,
                    "sim": "u_0"
                })

        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path,index=False)
        print(f"Control trajectories export succeeded : {output_path}")

    def export_bellman_values(self, filename: str = "bellman_values.csv") -> None:
        data = []
        for w in range(self.nb_weeks):
            for c_index, c in enumerate(range(0, 101, 2)):
                stock_percent = c  # stock exprimé en %
                for s in self.scenarios:
                    value = self.bv.bv[w, c_index, s]
                    data.append({
                        "week": w + 1,
                        "stock_percent": stock_percent,
                        "mcYear": s + 1,
                        "bellman_value": value
                    })

        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Bellman values export succeeded: {output_path}")

    def export_trajectories(self,filename:str="trajectories.csv") ->None:
        data = []

        for s in self.scenarios:
            for w in range(self.nb_weeks):
                hlevel =self.bv.trajectories[s,w]
                data.append({
                    "area": self.cost_function.name_area,
                    "hlevel": hlevel,
                    "week": w + 1,
                    "mcYear": s + 1,
                    "sim": "u_0"
                })
        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path,index=False)
        print(f"Stock trajectories export succeeded : {output_path}")
    

class ModifyAntaresStudy:
    def __init__(self, cost_function:Proxy, bv:BellmanValuesProxy):
        self.cost_function = cost_function
        self.bv = bv
        self.reservoir = cost_function.reservoir
        self.dir_study = cost_function.dir_study
        self.area = cost_function.name_area

    def overwrite_inflows(self) -> None:
        inflow_path = os.path.join(self.dir_study, "input", "hydro", "series", self.area, "mod.txt")
        inflow_backup_path = inflow_path.replace(".txt", "_old.txt")

        if os.path.exists(inflow_path):
            os.rename(inflow_path, inflow_backup_path)

        inflows = np.loadtxt(inflow_backup_path)
        inflows[:, :] = 0

        np.savetxt(inflow_path, inflows, fmt="%.8f", delimiter="\t")

    def overwrite_hydro_ini_file(self) -> None:
        hydro_ini_path = os.path.join(self.dir_study, "input", "hydro", "hydro.ini")
        config = ConfigParser()
        config.read(hydro_ini_path)
        config["reservoir"][f"{self.area}"] = "false"
        with open(hydro_ini_path, "w") as configfile:
            config.write(configfile)

    def create_st_cluster(self) -> None:
        contenu = f"""[lt_stock_proxy_{self.area}]
name = lt_stock_proxy_{self.area}
group = PSP_open
reservoircapacity = {self.bv.reservoir_capacity}
initiallevel = 0.500000
injectionnominalcapacity = {np.max(self.reservoir.max_daily_pumping / 24)}
withdrawalnominalcapacity = {np.max(self.reservoir.max_daily_generating / 24)}
efficiency = {self.cost_function.efficiency}
efficiencywithdrawal = {self.cost_function.turb_efficiency}
initialleveloptim = false
enabled = true
"""
        list_ini_path = os.path.join(self.dir_study, "input", "st-storage", "clusters", self.area, "list.ini")
        os.makedirs(os.path.dirname(list_ini_path), exist_ok=True)
        with open(list_ini_path, "a") as f:
            f.write(contenu)

    def create_pmax_file(self) -> None:
        pmax_injection_hourly = np.repeat(self.reservoir.max_daily_pumping, 24) / 24
        pmax_withdrawal_hourly = np.repeat(self.reservoir.max_daily_generating, 24) / 24

        modulation_injection = np.clip(
            pmax_injection_hourly / np.max(self.reservoir.max_daily_pumping / 24), 0, 1)
        modulation_withdrawal = np.clip(
            pmax_withdrawal_hourly / np.max(self.reservoir.max_daily_generating / 24), 0, 1)

        modulation_injection = np.concatenate([modulation_injection, np.full(24, modulation_injection[-1])])
        modulation_withdrawal = np.concatenate([modulation_withdrawal, np.full(24, modulation_withdrawal[-1])])

        folder_path = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.area, f"lt_stock_proxy_{self.area}"
        )
        os.makedirs(folder_path, exist_ok=True)
        np.savetxt(os.path.join(folder_path, "PMAX-injection.txt"), modulation_injection, fmt="%.8f")
        np.savetxt(os.path.join(folder_path, "PMAX-withdrawal.txt"), modulation_withdrawal, fmt="%.8f")

    def create_rule_curve_file(self) -> None:
        folder_path = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.area, f"lt_stock_proxy_{self.area}"
        )
        os.makedirs(folder_path, exist_ok=True)

        lower_arr = np.clip(self.bv.final_lower_rule_curve / self.bv.reservoir_capacity, 0, 1)
        lower_arr = np.floor(lower_arr * 1e6) / 1e6

        upper_arr = np.clip(self.bv.final_upper_rule_curve / self.bv.reservoir_capacity, 0, 1)
        upper_arr = np.ceil(upper_arr * 1e6) / 1e6

        np.savetxt(os.path.join(folder_path, "lower-rule-curve.txt"), lower_arr, fmt="%.6f")
        np.savetxt(os.path.join(folder_path, "upper-rule-curve.txt"), upper_arr, fmt="%.6f")

    def modify_scenario_builder(self) -> None:
        config = ConfigParser(strict=False)
        config.read(os.path.join(self.dir_study, "settings", "generaldata.ini"))
        nbyears = int(config["general"]["nbyears"])

        lines = []
        for mc in range(nbyears):
            trajectory = (mc % self.cost_function.net_load.shape[1]) + 1
            lines.append(f"\nsts,{self.area},{mc},lt_stock_proxy_{self.area}={trajectory}")

        path = os.path.join(self.dir_study, "settings", "scenariobuilder.dat")
        with open(path, "a") as f:
            f.writelines(lines)

    def create_inflows_sts(self) -> None:
        balance = np.zeros((168 * self.cost_function.nb_weeks, len(self.cost_function.scenarios)))
        for s in self.cost_function.scenarios:
            for w in range(self.cost_function.nb_weeks):
                hour_start = w * 168
                if w == 0:
                    hlevel_start = self.cost_function.reservoir.initial_level
                else:
                    hlevel_start = self.bv.trajectories[s, w - 1]
                hlevel_end = self.bv.trajectories[s, w]
                balance[hour_start, s] = hlevel_start - self.bv.reservoir_capacity / 2
                balance[hour_start + 167, s] = self.bv.reservoir_capacity / 2 - hlevel_end
                balance[hour_start:hour_start + 168, s] += (
                    np.repeat(self.bv.daily_inflow[w * 7:(w + 1) * 7, s], 24) / 24)

        balance = np.vstack([balance, np.zeros((24, len(self.cost_function.scenarios)))])
        path = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.area,
            f"lt_stock_proxy_{self.area}", "inflows.txt"
        )
        np.savetxt(path, balance, fmt="%.8f", delimiter="\t")

    def apply_all(self) -> None:
        self.overwrite_inflows()
        self.overwrite_hydro_ini_file()
        self.create_st_cluster()
        self.create_pmax_file()
        self.create_rule_curve_file()
        self.modify_scenario_builder()
        self.create_inflows_sts()
        print(f"✅ Antares study modified for area '{self.area}'")


class UndoAntaresModifications:
    def __init__(self, dir_study: str, area: str):
        self.dir_study = dir_study
        self.area = area

    def restore_inflows(self) -> None:
        inflow_path = os.path.join(
            self.dir_study, "input", "hydro", "series", self.area, "mod.txt"
        )
        inflow_backup_path = inflow_path.replace(".txt", "_old.txt")
        if os.path.exists(inflow_backup_path):
            if os.path.exists(inflow_path):
                os.remove(inflow_path)
            os.rename(inflow_backup_path, inflow_path)
            print("✔ inflows restored.")
        else:
            print("⚠ inflow backup not found. Nothing restored.")

    def restore_hydro_ini(self) -> None:
        path = os.path.join(self.dir_study, "input", "hydro", "hydro.ini")
        config = ConfigParser()
        config.read(path)
        if "reservoir" in config and f"{self.area}" in config["reservoir"]:
            config["reservoir"][f"{self.area}"] = "true"
            with open(path, "w") as configfile:
                config.write(configfile)
            print("✔ hydro.ini restored.")
        else:
            print(f"⚠ hydro.ini unchanged: missing [reservoir]/{self.area} section.")

    def remove_st_cluster_section(self) -> None:
        list_ini_path = os.path.join(
            self.dir_study, "input", "st-storage", "clusters", self.area, "list.ini"
        )
        if not os.path.exists(list_ini_path):
            print("⚠ list.ini not found.")
            return

        with open(list_ini_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        skip = False
        for line in lines:
            if line.strip().startswith(f"[lt_stock_proxy_{self.area}]"):
                skip = True
                continue
            elif skip and line.strip().startswith("["):
                skip = False
            if not skip:
                new_lines.append(line)

        with open(list_ini_path, "w") as f:
            f.writelines(new_lines)

        print("✔ st-cluster section removed.")

    def remove_st_series_folder(self) -> None:
        folder = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.area,
            f"lt_stock_proxy_{self.area}"
        )
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print("✔ st-series folder removed.")
        else:
            print("⚠ st-series folder not found.")

    def clean_scenariobuilder(self) -> None:
        path = os.path.join(self.dir_study, "settings", "scenariobuilder.dat")
        if not os.path.exists(path):
            print("⚠ scenariobuilder.dat not found.")
            return

        with open(path, "r") as f:
            lines = f.readlines()

        filtered = [line for line in lines if f"lt_stock_proxy_{self.area}" not in line]

        with open(path, "w") as f:
            f.writelines(filtered)

        print("✔ scenariobuilder cleaned.")

    def undo_all(self) -> None:
        print(f"\n🔁 Restoring Antares study for area: {self.area}")
        self.restore_inflows()
        self.restore_hydro_ini()
        self.remove_st_cluster_section()
        self.remove_st_series_folder()
        self.clean_scenariobuilder()
        print(f"✅ Restoration complete for area '{self.area}'\n")


class Launch:
    def __init__(self, dir_study: str, area: str, MC_years: int, alpha: float, coeff_cost: int, enable_logging: bool, global_export_dir: str | None = None):
        self.dir_study = dir_study
        self.name_area = area
        self.nb_scenarios = MC_years
        self.alpha = alpha
        self.coeff = coeff_cost
        self.enable_logging = enable_logging
        self.global_export_dir = global_export_dir

    def run(self, actions: list[str] | None = None) -> None:
        # Si uniquement undo_modifications, ne crée aucun dossier d'export ni objet inutile
        if actions is not None and len(actions) == 1 and actions[0] == "undo_modifications":
            UndoAntaresModifications(self.dir_study, self.name_area).undo_all()
            return

        if self.global_export_dir is None:
            raise ValueError("A per-area export_dir must be provided via global_export_dir in all cases except undo_modifications.")
        export_dir = os.path.join(self.global_export_dir, self.name_area)
        os.makedirs(export_dir, exist_ok=True)

        start = time.time()
        self.proxy = Proxy(dir_study=self.dir_study, name_area=self.name_area, nb_scenarios=self.nb_scenarios)
        self.bv = BellmanValuesProxy(self.proxy, alpha=self.alpha, coeff=self.coeff, enable_logging=self.enable_logging, export_dir=export_dir)
        end = time.time()
        print(f"Stage cost functions, Bellman values and trajectories computed in : {end-start} s.")

        self.plotter = Plotter(self.bv)
        self.exporter = Exporter(self.proxy, self.bv)
        self.modifier = ModifyAntaresStudy(self.proxy, self.bv)

        if actions is None:
            actions = ["modify_antares_data"]
        if actions == ["all"]:
            actions = [
                "export_bellman_values",
                "export_controls",
                "export_trajectories",
                "plot_trajectories",
                "plot_usage_values",
                "plot_usage_values_heatmap",
                "plot_all_trajectories_pyplot",
                "plot_adjusted_rule_curves",
                "modify_antares_data",
            ]

        for action in actions:
            if action == "export_bellman_values":
                self.exporter.export_bellman_values()
            elif action == "export_controls":
                self.exporter.export_controls()
            elif action == "export_trajectories":
                self.exporter.export_trajectories()
            elif action == "plot_trajectories":
                self.plotter.plot_trajectories()
            elif action == "plot_usage_values":
                self.plotter.plot_usage_values()
            elif action == "plot_usage_values_heatmap":
                self.plotter.plot_usage_values_heatmap()
            elif action == "plot_all_trajectories_pyplot":
                self.plotter.plot_all_trajectories_pyplot()
            elif action == "plot_adjusted_rule_curves":
                self.plotter.plot_adjusted_rule_curves()
            elif action == "modify_antares_data":
                self.modifier.apply_all()
            elif action == "undo_modifications":
                UndoAntaresModifications(self.dir_study, self.name_area).undo_all()
            else:
                print(f"Unknown action: {action}")

def run_for_area(area: str, dir_study: str, MC_years: int, alpha: float, coeff_cost: int, enable_logging: bool, actions: list[str] | None = None, global_export_dir: str | None = None) -> None:
    # Si uniquement undo_modifications, ne passe pas d'export dir
    if actions is not None and len(actions) == 1 and actions[0] == "undo_modifications":
        Launch(
            dir_study=dir_study,
            area=area,
            MC_years=MC_years,
            alpha=alpha,
            coeff_cost=coeff_cost,
            enable_logging=enable_logging,
            global_export_dir=None
        ).run(actions=actions)
    else:
        Launch(
            dir_study=dir_study,
            area=area,
            MC_years=MC_years,
            alpha=alpha,
            coeff_cost=coeff_cost,
            enable_logging=enable_logging,
            global_export_dir=global_export_dir
        ).run(actions=actions)

def main() -> None:
    parser = argparse.ArgumentParser(description="Lancer la génération des trajectoires pour plusieurs zones.")
    parser.add_argument("--dir_study", type=str, required=True, help="Répertoire d'entrée contenant les données.")
    parser.add_argument("--area", type=str, nargs='+', required=True, help="Liste des zones d'étude (séparées par un espace).")
    parser.add_argument("--MC_years", type=int, required=False, default=200, help="Nombre d'années Monte-Carlo à simuler.")
    parser.add_argument("--alpha", type=float, required=False,default=2, help="Coefficient alpha de la fonction de coût.")
    parser.add_argument("--coeff_cost", type=int, required=False,default=1000000000, help="Facteur d'échelle pour la fonction de coût.")
    parser.add_argument("--enable_logging", type=bool, default=False, help="Activer les logs.")
    parser.add_argument("--actions", type=str, nargs='*', default=None, help="Liste des actions à effectuer (ex: export_bellman_values, plot_trajectories, modify_antares_data, undo_modifications, etc.)")

    args = parser.parse_args()

    # Si uniquement undo_modifications, ne crée aucun dossier d'export
    if args.actions is not None and len(args.actions) == 1 and args.actions[0] == "undo_modifications":
        for area in args.area:
            run_for_area(
                area,
                args.dir_study,
                args.MC_years,
                args.alpha,
                args.coeff_cost,
                args.enable_logging,
                args.actions,
                None
            )
        return

    # Sinon, création du dossier global d'export (daté)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_export_dir = os.path.join(args.dir_study, f"exports_LT_storage_trajectories_{date_str}")
    os.makedirs(global_export_dir, exist_ok=True)

    if len(args.area) == 1:
        run_for_area(
            args.area[0],
            args.dir_study,
            args.MC_years,
            args.alpha,
            args.coeff_cost,
            args.enable_logging,
            args.actions,
            global_export_dir
        )
    else:
        with ProcessPoolExecutor() as executor:
            future_to_area = {
                executor.submit(
                    run_for_area,
                    area,
                    args.dir_study,
                    args.MC_years,
                    args.alpha,
                    args.coeff_cost,
                    args.enable_logging,
                    args.actions,
                    global_export_dir
                ): area for area in args.area
            }
            for future in as_completed(future_to_area):
                area = future_to_area[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Erreur pour la zone {area} : {e}")

if __name__ == "__main__":
    main()
