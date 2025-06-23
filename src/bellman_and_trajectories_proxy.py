from proxy_stage_cost_function import Proxy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rcParams
from scipy.interpolate import interp1d
import time
import plotly.graph_objects as go
import plotly.express as px
from type_definition import Callable
import pandas as pd
import os
import logging
from logging.handlers import RotatingFileHandler
import argparse

rcParams['font.family'] = 'Cambria'

class BellmanValuesHydro:
    def __init__(self, cost_function: Proxy,alpha:float,coeff:float,enable_logging: bool):
        
        self.cost_function=cost_function
        self.reservoir = cost_function.reservoir
        self.reservoir_capacity = self.reservoir.capacity
        self.initial_level=self.reservoir.initial_level

        self.bottom_rule_curve=cost_function.reservoir.bottom_rule_curve
        self.upper_rule_curve=cost_function.reservoir.upper_rule_curve
        self.daily_bottom_rule_curve=self.cost_function.reservoir.daily_bottom_rule_curve
        self.daily_upper_rule_curve=self.cost_function.reservoir.daily_upper_rule_curve

        self.inflow=cost_function.reservoir.inflow
        self.daily_inflow=cost_function.reservoir.daily_inflow
        
        self.nb_weeks=self.cost_function.nb_weeks
        self.scenarios=self.cost_function.scenarios
        
        self.cost_functions_turb_and_pump=self.cost_function.compute_stage_cost_functions(alpha,coeff)
        self.cost_functions=self.cost_functions_turb_and_pump[:,:,0]
        self.turb_functions=self.cost_functions_turb_and_pump[:,:,1]
        self.pump_functions=self.cost_functions_turb_and_pump[:,:,2]
        self.bv=np.zeros((self.nb_weeks,51,len(self.scenarios)))
        self.mean_bv=np.zeros((self.nb_weeks,51))
        
        self.export_dir = self.make_unique_export_dir()
        self.logger = self.setup_logger() if enable_logging else self.get_null_logger()
        self.compute_bellman_values()
        self.compute_usage_values()
        self.compute_trajectories()
        
    def log_section_title(self, title: str)->None:
        self.logger.debug("\n" + "=" * 70)
        self.logger.debug(f"{title.center(70)}")
        self.logger.debug("=" * 70 + "\n")

    def get_null_logger(self) -> logging.Logger:
        null_logger = logging.getLogger("NullLogger")
        null_logger.setLevel(logging.CRITICAL + 1)  # Ignore tout message
        if not null_logger.hasHandlers():
            null_logger.addHandler(logging.NullHandler())
        return null_logger
    
    def setup_logger(self, log_filename: str = "log", max_bytes: int = 10_000_000, backup_count: int = 5)->logging.Logger:
        logger = logging.getLogger("BellmanLogger")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # évite les doublons

        # Supprimer les handlers existants si redéfini plusieurs fois
        if logger.hasHandlers():
            logger.handlers.clear()

        log_path = os.path.join(self.export_dir, log_filename)
        handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count,encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

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
        self.log_section_title("CALCUL DES VALEURS DE BELLMAN")
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
        self.log_section_title("CALCUL DES TRAJECTOIRES")
        self.trajectories = np.zeros((len(self.scenarios),self.nb_weeks))
        self.optimal_controls = np.zeros((len(self.scenarios),self.nb_weeks))
        self.optimal_turb = np.zeros((len(self.scenarios),self.nb_weeks))
        self.optimal_pump = np.zeros((len(self.scenarios),self.nb_weeks))
        for s in self.scenarios:
            previous_stock=self.initial_level
            for w in range(self.nb_weeks):
                self.logger.debug("\n" + "-" * 60)
                self.logger.debug(f"---- Semaine {w+1}, scénario {s+1} ----")
                self.logger.debug("-" * 60)
                self.logger.debug(f"Stock précédent : {previous_stock:.2f} MWh")

                inflows_for_week = self.inflow[w,s]
                self.logger.debug(f"Inflow : {inflows_for_week:.2f} MWh")
                cost_function = self.cost_functions[w,s]
                penalty_function=self.penalty_rule_curves(w)
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
                    new_stock = previous_stock - control+inflows_for_week
                    cost = cost_function(control)
                    future_value = future_bellman_function(new_stock)
                    penalty=penalty_function(new_stock)
                    total_value = cost + future_value +penalty
                    self.logger.debug(f"Test contrôle (libre): {control:.2f}, stock suivant: {new_stock:.2f}, coût: {cost:.2f}, future_value: {future_value:.2f}, pénalité: {penalty:.2f}, total: {total_value:.2f}")

                    if total_value < best_value:
                        best_value = total_value
                        best_new_stock = new_stock
                        optimal_control=control
                        self.logger.debug(f"→ Nouveau meilleur contrôle retenu (libre) : {control:.2f}, stock suivant: {new_stock:.2f}, total: {total_value:.2f}")

                       
                for c_new in range(0,101,2):
                    max_week_turb=np.sum(self.cost_function.max_daily_generating[w * 7:(w + 1) * 7])
                    max_week_pump=np.sum(self.cost_function.max_daily_pumping[w * 7:(w + 1) * 7])
                    week_energy_var=previous_stock-c_new/100 * self.reservoir_capacity
                    if week_energy_var<-max_week_pump*self.cost_function.efficiency or week_energy_var>max_week_turb*self.cost_function.turb_efficiency:
                            continue
                    control=previous_stock-c_new/100 * self.reservoir_capacity
                    new_stock=previous_stock-control+inflows_for_week
                    cost = cost_function(control)
                    future_value = future_bellman_function(new_stock)
                    penalty=penalty_function(new_stock)
                    total_value = cost + future_value + penalty
                    self.logger.debug(f"Test contrôle (forcé): {control:.2f}, stock suivant: {new_stock:.2f}, coût: {cost:.2f}, future_value: {future_value:.2f}, pénalité: {penalty:.2f}, total: {total_value:.2f}")

                    if total_value < best_value:
                        best_value = total_value
                        best_new_stock = new_stock
                        optimal_control=control
                        self.logger.debug(f"→ Nouveau meilleur contrôle retenu (forcé) : {control:.2f}, stock suivant: {new_stock:.2f}, total: {total_value:.2f}")

                self.logger.debug(f"==> Stock retenu pour la semaine {w+1} : {best_new_stock:.2f} MWh\n")

                if best_new_stock is not None:
                    self.trajectories[s,w] =best_new_stock
                    self.optimal_controls[s,w] = optimal_control
                    self.optimal_turb[s,w] = self.turb_functions[w,s](optimal_control)
                    self.optimal_pump[s,w] = self.pump_functions[w,s](optimal_control)
                    lower_bound=self.bottom_rule_curve[w]
                    upper_bound=self.upper_rule_curve[w]
                    if not (lower_bound <= best_new_stock <= upper_bound):
                        print(
                            f"⚠️ Stock hors courbes guides - Semaine {w+1}, scénario {s+1} : "
                            f"{best_new_stock:.2f} ∉ [{lower_bound:.2f}, {upper_bound:.2f}]"
                        )
                    previous_stock=best_new_stock
                else:
                    self.trajectories[s,w]=None
                    self.optimal_controls[s,w]=None
                    self.optimal_pump[s,w]=None
                    self.optimal_turb[s,w]=None
                
                # previous_stock=best_new_stock
                

    # affichages

    def plot_bellman_value(self, week_index: int) -> None:
        if week_index < 0 or week_index >= self.nb_weeks:
            raise ValueError(f"Semaine invalide : {week_index}. Doit être entre 0 et {self.nb_weeks - 1}.")

        stock_levels = np.linspace(0, 100, 51)
        bellman_values = self.mean_bv[week_index,:]

        plt.figure(figsize=(10, 5))
        plt.plot(stock_levels, bellman_values, label=f"Semaine {week_index + 1}", color='tab:blue')

        plt.xlabel("Stock (%)")
        plt.ylabel("Valeur de Bellman")
        plt.title(f"Valeur de Bellman en fonction du stock - Semaine {week_index + 1}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_usage_values(self) -> None:
        
        stock_levels = np.linspace(2, 100, 50) 
        plt.figure(figsize=(12, 6))

        for w in range(self.nb_weeks):
            plt.plot(
                stock_levels, 
                self.usage_values[w],
                label=f"S {w+1}"
            )

        plt.xlabel('Stock (%)')
        plt.ylabel('Valeur d\'usage (MWh)')
        plt.title('Valeurs d\'usage en fonction du stock')
        plt.legend(
        loc='lower right',
        bbox_to_anchor=(1, -0.15),
        ncol=6  # Ajustez selon la place disponible
    )
        plt.tight_layout(rect=(0, 0.1, 1, 1))  # Laisse de la place en bas pour la légende
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_usage_values_heatmap(self) -> None:

        fig, ax = plt.subplots(figsize=(14, 6))

        norm = colors.Normalize(np.min(self.usage_values[:-1]), np.max(self.usage_values[:-1]))
        # norm = colors.Normalize(vmin=-30, vmax=0)


        im = ax.imshow(
        self.usage_values[:-1].T,
        aspect='auto',
        origin='lower',
        cmap='nipy_spectral',
        extent=(1, 52, 2, 100),
        norm=norm,
        interpolation='bilinear'  # lissage
    )

        cbar = fig.colorbar(im, ax=ax, ticks=np.linspace(np.min(self.usage_values[:-1]), np.max(self.usage_values[:-1]), 10))
        # cbar = fig.colorbar(im, ax=ax, ticks=np.linspace(-30, 0, 10))
        cbar.set_label("Valeur d’usage (MWh)")

        ax.set_xlabel("Semaine")
        ax.set_ylabel("Stock (%)")
        ax.set_title("Nappes de valeurs d’usage (Bellman)")

        plt.grid(False)
        plt.tight_layout()
        plt.show()


    def plot_trajectories(self) -> None:
        fig = go.Figure()
        weeks = list(range(1, self.nb_weeks + 2))

        upper_percent = self.upper_rule_curve / self.reservoir_capacity * 100
        fig.add_trace(go.Scatter(
            x=weeks,
            y=upper_percent,
            mode='lines',
            name='Upper rule curve',
            line=dict(dash='dash', color='green'),
            visible=True
        ))

        lower_percent = self.bottom_rule_curve / self.reservoir_capacity * 100
        fig.add_trace(go.Scatter(
            x=weeks,
            y=lower_percent,
            mode='lines',
            name='Lower rule curve',
            line=dict(dash='dash', color='red'),
            visible=True
        ))

        colors = px.colors.qualitative.Plotly

        for s in self.scenarios:
            visible = True if s == 0 else False
            color = colors[s % len(colors)]
            stock_percent = self.trajectories[s] / self.reservoir_capacity * 100
            fig.add_trace(go.Scatter(
                x=weeks,
                y=stock_percent,
                mode='lines',
                name=f'MC {s + 1}',
                line=dict(color=color),
                visible=visible
            ))

        n_scenarios = len(self.scenarios)
        n_shared_guides = 2 
        buttons = []

        for s in self.scenarios:
            visibility = [True] * n_shared_guides + [False] * n_scenarios
            visibility[n_shared_guides + s] = True
            buttons.append(dict(
                label=f"MC {s + 1}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title.text": f"Trajectoire du stock - MC {s + 1}"}
                ]
            ))

        visibility_all = [True] * (n_shared_guides + n_scenarios)
        buttons.append(dict(
            label=f"all MC",
            method="update",
            args=[
                {"visible": visibility_all},
                {"title.text": "Trajectoires du stock - All MC"}
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
            title=dict(text="Trajectoire du stock - MC 1", font=dict(family="Cambria", size=18)),
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
        html_path = os.path.join(self.export_dir, "trajectories_plot.html")
        fig.write_html(html_path)
        print(f"Interactive plot saved at: {html_path}")


    # exports
    def make_unique_export_dir(self) -> str:
        base_path = os.path.join(self.cost_function.dir_study, "exports_hydro_trajectories")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            return base_path

        i = 1
        while True:
            new_path = f"{base_path}_{i}"
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                return new_path
            i += 1

    def export_controls(self,filename:str="controls.csv") -> None:
        data = []
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                u = self.optimal_controls[s, w]
                t = self.optimal_turb[s,w]
                p = self.optimal_pump[s,w]
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

    def export_sts_inflows(self, filename: str = "sts_inflows.txt")->None:
        balance=np.zeros((168*self.nb_weeks,len(self.scenarios)))
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                hour_start=w*168
                if w==0:
                    hlevel_start=self.initial_level
                else:
                    hlevel_start=self.trajectories[s,w-1]
                hlevel_end=self.trajectories[s,w]
                balance[hour_start,s]=hlevel_start-self.reservoir_capacity/2
                balance[hour_start+167,s]=self.reservoir_capacity/2-hlevel_end
                balance[hour_start:hour_start+168,s]+=np.repeat(self.daily_inflow[w*7:(w+1)*7,s],24)/24
        
        balance=np.vstack([balance,np.zeros((24,len(self.scenarios)))])

        filepath = os.path.join(self.export_dir, filename)

        np.savetxt(filepath, balance, fmt="%.6f", delimiter="\t")

        print(f"Balance export succedeed : {filepath}")
    
    def export_bellman_values(self, filename: str = "bellman_values.csv") -> None:
        data = []
        for w in range(self.nb_weeks):
            for c_index, c in enumerate(range(0, 101, 2)):
                stock_percent = c  # stock exprimé en %
                for s in self.scenarios:
                    value = self.bv[w, c_index, s]
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


    def export_hourly_upper_rule_curves(self, filename: str = "hourly_upper_rule_curve.txt") -> None:
        n_days = 365
        daily_indices = np.arange(n_days)
        interpolator = interp1d(daily_indices, self.daily_upper_rule_curve/self.reservoir_capacity, kind='linear')

        hourly_indices = np.linspace(0, n_days - 1, n_days * 24)


        hourly_upper_rule_curve = interpolator(hourly_indices)


        filepath = os.path.join(self.export_dir, filename)

        np.savetxt(filepath, hourly_upper_rule_curve, fmt="%.6f")

        print(f"Hourly upper rule curve export succeeded: {filepath}")

    def export_hourly_bottom_rule_curves(self, filename: str = "hourly_bottom_rule_curve.txt") -> None:
        n_days = 365

        daily_indices = np.arange(n_days)
        interpolator = interp1d(daily_indices, self.daily_bottom_rule_curve/self.reservoir_capacity, kind='linear')

        hourly_indices = np.linspace(0, n_days - 1, n_days * 24)


        hourly_upper_rule_curve = interpolator(hourly_indices)


        filepath = os.path.join(self.export_dir, filename)

        np.savetxt(filepath, hourly_upper_rule_curve, fmt="%.6f")

        print(f"Hourly bottom rule curve export succeeded: {filepath}")


    def export_trajectories(self,filename:str="trajectories.csv") ->None:
        data = []

        for s in self.scenarios:
            for w in range(self.nb_weeks):
                hlevel =self.trajectories[s,w]
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
    
# launcher
class Launch:
    def __init__(self, dir_study:str, area :str, MC_years : int, alpha :float, coeff_cost :int, enable_logging:bool):
        self.dir_study=dir_study
        self.name_area=area
        self.nb_scenarios=MC_years
        self.alpha=alpha
        self.coeff=coeff_cost
        self.enable_logging=enable_logging

    def run(self)->None:
        start=time.time()
        self.proxy=Proxy(dir_study=self.dir_study,name_area=self.name_area,nb_scenarios=self.nb_scenarios)
        self.bv=BellmanValuesHydro(self.proxy,alpha=self.alpha,coeff=self.coeff,enable_logging=self.enable_logging)
        end=time.time()
        print(f"Stage cost functions, Bellman values and trajectories computed in : {end-start} s.")
        self.bv.export_bellman_values()
        self.bv.plot_trajectories()
        self.bv.export_controls()
        self.bv.export_sts_inflows()
        self.bv.export_trajectories()
        # self.bv.export_hourly_upper_rule_curves()
        # self.bv.export_hourly_bottom_rule_curves()
        self.bv.plot_usage_values()
        self.bv.plot_usage_values_heatmap()

def main()->None:
    parser = argparse.ArgumentParser(description="Lancer la génération des trajectoires.")
    parser.add_argument("--dir_study", type=str, required=True, help="Répertoire d'entrée contenant les données.")
    parser.add_argument("--area", type=str, required=True, help="Nom de la zone d'étude.")
    parser.add_argument("--MC_years", type=int, required=True, help="Nombre d'années Monte-Carlo à simuler.")
    parser.add_argument("--alpha", type=float, required=True, help="Coefficient alpha de la fonction de coût.")
    parser.add_argument("--coeff_cost", type=int, required=True, help="Facteur d'échelle pour la fonction de coût. Si alpha vaut 2, coeff_cost vaut 1e9.")
    parser.add_argument("--enable_logging", type=bool, default=False, help="Activer les logs.")

    args = parser.parse_args()

    launcher = Launch(dir_study=args.dir_study,area=args.area,MC_years=args.MC_years,alpha=args.alpha,coeff_cost=args.coeff_cost,enable_logging=args.enable_logging)
    launcher.run()


if __name__ == "__main__":
    main()