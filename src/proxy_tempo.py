from read_antares_data import Reservoir,NetLoad
import numpy as np
from typing import Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from type_definition import Callable
from scipy.interpolate import interp1d
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import rcParams
rcParams['font.family'] = 'Cambria'


"""
To run the calculations, export daily control trajectories, stock trajectories, and generate plots, use:

python "path/to/script.py" --dir_study "path_to_study" --area "area_name" --actions <list_of_actions> [--cvar <float>] [--lower_rc_red <list_of_21_integers>]

- --dir_study : Path to the Antares study directory (required).
- --area      : Name of the area to process (required).
- --actions   : Space-separated list of commands to execute (required). Possible commands:
    * export_trajectories     : Export weekly stock trajectories CSV.
    * export_daily_controls   : Export daily control trajectories CSV.
    * export_usage_values     : Export weekly usage values CSV.
    * plot_trajectories       : Generate interactive stock trajectories plot (HTML).
    * plot_usage_values_red   : Plot usage values for red day stocks.
    * plot_usage_values_wr    : Plot usage values for white+red day stocks.

- --cvar      : (Optional) CVaR parameter controlling risk aversion (default=1.0).
- --lower_rc_red : (Optionnal) Lower rule curve for red stock (list of 21 integers, default=0*21, space-separated).

Examples:

python proxy_tempo.py --dir_study "/path/to/study" --area "MyArea" --actions export_trajectories plot_trajectories --cvar 1.5
python proxy_tempo.py --dir_study "/path/to/study" --area "MyArea" --actions plot_trajectories \
    --lower_rc_red 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
"""

class GainFunctionTempo:
    def __init__(self, net_load: NetLoad):
        """
        Initialize gain function with net load data and maximum weekly control.
        Extends net_load data by repeating July-August period for continuity.
        """
        self.net_load = net_load.compute_net_load()
        july_august = self.net_load[24:65*24, :]  # Select July-August data (days 1-64)
        self.net_load = np.concatenate((self.net_load, july_august), axis=0)
        self.nb_scenarios = net_load.nb_scenarios
        self.daily_net_load = self.net_load.reshape(365 + 64, 24, self.nb_scenarios).sum(axis=1)  # Sum hourly to daily load

    def gain_for_week_control_and_scenario(self, week_index: int, control: int, scenario: int, max_control:int) -> float:
        """
        Compute gain for a given week index, control level, and scenario.
        Gains are sum of the top 'control' daily net loads in the considered week,
        limited by max_control.
        """
        week_start = week_index * 7 + 2  #If year begins on monday 1st July
        week_end = week_start + 7

        daily_load_week = self.daily_net_load[week_start:week_end, scenario]
        # Sort and take top 'max_control' values in descending order
        daily_load_week = np.sort(daily_load_week[:max_control])[::-1]

        gain = np.sum(daily_load_week[:control])
        return gain


class BellmanValuesTempo:

    def __init__(self, gain_function: GainFunctionTempo,
                 capacity: int,
                 start_week: int,
                 end_week: int,
                 max_control:int,
                 CVar: float=1.0,
                 lower_rule_curve : np.ndarray = np.repeat(0,61),
                 upper_rule_curve : np.ndarray|None = None ):
        """
        Initialize Bellman value calculator over a period of weeks with given capacity and CVar.
        Prepares arrays to store Bellman values, their mean with CVar risk measure, and usage values.
        """
        self.max_control = max_control
        self.gain_function = gain_function
        self.start_week = start_week
        self.end_week = end_week

        self.capacity = capacity
        self.nb_scenarios = self.gain_function.nb_scenarios
        self.CVar = CVar

        # 61 = time horizon of Tempo calendar :
        # Tempo Red : 1st November -> 31st March
        # Tempo White : 1st September -> 31st August
        # Time horizon for resolution : 1st July (start of year in Antares) -> 30 June (61 weeks)
        self.mean_bv = np.zeros((61, self.capacity + 1))  # CVaR aggregated values over scenarios
        self.usage_values = np.zeros((61, self.capacity))
        self.lower_rule_curve = lower_rule_curve
        self.upper_rule_curve = upper_rule_curve if upper_rule_curve is not None else np.repeat(self.capacity,61)

        self.compute_bellman_values()
        self.compute_usage_values()

    def penalty(self,week : int) -> Callable:
        """
        Return a penalty function (linear interpolation) heavily penalizing states outside [0, capacity].
        """
        penalty = interp1d([
                self.lower_rule_curve[week] - 1,
                self.lower_rule_curve[week],
                self.upper_rule_curve[week],
                self.upper_rule_curve[week] + 1,
            ],
            [-1e9, 0, 0, -1e9],
            kind='linear', fill_value='extrapolate')
        # Alternative no penalty: penalty = lambda x: 0
        return penalty

    def compute_bellman_values(self) -> None:
        """
        Compute Bellman values backward in time with risk-adjusted expectations (CVaR).
        For each week, capacity level, and scenario, optimize over possible control actions.
        """
        self.mean_bv[self.end_week] = np.zeros(self.capacity + 1)
        for w in reversed(range(self.start_week, self.end_week)):
            penalty = self.penalty(w+1)
            for c in range(self.capacity + 1):
                bv = []
                for s in range(self.nb_scenarios):
                    best_value = -np.inf
                    for control in range(self.max_control + 1):

                        # Week W+1 selected because Bellman values are computed at end of week w
                        gain = self.gain_function.gain_for_week_control_and_scenario(w + 1, control, s,self.max_control)
                        future_value = self.mean_bv[w + 1, c - control]
                        total_value = gain + future_value + penalty(c - control)
                        if total_value > best_value:
                            best_value = total_value
                    bv.append(best_value)

                alpha = self.CVar
                bellman_values = bv
                sorted_bv = np.sort(bellman_values)
                cutoff_index = int((1 - alpha) * len(sorted_bv))
                self.mean_bv[w, c] = np.mean(sorted_bv[cutoff_index:])

    def compute_usage_values(self) -> None:
        """
        Compute marginal usage values as difference of Bellman mean values between successive capacity levels.
        """
        for w in range(self.start_week, self.end_week + 1):
            for c in range(1, self.capacity + 1):
                self.usage_values[w, c - 1] = self.mean_bv[w, c] - self.mean_bv[w, c - 1]


class TrajectoriesTempo:
    def __init__(self,
                 bv: BellmanValuesTempo,
                 stock_trajectories_red: Optional[np.ndarray] = None):
        """
        Compute optimal stock and control trajectories based on Bellman values,
        optionally constrained by 'red' stock trajectories.
        """
        self.bv = bv
        self.capacity = self.bv.capacity
        self.nb_scenarios = self.bv.nb_scenarios
        self.start_week = self.bv.start_week
        self.end_week = self.bv.end_week
        self.max_control = self.bv.max_control

        self.stock_trajectories_red = stock_trajectories_red

        self.control_trajectories,self.stock_trajectories=self.compute_trajectories()
        self.control_trajectories_white,self.stock_trajectories_white=self.compute_trajectories_white()

    def compute_trajectories(self) -> tuple:
        """
        Compute stock and control trajectories over all scenarios and weeks,
        considering possible constraints from reduced stock trajectories.
        """
        control_trajectories = np.zeros((self.nb_scenarios, 61))
        stock_trajectories = np.zeros((self.nb_scenarios, 61))

        for s in range(self.nb_scenarios):
            stock_trajectories[s, :self.start_week] = self.capacity
            for w in range(self.start_week, self.end_week + 1):
                penalty = self.bv.penalty(w)
                if self.start_week>0 and stock_trajectories[s, w - 1] == 0:
                    stock_trajectories[s, w:] = 0
                    break
                best_value = -np.inf
                best_control = None
                for c in range(self.max_control + 1):
                    gain = self.bv.gain_function.gain_for_week_control_and_scenario(w, c, s,self.max_control)
                    future_value = self.bv.mean_bv[w, int(stock_trajectories[s, w - 1]) - c]
                    total_value = gain + future_value + penalty(int(stock_trajectories[s, w - 1]) - c)
                    if total_value > best_value:
                        best_control = c
                        best_value = total_value
                if self.stock_trajectories_red is not None and best_control is not None:
                    # Do not allow negative stocks below red stocks
                    if stock_trajectories[s, w - 1] - best_control < self.stock_trajectories_red[s, w]:
                        best_control = stock_trajectories[s, w - 1] - self.stock_trajectories_red[s, w]
                    # Do not allow negative controls below red controls
                    if best_control < self.stock_trajectories_red[s, w - 1] - self.stock_trajectories_red[s, w]:
                        best_control = self.stock_trajectories_red[s, w - 1] - self.stock_trajectories_red[s, w]

                if best_control is not None:
                    stock_trajectories[s, w] = stock_trajectories[s, w - 1] - best_control
                    control_trajectories[s, w] = best_control
                else:
                    stock_trajectories[s, w] = np.nan
                    control_trajectories[s, w] = np.nan

        return control_trajectories, stock_trajectories

    def compute_trajectories_white(self) -> tuple:
        """
        Compute the 'white' stock and control trajectories as residuals after
        subtracting the 'red' stock trajectories, if provided.
        """
        stock_trajectories_white = np.zeros((self.nb_scenarios, 61))
        control_trajectories_white = np.zeros((self.nb_scenarios, 61))

        if self.stock_trajectories_red is not None:
            stock_trajectories_white = self.stock_trajectories - self.stock_trajectories_red
            for s in range(self.nb_scenarios):
                for w in range(1, 61):
                    control_trajectories_white[s, w] = stock_trajectories_white[s, w - 1] - stock_trajectories_white[s, w]

        return control_trajectories_white, stock_trajectories_white

    def control_trajectory_for_scenario(self, scenario: int) -> np.ndarray:
        """
        Return control trajectory for a given scenario.
        """
        return self.control_trajectories[scenario]

    def stock_trajectory_for_scenario(self, scenario: int) -> np.ndarray:
        """
        Return stock trajectory for a given scenario.
        """
        return self.stock_trajectories[scenario]

    def control_trajectory_for_scenario_white(self, scenario: int) -> np.ndarray:
        """
        Return white control trajectory for a given scenario, or empty if none.
        """
        if self.control_trajectories_white.shape[0] > 0:
            return self.control_trajectories_white[scenario]
        else:
            return np.array([])

    def stock_trajectory_for_scenario_white(self, scenario: int) -> np.ndarray:
        """
        Return white stock trajectory for a given scenario, or empty if none.
        """
        if self.stock_trajectories_white.shape[0] > 0:
            return self.stock_trajectories_white[scenario]
        else:
            return np.array([])


class LaunchTempo:
    def __init__(self, dir_study: str, area: str, CVar: float, lower_rc:list):
        """
        Initialize the launcher with study directory, area name, and CVaR parameter.
        Automatically creates a unique export directory.
        """
        self.dir_study = dir_study
        self.area = area
        self.CVar = CVar
        self.lower_rc = np.array(lower_rc)
        self.export_dir = self.make_unique_export_dir()

    def make_unique_export_dir(self) -> str:
        """
        Create a unique export directory under the study path,
        named with the current date and time.
        Returns the path of the created directory.
        """
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(
            self.dir_study, "user", f"exports_tempo_{date_str}"
        )
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    def export_stock_trajectories(self, trajectories_r: 'TrajectoriesTempo',
                                 trajectories_wr: 'TrajectoriesTempo',
                                 filename: str = "stock_trajectories.csv") -> None:
        """
        Export red and white stock trajectories for all scenarios and weeks into a CSV file.
        """
        data = []
        nb_scenarios = trajectories_wr.nb_scenarios

        for s in range(nb_scenarios):
            stock_r = trajectories_r.stock_trajectory_for_scenario(s)
            stock_w = trajectories_wr.stock_trajectory_for_scenario_white(s)

            for week in range(61):
                data.append({
                    "MC": s + 1,
                    "week": week + 1,
                    "red_days_remaining": stock_r[week],
                    "white_days_remaining": stock_w[week]
                })

        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Stock trajectories export succeeded: {output_path}")

    def export_daily_control_trajectories(self, trajectories_r: 'TrajectoriesTempo',
                                          trajectories_wr: 'TrajectoriesTempo',
                                          filename: str = "daily_control_trajectories.csv") -> None:
        """
        Export daily control trajectories (red and white) for all scenarios and weeks to CSV.
        The daily net loads are sorted and matched to the controls.
        """
        data = []
        nb_scenarios = trajectories_wr.nb_scenarios
        net_load = trajectories_wr.bv.gain_function.net_load

        for s in range(nb_scenarios):
            control_r = trajectories_r.control_trajectory_for_scenario(s)
            control_w = trajectories_wr.control_trajectory_for_scenario_white(s)

            for week in range(61):
                week_start = week * 7 + 2
                week_end = week_start + 7
                week_days = net_load[week_start * 24: week_end * 24, s].reshape(7, 24).sum(axis=1)
                week_days_r = week_days[:5]
                week_days_w = week_days[:6]

                sorted_days_r = np.argsort(week_days_r)[::-1]
                sorted_days_w = np.argsort(week_days_w)[::-1]

                r = int(control_r[week]) if control_r[week] is not None else None
                w = int(control_w[week]) if control_w[week] is not None else None

                used_days = set()

                for d in sorted_days_r:
                    if r is not None and r > 0:
                        color = "red"
                        used_days.add(d)
                        r -= 1
                        data.append({
                            "MC": s + 1,
                            "week": week + 1,
                            "day": int(d + 1),
                            "color": color,
                            "net_load": float(week_days[d])
                        })

                for d in sorted_days_w:
                    if w is not None and w > 0 and d not in used_days:
                        color = "white"
                        used_days.add(d)
                        w -= 1
                        data.append({
                            "MC": s + 1,
                            "week": week + 1,
                            "day": int(d + 1),
                            "color": color,
                            "net_load": float(week_days[d])
                        })

        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Daily control trajectories export succeeded: {output_path}")

    def export_usage_values(self, bv_r: 'BellmanValuesTempo', bv_wr: 'BellmanValuesTempo',
                            filename: str = "usage_values.csv") -> None:
        """
        Export usage values (marginal values) for red and white stocks over weeks and stock levels.
        """
        data = []
        max_capacity = max(bv_r.capacity, bv_wr.capacity)

        for week in range(61):
            for stock in range(max_capacity):
                val_r = bv_r.usage_values[week, stock] if (week < bv_r.usage_values.shape[0] and stock < bv_r.capacity) else np.nan
                val_wr = bv_wr.usage_values[week, stock] if (week < bv_wr.usage_values.shape[0] and stock < bv_wr.capacity) else np.nan
                data.append({
                    "week": week + 1,
                    "remaining_stock": stock,
                    "usage_value_red": val_r,
                    "usage_value_white_red": val_wr,
                })

        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Usage values export succeeded: {output_path}")

    def plot_stock_trajectories(
        self,
        bellman_values_r: BellmanValuesTempo,
        trajectories_r: 'TrajectoriesTempo',
        trajectories_wr: 'TrajectoriesTempo'
    ) -> None:

        nb_scenarios = trajectories_wr.nb_scenarios
        weeks = np.arange(1, 62)
        fig = go.Figure()

        # Courbe guide basse
        fig.add_trace(go.Scatter(
            x=weeks,
            y=bellman_values_r.lower_rule_curve,
            mode='lines',
            name='Lower rule curve Tempo red',
            line=dict(dash='dash', color='red')
        ))

        # Traces Red/White par MC
        for s in range(nb_scenarios):
            fig.add_trace(go.Scatter(
                x=weeks,
                y=trajectories_r.stock_trajectory_for_scenario(s),
                name=f"Red stock",
                line=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                x=weeks,
                y=trajectories_wr.stock_trajectory_for_scenario_white(s),
                name=f"White stock",
                line=dict(color='green')
            ))

        # Traces "All MC Red" et "All MC White"
        palette = px.colors.qualitative.Bold
        for s in range(nb_scenarios):
            fig.add_trace(go.Scatter(
                x=weeks,
                y=trajectories_r.stock_trajectory_for_scenario(s),
                name=f"All MC Red {s+1}",
                line=dict(color=palette[s % len(palette)])
            ))
        for s in range(nb_scenarios):
            fig.add_trace(go.Scatter(
                x=weeks,
                y=trajectories_wr.stock_trajectory_for_scenario_white(s),
                name=f"All MC White {s+1}",
                line=dict(color=palette[s % len(palette)])
            ))

        total_traces = 1 + 4 * nb_scenarios

        # Visibilité initiale
        init_vis = [False] * total_traces
        init_vis[0] = True
        init_vis[1] = True
        init_vis[2] = True
        for i, trace in enumerate(fig.data):
            trace.visible = init_vis[i]

        # Boutons
        buttons = []
        for s in range(nb_scenarios):
            vis_mc = [False] * total_traces
            vis_mc[0] = True
            vis_mc[1 + 2 * s] = True
            vis_mc[1 + 2 * s + 1] = True
            buttons.append(dict(
                label=f"MC {s+1}",
                method="update",
                args=[{"visible": vis_mc}, {"title": {"text": f"Tempo Day Stocks - MC {s+1}"}}]
            ))

        vis_all_red = [False] * total_traces
        vis_all_red[0] = True
        for s in range(nb_scenarios):
            vis_all_red[1 + 2 * nb_scenarios + s] = True
        buttons.append(dict(
            label="All MC Red",
            method="update",
            args=[{"visible": vis_all_red}, {"title": {"text": "Red Day Stocks - All MC"}}]
        ))

        vis_all_white = [False] * total_traces
        vis_all_white[0] = True
        for s in range(nb_scenarios):
            vis_all_white[1 + 3 * nb_scenarios + s] = True
        buttons.append(dict(
            label="All MC White",
            method="update",
            args=[{"visible": vis_all_white}, {"title": {"text": "White Day Stocks - All MC"}}]
        ))

        # Layout
        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=0.85, y=1.15,
                showactive=True
            )],
            title=dict(text="Tempo Day Stocks - MC 1", x=0.5),
            xaxis=dict(title="Week", showgrid=True, gridcolor="rgba(100,100,100,0.2)", gridwidth=1, dtick=1, tick0=1),
            yaxis=dict(title="Remaining Day Stock", showgrid=True, gridcolor="rgba(100,100,100,0.2)", gridwidth=1, dtick=1, tick0=0),
            font=dict(family="Cambria", size=14),
            legend=dict(visible=True),
            margin=dict(t=100, b=120),
            hovermode="x unified"   # affiche x et y pour toutes les courbes à la même abscisse
        )

        fig.show()
        html_path = os.path.join(self.export_dir, "trajectories_plot.html")
        fig.write_html(html_path)
        print(f"Interactive plot saved at: {html_path}")


    def plot_usage_values(self, bv: 'BellmanValuesTempo') -> None:
        """
        Plot usage values as a function of stock for each week.
        Legend is placed outside the plot for clarity.
        """
        stock_levels = np.arange(1, bv.capacity + 1)

        fig, ax = plt.subplots(figsize=(12, 6))

        for w in range(bv.start_week, bv.end_week + 1):
            ax.plot(stock_levels, bv.usage_values[w], label=f"W{w + 1}")

        ax.set_xlabel('Stock (remaining days)', fontsize=14)
        ax.set_ylabel("Usage Value (MWh/load reduction)", fontsize=14)
        ax.set_title("Weekly Usage Values by Stock Level", fontsize=14)
        ax.grid(True)

        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            fontsize=8,
            frameon=False
        )

        plt.show()
    
    def run(self, actions: Optional[list[str]] = None) -> None:
        """
        Run the specified list of actions, including calculation, export, and plotting.
        """
        if actions is None:
            raise ValueError("Actions must be provided")
        
        start = time.time()
        net_load = NetLoad(reservoir=Reservoir(dir_study=self.dir_study,
                                               name_area=self.area),
                            dir_study=self.dir_study,
                            name_area=self.area)

        gain_function_tempo = GainFunctionTempo(net_load=net_load)

        bellman_values_r = BellmanValuesTempo(gain_function=gain_function_tempo, capacity=22,
                                              start_week=18, end_week=38, CVar=self.CVar, max_control=5,
                                              lower_rule_curve=
                                              np.concatenate([np.repeat(22,18),self.lower_rc,np.repeat(0,22)]))
        
        bellman_values_wr = BellmanValuesTempo(gain_function=gain_function_tempo, capacity=65,
                                               start_week=9, end_week=60, CVar=self.CVar, max_control=6,
                                              lower_rule_curve=
                                              np.concatenate([np.repeat(22,18),self.lower_rc,np.repeat(0,22)]))

        trajectories_r = TrajectoriesTempo(bv=bellman_values_r)
        trajectories_white_and_red = TrajectoriesTempo(bv=bellman_values_wr, stock_trajectories_red=trajectories_r.stock_trajectories)
        end = time.time()
        print(f"Execution time: {end - start:.2f} seconds")



        for action in actions:
            if action == "export_trajectories":
                self.export_stock_trajectories(trajectories_r=trajectories_r, trajectories_wr=trajectories_white_and_red)
            elif action == "export_daily_controls":
                self.export_daily_control_trajectories(trajectories_r=trajectories_r, trajectories_wr=trajectories_white_and_red)
            elif action == "export_usage_values":
                self.export_usage_values(bv_r=bellman_values_r, bv_wr=bellman_values_wr)
            elif action == "plot_trajectories":
                self.plot_stock_trajectories(bellman_values_r=bellman_values_r,trajectories_r=trajectories_r, trajectories_wr=trajectories_white_and_red)
            elif action == "plot_usage_values_red":
                self.plot_usage_values(bv=bellman_values_r)
            elif action == "plot_usage_values_wr":
                self.plot_usage_values(bv=bellman_values_wr)
            else:
                print(f"Unknown action: {action}")

def main() -> None:
    """
    Main entry point: parse CLI arguments and launch the processing.
    """
    parser = argparse.ArgumentParser(description="Launch Tempo trajectories generation.")
    parser.add_argument("--dir_study", type=str, required=True, help="Input directory containing the data.")
    parser.add_argument("--area", type=str, required=True, help="Study area name.")
    parser.add_argument("--actions", type=str, nargs='*', required=True, help="List of commands to execute.")
    parser.add_argument("--lower_rc_red", type=int,nargs=21,required=False,default=[0]*21, help="Lower rule curve for red days (21 values).")
    parser.add_argument("--cvar", type=float, default=1.0, help="CVaR parameter for trajectory generation.")

    args = parser.parse_args()

    launcher = LaunchTempo(dir_study=args.dir_study, area=args.area, CVar=args.cvar,lower_rc=args.lower_rc_red)
    launcher.run(args.actions)


if __name__ == "__main__":
    main()


