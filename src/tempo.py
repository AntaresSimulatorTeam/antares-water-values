from read_antares_data import NetLoad
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


"""To launch the calculations, export daily contro, stock trajectories and plots:
python "path"/tempo.py --dir_study "path_to_study" --area "name_area" --cvar float(CVar parameter, default 1) """

class GainFunctionTempo:
    def __init__(self, net_load : NetLoad, max_control:int):
        
        self.net_load=net_load.net_load
        self.nb_scenarios=net_load.nb_scenarios
        self.daily_net_load=net_load.net_load.reshape(365+64,24,self.nb_scenarios).sum(axis=1)
        self.max_control=max_control

    def gain_for_week_control_and_scenario(self, week_index: int, control : int, scenario : int) -> float:

        week_start=week_index*7+2
        week_end=week_start+7
        
        self.daily_net_load_for_week=self.daily_net_load[week_start:week_end,scenario]

        self.daily_net_load_for_week=(np.sort(self.daily_net_load_for_week[:self.max_control]))[::-1]

        gain=np.sum(self.daily_net_load_for_week[:control])
        return gain


class BellmanValuesTempo:
    
    def __init__(self, gain_function : GainFunctionTempo,
                 capacity:int, 
                 start_week:int,
                 end_week:int,
                 CVar:float):
        
        self.max_control=gain_function.max_control
        self.gain_function=gain_function
        self.start_week=start_week
        self.end_week=end_week

        self.capacity=capacity
        self.nb_scenarios=self.gain_function.nb_scenarios
        self.CVar=CVar

        self.bv=np.zeros((61,self.capacity+1,self.nb_scenarios))
        self.mean_bv=np.zeros((61,self.capacity+1))
        self.usage_values=np.zeros((61,self.capacity))
        
        self.compute_bellman_values()
        # self.compute_usage_values()

    def penalty(self)->Callable:
        penalty=interp1d([-1,0,self.capacity,self.capacity+1],
                         [-1e9,0,0,-1e9], kind='linear',fill_value='extrapolate')
        # penalty=lambda x:0
        return penalty

    def compute_bellman_values(self) -> None:
        self.mean_bv[self.end_week]=np.zeros(self.capacity+1)
        for w in reversed(range(self.start_week,self.end_week)):
            for c in range(self.capacity+1):
                for s in range(self.nb_scenarios):
                    best_value=0
                    for control in range(self.max_control+1):
                        gain = self.gain_function.gain_for_week_control_and_scenario(w+1,control,s)
                        penalty=self.penalty()
                        future_value=self.mean_bv[w+1,c-control]
                        total_value=gain+future_value+penalty(c-control)
                        if total_value>best_value:
                            best_value=total_value
                    self.bv[w,c,s]=best_value

                alpha = self.CVar
                bellman_values = self.bv[w, c]
                sorted_bv = np.sort(bellman_values)
                cutoff_index = int((1-alpha) * len(sorted_bv))
                self.mean_bv[w, c] = np.mean(sorted_bv[cutoff_index:])

    def compute_usage_values(self) -> None:
        for w in range(self.start_week,self.end_week+1):
            for c in range(1,self.capacity+1):
                self.usage_values[w,c-1]=self.mean_bv[w,c]-self.mean_bv[w,c-1]
    

class TrajectoriesTempo :
    def __init__(self,
                bv:BellmanValuesTempo,
                stock_trajectories_red:Optional[np.ndarray]=None):
        
        self.bv=bv
        self.usage_values=bv.usage_values
        self.net_load=self.bv.gain_function.net_load
        self.capacity=self.bv.capacity
        self.nb_scenarios=self.bv.nb_scenarios
        self.start_week=self.bv.start_week
        self.end_week=self.bv.end_week
        self.max_control=self.bv.max_control

        self.control_trajectories=np.zeros((self.nb_scenarios,61)) 
        self.stock_trajectories=np.zeros((self.nb_scenarios,61))
        
        self.stock_trajectories_red=stock_trajectories_red

        self.compute_trajectories()
        self.compute_trajectories_white()
    

    def compute_trajectories(self) -> None:
        for s in range(self.nb_scenarios):
            self.stock_trajectories[s,:self.start_week]=self.capacity
            for w in range(self.start_week,self.end_week+1):
                if self.stock_trajectories[s,w-1]==0:
                    self.stock_trajectories[s,w:]=0
                    break
                best_value=0
                best_control=None
                for c in range(self.max_control+1):
                    gain = self.bv.gain_function.gain_for_week_control_and_scenario(w,c,s)
                    future_value=self.bv.mean_bv[w,int(self.stock_trajectories[s,w-1])-c]
                    penalty=self.bv.penalty()
                    total_value=gain+future_value+penalty(int(self.stock_trajectories[s,w-1])-c)
                    if total_value>best_value:
                        best_control=c
                        best_value=total_value
                if self.stock_trajectories_red is not None and best_control is not None:
                    # interdit les stocks négatifs pour les stocks blancs
                    if self.stock_trajectories[s,w-1]-best_control<self.stock_trajectories_red[s,w]:
                        best_control=self.stock_trajectories[s,w-1]-self.stock_trajectories_red[s,w]
                    # interdit les controles négatifs pour les stocks blancs
                    if best_control<self.stock_trajectories_red[s,w-1]-self.stock_trajectories_red[s,w]:
                        best_control=self.stock_trajectories_red[s,w-1]-self.stock_trajectories_red[s,w]

                if best_control is not None:
                    self.stock_trajectories[s,w]=self.stock_trajectories[s,w-1]-best_control
                    self.control_trajectories[s,w]=best_control
                else:
                    self.stock_trajectories[s,w]=None
                    self.control_trajectories[s,w]=None


    def compute_trajectories_white(self)->None:
        if self.stock_trajectories_red is not None:
            self.stock_trajectories_white=self.stock_trajectories-self.stock_trajectories_red
        else:
            self.stock_trajectories_white=np.array([[]])
        self.control_trajectories_white=np.zeros((self.nb_scenarios,61))
        if self.stock_trajectories_red is not None:
            for s in range(self.nb_scenarios):
                for w in range(1,61):
                    self.control_trajectories_white[s,w]=self.stock_trajectories_white[s,w-1]-self.stock_trajectories_white[s,w]

    def control_trajectory_for_scenario(self, scenario:int)-> np.ndarray:
        return self.control_trajectories[scenario]

    def stock_trajectory_for_scenario(self,scenario:int) -> np.ndarray:
        return self.stock_trajectories[scenario]
    
    def control_trajectory_for_scenario_white(self, scenario: int) -> np.ndarray:
        return self.control_trajectories_white[scenario] if self.control_trajectories_white[scenario].shape[0]>0 else np.array([])

    def stock_trajectory_for_scenario_white(self, scenario: int) -> np.ndarray:
        return self.stock_trajectories_white[scenario] if self.stock_trajectories_white[scenario].shape[0]>0 else np.array([])


class LaunchTempo :
    def __init__(self, dir_study: str, area: str, CVar: float):
            self.dir_study = dir_study
            self.area = area
            self.CVar = CVar
            self.export_dir = self.make_unique_export_dir()

    def make_unique_export_dir(self) -> str:
        base_path = os.path.join(self.dir_study, "exports_tempo")
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

    def export_stock_trajectories(self, trajectories_r: TrajectoriesTempo,
                            trajectories_wr: TrajectoriesTempo,
                            filename: str = "stock_trajectories.csv") -> None:
        data = []

        nb_scenarios = trajectories_wr.nb_scenarios

        for s in range(nb_scenarios):
            stock_r = trajectories_r.stock_trajectory_for_scenario(s)
            stock_w = trajectories_wr.stock_trajectory_for_scenario_white(s)

            for week in range(61):
                data.append({
                    "MC": s + 1,
                    "semaine": week + 1,
                    "jours_rouges_restants": stock_r[week],
                    "jours_blancs_restants": stock_w[week]
                })

        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Stock trajectories export succeeded : {output_path}")

    def export_daily_control_trajectories(self, trajectories_r: TrajectoriesTempo,
                    trajectories_wr: TrajectoriesTempo,
                    filename: str = "daily_control_trajectories.csv") -> None:
        data = []
        nb_scenarios = trajectories_wr.nb_scenarios
        net_load = trajectories_wr.bv.gain_function.net_load

        for s in range(nb_scenarios):
            tirage_r = trajectories_r.control_trajectory_for_scenario(s)
            tirage_w = trajectories_wr.control_trajectory_for_scenario_white(s)

            for week in range(61):
                week_start = week* 7 + 2
                week_end = week_start + 7
                week_days = net_load[week_start * 24: week_end * 24, s].reshape(7, 24).sum(axis=1)
                week_days_r = week_days[:5]
                week_days_w = week_days[:6]
 
                sorted_days_r = np.argsort(week_days_r)[::-1]
                sorted_days_w = np.argsort(week_days_w)[::-1]

                r = int(tirage_r[week]) if tirage_r[week] is not None else None
                w = int(tirage_w[week]) if tirage_w[week] is not None else None

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
                    if w is not None and w> 0 and d not in used_days:
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
        print(f"Daily control trajectories export succeeded : {output_path}")

    def plot_stock_trajectories(self, trajectories_r: TrajectoriesTempo, trajectories_wr: TrajectoriesTempo) -> None:
        nb_scenarios = trajectories_wr.nb_scenarios
        weeks = np.arange(1, 62)

        fig = go.Figure()

        color_palette = px.colors.qualitative.Bold

        for s in range(nb_scenarios):
            stock_r = trajectories_r.stock_trajectory_for_scenario(s)
            stock_w = trajectories_wr.stock_trajectory_for_scenario_white(s)

            fig.add_trace(go.Scatter(
                x=weeks,
                y=stock_r,
                name=f"Stock jours rouges MC {s + 1}",
                visible=(s == 0),
                line=dict(color='red'),
                showlegend=False,
                hovertemplate=(
                    "Semaine %{x}<br>" +
                    "Stock: %{y}<br>" +
                    f"MC {s + 1}<br>" +
                    "rouge<br>" +
                    "<extra></extra>"
                )
            ))

            fig.add_trace(go.Scatter(
                x=weeks,
                y=stock_w,
                name=f"Stock jours blancs MC {s + 1}",
                visible=(s == 0),
                line=dict(color='green'),
                showlegend=False,
                hovertemplate=(
                    "Semaine %{x}<br>" +
                    "Stock: %{y}<br>" +
                    f"MC {s + 1}<br>" +
                    "blanc<br>" +
                    "<extra></extra>"
                )
            ))

        for s in range(nb_scenarios):
            stock_r = trajectories_r.stock_trajectory_for_scenario(s)
            color = color_palette[s % len(color_palette)]
            fig.add_trace(go.Scatter(
                x=weeks,
                y=stock_r,
                name=f"All MC rouges {s + 1}",
                visible=False,
                line=dict(color=color),
                showlegend=False,
                hovertemplate=(
                    "Semaine %{x}<br>" +
                    "Stock: %{y}<br>" +
                    f"MC {s + 1}<br>" +
                    "rouge<br>" +
                    "<extra></extra>"
                )
            ))

        for s in range(nb_scenarios):
            stock_w = trajectories_wr.stock_trajectory_for_scenario_white(s)
            color = color_palette[s % len(color_palette)]
            fig.add_trace(go.Scatter(
                x=weeks,
                y=stock_w,
                name=f"All MC blancs {s + 1}",
                visible=False,
                line=dict(color=color),
                showlegend=False,
                hovertemplate=(
                    "Semaine %{x}<br>" +
                    "Stock: %{y}<br>" +
                    f"MC {s + 1}<br>" +
                    "blanc<br>" +
                    "<extra></extra>"
                )
            ))

        buttons = []
        for s in range(nb_scenarios):
            visibility = [False] * (4 * nb_scenarios)
            visibility[2 * s] = True
            visibility[2 * s + 1] = True
            buttons.append(dict(
                label=f"MC {s + 1}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title.text": f"Stocks des jours Tempo - MC {s + 1}"}
                ]
            ))

        visibility_all_rouge = [False] * (4 * nb_scenarios)
        for s in range(nb_scenarios):
            visibility_all_rouge[2 * nb_scenarios + s] = True
        buttons.append(dict(
            label="all MC rouges",
            method="update",
            args=[
                {"visible": visibility_all_rouge},
                {"title.text": "Stocks des jours rouges - All MC"}
            ]
        ))

        visibility_all_blanc = [False] * (4 * nb_scenarios)
        for s in range(nb_scenarios):
            visibility_all_blanc[3 * nb_scenarios + s] = True
        buttons.append(dict(
            label="all MC blancs",
            method="update",
            args=[
                {"visible": visibility_all_blanc},
                {"title.text": "Stocks des jours blancs - All MC"}
            ]
        ))

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=0.85,
                y=1.15,
                showactive=True
            )],
            title=dict(text="Stocks des jours Tempo - MC 1", x=0.5),
            xaxis=dict(
                title="Semaine",
                showgrid=True,
                gridcolor="rgba(100,100,100,0.2)",
                gridwidth=1,
                dtick=1,
                tick0=1
            ),
            yaxis=dict(
                title="Stock de jours restants",
                showgrid=True,
                gridcolor="rgba(100,100,100,0.2)",
                gridwidth=1,
                dtick=1,
                tick0=0
            ),
            font=dict(
                family="Cambria",
                size=14
            ),
            legend=dict(visible=False),
            margin=dict(t=100, b=120)
        )

        fig.show()

        html_path = os.path.join(self.export_dir, "trajectories_plot.html")
        fig.write_html(html_path)
        print(f"Interactive plot saved at: {html_path}")

    def run(self)->None:
        start=time.time()    
        net_load=NetLoad(dir_study=self.dir_study,name_area=self.area)

        gain_function_tempo_r=GainFunctionTempo(net_load=net_load,max_control=5)
        gain_function_tempo_wr=GainFunctionTempo(net_load=net_load,max_control=6)

        bellman_values_r=BellmanValuesTempo(gain_function=gain_function_tempo_r,capacity=22,start_week=18,end_week=38,CVar=self.CVar)
        bellman_values_wr=BellmanValuesTempo(gain_function=gain_function_tempo_wr,capacity=65,start_week=9,end_week=60,CVar=self.CVar)

        trajectories_r=TrajectoriesTempo(bv=bellman_values_r)
        trajectories_white_and_red=TrajectoriesTempo(bv=bellman_values_wr,stock_trajectories_red=trajectories_r.stock_trajectories)
        end=time.time()
        print("Execution time: ", end-start)
        self.export_stock_trajectories(trajectories_r=trajectories_r,trajectories_wr=trajectories_white_and_red)
        self.export_daily_control_trajectories(trajectories_r=trajectories_r,trajectories_wr=trajectories_white_and_red)
        self.plot_stock_trajectories(trajectories_r=trajectories_r,trajectories_wr=trajectories_white_and_red)
        

def main()->None:

    parser = argparse.ArgumentParser(description="Lancer la génération des trajectoires Tempo.")
    parser.add_argument("--dir_study", type=str, required=True, help="Répertoire d'entrée contenant les données.")
    parser.add_argument("--area", type=str, required=True, help="Nom de la zone d'étude.")
    parser.add_argument("--cvar", type=float, default=1.0, help="Paramètre CVaR pour la génération des trajectoires.")

    args = parser.parse_args()

    launcher = LaunchTempo(dir_study=args.dir_study, area=args.area, CVar=args.cvar)
    launcher.run()


if __name__ == "__main__":
    main()

