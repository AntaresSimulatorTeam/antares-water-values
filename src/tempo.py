from read_antares_data import NetLoad
import numpy as np
from typing import Optional
import plotly.graph_objects as go
import pandas as pd
import os

class GainFunctionTempo:
    def __init__(self, net_load : NetLoad, max_control:int):
        
        self.net_load=net_load.net_load
        self.nb_scenarios=net_load.nb_scenarios
        self.daily_net_load=net_load.net_load.reshape(365+64,24,self.nb_scenarios).sum(axis=1)
        self.max_control=max_control

    def gain_for_week_control_and_scenario(self, week_index: int, control : int, scenario : int) -> float:

        week_start=week_index*7+2
        week_end=week_start+7
        

        self.daily_net_load_for_week=self.daily_net_load[week_start:week_end,scenario] #consommation résiduelle par jour pour la semaine considérée


        self.daily_net_load_for_week=(np.sort(self.daily_net_load_for_week[:self.max_control]))[::-1]

        gain=np.sum(self.daily_net_load_for_week[:control])
        return gain


class BellmanValuesTempo:
    
    def __init__(self, gain_function : GainFunctionTempo,
                 capacity:int, 
                 nb_week:int, 
                 start_week:int,
                 CVar:float):
        
        self.max_control=gain_function.max_control
        self.gain_function=gain_function
        self.nb_week=nb_week
        self.start_week=start_week

        self.capacity=capacity
        self.nb_scenarios=self.gain_function.nb_scenarios
        self.end_week=self.start_week+nb_week
        self.CVar=CVar

        self.bv=np.zeros((62,self.capacity+1,self.nb_scenarios))
        self.mean_bv=np.zeros((62,self.capacity+1))
        self.usage_values=np.zeros((62,self.capacity))
        
        self.compute_bellman_values()
        self.compute_usage_values()

    def compute_bellman_values(self) -> None:
        for w in reversed(range(self.start_week,self.end_week-1)):
            for c in range(self.capacity+1):
                for s in range(self.nb_scenarios):
                    best_value=0
                    for control in range(min(self.max_control,c)+1):
                        gain = self.gain_function.gain_for_week_control_and_scenario(w,control,s)
                        
                        future_value=self.mean_bv[w+1,c-control]
                        total_value=gain+future_value
                        if total_value>best_value:
                            best_value=total_value
                    self.bv[w,c,s]=best_value

                #calcul de la CVar : si alpha = 1 calcul en espérance                
                alpha = self.CVar
                bellman_values = self.bv[w, c]
                sorted_bv = np.sort(bellman_values)
                cutoff_index = int((1-alpha) * len(sorted_bv))
                # CVaR = moyenne des pires cas (sous le quantile 1 - alpha)
                self.mean_bv[w, c] = np.mean(sorted_bv[cutoff_index:])

    def compute_usage_values(self) -> None:
        for w in range(self.start_week,self.end_week):
            for c in range(1,self.capacity+1):
                self.usage_values[w,c-1]=self.mean_bv[w,c]-self.mean_bv[w,c-1]


class TrajectoriesTempo :
    def __init__(self,
                bv:BellmanValuesTempo,
                trajectories_red:Optional[np.ndarray]=None):
        
        self.bv=bv
        self.usage_values=bv.usage_values
        self.net_load=self.bv.gain_function.net_load
        self.capacity=self.bv.capacity
        self.nb_week=self.bv.nb_week
        self.nb_scenarios=self.bv.nb_scenarios
        self.start_week=self.bv.start_week
        self.end_week=self.bv.end_week
        self.max_control=self.bv.max_control

        self.trajectories=np.zeros((self.nb_scenarios,self.nb_week-1)) 
        self.stock=np.zeros((self.nb_scenarios,self.nb_week))
        
        self.trajectories_red=trajectories_red

        self.compute_trajectories()
        self.compute_stock()

        self.compute_white_trajectories()
        self.compute_white_stock()
    
    def compute_trajectories(self)-> None:

        for s in range(self.nb_scenarios):
            remaining_capacity=self.capacity
            for w in range(self.start_week+1,self.end_week):
                if remaining_capacity==0:
                    self.trajectories[s,w-self.start_week-1]=0
                    continue
                
                thresholds=self.usage_values[w,:remaining_capacity]
                week_start=(w-1)*7+2
                week_end=week_start+7
                week_days=self.net_load[week_start*24:week_end*24,s].reshape(7,24).sum(axis=1)
                sorted_load=(np.sort(week_days[:self.max_control]))[::-1]

                control = 0
                for d in range(min(self.max_control, remaining_capacity)):
                    idx = remaining_capacity - 1 - d
                    if sorted_load[d] < thresholds[idx]: 
                        break         # puis on s’arrête
                    control += 1      # sinon on continue

                if self.trajectories_red is not None:
                    if w <= 18:
                        # On protège les jours rouges à venir
                        if remaining_capacity - control < 22:
                            control = int(max(0, remaining_capacity - 22))

                    elif 19 <= w < 40:
                        red_weeks_idx = w - 18
                        control_r = self.trajectories_red[s, red_weeks_idx-1]

                        # Imposer que control_wr >= control_r
                        control = int(max(control, control_r))

                        # Ne pas dépasser ce qu'on peut consommer
                        control = int(min(control, remaining_capacity))


                self.trajectories[s, w - self.start_week -1] = control
                remaining_capacity -= control
                remaining_capacity=int(max(remaining_capacity,0))
        
    
    def compute_white_trajectories(self)->None:
        if self.trajectories_red is not None:
            self.trajectories_white=np.copy(self.trajectories)
            for s in range(self.nb_scenarios):
                for w in range(9,30):
                    index=w
                    red_index=w-9
                    self.trajectories_white[s,index]-=self.trajectories_red[s,red_index]
        else:
            self.trajectories_white=np.array([])

    def compute_stock(self) -> None:
        for s in range(self.nb_scenarios):
            self.stock[s,0]=self.capacity
            remaining_capacity=self.capacity
            for w in range(self.start_week+1,self.end_week):
                self.stock[s,w-self.start_week]=remaining_capacity-self.trajectories[s,w-self.start_week-1]
                if self.stock[s,w-self.start_week]<0:
                    print(f"WARNING Stock Négatif : Scénario {s}, Semaine {w}")
                remaining_capacity-=self.trajectories[s,w-self.start_week-1]
            
    def compute_white_stock(self)-> None:
        if self.trajectories_red is not None:
            self.stock_white=np.zeros_like(self.stock)
            for s in range(self.nb_scenarios):
                self.stock_white[s,0]=self.capacity
                self.stock_white[s,:self.start_week]=self.stock[s,:self.start_week]-np.full(self.start_week,22)
                for w in range(self.start_week+1,self.end_week):
                    index=w-self.start_week
                    control_w=self.trajectories_white[s,index-1]
                    self.stock_white[s,index]=self.stock_white[s,index-1]-control_w
                    if self.stock_white[s,w-self.start_week]<0:
                        print(f"WARNING Stock Tempo Blancs Négatif : Scénario {s}, Semaine {w}")
        else:
            self.stock_white=np.array([])

    def trajectory_for_scenario(self, scenario:int)-> np.ndarray:
        return self.trajectories[scenario]

    def stock_for_scenario(self,scenario:int) -> np.ndarray:
        return self.stock[scenario]
    
    def white_trajectory_for_scenario(self, scenario: int) -> np.ndarray:
        return self.trajectories_white[scenario] if self.trajectories_white[scenario].shape[0]>0 else np.array([])

    def white_stock_for_scenario(self, scenario: int) -> np.ndarray:
        return self.stock_white[scenario] if self.stock_white[scenario].shape[0]>0 else np.array([])


class LaunchTempo :
    def __init__(self, dir_study :str, area:str,CVar:float):
        self.dir_study=dir_study
        self.area=area
        self.CVar=CVar
        self.compute_tempo(self.dir_study,self.area)

    def export_trajectories(self,trajectories_r: TrajectoriesTempo, 
                            trajectories_wr: TrajectoriesTempo,
                            filename: str = "stocks_tempo.csv") -> None:
        data = []

        nb_scenarios = trajectories_wr.nb_scenarios
        nb_week = trajectories_wr.nb_week

        for s in range(nb_scenarios):
            stock_r = trajectories_r.stock_for_scenario(s)
            stock_w = trajectories_wr.white_stock_for_scenario(s)
            stock_r_shifted = [None]*9 + list(stock_r[:])+[None]*22

            for week in range(nb_week):
                data.append({
                    "scenario": s+1,
                    "semaine": week+1,
                    "jours_rouges_restants": stock_r_shifted[week],
                    "jours_blancs_restants": stock_w[week]
                })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Export des trajectoires terminé : {filename}")


    def plot_trajectories(self,trajectories_r:TrajectoriesTempo,trajectories_wr:TrajectoriesTempo)->None:
        nb_scenarios = trajectories_wr.nb_scenarios
        weeks = np.arange(trajectories_wr.nb_week)

        fig = go.Figure()

        for s in range(nb_scenarios):
            stock_r = trajectories_r.stock_for_scenario(s)
            stock_w = trajectories_wr.white_stock_for_scenario(s)

            visible = (s == 0)

            stock_r_shifted = [None]*9 + list(stock_r[:])

            fig.add_trace(go.Scatter(
                x=weeks,
                y=stock_r_shifted,
                name="Stock jours rouges",
                visible=visible,
                line=dict(color='red'),
                legendgroup=f"scen{s}",
                showlegend=True if s == 0 else False
            ))

            fig.add_trace(go.Scatter(
                x=weeks,
                y=stock_w,
                name="Stock jours blancs",
                visible=visible,
                line=dict(color='green'),
                legendgroup=f"scen{s}",
                showlegend=True if s == 0 else False
            ))

        buttons = []

        for s in range(nb_scenarios):
            visibility = [False] * (2 * nb_scenarios)
            visibility[2*s] = visibility[2*s + 1] = visibility[2*s + 1] = True

            buttons.append(dict(
                label=f"Scénario {s}",
                method="update",
                args=[{"visible": visibility},
                    {"title": f"Stocks des jours Tempo - Scénario {s}"}]
            ))

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=1.05,
                y=1,
                showactive=True
            )],
            title="Stocks des jours Tempo - Scénario 0",
            xaxis_title="Semaine",
            yaxis_title="Stock de jours restants",
            legend=dict(x=0, y=-0.2, orientation="h"),
            margin=dict(t=80)
        )

        fig.show()


    def compute_tempo(self,dir_study:str, area:str)->None:

        net_load=NetLoad(dir_study=dir_study,name_area=area)

        gain_function_tempo_r=GainFunctionTempo(net_load=net_load,max_control=5)
        gain_function_tempo_wr=GainFunctionTempo(net_load=net_load,max_control=6)

        bellman_values_r=BellmanValuesTempo(gain_function=gain_function_tempo_r,capacity=22,nb_week=22,start_week=18,CVar=self.CVar)
        bellman_values_wr=BellmanValuesTempo(gain_function=gain_function_tempo_wr,capacity=65,nb_week=53,start_week=9,CVar=self.CVar)

        trajectories_r=TrajectoriesTempo(bv=bellman_values_r)
        trajectories_white_and_red=TrajectoriesTempo(bv=bellman_values_wr,trajectories_red=trajectories_r.trajectories)

        self.export_trajectories(trajectories_r=trajectories_r,trajectories_wr=trajectories_white_and_red)
        self.plot_trajectories(trajectories_r=trajectories_r,trajectories_wr=trajectories_white_and_red)


dir_study = "test_data/one_node_(1)"
area = "area"
LaunchTempo(dir_study=dir_study,area=area,CVar=0.2)
