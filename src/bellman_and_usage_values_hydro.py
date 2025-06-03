from gain_function_hydro import GainFunctionHydro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import interp1d
import time
import plotly.graph_objects as go
from type_definition import Callable
import pandas as pd


rcParams['font.family'] = 'Cambria'

class BellmanValuesHydro:
    def __init__(self, gain_function: GainFunctionHydro,
                ):
        self.gain_function=gain_function
        self.reservoir = gain_function.reservoir
        self.reservoir_capacity = self.reservoir.capacity

        self.bottom_rule_curve=gain_function.reservoir.bottom_rule_curve
        self.upper_rule_curve=gain_function.reservoir.upper_rule_curve

        self.inflow=gain_function.reservoir.inflow
        
        self.nb_weeks=self.gain_function.nb_weeks
        self.scenarios=self.gain_function.scenarios
        
        self.gain_functions=self.gain_function.compute_gain_functions(2)

        self.bv=np.zeros((self.nb_weeks+1,51,len(self.scenarios)))
        self.mean_bv=np.zeros((self.nb_weeks+1,51))

        self.compute_bellman_values()
        self.compute_usage_values()
        self.compute_trajectories()

    def penalty(self,week_idx:int)->Callable:
        if week_idx==self.nb_weeks-1:
            penalty = lambda x: abs(x-self.reservoir.initial_level)/1e3
        else :
            first_point_x=max(0,self.reservoir.bottom_rule_curve[week_idx]-(self.reservoir.capacity-self.reservoir.upper_rule_curve[week_idx]))
            penalty = interp1d(
                [
                    first_point_x,
                    self.reservoir.bottom_rule_curve[week_idx],
                    self.reservoir.upper_rule_curve[week_idx],
                    self.reservoir.capacity],
                [1e3,0,0,1e3],
                kind="linear",fill_value="extrapolate")
            
        return penalty

    def compute_bellman_values(self) -> None:
        for w in reversed(range(self.nb_weeks)): 
            for c in range(0, 101, 2):
                penalty_function=self.penalty(w)
                for s in self.scenarios:
                    inflows_for_week = self.inflow[w, s]
                    current_stock = (c /100) * self.reservoir_capacity
                    gain_function=self.gain_functions[w,s]
                    controls=gain_function.x
                    future_bellman_function = interp1d(
                        np.linspace(0, self.reservoir_capacity, 51),
                        self.mean_bv[w + 1], 
                        kind="linear",
                        fill_value="extrapolate", 
                        )

                    best_value = np.inf 
                    for control in controls:     
                        control = min(control, current_stock+inflows_for_week)

                        next_stock = current_stock - control + inflows_for_week
                        gain = gain_function(control)
                        future_value = future_bellman_function(next_stock)
                        penalty=penalty_function(next_stock)
                        total_value = gain + future_value + penalty

                        if total_value < best_value:
                            best_value = total_value
                            
                    for c_new in range(0,101,2):
                        max_week_energy=np.sum(self.gain_function.max_daily_generating[w * 7:(w + 1) * 7])
                        week_energy_turbine=current_stock-c_new/100 * self.reservoir_capacity
                        if week_energy_turbine<0 or week_energy_turbine>max_week_energy:
                            continue
                        control=min(current_stock-c_new/100 * self.reservoir_capacity,current_stock+inflows_for_week)                    
                        next_stock=current_stock-control+inflows_for_week
                        penalty=penalty_function(next_stock)
                        gain = gain_function(control)
                        future_value = future_bellman_function(next_stock)
                        total_value = gain + future_value +penalty

                        if total_value < best_value:
                            best_value = total_value


                    self.bv[w, c // 2, s] = best_value

                self.mean_bv[w, c // 2] = np.mean(self.bv[w, c // 2])

    def compute_usage_values(self) -> None:
        self.usage_values=np.zeros((self.nb_weeks+1,50))
        for w in range(self.nb_weeks+1):
            for c in range(2,102,2):
                self.usage_values[w,(c//2)-1]=self.mean_bv[w,c//2]-self.mean_bv[w,(c//2)-1]

    def compute_trajectories(self) -> None:
        self.trajectories = np.zeros((len(self.scenarios),self.nb_weeks+1))
        self.optimal_controls = np.zeros((len(self.scenarios),self.nb_weeks))
        self.trajectories[:,0] = self.reservoir.initial_level
        
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                penalty_function=self.penalty(w)
                current_stock = self.trajectories[s,w]
                inflows_for_week = self.inflow[w,s]
                gain_function = self.gain_functions[w,s]
                controls = gain_function.x
                future_bellman_function = interp1d(
                        np.linspace(0, self.reservoir_capacity, 51),
                        self.mean_bv[w + 1], 
                        kind="linear",
                        fill_value="extrapolate", 
                    )

                best_value = np.inf
                best_next_stock = None

                for control in controls:
                    control = min(control,current_stock+inflows_for_week)
                    next_stock = current_stock - control+inflows_for_week
                    gain = gain_function(control)
                    future_value = future_bellman_function(next_stock)
                    penalty=penalty_function(next_stock)
                    total_value = gain + future_value + penalty
                    if total_value < best_value:
                        best_value = total_value
                        best_next_stock = next_stock
                        optimal_control=control
                       
                for c_new in range(0,101,2):
                    max_week_energy=np.sum(self.gain_function.max_daily_generating[w * 7:(w + 1) * 7])
                    week_energy_turbine=current_stock-c_new/100 * self.reservoir_capacity
                    if week_energy_turbine<0 or week_energy_turbine>max_week_energy:
                        continue
                    control=min(current_stock-c_new/100 * self.reservoir_capacity ,current_stock+inflows_for_week)
                    next_stock=current_stock-control+inflows_for_week
                    gain = gain_function(control)
                    future_value = future_bellman_function(next_stock)
                    penalty=penalty_function(next_stock)
                    total_value = gain + future_value + penalty
                    if total_value < best_value:
                        best_value = total_value
                        best_next_stock = next_stock
                        optimal_control=control
                       

                if best_next_stock is not None:
                    self.trajectories[s,w+1] =best_next_stock
                    self.optimal_controls[s,w] = optimal_control
                    lower_bound=self.reservoir.bottom_rule_curve[w]
                    upper_bound=self.reservoir.upper_rule_curve[w]
                    if not (lower_bound <= best_next_stock <= upper_bound):
                        print(
                            f"⚠️ Stock hors courbes guides - Semaine {w+1}, scénario {s+1} : "
                            f"{best_next_stock:.2f} ∉ [{lower_bound:.2f}, {upper_bound:.2f}]"
                        )
                else:
                    self.trajectories[s,w+1]=None
                    self.optimal_controls[s,w]=None
                
        self.trajectories=self.trajectories[:,1:]

    # affichages

    def plot_usage_values(self,usage_values: np.ndarray) -> None:
        
        stock_levels = np.linspace(2, 100, 50) 
        plt.figure(figsize=(12, 6))

        for w in range(self.nb_weeks+1):
            plt.plot(
                stock_levels, 
                usage_values[w],
                label=f"S {w}"
            )

        plt.xlabel('Stock (%)')
        plt.ylabel('Valeur d\'usage')
        plt.title('Valeurs d\'usage en fonction du stock')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trajectories(self) -> None:
        fig = go.Figure()
        weeks = list(range(1, self.nb_weeks + 2))

        # Courbes guide (toujours visibles)
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

        # Trajectoires par scénario
        for s in self.scenarios:
            visible = True if s == 0 else False
            stock_percent = self.trajectories[s] / self.reservoir_capacity * 100
            fig.add_trace(go.Scatter(
                x=weeks,
                y=stock_percent,
                mode='lines+markers',
                name=f'MC {s + 1}',
                visible=visible
            ))

        # Configuration des boutons
        n_scenarios = len(self.scenarios)
        n_shared_guides = 2  # upper and lower rule curves
        buttons = []

        for s in self.scenarios:
            visibility = [True] * n_shared_guides + [False] * n_scenarios
            visibility[n_shared_guides + s] = True  # afficher uniquement le scénario s
            buttons.append(dict(
                label=f"MC {s + 1}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title.text": f"Trajectoire du stock - MC {s + 1}"}
                ]
            ))

        # Mise à jour du layout
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
            margin=dict(t=80)
        )

        fig.show()

    # exports

    def export_controls(self) -> pd.DataFrame:
        data = []

        for s in self.scenarios:
            for w in range(self.nb_weeks):
                u = self.optimal_controls[s, w]
                data.append({
                    "area": self.gain_function.name_area,
                    "u": u,
                    "week": w + 1,
                    "mcYear": s + 1,
                    "sim": "u_0"
                })

        df = pd.DataFrame(data)
        return df

    def export_trajectories(self) ->pd.DataFrame:
        data = []

        for s in self.scenarios:
            for w in range(self.nb_weeks):
                hlevel =self.trajectories[s,w]
                data.append({
                    "area": self.gain_function.name_area,
                    "hlevel": hlevel,
                    "week": w + 1,
                    "mcYear": s + 1,
                    "sim": "u_0"
                })
        df = pd.DataFrame(data)
        return df
    

start=time.time()
gain=GainFunctionHydro("/test_data/one_node_(1)", "area")
bv=BellmanValuesHydro(gain)
end=time.time()

print("Execution time: ", end-start)

bv.plot_trajectories()

# trajectories=bv.export_trajectories()
# trajectories.to_csv("C:/Users/brescianomat/Desktop/trajectories.csv", index=False)

# constraint_values = bv.export_controls()
# constraint_values.to_csv("C:/Users/brescianomat/Desktop/constraint_values.csv", index=False)

