from gain_function_hydro import GainFunctionHydro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import interp1d
import time
import plotly.graph_objects as go
from type_definition import Callable

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
        
        self.controls=self.gain_function.compute_controls()
        self.gain_functions=self.gain_function.compute_gain_functions(self.controls,2)
    
        self.bv=np.zeros((self.nb_weeks+1,51,len(self.scenarios)))
        self.mean_bv=np.zeros((self.nb_weeks+1,51))

        self.compute_bellman_values()
        self.compute_usage_values()
        self.compute_trajectories()

    def penalty(self,week_idx:int)->Callable:
        if week_idx==self.nb_weeks:
            penalty = lambda x: 10000000000*abs(x-self.reservoir.initial_level)
        else :
            penalty = interp1d(
                [0,self.reservoir.bottom_rule_curve[week_idx],self.reservoir.upper_rule_curve[week_idx],self.reservoir.capacity],
                [10000000000* (self.reservoir.bottom_rule_curve[week_idx]),0,0,10000000000*(self.reservoir.capacity - self.reservoir.upper_rule_curve[week_idx])],
                kind="linear",fill_value="extrapolate")
        return penalty

    def compute_bellman_values(self) -> None:
        for w in reversed(range(self.nb_weeks)): 
            for c in range(0, 101, 2):
                penalty_function=self.penalty(w+1)
                for s in self.scenarios:
                    inflows_for_week = self.inflow[w, s]
                    current_stock = (c /100) * self.reservoir_capacity
                    controls=self.controls[w,s]
                    future_bellman_function = interp1d(
                        np.linspace(0, self.reservoir_capacity, 51),
                        self.mean_bv[w + 1], 
                        kind="linear",
                        fill_value="extrapolate", 
                        )

                    best_value = np.inf 
                    for control in controls:     
                        # control = min(control, current_stock)         
                        control = min(control-inflows_for_week, current_stock)

                        next_stock = current_stock - control
                        gain = self.gain_functions[w, s](control)
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
                        # control=min(current_stock-c_new/100 * self.reservoir_capacity,current_stock)                    
                        control=min(current_stock-c_new/100 * self.reservoir_capacity-inflows_for_week,current_stock)                    
                        next_stock=current_stock-control
                        penalty=penalty_function(next_stock)
                        gain = self.gain_functions[w,s](control)
                        future_value = future_bellman_function(next_stock)
                        total_value = gain + future_value +penalty

                        if total_value < best_value:
                            best_value = total_value


                    self.bv[w, c // 2, s] = best_value

                self.mean_bv[w, c // 2] = np.mean(self.bv[w, c // 2])

    def compute_usage_values(self) -> None:
        self.usage_values=np.zeros((self.nb_weeks+1,51))
        for w in range(self.nb_weeks+1):
            for c in range(2,102,2):
                self.usage_values[w,(c//2)-1]=self.mean_bv[w,c//2]-self.mean_bv[w,(c//2)-1]

    def compute_trajectories(self) -> None:
        self.trajectories = np.zeros((len(self.scenarios),self.nb_weeks+1))
        self.trajectories[:,0] = self.reservoir.initial_level
        # self.trajectories[:,0]=0.4*self.reservoir_capacity
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                penalty_function=self.penalty(w+1)
                current_stock = self.trajectories[s,w]
                inflows_for_week = self.inflow[w,s]
                gain_function = self.gain_functions[w,s]
                controls = self.controls[w,s]
                future_bellman_function = interp1d(
                        np.linspace(0, self.reservoir_capacity, 51),
                        self.mean_bv[w + 1], 
                        kind="linear",
                        fill_value="extrapolate", 
                    )

                best_value = np.inf
                best_next_stock = None

                for control in controls:
                    # control = min(control, current_stock)
                    control = min(control-inflows_for_week,current_stock)
                    next_stock = current_stock - control
                    gain = gain_function(control)
                    future_value = future_bellman_function(next_stock)
                    penalty=penalty_function(next_stock)
                    total_value = gain + future_value + penalty
                    if total_value < best_value:
                        best_value = total_value
                        best_next_stock = next_stock

                for c_new in range(0,101,2):
                    max_week_energy=np.sum(self.gain_function.max_daily_generating[w * 7:(w + 1) * 7])
                    week_energy_turbine=current_stock-c_new/100 * self.reservoir_capacity
                    if week_energy_turbine<0 or week_energy_turbine>max_week_energy:
                        continue
                    # control=min(current_stock-c_new/100 * self.reservoir_capacity,current_stock)
                    control=min(current_stock-c_new/100 * self.reservoir_capacity - inflows_for_week,current_stock)
                    next_stock=current_stock-control
                    gain = gain_function(control)
                    future_value = future_bellman_function(next_stock)
                    penalty=penalty_function(next_stock)
                    total_value = gain + future_value + penalty
                    if total_value < best_value:
                        best_value = total_value
                        best_next_stock = c_new/100 * self.reservoir_capacity

                if best_next_stock is not None:
                    self.trajectories[s,w+1] =best_next_stock

                else:
                    self.trajectories[s,w+1]=None

    def plot_usage_values(self,usage_values: np.ndarray) -> None:
        
        stock_levels = np.linspace(0, 100, 51) 
        plt.figure(figsize=(12, 6))

        for w in range(self.nb_weeks+1):
            plt.plot(
                stock_levels, 
                usage_values[w],
                label=f"S {w+1}"
            )

        plt.xlabel('Stock (%)')
        plt.ylabel('Valeur d\'usage')
        plt.title('Valeurs d\'usage en fonction du stock')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trajectories_with_buttons(self)-> None:
        fig = go.Figure()

        for s in self.scenarios:
            visible = True if s == 0 else False
            stock_percent = self.trajectories[s] / self.reservoir_capacity * 100
            fig.add_trace(go.Scatter(
                x=list(range(self.nb_weeks+1)),
                y=stock_percent,
                mode='lines+markers',
                name=f'Scénario {s}',
                visible=visible
            ))

        buttons = []
        for s in self.scenarios:
            visibility = [False] * len(self.scenarios)
            visibility[s] = True
            buttons.append(dict(
                label=f"Scénario {s}",
                method="update",
                args=[{"visible": visibility},
                    {"title": f"Trajectoire du stock - Scénario {s}"}]
            ))

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=1.1,
                y=1.15,
                showactive=True
            )],
            title="Trajectoire du stock - Scénario 0",
            xaxis_title="Semaine",
            yaxis_title="Stock (%)",
            legend=dict(x=0, y=-0.2, orientation="h"),
            margin=dict(t=80)
        )

        fig.show()


start=time.time()
gain=GainFunctionHydro("C:/Users/brescianomat/Documents/Etudes Antares/BP23_A-Reference_2036", "fr")
bv=BellmanValuesHydro(gain)
end=time.time()

print("Execution time: ", end-start)
# print(bv.trajectories)
# print(bv.mean_bv)
# print(bv.usage_values)
bv.plot_trajectories_with_buttons()
bv.plot_usage_values(bv.usage_values)

