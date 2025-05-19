from read_antares_data import Reservoir
from read_antares_data import NetLoad
import numpy as np
from scipy.interpolate import interp1d
from type_definition import Callable
import matplotlib.pyplot as plt
import pandas as pd
import time

class GainFunctionHydro:
    def __init__(self, dir_study: str, name_area: str) -> None:
        self.dir_study=dir_study
        self.name_area=name_area
        self.reservoir = Reservoir(self.dir_study, self.name_area)
        self.net_load = np.maximum(NetLoad(dir_study, name_area).net_load,0)
        self.max_daily_generating=self.reservoir.max_daily_generating

        self.nb_weeks=self.reservoir.inflow.shape[0]
        # self.nb_scenarios=self.net_load.shape[1]
        self.scenarios=range(200)
        # self.scenarios=[4,5,18,48,66,78,82,117,129,138,152]

    #1ere méthode 
 
    def get_controls_for_week_and_scenario_simplified(self, week_index:int, control_discretization:int) -> np.ndarray:
        max_week_energy=np.sum(self.max_daily_generating[week_index * 7:(week_index + 1) * 7])
        controls=np.linspace(0, max_week_energy, control_discretization)
        return controls
  
    def compute_controls(self)->np.ndarray:
        controls=np.array([[self.get_controls_for_week_and_scenario_simplified(w,50) for s in self.scenarios] for w in range(self.nb_weeks)])
        return controls 

    def compute_curtailed_net_load(self, week_index: int, control: float, scenario: int) -> np.ndarray:
        # Données pour la semaine
        net_load_for_week = self.net_load[week_index * 168:(week_index + 1) * 168, scenario].copy()
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7:(week_index + 1) * 7], 24) / 24

        # Initialisation
        curtailed = np.zeros_like(net_load_for_week)
        energy_used = 0.0

        # Trier les indices des heures par charge décroissante
        sorted_indices = np.argsort(net_load_for_week)[::-1]

        for h in sorted_indices:
            if energy_used >= control:
                break
        

            available = min(
                net_load_for_week[h],                   # on ne peut pas écrêter plus que la charge
                max_energy_hour[h],                     # ni plus que la capacité max par heure
                control - energy_used                   # ni plus que le reste d'énergie à utiliser
            )

            curtailed[h] = available
            energy_used += available
            
        # Calcul de la charge résiduelle écrêtée
        curtailed_net_load = net_load_for_week - curtailed
        return curtailed_net_load
    
    def gain_function(self, week_index:int, controls:np.ndarray, scenario:int, alpha:float)->interp1d:
        controls_for_week_and_scenario=controls[week_index,scenario]
        curtailed_net_loads=np.array([self.compute_curtailed_net_load(week_index, control, scenario) for control in controls_for_week_and_scenario])
        gains=np.sum(curtailed_net_loads**alpha/1E9,axis=1)
        return interp1d(controls_for_week_and_scenario,gains,fill_value="extrapolate")
        
    def compute_gain_functions(self,alpha:float)->np.ndarray:
        controls=self.compute_controls()
        gain_functions=np.array([[self.gain_function(w,controls,s,alpha) for s in self.scenarios] for w in range(self.nb_weeks)])
        return gain_functions
    
    
    # 2eme méthode

    def gain_function_1(self, week_index: int, scenario: int, alpha: float) -> interp1d:
        net_load_for_week = self.net_load[week_index * 168:(week_index + 1) * 168, scenario].copy()
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7:(week_index + 1) * 7], 24) / 24
        
        control = np.minimum(net_load_for_week, max_energy_hour)
        curtailed_load = net_load_for_week - control
        gain = np.sum(curtailed_load ** alpha / 1e9)
        control_total = np.sum(control)

        gain_list = [gain]
        control_list = [control_total]

        sorted_indices = np.argsort(curtailed_load)
        control_sorted = control[sorted_indices]
        cumulative_energy = np.cumsum(control_sorted)
        total_energy = cumulative_energy[-1]

        n_points = 50
        target_energy_levels = np.linspace(0, total_energy, n_points)
        selected_indices = np.searchsorted(cumulative_energy, target_energy_levels)
        selected_indices = np.unique(np.clip(selected_indices, 0, len(sorted_indices) - 1))
        thresholds = net_load_for_week[sorted_indices[selected_indices]]

        for threshold in thresholds:
            new_curtailed_load = np.minimum(net_load_for_week, np.maximum(threshold, curtailed_load))
            delta = new_curtailed_load - curtailed_load
            gain += np.sum((new_curtailed_load ** alpha - curtailed_load ** alpha) / 1e9)
            new_control = control - delta
            new_control_total = max(0, np.sum(new_control))

            if abs(new_control_total - control_total) > 1e-8:
                gain_list.append(gain)
                control_list.append(new_control_total)

            curtailed_load = new_curtailed_load
            control = new_control
            control_total = new_control_total

        return interp1d(control_list, gain_list, fill_value='extrapolate')


    def compute_gain_functions_1(self,alpha:float)->np.ndarray: 
        gain_functions=np.array([[self.gain_function_1(w,s,alpha) for s in self.scenarios] for w in range(self.nb_weeks)])
        return gain_functions
    
    
    # affichages

    def plot_gain_function(self,week_index:int,scenario:int,alpha:float)->None:
        controls = self.compute_controls()
        gain_func = self.gain_function(week_index, controls, scenario, alpha)
        control_values = controls[week_index, scenario]
        gain_values = gain_func(control_values)
        plt.figure(figsize=(8, 5))
        plt.plot(control_values, gain_values, marker='o')
        plt.xlabel("Énergie turbinée hebdomadaire [MWh]")
        plt.ylabel(f"Gain (somme des net loads^α, α={alpha})")
        plt.title(f"Courbe de gain - Semaine {week_index}, Scénario {scenario}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_load(self, week_index: int, total_energy: float, scenario: int) -> None:
        original_residual_load = self.net_load[week_index * 168:(week_index + 1) * 168, scenario]
        curtailed_residual_load = self.compute_curtailed_net_load(week_index, total_energy, scenario)
        
        # sorted_original = np.sort(original_residual_load)[::-1]
        # sorted_curtailed = np.sort(curtailed_residual_load)[::-1]
        print(np.sum(curtailed_residual_load**2))
        hours = np.arange(len(original_residual_load))

        width = 0.45

        plt.figure(figsize=(14, 6))
        plt.bar(hours - width/2, original_residual_load, width=width, label='Initiale triée', color='skyblue')
        plt.bar(hours + width/2, curtailed_residual_load, width=width, label='Écrêtée triée', color='salmon')
        plt.title("Histogramme des consommations résiduelles triées (ordre décroissant)")
        plt.xlabel("Heures triées")
        plt.ylabel("Consommation Résiduelle")
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    


# gain=GainFunctionHydro("C:/Users/brescianomat/Documents/Etudes Antares/BP23_A-Reference_2036", "fr")
# start=time.time()
# gain.compute_gain_functions(alpha=2)
# end=time.time()
# print('calculation time : ',end-start)