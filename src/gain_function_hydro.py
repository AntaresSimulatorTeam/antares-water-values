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
        self.scenarios=range(10)

 
    def get_controls_for_week(self, week_index:int, control_discretization:int) -> np.ndarray:
        max_week_energy=np.sum(self.max_daily_generating[week_index * 7:(week_index + 1) * 7])

        x=np.linspace(0,1,control_discretization)
        x_transformed=x**2
        controls=x_transformed*max_week_energy
        return controls
  
    def compute_controls(self)->np.ndarray:
        controls=np.array([[self.get_controls_for_week(w,50) for s in self.scenarios] for w in range(self.nb_weeks)])
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

    def gain_function_reversed(self,week_index:int,scenario:int,alpha:float)->tuple:
        net_load_for_week = self.net_load[week_index * 168:(week_index + 1) * 168, scenario]
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7:(week_index + 1) * 7], 24) / 24

        curtail=np.minimum(net_load_for_week,max_energy_hour)
        control=np.sum(curtail)
        control_list=[control]
        curtailed_energy=net_load_for_week-curtail
        gain=np.sum(curtailed_energy**alpha/1e9)
        gain_list=[gain]
        curtailed_energy_list=[curtailed_energy]

        sorted_indices=np.argsort(curtailed_energy)

        for i in sorted_indices[::10]:
            threshold=net_load_for_week[i]
            new_curtailed_energy=np.minimum(net_load_for_week,np.maximum(threshold,curtailed_energy))
            delta=new_curtailed_energy-curtailed_energy
            control-=np.sum(delta)
            gain+=np.sum((new_curtailed_energy**alpha-curtailed_energy**alpha)/1e9)
            control_list.append(control)
            gain_list.append(gain)

            curtailed_energy=new_curtailed_energy
            curtailed_energy_list.append(curtailed_energy)
        
        return interp1d(control_list[::-1],gain_list[::-1],fill_value="extrapolate"), curtailed_energy_list[::-1]
    

    def compute_gain_functions(self,alpha:float)->np.ndarray:
        controls=self.compute_controls()
        # gain_functions=np.array([[self.gain_function_reversed(w,s,alpha)[0] for s in self.scenarios] for w in range(self.nb_weeks)])
        gain_functions=np.array([[self.gain_function(w,controls,s,alpha) for s in self.scenarios] for w in range(self.nb_weeks)])
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

    def plot_load(self, week_index: int, control: float, scenario: int) -> None:
        original_load = self.net_load[week_index * 168:(week_index + 1) * 168, scenario]
        curtailed_load = self.compute_curtailed_net_load(week_index, control, scenario)
        
        hours = np.arange(len(original_load))

        width = 0.45

        plt.figure(figsize=(14, 6))
        plt.bar(hours - width/2, original_load, width=width, label='Initiale', color='skyblue')
        plt.bar(hours + width/2, curtailed_load, width=width, label='Écrêtée', color='salmon')
        plt.title("Histogramme des consommations résiduelles")
        plt.xlabel("Heures")
        plt.ylabel("Consommation Résiduelle")
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    def plot_load_reversed(self,week_index:int,scenario:int,alpha:float,nb_heures_ecretees:int)->None:
        original_load = self.net_load[week_index * 168:(week_index + 1) * 168, scenario]
        curtailed_loads = self.gain_function_reversed(week_index, scenario,alpha)[1]
        curtailed_load=curtailed_loads[nb_heures_ecretees]
        # print(np.sum(original_load-curtailed_load))
        
        hours = np.arange(len(original_load))

        width = 0.45

        plt.figure(figsize=(14, 6))
        plt.bar(hours - width/2, original_load, width=width, label='Initiale', color='skyblue')
        plt.bar(hours + width/2, curtailed_load, width=width, label='Écrêtée', color='salmon')
        plt.title("Histogramme des consommations résiduelles")
        plt.xlabel("Heures")
        plt.ylabel("Consommation Résiduelle")
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()        

