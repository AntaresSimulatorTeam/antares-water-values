from read_antares_data import Reservoir
from read_antares_data import ResidualLoad
import numpy as np
from scipy.interpolate import interp1d
from type_definition import Callable
import matplotlib.pyplot as plt

class GainFunctionHydro:
    def __init__(self, dir_study: str, name_area: str) -> None:
        self.reservoir = Reservoir(dir_study, name_area)
        self.residual_load = np.maximum(ResidualLoad(dir_study, name_area).residual_load,0)
        self.max_daily_generating=self.reservoir.max_daily_generating

        self.nb_weeks=self.reservoir.inflow.shape[0]
        # self.nb_scenarios=self.residual_load.shape[1]
        self.nb_scenarios=10

    def get_controls_for_week_and_scenario_simplified(self, week_index:int, control_discretization:int) -> np.ndarray:
        max_week_energy=np.sum(self.max_daily_generating[week_index * 7:(week_index + 1) * 7])
        controls=np.linspace(0, max_week_energy, control_discretization)
        return controls

    def get_controls_for_week_and_scenario(self, week_index: int, scenario: int) -> np.ndarray:
        residual_load_for_week = self.residual_load[week_index * 168:(week_index + 1) * 168, scenario]
        sorted_indices = np.argsort(residual_load_for_week)[::-1]
        sorted_residual_load = residual_load_for_week[sorted_indices] 

        controls = np.zeros(168) 
        energy_used_per_hour=np.zeros(168)
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7: (week_index + 1) * 7], 24) / 24

        total_energy_used = 0 
        max_week_energy=np.sum(self.max_daily_generating[week_index * 7:(week_index + 1) * 7])

        for i in range(len(sorted_indices) - 1):
            delta = sorted_residual_load[i] - sorted_residual_load[i + 1]

            hours_to_curtail = sorted_indices[:i + 1]

            for h in hours_to_curtail:
                available_hour_energy=max_energy_hour[h] - energy_used_per_hour[h] 
                available_energy = min(delta, available_hour_energy)

                if available_energy > 0:
                    controls[i] = total_energy_used + available_energy 
                    total_energy_used += available_energy  
                    energy_used_per_hour[h] += available_energy  

                if total_energy_used >= max_week_energy:
                    break

            if total_energy_used >= max_week_energy:
                break

        return np.maximum(controls,0)

    def compute_controls(self)->np.ndarray:
        # controls = np.array([[self.get_controls_for_week_and_scenario(w, s) for s in range(self.nb_scenarios)] for w in range(self.nb_weeks)])
        controls = np.array([[self.get_controls_for_week_and_scenario_simplified(w, 50) for s in range(self.nb_scenarios)] for w in range(self.nb_weeks)])
        return controls

    def compute_curtailed_residual_load(self,week_index:int, control:float, scenario : int) -> np.ndarray:
        
        residual_load_for_week=self.residual_load[week_index*168:(week_index+1)*168,scenario]
        sorted_indices=np.argsort(residual_load_for_week)[::-1]
        sorted_residual_load=residual_load_for_week[sorted_indices]

        curtailed_residual_load=residual_load_for_week.copy()


        energy_used_per_hour=np.zeros(168)
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7: (week_index + 1) * 7], 24) / 24        
        total_energy_used=0

        for i in range(len(sorted_indices)-1):
            delta=sorted_residual_load[i]-sorted_residual_load[i+1]

            hours_to_curtail=sorted_indices[:i+1]
            

            for h in (hours_to_curtail):
                available_hour_energy=max_energy_hour[h] - energy_used_per_hour[h] 
                remaining_week_energy=control-total_energy_used
                available_energy=min(delta, available_hour_energy, remaining_week_energy)

                if available_energy>0:
                    curtailed_residual_load[h]-=available_energy
                    total_energy_used+=available_energy
                    energy_used_per_hour[h]+=available_energy
                
                else:
                    continue

                if total_energy_used>=control:
                    break

        return curtailed_residual_load
       
    def gain_function(self, week_index:int, controls:np.ndarray, scenario:int, alpha:float)->Callable:
        controls_for_week_and_scenario=controls[week_index,scenario]

        curtailed_residual_loads=np.array([self.compute_curtailed_residual_load(week_index, control, scenario) for control in controls_for_week_and_scenario])
        gains=np.sum(curtailed_residual_loads**alpha,axis=1)
        return interp1d(controls_for_week_and_scenario,gains,fill_value="extrapolate")
        
    def compute_gain_functions(self,controls:np.ndarray,alpha:int)->np.ndarray: 
        gains_functions=np.array([[self.gain_function(w,controls,s,alpha) for s in range(self.nb_scenarios)] for w in range(self.nb_weeks)])
        return gains_functions