from read_antares_data import Residual_load
from calculate_gain_func_tempo_red import GainFunction
from dataclasses import dataclass
import numpy as np

@dataclass
class Bellman_values:
    
    def __init__(self,residual_load : Residual_load, gain_function : GainFunction, capacity : int):
        self.residual_load=residual_load
        self.gain_function=gain_function
        self.capacity=capacity
        self.nb_week=22
        self.nb_scenarios=residual_load.nb_scenarios
        self.bv=np.zeros((self.nb_week,self.capacity,self.nb_scenarios))
        self.mean_bv=np.zeros((self.nb_week,self.capacity))

    def compute_bellman_values(self) -> None:
        for t in reversed(range(1,self.nb_week-1)):
            for c in range(self.capacity):
                for s in range(self.nb_scenarios):
                    best_value=0
                    for u in range(min(5,c)+1):
                        gain = self.gain_function.gain_for_week_scenario_and_control(t,u,s)
                        
                        future_value=self.mean_bv[t+1,c-u]
                        total_value=gain+future_value
                        if total_value>best_value:
                            best_value=total_value
                    self.bv[t,c,s]=best_value
                self.mean_bv[t,c]=np.mean(self.bv[t,c])
    
