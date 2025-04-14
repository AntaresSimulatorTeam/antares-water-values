from read_antares_data import Residual_load
from calculate_gain_func_tempo_red import GainFunction
from dataclasses import dataclass
import numpy as np

@dataclass
class Bellman_values:
    
    def __init__(self,residual_load : Residual_load, gain_function : GainFunction, capacity : int, nb_week:int):
        self.residual_load=residual_load
        self.gain_function=gain_function
        self.capacity=capacity
        self.nb_week=nb_week
        self.nb_scenarios=residual_load.nb_scenarios
        self.bv=np.zeros((52,self.capacity+1,self.nb_scenarios))
        self.mean_bv=np.zeros((52,self.capacity+1))
        self.compute_bellman_values()

    def compute_bellman_values(self) -> None:
        for w in reversed(range(18,18+self.nb_week-1)):
            for c in range(self.capacity+1):
                for s in range(self.nb_scenarios):
                    best_value=0
                    for control in range(min(5,c)+1):
                        gain = self.gain_function.gain_for_week_control_and_scenario(w,control,s)
                        
                        future_value=self.mean_bv[w+1,c-control]
                        total_value=gain+future_value
                        if total_value>best_value:
                            best_value=total_value
                    self.bv[w,c,s]=best_value
                self.mean_bv[w,c]=np.mean(self.bv[w,c])
    