from gain_function import GainFunctionTempo
from dataclasses import dataclass
import numpy as np

@dataclass
class BellmanValues:
    
    def __init__(self, gain_function : GainFunctionTempo,
                 capacity:int, 
                 nb_week:int, 
                 start_week:int):
        
        self.max_control=gain_function.max_control
        self.gain_function=gain_function
        self.nb_week=nb_week
        self.start_week=start_week

        self.capacity=capacity
        self.nb_scenarios=self.gain_function.nb_scenarios
        self.end_week=self.start_week+nb_week

        self.bv=np.zeros((62,self.capacity+1,self.nb_scenarios))
        self.mean_bv=np.zeros((62,self.capacity+1))
        self.compute_bellman_values()

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
                self.mean_bv[w,c]=np.mean(self.bv[w,c])
    