from read_antares_data import Residual_load
from gain_function_TEMPO import GainFunctionTEMPO
from bellman_values import Bellman_values
from usage_values import UV_tempo
from dataclasses import dataclass
import numpy as np

@dataclass
class Trajectories :
    def __init__(self, 
                residual_load:Residual_load, 
                gain_function:GainFunctionTEMPO, 
                bellman_values:Bellman_values, 
                usage_values:UV_tempo,
                capacity:int, 
                nb_week:int,
                max_control:int):
        self.residual_load=residual_load
        self.gain_function=gain_function
        self.bellman_values=bellman_values
        self.usage_values=usage_values
        self.capacity=capacity
        self.nb_week=nb_week
        self.nb_scenarios=self.residual_load.nb_scenarios
        self.trajectories=np.zeros((self.nb_scenarios,self.nb_week))
        self.start_week=self.bellman_values.start_week
        self.end_week=self.bellman_values.end_week
        self.max_control=max_control
        self.calculate_trajectories()
    
    def calculate_trajectories(self)-> None:

        for s in range(self.nb_scenarios):
            remaining_capacity=self.capacity
            for w in range(self.start_week+1,self.end_week):
                if remaining_capacity==0:
                    self.trajectories[s,w-self.start_week]=0
                    continue
                
                thresholds=self.usage_values.usage_values[w,:remaining_capacity]
                week_start=(w-1)*7+2
                week_end=week_start+7
                week_days=self.residual_load.residual_load[week_start*24:week_end*24,s].reshape(7,24).sum(axis=1)
                sorted_load=(np.sort(week_days[:self.max_control]))[::-1]

                control = 0
                for d in range(min(self.max_control, remaining_capacity)):
                    idx = remaining_capacity - 1 - d
                    if sorted_load[d] < thresholds[idx]: 
                        break         # puis on s’arrête
                    control += 1      # sinon on continue


                self.trajectories[s, w - self.start_week] = control
                remaining_capacity -= control

                if remaining_capacity<0:
                    remaining_capacity=0
    
    def trajectory_for_scenario(self, scenario:int)-> np.ndarray:
        return self.trajectories[scenario]
