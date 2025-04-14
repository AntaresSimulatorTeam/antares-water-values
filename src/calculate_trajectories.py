from read_antares_data import Residual_load
from calculate_gain_func_tempo_red import GainFunction
from calculate_bellman_values_with_reward_func import Bellman_values
from calculate_uv_tempo import UV_tempo_red
from dataclasses import dataclass
import numpy as np

@dataclass
class Trajectories :
    def __init__(self, 
                residual_load:Residual_load, 
                gain_function:GainFunction, 
                bellman_values:Bellman_values, 
                usage_values:UV_tempo_red,
                capacity:int, 
                nb_week:int):
        self.residual_load=residual_load
        self.gain_function=gain_function
        self.bellman_values=bellman_values
        self.usage_values=usage_values
        self.capacity=capacity
        self.nb_week=nb_week
        self.nb_scenarios=self.residual_load.nb_scenarios
        self.trajectories=np.zeros((self.nb_scenarios,self.nb_week))
        self.calculate_trajectories()
    
    def calculate_trajectories(self)-> None:

        for s in range(self.nb_scenarios):
            remaining_capacity=self.capacity
            for w in range(18,18+self.nb_week):
                if remaining_capacity==0:
                    self.trajectories[s,w-18]=0
                    continue
                
                thresholds=self.usage_values.usage_values[w,:remaining_capacity]
                week_start=w*7+2
                week_end=week_start+5
                week_days=self.residual_load.residual_load[week_start*24:week_end*24,s].reshape(5,24).sum(axis=1)
                sorted_load=np.sort(week_days)[::-1]

                control = 0
                for d in range(min(5,remaining_capacity)):
                    idx=remaining_capacity-1-d
                    if sorted_load[d]>thresholds[idx] and idx>=0:
                        control+=1
                        print("conso res: ", sorted_load[d], " seuil: ", thresholds[idx], " semaine : ",w," stock : ", remaining_capacity-control, " scenario : " , s)

                    else:
                        break

                self.trajectories[s, w - 18] = control
                remaining_capacity -= control

                if remaining_capacity<0:
                    remaining_capacity=0
    
    def trajectory_for_scenario(self, scenario:int)-> np.ndarray:
        return self.trajectories[scenario]
