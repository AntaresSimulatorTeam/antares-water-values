from read_antares_data import Residual_load
from calculate_gain_func_tempo_red import GainFunction
from dataclasses import dataclass
from calculate_bellman_values_with_reward_func import Bellman_values
import numpy as np

@dataclass
class UV_tempo_red:
    def __init__(self,residual_load : Residual_load, gain_function : GainFunction, bellman_values: Bellman_values, capacity : int, nb_week:int):
        self.residual_load=residual_load
        self.gain_function=gain_function
        self.bellman_values=bellman_values
        self.capacity=capacity
        self.nb_week=nb_week
        self.usage_values=np.zeros((52,self.capacity))
        self.compute_usage_values()

    def compute_usage_values(self) -> None:
        for w in range(18,18+self.nb_week):
            for c in range(1,self.capacity+1):
                self.usage_values[w,c-1]=self.bellman_values.mean_bv[w,c]-self.bellman_values.mean_bv[w,c-1]

        

