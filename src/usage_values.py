from read_antares_data import Residual_load
from gain_function_TEMPO import GainFunctionTEMPO
from bellman_values import Bellman_values
from dataclasses import dataclass
import numpy as np

@dataclass
class UV_tempo:
    def __init__(self,residual_load : Residual_load, 
                 gain_function : GainFunctionTEMPO, 
                 bellman_values: Bellman_values):
        self.residual_load=residual_load
        self.gain_function=gain_function
        self.bellman_values=bellman_values
        self.capacity=self.bellman_values.capacity
        self.nb_week=self.bellman_values.nb_week
        self.start_week=bellman_values.start_week
        self.end_week=bellman_values.end_week
        self.usage_values=np.zeros((61,self.capacity))
        self.compute_usage_values()

    def compute_usage_values(self) -> None:
        for w in range(self.start_week,self.end_week):
            for c in range(1,self.capacity+1):
                self.usage_values[w,c-1]=self.bellman_values.mean_bv[w,c]-self.bellman_values.mean_bv[w,c-1]

        

