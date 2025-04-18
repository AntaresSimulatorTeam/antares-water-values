from bellman_values import BellmanValues
from dataclasses import dataclass
import numpy as np

@dataclass
class UsageValuesTempo:
    def __init__(self, 
                 bellman_values: BellmanValues):
        self.bellman_values=bellman_values

        self.capacity=self.bellman_values.capacity
        self.nb_week=self.bellman_values.nb_week
        self.start_week=bellman_values.start_week
        self.end_week=bellman_values.end_week

        self.usage_values=np.zeros((62,self.capacity))
        self.compute_usage_values()

    def compute_usage_values(self) -> None:
        for w in range(self.start_week,self.end_week):
            for c in range(1,self.capacity+1):
                self.usage_values[w,c-1]=self.bellman_values.mean_bv[w,c]-self.bellman_values.mean_bv[w,c-1]

        

