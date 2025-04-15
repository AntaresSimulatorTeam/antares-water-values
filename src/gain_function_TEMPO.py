import numpy as np
from read_antares_data import Residual_load
from dataclasses import dataclass


@dataclass
class GainFunctionTEMPO:
    def __init__(self, residual_load : Residual_load, max_control:int):
        self.residual_load=residual_load.residual_load
        self.nb_scenarios=residual_load.nb_scenarios
        self.daily_residual_load=residual_load.residual_load.reshape(365+65,24,self.nb_scenarios).sum(axis=1)
        self.max_control=max_control

    def gain_for_week_control_and_scenario(self, week_index: int, control : int, scenario : int) -> float:

        week_start=week_index*7+2
        week_end=week_start+7
        

        self.daily_residual_load_for_week=self.daily_residual_load[week_start:week_end,scenario] #consommation résiduelle par jour pour la semaine considérée


        self.daily_residual_load_for_week=(np.sort(self.daily_residual_load_for_week[:self.max_control]))[::-1]

        gain=np.sum(self.daily_residual_load_for_week[:control])
        return gain