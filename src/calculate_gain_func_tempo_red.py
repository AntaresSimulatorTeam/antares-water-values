import numpy as np
from read_antares_data import Residual_load
from dataclasses import dataclass
from itertools import islice


@dataclass
class GainFunction:
    def __init__(self, residual_load : Residual_load):
        self.residual_load=residual_load.residual_load
        self.nb_scenarios=residual_load.nb_scenarios
        self.daily_residual_load=residual_load.residual_load.reshape(365,24,self.nb_scenarios).sum(axis=1)

    def gain_for_week_scenario_and_control(self, week_index: int, u : int, scenario : int) -> float:

        week_start=week_index*7
        week_end=week_start+5

        daily_residual_load_for_week=self.daily_residual_load[week_start:week_end,scenario-1]
        daily_residual_load_for_week=(np.sort(daily_residual_load_for_week))[::-1]

        gain=np.sum(daily_residual_load_for_week[:u])
        return gain