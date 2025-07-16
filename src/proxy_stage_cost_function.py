from read_antares_data import Reservoir
from read_antares_data import NetLoad
import numpy as np
from scipy.interpolate import interp1d

class Proxy:
    def __init__(self, dir_study: str, name_area: str, MC_years:int, alpha:float, coeff:float) -> None:
        self.dir_study = dir_study
        self.name_area = name_area
        self.reservoir = Reservoir(dir_study, name_area)

        # self.max_daily_generating=self.reservoir.max_daily_generating-1
        # self.max_daily_pumping=self.reservoir.max_daily_pumping-1 if not np.allclose(self.reservoir.max_daily_pumping,0) else self.reservoir.max_daily_pumping

        self.turb_efficiency=1
        self.alpha=alpha
        self.coeff = coeff

        self.nb_weeks=52
        self.scenarios=range(MC_years)
        
        self.weighted_net_load = self.compute_weighted_net_load()

    
    def compute_weighted_net_load(self)-> np.ndarray:
        weighted_net_load = np.zeros((365 * 24, 200))
        for key, value in self.reservoir.allocation_dict.items():
            weighted_net_load += value * NetLoad(self.dir_study, key).compute_net_load()

        return weighted_net_load
    
    def compute_turb_and_pump_with_thresholds(self, 
                                              turb_thresholds : np.ndarray, 
                                              weekly_net_load : np.ndarray, 
                                              max_hourly_turb : np.ndarray, 
                                              max_hourly_pump : np.ndarray, 
                                              null_pump : bool) -> tuple:
        hourly_turb = []
        hourly_pump = []
        weekly_control = []
        costs = []
        
        for turb_threshold in turb_thresholds:
            clipped_net_load = np.minimum(weekly_net_load, np.maximum(turb_threshold, weekly_net_load - max_hourly_turb))
            turb = weekly_net_load - clipped_net_load

            if not null_pump:
                if turb_threshold < 0:
                    pump_threshold = turb_threshold
                else:
                    pump_threshold = ((self.reservoir.efficiency / self.turb_efficiency) ** (1 / (self.alpha - 1))) * turb_threshold
                potential_pump = pump_threshold - clipped_net_load
                mask = clipped_net_load < pump_threshold
                pump = np.minimum(potential_pump, max_hourly_pump) * mask
            else:
                pump = np.zeros_like(clipped_net_load)

            clipped_net_load += pump

            hourly_turb.append(np.sum(turb * self.turb_efficiency))
            hourly_pump.append(np.sum(pump * self.reservoir.efficiency))
            hourly_control = turb * self.turb_efficiency - pump * self.reservoir.efficiency
            weekly_control.append(np.sum(hourly_control))
            cost = np.sum(np.abs(clipped_net_load) ** self.alpha / self.coeff)
            costs.append(cost)
        
        return hourly_turb,hourly_pump,weekly_control,costs



    def stage_cost_function(self, week: int, scenario: int) -> np.ndarray:
        weekly_net_load = self.weighted_net_load[week * 168:(week + 1) * 168, scenario]
        max_hourly_turb = self.reservoir.max_hourly_turb[(week)*168:(week+1)*168]
        max_hourly_pump = self.reservoir.max_hourly_pump[(week)*168:(week+1)*168]
        null_pump = np.allclose(self.reservoir.max_hourly_pump, 0)
        
        turb_thresholds = np.quantile(
            np.linspace(
                np.min(weekly_net_load - max_hourly_turb),
                np.max(weekly_net_load + max_hourly_pump) * (self.turb_efficiency / self.reservoir.efficiency) ** (1 / (self.alpha - 1))
            ),
            np.linspace(0, 1, 25)
        )

        hourly_turb, hourly_pump, weekly_control, costs = self.compute_turb_and_pump_with_thresholds(
            turb_thresholds=turb_thresholds,
            weekly_net_load=weekly_net_load,
            max_hourly_turb=max_hourly_turb,
            max_hourly_pump=max_hourly_pump,
            null_pump=null_pump
        )

        return np.array([
            interp1d(weekly_control, costs, fill_value="extrapolate"),
            interp1d(weekly_control, hourly_turb, fill_value="extrapolate"),
            interp1d(weekly_control, hourly_pump, fill_value="extrapolate")
        ])


    def compute_stage_cost_functions(self)->np.ndarray:
        cost_functions=np.array([[self.stage_cost_function(w,s) for s in self.scenarios] for w in range(self.nb_weeks)])
        return cost_functions
