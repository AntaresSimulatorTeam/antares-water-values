from tracemalloc import start
from read_antares_data import Reservoir,NetLoad
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm



class ProxyStageCostFunction:
    def __init__(self, dir_study: str, name_area: str, MC_years:int, alpha:float,pbar:tqdm) -> None:
        """
        Initialize the object with study directory, area, number of Monte-Carlo scenarios,
        cost exponent alpha.

        Args:
            dir_study (str): Path to study directory.
            name_area (str): Name of the area.
            MC_years (int): Number of Monte-Carlo scenarios.
            alpha (float): Exponent for cost function.
        """
        self.dir_study = dir_study
        self.name_area = name_area
        self.reservoir = Reservoir(dir_study, name_area)
        self.pbar = pbar
        

        self.turb_efficiency=1
        self.alpha=alpha

        self.nb_weeks=52
        self.scenarios=range(self.reservoir.nb_scenarios)[:MC_years]
        
        self.weighted_net_load = self.compute_weighted_net_load()
        self.stage_cost_functions = self.compute_stage_cost_functions()

    
    def compute_weighted_net_load(self)-> np.ndarray:
        """
        Compute the weighted net load across all areas using the allocation weights.

        Returns:
            np.ndarray: A (8760, nb_scenarios) array of hourly weighted net load for each scenario.
        """
        weighted_net_load = np.zeros((365 * 24, len(self.scenarios)))
        for key, value in self.reservoir.allocation_dict.items():
            net_load = NetLoad(self.reservoir,self.dir_study, key).net_load
            weighted_net_load += value * net_load[:,:len(self.scenarios)]

        return weighted_net_load
    
    def compute_turb_and_pump_with_thresholds(self, 
                                              turb_thresholds : np.ndarray, 
                                              weekly_net_load : np.ndarray, 
                                              max_hourly_turb : np.ndarray, 
                                              max_hourly_pump : np.ndarray, 
                                              null_pump : bool) -> tuple:
        """
        Compute weekly turbine and pump energy, control, and cost for given thresholds 
        of net load.

        Returns:
            tuple: (hourly_turb, hourly_pump, weekly_control, costs)
        """  
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
            cost = np.sum(np.abs(clipped_net_load) ** self.alpha)
            costs.append(cost)
        
        return hourly_turb,hourly_pump,weekly_control,costs



    def stage_cost_function(self, week: int, scenario: int) -> np.ndarray:
        """
        Compute the cost, turbined energy, and pumped energy for a given week and scenario,
        as functions of the weekly energy control.

        For each turbining threshold, the corresponding pumping threshold is computed using:
            pump_threshold = turb_threshold * (η_pump / η_turb)^{1 / (α - 1)}

        where η_pump is the pumping efficiency, η_turb the turbining efficiency, and α is the convexity
        exponent of the cost function.

        Returns:
            np.ndarray: Array of three interpolators (scipy interp1d):
                - cost(control),
                - turbined_energy(control),
                - pumped_energy(control)
        """
        weekly_net_load = self.weighted_net_load[week * 168:(week + 1) * 168, scenario]
        max_hourly_turb = self.reservoir.max_hourly_turb[(week)*168:(week+1)*168]
        max_hourly_pump = self.reservoir.max_hourly_pump[(week)*168:(week+1)*168]
        null_pump = np.allclose(self.reservoir.max_hourly_pump, 0)
        
        low = np.min(weekly_net_load - max_hourly_turb)
        raw_high = np.max(weekly_net_load + max_hourly_pump) * (self.turb_efficiency / self.reservoir.efficiency) ** (1 / (self.alpha - 1))
        high = np.max(weekly_net_load) if null_pump else raw_high

        turb_thresholds = np.linspace(low, high, 25)

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
        """
        Compute and store cost-related interpolators for all weeks and scenarios.

        For each (week, scenario) pair, computes:
            - cost(control),
            - turbined_energy(control),
            - pumped_energy(control)

        Returns:
            np.ndarray: Array of shape (nb_weeks, nb_scenarios), containing
            3-element arrays of scipy interp1d interpolators.
        """
        cost_functions = np.empty(
            (self.nb_weeks, len(self.scenarios), 3), 
            dtype=object
        )
        if hasattr(self, "pbar"):
            self.pbar.set_postfix_str("Stage cost functions computing")        
        for w in range(self.nb_weeks):
            for s in self.scenarios:
                if hasattr(self,"pbar"):
                    self.pbar.update(1)
                cost_functions[w,s]=self.stage_cost_function(w,s)
        return cost_functions
        

    def upper_bound_cost(self, week: int) -> float:
        """
        Compute a conservative upper bound on the stage cost for a given week.

        For each scenario at the given week, this inspects the interpolated
        stage-cost function c_w^s(u) at the two extreme control values available
        in its grid (controls[0] and controls[-1]), and returns the maximum over
        both extremes and all scenarios:

            ub_cost = max_s  max( c_w^s(u_min), c_w^s(u_max) )

        Args:
            week (int): Week index.

        Returns:
            float: Upper bound of the stage cost for the given week across scenarios.
        """
        ub_cost=0
        for s in self.scenarios:
            stage_cost_function = self.stage_cost_functions[week,s][0]
            controls = stage_cost_function.x
            max_cost_turb = stage_cost_function(controls[-1])
            max_cost_pump = stage_cost_function(controls[0])
            ub_cost_new = max(max_cost_turb,max_cost_pump)
            ub_cost=max(ub_cost,ub_cost_new)
        return ub_cost