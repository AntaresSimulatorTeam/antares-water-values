from tracemalloc import start
from read_antares_data import Reservoir,NetLoad
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

THRESHOLDS = 25

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
        Compute the weighted net load using the allocation weights.

        Returns:
            np.ndarray: A (8760, nb_scenarios) array of hourly weighted net load for each scenario.
        """
        weighted_net_load = np.zeros((365 * 24, len(self.scenarios)))
        for key, value in self.reservoir.allocation_dict.items():
            net_load = NetLoad(self.reservoir,self.dir_study, key).net_load
            weighted_net_load += value * net_load[:,:len(self.scenarios)]

        return weighted_net_load

    def compute_control_with_thresholds(
        self,
        turb_thresholds: np.ndarray,
        weekly_net_load: np.ndarray,
        max_hourly_turb: np.ndarray,
        max_hourly_pump: np.ndarray,
        null_pump: bool
    ) -> tuple:
        """
        Compute weekly control and cost using
        threshold-based hourly policies.

        Each policy is defined by a turbining threshold applied to the hourly
        net load: turbining is activated when the net load is above this
        threshold and limited by the maximum hourly turbining capacity.
        When pumping is allowed, a corresponding pumping threshold is defined as:
            pump_threshold =
                turb_threshold                                      if turb_threshold < 0
                turb_threshold * (eta_pump / eta_turb)^(1/(alpha-1)) otherwise

        Pumping is activated when the net load is below the pumping threshold,
        within the limits of the maximum hourly pumping capacity. Hourly
        turbining and pumping are aggregated over the week to compute the
        net weekly control and the associated transition cost.
        """
        nl = weekly_net_load[None, :] 
        mt = max_hourly_turb[None, :] 
        mp = max_hourly_pump[None, :]    
        tt = turb_thresholds[:, None]

        # Turbining clip
        clipped = np.minimum(nl, np.maximum(tt, nl - mt))  
        turb = nl - clipped                                

        # Pumping
        if null_pump:
            pump = np.zeros_like(clipped)               
        else:
            ratio = (self.reservoir.efficiency / self.turb_efficiency) ** (1 / (self.alpha - 1))

            pump_thresholds = np.where(
                turb_thresholds < 0,
                turb_thresholds,
                ratio * turb_thresholds
            )                                              

            pt = pump_thresholds[:, None]                  
            potential_pump = pt - clipped             
            mask = clipped < pt                        

            pump = np.where(mask, np.minimum(potential_pump, mp), 0.0)

        net_after = clipped + pump                         

        # Aggregations
        weekly_control = (turb * self.turb_efficiency - pump * self.reservoir.efficiency).sum(axis=1)
        costs = (np.abs(net_after) ** self.alpha).sum(axis=1)

        return weekly_control, costs


    def stage_cost_function(self, week: int, scenario: int) -> tuple:
        """
        Compute the cost for a given week and scenario,
        as functions of the weekly energy control.

        Returns:
            Interpolator (scipy interp1d): cost(control),
        """
        weekly_net_load = self.weighted_net_load[week * 168:(week + 1) * 168, scenario]
        max_hourly_turb = self.reservoir.max_hourly_turb[(week)*168:(week+1)*168]
        max_hourly_pump = self.reservoir.max_hourly_pump[(week)*168:(week+1)*168]
        null_pump = np.allclose(self.reservoir.max_hourly_pump, 0)
        
        low = np.min(weekly_net_load - max_hourly_turb)
        raw_high = np.max(weekly_net_load + max_hourly_pump) * (self.turb_efficiency / self.reservoir.efficiency) ** (1 / (self.alpha - 1))
        high = np.max(weekly_net_load) if null_pump else raw_high

        turb_thresholds = np.unique(np.linspace(low, high, THRESHOLDS))

        weekly_control, costs = self.compute_control_with_thresholds(
            turb_thresholds=turb_thresholds,
            weekly_net_load=weekly_net_load,
            max_hourly_turb=max_hourly_turb,
            max_hourly_pump=max_hourly_pump,
            null_pump=null_pump
        )
        
        idx = np.argsort(weekly_control)
        weekly_control = weekly_control[idx]
        costs = costs[idx]

        ub_cost = float(max(costs[0], costs[-1]))

        stage_cost_function = interp1d(weekly_control, costs, fill_value="extrapolate")
        
        return stage_cost_function,ub_cost


    def compute_stage_cost_functions(self)->np.ndarray:
        """
        Compute and store cost-related interpolators for all weeks and scenarios.

        Returns:
            np.ndarray: Array of shape (nb_weeks, nb_scenarios), containing
            stage cost function.
        """
        cost_functions = np.empty(
            (self.nb_weeks, len(self.scenarios)), 
            dtype=object
        )

        self.stage_cost_upper_bounds = np.empty((self.nb_weeks, len(self.scenarios)), dtype=float)

        if hasattr(self, "pbar"):
            self.pbar.set_postfix_str("Stage cost functions computing")        
        for w in range(self.nb_weeks):
            for s in self.scenarios:
                if hasattr(self,"pbar"):
                    self.pbar.update(1)

                stage_cost_function, ub = self.stage_cost_function(w, s)
                cost_functions[w, s] = stage_cost_function
                self.stage_cost_upper_bounds[w, s] = ub

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
        return float(self.stage_cost_upper_bounds[week].max())