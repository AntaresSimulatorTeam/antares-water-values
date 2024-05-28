import numpy as np
from typing import Dict, Union, Any
from read_antares_data import TimeScenarioParameter

class Estimator:
    """ Generic class for estimators/interpolators """
    def __init__(self, param:TimeScenarioParameter) -> None:
        ...
        
    def update(self,
               param:TimeScenarioParameter,
               objective_values:np.ndarray,
               control_duals:Dict[str, np.ndarray]) -> None:
        ...
    
    def __getitem__(self, ws: Union[tuple[int,int], int]) -> Any:
        ...

        
class LinearInterpolator:
    """ Class to enable use of n-dimensionnal linear interpolation """
    def __init__(self, inputs:np.ndarray, costs:np.ndarray, duals:np.ndarray):
        """
        Instanciates a Linear Interpolator
        
        Parameters
        ----------
            inputs:np.ndarray: The coordinates for which costs / duals are obtained
                must have the same shape as what we'll call our interpolator with,
            costs:np.ndarray: Cost for every input,
            duals:np.ndarray: Duals for every input first dimension should be the same as inputs,
        """
        self.inputs = inputs
        self.costs = costs.ravel()
        self.duals = np.array([dual.ravel() for dual in duals])
    

    def __call__(self, x:np.ndarray) -> Union[np.ndarray, float]:
        """
        Interpolates between the saved points
        
        Parameters
        ----------
            x:np.ndarray: Array of coordinates for which we want to have the interpolation
                should have the same shape as the inputs
                
        Returns
        -------
            np.ndarray: 'best'/maximum interpolation for every coordinate
        """
        return np.max([self.costs[id] + np.dot(val-x, self.duals[:, id]) for id, val in enumerate(self.inputs)], axis=0)
    
    def alltile(self, x:np.ndarray) -> np.ndarray:
        """
        Debugging function used to see the interpolation proposed by every hyperplane we have
        
        Parameters
        ----------
            x:np.ndarray: Array of coordinates for which we want to have the interpolation
                should have the same shape as the inputs
        
        Returns
        -------
            np.ndarray All possible interpolations
        """
        return np.array([self.costs[id] + np.dot(val-x, self.duals[:, id]) for id, val in enumerate(self.inputs)])

class LinearCostEstimator(Estimator):
    """ A class to contain an ensemble of Interpolators for every week and scenario """
    def __init__(self, 
                 param:TimeScenarioParameter,
                 controls:np.ndarray,
                 costs:np.ndarray,
                 duals:np.ndarray) -> None:
        """
        Instanciates a LinearCostEstimator
        
        Parameters
        ----------
            param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on
            controls:np.ndarray: The coordinates for which costs / duals are obtained
                must have the same shape as what we'll call our interpolator with,
            costs:np.ndarray: Cost for every input,
            duals:np.ndarray: Duals for every input first dimension should be the same as inputs,
        """
        self.estimators = np.array([[LinearInterpolator(inputs=controls[week, scenario],
                                                        costs=costs[week, scenario],
                                                        duals=duals[week, scenario])\
            for scenario in range(param.len_scenario)]\
            for week in range(param.len_week)])
    
    def __getitem__(self, ws: Union[tuple[int,int], int]) -> LinearInterpolator:
        """
        Gets a LinearInterpolators
        
        Parameters
        ----------
            ws:tuple[int, int] / int: index of the week or scenario we want,
        
        Returns
        -------
            Array of ... or LinearInterpolator
        """
        return self.estimators[ws]
    
    def __call__(self, week: int, scenario: int, control:np.ndarray) -> float:
        """
        Directly gets an interpolation at given week / scenario
        
        Parameters
        ----------
            week:int: index of the week,
            scenario:int: index of the scenario,
            control:np.ndarray: control(s) to interpolate at
        
        Returns
        -------
            np.ndarray: interpolation(s) at control(s)
        """
        return self.estimators[week, scenario](control=control)