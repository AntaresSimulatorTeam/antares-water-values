import numpy as np
from typing import Dict, Union, Any
from read_antares_data import TimeScenarioParameter

class Estimator:
    """ Generic class for estimators/interpolators """
    def __init__(self, param:TimeScenarioParameter) -> None:
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
        self.duals = duals
    
    def update(self, inputs:np.ndarray, costs:np.ndarray, duals:np.ndarray) -> None:
        """
        Updates the parameters of the Linear Interpolator
        
        Parameters
        ----------
            inputs:np.ndarray: The coordinates for which costs / duals are obtained
                must have the same shape as what we'll call our interpolator with,
            costs:np.ndarray: Cost for every input,
            duals:np.ndarray: Duals for every input first dimension should be the same as inputs,
        """
        self.inputs = np.concatenate([self.inputs, inputs])
        self.costs = np.concatenate([self.costs, costs])
        self.duals = np.concatenate([self.duals, duals])
        
    def remove(self, ids:Union[list[int], np.ndarray]) -> None:
        """
        Remove some approximation from the Linear Interpolator
        
        Parameters
        ----------
            ids:np.ndarray: ids of the approximations to remove
            
        """
        self.inputs = np.delete(self.inputs, ids, axis=0)
        self.costs = np.delete(self.costs, ids)
        self.duals = np.delete(self.duals, ids, axis=0)

    def __call__(self, x:np.ndarray) -> Union[np.ndarray, float]:
        """
        Interpolates between the saved points
        
        Parameters
        ----------
            x:np.ndarray: Array of coordinates for which we want to have the interpolation
                should have the same shape as the inputs
                
        Returns
        -------
            np.ndarray: 'best'/maximum interpolation for every point given
        """
        return np.max([self.costs[id] + np.dot(val-x, -self.duals[id]) for id, val in enumerate(self.inputs)], axis=0)
    
    
    def get_owner(self, x:np.ndarray) -> Any:
        """
        Interpolates between the saved points and returns the id of the subgradient active for each interpolation 
        
        Parameters
        ----------
            x:np.ndarray: Array of coordinates for which we want to have the interpolation
                should have the same shape as the inputs
                
        Returns
        -------
            np.ndarray: id of 'best'/maximum subgradient for every point given
        """
        return np.argmax([self.costs[id] + np.dot(val-x, -self.duals[id]) for id, val in enumerate(self.inputs)], axis=0)
    
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
        return np.array([self.costs[id] + np.dot(val-x, -self.duals[id]) for id, val in enumerate(self.inputs)])
    
    def remove_approximations(self, controls:np.ndarray, real_costs:np.ndarray) -> None:
        """
        Removes all hyperplanes above the real values at specified controls
        
        Parameters
        ----------
            controls:np.ndarray: controls at which we know the real costs,
            real_costs:np.ndarray: real costs at controls
        """
        estimated_costs = self.alltile(controls)
        #Abnormality is when an hyperplane gives an estimation over the real price (when it should be under)
        are_abnormal = estimated_costs.T - real_costs[:, None] >= 0
        has_abnormality = np.sum(are_abnormal, axis=0) > 0
        ids = [i for i, abn in enumerate(has_abnormality) if abn]
        self.remove(ids=ids)
        
    def limited(self, x:np.ndarray, chosen_pts:Union[list[bool], np.ndarray]) -> Union[np.ndarray, float]:
        """
        Interpolates between the chosen points
        
        Parameters
        ----------
            x:np.ndarray: Array of coordinates for which we want to have the interpolation
                should have the same shape as the inputs,
            
            chosen_pts:Union[list, np.ndarray]: Points allowed to participate to interpolation
                
        Returns
        -------
            np.ndarray: 'best'/maximum interpolation for every coordinate using those points
        """
        return np.max([self.costs[id] + np.dot(val-x, -self.duals[id]) for id, (val, chosen) in enumerate(zip(self.inputs, chosen_pts)) if chosen], axis=0)
    
    def count_redundant(self, tolerance:float, remove:bool=False) -> tuple[int, list]:
        """
        Counts the number of estimation points that are already given by the others
        
        Parameters
        ----------
            precision:float: Tolerance on how different the points information from the ones we currently 
            remove:bool: Whether or not we want to delete the observations described as redundant
        
        Returns
        -------
            n:int: number of points that could be removed without damaging the prediction
            non_redundants:list: ids of non redundant points
        """
        n_inputs =self.inputs.shape[0]
        
        # Initiate the good points list
        non_redundants = [True] + [False]*(n_inputs - 1)
        #We won't look at the difference in derivative at a point, even though it could be important
        for i in range(1, n_inputs):
            non_redundants[i] = np.abs(self.limited(self.inputs[i], non_redundants) - self.costs[i]) >= tolerance
            
        redundants = [i for i, not_reddt in enumerate(non_redundants) if not(not_reddt)]
        #Remove the redundant observations (might be needed if bellman problems become too heavy, might never happen tho)
        if remove:
            self.remove(ids=redundants)
        return len(redundants), non_redundants
        

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
    
    def update(self, inputs:np.ndarray, costs:np.ndarray, duals:np.ndarray) -> None:
        """
        Updates the parameters of the Linear Interpolators
        
        Parameters
        ----------
            inputs:np.ndarray: The coordinates for which costs / duals are obtained
                must have the same shape as what we'll call our interpolator with,
            costs:np.ndarray: Cost for every input,
            duals:np.ndarray: Duals for every input first dimension should be the same as inputs,
        """
        for week, (inputs_w, costs_w, duals_w) in enumerate(zip(inputs, costs, duals)):
            for scenario, (inputs, costs, duals) in enumerate(zip(inputs_w, costs_w, duals_w)):
                self.estimators[week, scenario].update(input=inputs, costs=costs, duals=duals)