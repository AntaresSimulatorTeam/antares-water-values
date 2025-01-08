import numpy as np
from scipy.interpolate import interp1d

from hyperplane_decomposition import decompose_hyperplanes
from hyperplane_interpolation import get_interpolation
from read_antares_data import TimeScenarioIndex, TimeScenarioParameter
from reservoir_management import MultiStockManagement
from stock_discretization import StockDiscretization
from type_definition import Array1D, Callable, Dict, List, Optional, Union


class Estimator:

    def __init__(self) -> None:
        raise NotImplementedError

    def update(
        self,
        Vu: float,
        slope: Dict[str, float],
        n_scenario: int,
        idx: int,
        list_areas: List[str],
    ) -> None:
        raise NotImplementedError

    def get_value(self, x: Dict[str, float]) -> float:
        return NotImplemented


class PieceWiseLinearInterpolator:

    def __init__(
        self,
        controls: Array1D,
        costs: Array1D,
    ):
        self.inputs = controls
        self.costs = costs

    def __call__(self, x: float) -> float:
        fn = interp1d(self.inputs, self.costs)
        return fn(x)


class UniVariateEstimator(Estimator):

    def __init__(
        self,
        estimators: Dict[str, PieceWiseLinearInterpolator],
    ):
        self.estimators = estimators

    def __getitem__(self, key: str) -> PieceWiseLinearInterpolator:
        return self.estimators[key]

    def update(
        self,
        Vu: float,
        slope: Dict[str, float],
        n_scenario: int,
        idx: int,
        list_areas: List[str],
    ) -> None:
        for area in list_areas:
            self.estimators[area].costs[idx] += -Vu / n_scenario

    def get_value(self, x: Dict[str, float]) -> float:
        return -sum([self.estimators[area](y) for area, y in x.items()])


class BellmanValueEstimation(Estimator):

    def __init__(
        self, value: Dict[str, Array1D], stock_discretization: StockDiscretization
    ):
        self.V = value
        self.discretization = stock_discretization

    def __getitem__(self, key: str) -> Array1D:
        return self.V[key]

    def update(
        self,
        Vu: float,
        slope: Dict[str, float],
        n_scenario: int,
        idx: int,
        list_areas: List[str],
    ) -> None:
        self.V["intercept"][idx] += Vu / n_scenario
        for area in list_areas:
            self.V[f"slope_{area}"][idx] += slope[area] / n_scenario

    def get_value(self, x: Dict[str, float]) -> float:
        return max(
            [
                sum(
                    [
                        self.V[f"slope_{area}"][idx]
                        * (
                            x[area.area]
                            - self.discretization.list_discretization[area][idx[i]]
                        )
                        for i, area in enumerate(
                            self.discretization.list_discretization.keys()
                        )
                    ]
                )
                + self.V["intercept"][idx]
                for idx in self.discretization.get_product_stock_discretization()
            ]
        )


class MultiVariateEstimator:

    def __init__(
        self, controls: Optional[np.ndarray], costs: np.ndarray, duals: np.ndarray
    ):
        if controls is not None:
            self.true_inputs = controls
            self.inputs = controls
        self.costs = costs.ravel()
        self.duals = duals
        self.true_costs = costs.ravel()
        self.true_duals = duals

    def update(
        self,
        costs: Union[np.ndarray, float],
        duals: Union[np.ndarray, float],
        controls: Optional[np.ndarray],
    ) -> None:
        raise NotImplementedError

    def count_redundant(
        self, tolerance: float, remove: bool = False
    ) -> tuple[int, list[bool]]:
        return 0, [True]

    def remove_interps(self) -> None:
        pass

    def round(self, precision: int = 6) -> None:
        pass

    def __call__(self, x: np.ndarray) -> float:
        return NotImplemented

    def to_julia_dict(self) -> Dict[str, np.ndarray]:
        return NotImplemented


class LinearInterpolator(MultiVariateEstimator):
    """Class to enable use of n-dimensionnal linear interpolation"""

    def __init__(
        self,
        controls: Optional[np.ndarray],
        costs: np.ndarray,
        duals: np.ndarray,
        interp_mode: Optional[bool] = False,
    ):
        """
        Instanciates a Linear Interpolator

        Parameters
        ----------
            inputs:np.ndarray: The coordinates for which costs / duals are obtained
                must have the same shape as what we'll call our interpolator with,
            costs:np.ndarray: Cost for every input,
            duals:np.ndarray: Duals for every input first dimension should be the same as inputs,
        """
        assert controls is not None
        self.inputs = controls
        self.costs = costs.ravel()
        self.duals = duals
        self.true_inputs = controls
        self.true_costs = costs.ravel()
        self.true_duals = duals
        if interp_mode:
            self.add_interpolations()
            self.remove_incoherence()

    def update(
        self,
        costs: Union[np.ndarray, float],
        duals: Union[np.ndarray, float],
        controls: Optional[np.ndarray],
        interp_mode: Optional[bool] = False,
    ) -> None:
        """
        Updates the parameters of the Linear Interpolator

        Parameters
        ----------
            inputs:np.ndarray: The coordinates for which costs / duals are obtained
                must have the same shape as what we'll call our interpolator with,
            costs:np.ndarray: Cost for every input,
            duals:np.ndarray: Duals for every input first dimension should be the same as inputs,
        """
        assert controls is not None
        self.inputs = np.concatenate([self.inputs, controls])
        self.costs = np.concatenate([self.costs, costs])
        self.duals = np.concatenate([self.duals, duals])
        self.true_inputs = np.concatenate([self.true_inputs, controls])
        self.true_costs = np.concatenate([self.true_costs, costs])
        self.true_duals = np.concatenate([self.true_duals, duals])
        if interp_mode:
            self.add_interpolations()
            self.remove_incoherence()

    def remove(self, ids: Union[list[int], np.ndarray]) -> None:
        """
        Remove some approximation from the Linear Interpolator

        Parameters
        ----------
            ids:np.ndarray: ids of the approximations to remove

        """
        inputs_shape = self.inputs.shape
        self.inputs = np.delete(self.inputs, ids, axis=0)
        self.costs = np.delete(self.costs, ids)
        self.duals = np.delete(self.duals, ids, axis=0)
        if self.inputs.size == 0:
            self.inputs = np.zeros(inputs_shape)
            self.costs = np.array([0])
            self.duals = np.zeros(inputs_shape)

    def __call__(self, x: np.ndarray) -> float:
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
        return np.max(
            [
                self.costs[id] + np.dot(x - val, self.duals[id])
                for id, val in enumerate(self.inputs)
            ],
            axis=0,
        )

    def dualize(self, x: np.ndarray) -> float:
        return self.duals[
            np.argmax(
                [
                    self.costs[id] + np.dot(x - val, self.duals[id])
                    for id, val in enumerate(self.inputs)
                ],
                axis=0,
            )
        ]

    def get_owner(self, x: np.ndarray) -> List[int]:
        """
        Interpolates between the saved points and returns the id of the subgradient active for each interpolation

        Parameters
        ----------
            x:np.ndarray: Array of coordinates for which we want to have the interpolation
                should have the same shape as the inputs

        Returns
        -------
            np.ndarray: id of 'best'/maximum subgradivalent for every point given
        """
        return np.argmax(
            [
                self.costs[id] + np.dot(val - x, -self.duals[id])
                for id, val in enumerate(self.inputs)
            ],
            axis=0,
        )

    def alltile(self, x: np.ndarray) -> np.ndarray:
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
        return np.array(
            [
                self.costs[id] + np.dot(x - val, self.duals[id])
                for id, val in enumerate(self.inputs)
            ]
        )

    def add_interpolations(self, n_splits: int = 3) -> None:
        new_conts, new_costs, new_slopes = get_interpolation(
            controls_init=self.true_inputs,
            costs=self.true_costs,
            slopes=self.true_duals,
            n_splits=n_splits,
        )
        self.inputs = new_conts
        self.costs = new_costs
        self.duals = new_slopes

    def remove_incoherence(self) -> None:
        """
        Removes all hyperplanes above the real values at specified controls

        Parameters
        ----------
            controls:np.ndarray: controls at which we know the real costs,
            real_costs:np.ndarray: real costs at controls
        """
        estimated_costs = self.alltile(self.true_inputs)
        # Abnormality is when an hyperplane gives an estimation over the real price (when it should be under)
        are_abnormal = estimated_costs.T - self.true_costs[:, None] > 100
        has_abnormality = np.sum(are_abnormal, axis=0) > 0
        ids = [i for i, abn in enumerate(has_abnormality) if abn]
        self.remove(ids=ids)

    def remove_interps(self) -> None:
        self.inputs = self.true_inputs
        self.costs = self.true_costs
        self.duals = self.true_duals

    def limited(
        self, x: np.ndarray, chosen_pts: Union[list[bool], np.ndarray]
    ) -> Union[np.ndarray, float]:
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
        return np.max(
            [
                self.costs[id] + np.dot(val - x, -self.duals[id])
                for id, (val, chosen) in enumerate(zip(self.inputs, chosen_pts))
                if chosen
            ],
            axis=0,
        )

    def count_redundant(
        self, tolerance: float, remove: bool = False
    ) -> tuple[int, list[bool]]:
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
        n_inputs = self.inputs.shape[0]

        # Initiate the good points list
        non_redundants = [True] + [False] * (n_inputs - 1)

        # We won't look at the difference in derivative at a point, even though it could be important
        for i in range(1, n_inputs):
            non_redundants[i] = (
                np.abs(self.limited(self.inputs[i], non_redundants) - self.costs[i])
                > tolerance
            )

        redundants = [
            i for i, not_reddt in enumerate(non_redundants) if not (not_reddt)
        ]
        # Remove the redundant observations (might be needed if bellman problems become too heavy, might never happen tho)
        if remove:
            self.remove(ids=redundants)
        return len(redundants), non_redundants

    def round(self, precision: int = 6) -> None:
        self.inputs = np.round(self.inputs, precision)
        self.costs = np.round(self.costs, precision)
        self.duals = np.round(self.duals, precision)
        self.true_inputs = np.round(self.true_inputs, precision)
        self.true_costs = np.round(self.true_costs, precision)
        self.true_duals = np.round(self.true_duals, precision)

    def to_julia_dict(self) -> Dict[str, np.ndarray]:
        return {"inputs": self.inputs, "costs": self.costs, "duals": self.duals}


class LinearDecomposer(LinearInterpolator):
    """Class intended to superpose linear interpolators as a way to decompose a multivariate function
    as a sum of monovariate functions"""

    def __init__(
        self,
        inputs: np.ndarray,
        costs: np.ndarray,
        duals: np.ndarray,
        correlations: Optional[np.ndarray] = None,
    ):
        """
        Instanciates a Linear Interpolator

        Parameters
        ----------
            inputs:np.ndarray: The coordinates for which costs / duals are obtained
                must have the same shape as what we'll call our interpolator with,
            costs:np.ndarray: Cost for every input,
            duals:np.ndarray: Duals for every input first dimension should be the same as inputs,
        """
        n_reservoirs = inputs.shape[-1]
        n_inputs = inputs.shape[0]
        if correlations is None:
            correlations = np.eye(n_reservoirs)

        inputs_decomp, _, duals_decomp = decompose_hyperplanes(
            inputs=inputs, costs=costs, slopes=duals, correlations=correlations
        )

        self.correlations = correlations
        self.lower_bound = LinearInterpolator(controls=inputs, costs=costs, duals=duals)
        self.layers: list[LinearInterpolator] = [
            LinearInterpolator(inp, np.zeros(inp.shape[0]), slp)
            for inp, slp in zip(inputs_decomp, duals_decomp)
        ]
        # Approximate inputs
        self.inputs = np.concatenate(
            [inputs, inputs_decomp.reshape(n_inputs * n_reservoirs, n_reservoirs)]
        )
        self.costs = np.concatenate([costs, np.zeros(n_inputs * n_reservoirs)])
        self.duals = np.concatenate(
            [duals, duals_decomp.reshape(n_inputs * n_reservoirs, n_reservoirs)]
        )
        self.true_inputs = inputs
        self.true_costs = costs
        self.true_duals = duals
        self.remove_inconsistence()
        self.remove_incoherence()

    def __call__(self, x: np.ndarray) -> float:
        if len(x.shape) > 1:
            x = x[0]
        return np.maximum(
            self.lower_bound(x), np.sum([layer(x) for layer in self.layers], axis=0)
        )

    def remove_inconsistence(self, tolerance: float = 1) -> None:
        inputs, costs = self.lower_bound.inputs, self.lower_bound.costs
        assert all(costs + tolerance > 0)
        guesses = np.array([layer(inputs) for layer in self.layers])  # N_res * N_inp
        while any(np.sum(guesses, axis=0) > costs + tolerance):
            # Identify likely source of error
            bad_guesses = np.sum(guesses, axis=0) > costs + tolerance
            # Removing first potential source of pb
            first_pb_inp = inputs[bad_guesses][0]
            bad_lay = [layer for layer in self.layers if layer(first_pb_inp) > 0][-1]
            bad_lay.remove(bad_lay.get_owner(first_pb_inp))
            guesses = np.array(
                [layer(inputs) for layer in self.layers]
            )  # N_res * N_inp

    def update(
        self,
        costs: Union[np.ndarray, float],
        duals: Union[np.ndarray, float],
        controls: Optional[np.ndarray],
        interp_mode: Optional[bool] = None,
    ) -> None:

        assert controls is not None
        n_reservoirs = controls.shape[-1]
        n_inputs = controls.shape[0]
        # Compute decomposition
        inputs_decomp, _, duals_decomp = decompose_hyperplanes(
            inputs=controls,
            costs=np.array(costs),
            slopes=np.array(duals),
            correlations=self.correlations,
        )

        # Update lower bound
        self.lower_bound.update(controls=controls, costs=costs, duals=duals)

        # Update decomposed hyper planes
        for estimator, new_inps, new_duals in zip(
            self.layers, inputs_decomp, duals_decomp
        ):
            estimator.update(
                controls=new_inps, costs=np.zeros(new_inps.shape[0]), duals=new_duals
            )

        # Update approx
        self.inputs = np.concatenate(
            [
                self.inputs,
                controls,
                inputs_decomp.reshape(n_inputs * n_reservoirs, n_reservoirs),
            ]
        )
        self.costs = np.concatenate(
            [self.costs, costs, np.zeros(n_inputs * n_reservoirs)]
        )
        self.duals = np.concatenate(
            [
                self.duals,
                duals,
                duals_decomp.reshape(n_inputs * n_reservoirs, n_reservoirs),
            ]
        )
        self.true_inputs = np.concatenate([self.true_inputs, controls])
        self.true_costs = np.concatenate([self.true_costs, costs])
        self.true_duals = np.concatenate([self.true_duals, duals])
        self.remove_incoherence()
        self.remove_inconsistence()

    def remove_interps(self) -> None:
        self.inputs = self.lower_bound.inputs
        self.costs = self.lower_bound.costs
        self.duals = self.lower_bound.duals

    def round(self, precision: int = 1) -> None:
        self.lower_bound.round(precision=precision)
        for layer in self.layers:
            layer.round(precision=precision)
        self.inputs = np.round(self.inputs, precision)
        self.costs = np.round(self.costs, precision)
        self.duals = np.round(self.duals, precision)


class RewardApproximation(MultiVariateEstimator):
    """Class to store and update reward approximation for a given week and a given scenario"""

    def __init__(self, lb_control: float, ub_control: float, ub_reward: float) -> None:
        """
        Create a new reward approximation

        Parameters
        ----------
        lb_control:float :
            Lower possible bound on control
        ub_control:float :
            Upper possible bound on control
        ub_reward:float :
            Upper bound on reward

        Returns
        -------
        None
        """
        self.breaking_point = [lb_control, ub_control]
        self.list_cut = [(0.0, ub_reward)]

    def reward_function(self) -> Callable:
        """Return a function to evaluate reward at any point based on the current approximation."""
        return lambda x: min([cut[0] * x + cut[1] for cut in self.list_cut])

    def update(
        self,
        costs: Union[np.ndarray, float],
        duals: Union[np.ndarray, float],
        controls: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update reward approximation by adding a new cut

        Returns
        -------
        None
        """
        if isinstance(duals, np.ndarray):
            slope_new_cut = float(duals[0])
        else:
            slope_new_cut = duals
        if isinstance(costs, np.ndarray):
            intercept_new_cut = float(costs[0])
        else:
            intercept_new_cut = costs
        previous_reward = self.reward_function()
        new_cut: Callable = lambda x: slope_new_cut * x + intercept_new_cut
        new_reward: list[tuple[float, float]] = []
        new_points = [self.breaking_point[0]]

        if len(self.breaking_point) != len(self.list_cut) + 1:
            raise (ValueError)

        for i in range(len(self.breaking_point)):
            if i == len(self.breaking_point) - 1:
                new_points.append(self.breaking_point[-1])
            else:
                new_cut_below_previous_reward_at_i = self.check_relative_position(
                    previous_reward, new_cut, i
                )
                new_cut_above_previous_reward_at_i = self.check_relative_position(
                    new_cut, previous_reward, i
                )
                new_cut_below_previous_reward_at_i_plus_1 = (
                    self.check_relative_position(previous_reward, new_cut, i + 1)
                )
                new_cut_above_previous_reward_at_i_plus_1 = (
                    self.check_relative_position(new_cut, previous_reward, i + 1)
                )
                slopes_are_different = (slope_new_cut - self.list_cut[i][0]) != 0
                if i == 0:
                    if new_cut_below_previous_reward_at_i:
                        new_reward.append((slope_new_cut, intercept_new_cut))
                    elif new_cut_above_previous_reward_at_i:
                        new_reward.append(self.list_cut[i])
                    elif new_cut_below_previous_reward_at_i_plus_1:
                        new_reward.append((slope_new_cut, intercept_new_cut))
                    else:
                        new_reward.append(self.list_cut[i])
                if (new_cut_below_previous_reward_at_i) and (
                    new_cut_above_previous_reward_at_i_plus_1
                ):
                    if slopes_are_different:
                        new_reward.append(self.list_cut[i])
                        new_points.append(
                            self.calculate_breaking_point(
                                slope_new_cut=slope_new_cut,
                                intercept_new_cut=intercept_new_cut,
                                i=i,
                            )
                        )
                elif new_cut_above_previous_reward_at_i:
                    if i != 0:
                        new_reward.append(self.list_cut[i])
                        new_points.append(self.breaking_point[i])
                    if new_cut_below_previous_reward_at_i_plus_1:
                        if slopes_are_different:
                            new_reward.append((slope_new_cut, intercept_new_cut))
                            new_points.append(
                                self.calculate_breaking_point(
                                    slope_new_cut=slope_new_cut,
                                    intercept_new_cut=intercept_new_cut,
                                    i=i,
                                )
                            )
                elif (
                    not (new_cut_below_previous_reward_at_i)
                    and not (new_cut_above_previous_reward_at_i)
                    and i != 0
                ):
                    new_points.append(self.breaking_point[i])
                    if new_cut_below_previous_reward_at_i_plus_1:
                        new_reward.append((slope_new_cut, intercept_new_cut))
                    else:
                        new_reward.append(self.list_cut[i])

        self.breaking_point = new_points
        self.list_cut = new_reward

    def calculate_breaking_point(
        self,
        intercept_new_cut: float,
        slope_new_cut: float,
        i: int,
    ) -> float:
        intercept_previous_cut = self.list_cut[i][1]
        slope_previous_cut = self.list_cut[i][0]
        return -(intercept_new_cut - intercept_previous_cut) / (
            slope_new_cut - slope_previous_cut
        )

    def check_relative_position(
        self, previous_reward: Callable, new_cut: Callable, i: int
    ) -> bool:
        return new_cut(self.breaking_point[i]) < previous_reward(self.breaking_point[i])


class LinearCostEstimator:
    """A class to contain an ensemble of Interpolators for every week and scenario"""

    def __init__(
        self,
        param: TimeScenarioParameter,
        controls: np.ndarray,
        costs: np.ndarray,
        duals: np.ndarray,
        correlations: Optional[np.ndarray] = None,
    ) -> None:
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
        estimators: Dict[TimeScenarioIndex, MultiVariateEstimator] = {}
        for week in range(param.len_week):
            for scenario in range(param.len_scenario):
                r = LinearDecomposer(
                    inputs=controls[week, scenario],
                    costs=costs[week, scenario],
                    duals=duals[week, scenario],
                    correlations=correlations,
                )
                estimators[TimeScenarioIndex(week, scenario)] = r
        self.estimators = estimators
        self.param = param

    def __getitem__(self, index: TimeScenarioIndex) -> MultiVariateEstimator:
        """
        Gets a LinearInterpolators

        Parameters
        ----------
            ws:tuple[int, int] / int: index of the week or scenario we want,

        Returns
        -------
            Array of ... or LinearInterpolator
        """
        return self.estimators[index]

    def update(
        self,
        controls: np.ndarray,
        costs: np.ndarray,
        duals: np.ndarray,
    ) -> None:
        """
        Updates the parameters of the Linear Interpolators

        Parameters
        ----------
            inputs:np.ndarray: The coordinates for which costs / duals are obtained
                must have the same shape as what we'll call our interpolator with,
            costs:np.ndarray: Cost for every input,
            duals:np.ndarray: Duals for every input first dimension should be the same as inputs,
        """
        for week, (inputs_w, costs_w, duals_w) in enumerate(
            zip(controls, costs, duals)
        ):
            for scenario, (controls, costs, duals) in enumerate(
                zip(inputs_w, costs_w, duals_w)
            ):
                self.estimators[TimeScenarioIndex(week, scenario)].update(
                    controls=controls,
                    costs=costs,
                    duals=duals,
                )

    def remove_redundants(
        self,
        tolerance: float = 1e-7,
    ) -> None:
        for estimator in self.estimators.values():
            estimator.count_redundant(tolerance=tolerance, remove=True)

    def remove_interpolations(
        self,
    ) -> None:
        for estimator in self.estimators.values():
            estimator.remove_interps()

    def round(
        self,
        precision: int = 6,
    ) -> None:
        for estimator in self.estimators.values():
            estimator.round(precision)

    def to_julia_compatible_structure(self) -> Array1D:
        julia_structure = np.array(
            [
                [
                    self.estimators[TimeScenarioIndex(week, scenario)].to_julia_dict()
                    for scenario in range(self.param.len_scenario)
                ]
                for week in range(self.param.len_week)
            ]
        )
        return julia_structure
