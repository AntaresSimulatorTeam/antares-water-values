from itertools import product

import numpy as np
from scipy.interpolate import interp1d

from estimation import PieceWiseLinearInterpolator, RewardApproximation
from read_antares_data import Reservoir, TimeScenarioIndex, TimeScenarioParameter
from type_definition import Array1D, Array2D, Callable, Dict, Iterable, List, Optional


class ReservoirManagement:

    def __init__(
        self,
        reservoir: Reservoir,
        penalty_bottom_rule_curve: float = 0,
        penalty_upper_rule_curve: float = 0,
        penalty_final_level: float = 0,
        force_final_level: bool = False,
        final_level: Optional[float] = None,
        overflow: bool = True,
    ) -> None:
        """Class to describe reservoir management parameters.

        Args:
            reservoir (Reservoir): Reservoir in question
            penalty_bottom_rule_curve (float, optional): Penalty for violating bottom rule curve. Defaults to 0.
            penalty_upper_rule_curve (float, optional): Penalty for violating upper rule curve. Defaults to 0.
            penalty_final_level (float, optional): Penalty for not respecting final level. Defaults to 0.
            force_final_level (bool, optional): Whether final level is imposed. Defaults to False.
            final_level (Optional[float], optional): Final level to impose, if not specified is equal to initial level. Defaults to None.
            overflow (bool, optional) : Whether overflow is possible or forbiden. Defaults to True.
        """

        self.reservoir = reservoir
        self.penalty_bottom_rule_curve = penalty_bottom_rule_curve
        self.penalty_upper_rule_curve = penalty_upper_rule_curve
        self.overflow = overflow

        if force_final_level:
            self.penalty_final_level = penalty_final_level
            if final_level:
                self.final_level = final_level
            else:
                self.final_level = reservoir.initial_level
        else:
            self.final_level = False

    def get_penalty(self, week: int, len_week: int) -> Callable:
        """
        Return a function to evaluate penalities for violating rule curves for any level of stock.

        Parameters
        ----------
        week:int :
            Week considered
        len_week:int :
            Total number of weeks

        Returns
        -------

        """
        if week == len_week - 1 and self.final_level:
            pen = interp1d(
                [
                    0,
                    self.final_level,
                    self.reservoir.capacity,
                ],
                [
                    -self.penalty_final_level * (self.final_level),
                    0,
                    -self.penalty_final_level
                    * (self.reservoir.capacity - self.final_level),
                ],
            )
        else:
            pen = interp1d(
                [
                    0,
                    self.reservoir.bottom_rule_curve[week],
                    self.reservoir.upper_rule_curve[week],
                    self.reservoir.capacity,
                ],
                [
                    -self.penalty_bottom_rule_curve
                    * (self.reservoir.bottom_rule_curve[week]),
                    0,
                    0,
                    -self.penalty_upper_rule_curve
                    * (self.reservoir.capacity - self.reservoir.upper_rule_curve[week]),
                ],
            )
        return pen


class MultiStockManagement:

    def __init__(self, list_reservoirs: List[ReservoirManagement]) -> None:
        """Describes reservoir management for all stocks

        Args:
            list_reservoirs (List[ReservoirManagement]): List of reservoir management
        """
        self.dict_reservoirs = {}
        for res in list_reservoirs:
            self.dict_reservoirs[res.reservoir.area] = res

    def get_disc(
        self,
        method: str,
        week: int,
        xNsteps: int,
        reference_pt: np.ndarray,
        correlation_matrix: np.ndarray,
        alpha: float = 1.0,
        in_out_ratio: float = 3.0,
    ) -> Array1D:
        n_reservoirs = len(self.dict_reservoirs)
        if len(reference_pt.shape) > 1:
            reference_pt = np.mean(reference_pt, axis=1)
        lbs = np.array(
            [
                mng.reservoir.bottom_rule_curve[week] * alpha
                for mng in self.dict_reservoirs.values()
            ]
        )
        ubs = np.array(
            [
                mng.reservoir.upper_rule_curve[week] * alpha
                + mng.reservoir.capacity * (1 - alpha)
                for mng in self.dict_reservoirs.values()
            ]
        )
        full = np.array(
            [mng.reservoir.capacity for mng in self.dict_reservoirs.values()]
        )
        empty = np.array([0] * n_reservoirs)
        n_pts_above = np.maximum(
            1 + 2 * (full - ubs > 0),
            np.round(
                xNsteps
                * (full - ubs)
                / (full - empty + (in_out_ratio - 1) * (ubs - lbs))
            ).astype(int),
        )
        n_pts_in = np.round(
            xNsteps
            * (ubs - lbs)
            * in_out_ratio
            / (full - empty + (in_out_ratio - 1) * (ubs - lbs))
        ).astype(int)
        n_pts_below = np.maximum(
            1,
            np.round(
                xNsteps
                * (lbs - empty)
                / (full - empty + (in_out_ratio - 1) * (ubs - lbs))
            ).astype(int),
        )
        n_pts_in += xNsteps - (
            n_pts_below + n_pts_in + n_pts_above
        )  # Make sure total adds up
        if method == "lines":
            above_curve_pts = [
                np.linspace(ubs[r], full[r], n_pts_above[r], endpoint=True)
                for r in range(n_reservoirs)
            ]
            in_curve_pts = [
                np.linspace(lbs[r], ubs[r], n_pts_in[r], endpoint=False)
                for r in range(n_reservoirs)
            ]
            below_curve_pts = [
                np.linspace(empty[r], lbs[r], n_pts_below[r], endpoint=False)
                for r in range(n_reservoirs)
            ]
            all_pts = np.array(
                [
                    np.concatenate(
                        (below_curve_pts[r], in_curve_pts[r], above_curve_pts[r])
                    )
                    for r in range(n_reservoirs)
                ]
            ).T
            diffs_to_ref = all_pts[:, None] - reference_pt[None, :]  # Disc * R
            diffs_to_ref = (
                diffs_to_ref[:, :, None] * np.eye(n_reservoirs)[None, :, :]
            )  # Disc * R * R
            diffs_to_ref = np.dot(diffs_to_ref, correlation_matrix)  # Disc * R * R
            new_levels = reference_pt[None, None, :] + diffs_to_ref  # Disc * R * R
            new_levels = np.maximum(
                new_levels, empty[None, None, :]
            )  # Do or do not make other points leave guiding curves ?
            new_levels = np.minimum(
                new_levels, full[None, None, :]
            )  # We chose the former
            levels = np.reshape(
                new_levels, (xNsteps * n_reservoirs, n_reservoirs)
            )  # (Disc * R) * R
        else:
            # Listing all levels
            levels_discretization = product(
                *[
                    np.concatenate(
                        [
                            [0],
                            np.linspace(lbs[i], ubs[i], xNsteps - 2),
                            [(alpha + 1) / 2 * mng.reservoir.capacity],
                        ]
                    )
                    for i, mng in enumerate(self.dict_reservoirs.values())
                ]
            )
            levels = np.array([level for level in levels_discretization])
        return levels


class BellmanValueCalculation:

    def __init__(
        self,
        param: TimeScenarioParameter,
        reward: Dict[TimeScenarioIndex, RewardApproximation],
        reservoir_management: ReservoirManagement,
        stock_discretization: Array1D,
    ) -> None:
        self.time_scenario_param = param
        self.reward_approximation = reward
        self.reservoir_management = reservoir_management
        self.stock_discretization = stock_discretization

        self.reward_fn: Dict[TimeScenarioIndex, Callable] = {}
        self.penalty_fn: Dict[TimeScenarioIndex, Callable] = {}
        for week in range(self.time_scenario_param.len_week):
            for scenario in range(self.time_scenario_param.len_scenario):
                self.reward_fn[TimeScenarioIndex(week=week, scenario=scenario)] = (
                    self.reward_approximation[
                        TimeScenarioIndex(week=week, scenario=scenario)
                    ].reward_function()
                )
                self.penalty_fn[TimeScenarioIndex(week=week, scenario=scenario)] = (
                    self.reservoir_management.get_penalty(
                        week=week, len_week=param.len_week
                    )
                )

    def solve_weekly_problem_with_approximation(
        self,
        week: int,
        scenario: int,
        level_i: float,
        V_fut: PieceWiseLinearInterpolator,
    ) -> tuple[float, float, float]:
        """
        Optimize control of reservoir during a week based on reward approximation and current Bellman values.

        Parameters
        ----------
        level_i:float :
            Initial level of reservoir at the beginning of the week
        V_fut:callable :
            Bellman values at the end of the week

        Returns
        -------
        Vu:float :
            Optimal objective value
        xf:float :
            Final level of sotck
        control:float :
            Optimal control
        """

        Vu = float("-inf")
        stock = self.reservoir_management.reservoir
        pen = self.penalty_fn[TimeScenarioIndex(week=week, scenario=scenario)]
        reward_fn = self.reward_fn[TimeScenarioIndex(week=week, scenario=scenario)]
        points = self.reward_approximation[
            TimeScenarioIndex(week=week, scenario=scenario)
        ].breaking_point
        X = self.stock_discretization

        for i_fut in range(len(X)):
            u = -X[i_fut] + level_i + stock.inflow[week, scenario]
            if -stock.max_pumping[week] * stock.efficiency <= u:
                if (
                    self.reservoir_management.overflow
                    or u <= stock.max_generating[week]
                ):
                    u = min(u, stock.max_generating[week])
                    G = reward_fn(u)
                    penalty = pen(X[i_fut])
                    if (G + V_fut(X[i_fut]) + penalty) > Vu:
                        Vu = G + V_fut(X[i_fut]) + penalty
                        xf = X[i_fut]
                        control = u

        for u in range(len(points)):
            state_fut = level_i - points[u] + stock.inflow[week, scenario]
            if 0 <= state_fut <= stock.capacity:
                penalty = pen(state_fut)
                G = reward_fn(points[u])
                if (G + V_fut(state_fut) + penalty) > Vu:
                    Vu = G + V_fut(state_fut) + penalty
                    xf = state_fut
                    control = points[u]

        Umin = level_i + stock.inflow[week, scenario] - stock.bottom_rule_curve[week]
        if (
            -stock.max_pumping[week] * stock.efficiency
            <= Umin
            <= stock.max_generating[week]
        ):
            state_fut = level_i - Umin + stock.inflow[week, scenario]
            penalty = pen(state_fut)
            if (reward_fn(Umin) + V_fut(state_fut) + penalty) > Vu:
                Vu = reward_fn(Umin) + V_fut(state_fut) + penalty
                xf = state_fut
                control = Umin

        Umax = level_i + stock.inflow[week, scenario] - stock.upper_rule_curve[week]
        if (
            -stock.max_pumping[week] * stock.efficiency
            <= Umax
            <= stock.max_generating[week]
        ):
            state_fut = level_i - Umax + stock.inflow[week, scenario]
            penalty = pen(state_fut)
            if (reward_fn(Umax) + V_fut(state_fut) + penalty) > Vu:
                Vu = reward_fn(Umax) + V_fut(state_fut) + penalty
                xf = state_fut
                control = Umax

        control = min(
            -(xf - level_i - stock.inflow[week, scenario]),
            stock.max_generating[week],
        )
        return (Vu, xf, control)

    def calculate_VU(
        self,
        final_values: Array1D = np.zeros(1, dtype=np.float32),
    ) -> Dict[int, PieceWiseLinearInterpolator]:
        """
        Calculate Bellman values for every week based on reward approximation

        Parameters
        ----------

        Returns
        -------

        """
        X = self.stock_discretization
        V = {
            week: np.zeros(
                (len(X), self.time_scenario_param.len_scenario), dtype=np.float32
            )
            for week in range(self.time_scenario_param.len_week + 1)
        }
        if len(final_values) == len(X):
            for scenario in range(self.time_scenario_param.len_scenario):
                V[self.time_scenario_param.len_week][:, scenario] = final_values

        for week in range(self.time_scenario_param.len_week - 1, -1, -1):

            for scenario in range(self.time_scenario_param.len_scenario):
                V_fut = PieceWiseLinearInterpolator(X, V[week + 1][:, scenario])
                for i in range(len(X)):

                    Vu, _, _ = self.solve_weekly_problem_with_approximation(
                        level_i=X[i],
                        V_fut=V_fut,
                        week=week,
                        scenario=scenario,
                    )

                    V[week][i, scenario] = Vu + V[week][i, scenario]

            V[week] = np.repeat(
                np.mean(V[week], axis=1, keepdims=True),
                self.time_scenario_param.len_scenario,
                axis=1,
            )
        return {
            week: PieceWiseLinearInterpolator(X, np.mean(v, axis=1))
            for (week, v) in V.items()
        }


class MultiStockBellmanValueCalculation:

    def __init__(self, list_reservoirs: List[BellmanValueCalculation]) -> None:
        self.dict_reservoirs = {}
        for res in list_reservoirs:
            self.dict_reservoirs[res.reservoir_management.reservoir.area] = res

    def get_product_stock_discretization(self) -> Iterable:
        return product(
            *[
                [i for i in range(len(res_man.stock_discretization))]
                for res_man in self.dict_reservoirs.values()
            ]
        )
