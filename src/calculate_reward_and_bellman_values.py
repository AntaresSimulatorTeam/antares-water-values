import numpy as np
from scipy.interpolate import interp1d

from read_antares_data import Reservoir, TimeScenarioIndex, TimeScenarioParameter
from type_definition import Array1D, Array2D, Callable, Dict, List, Optional


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
        if week == len_week and self.final_level:
            # penalty at the beginning of week len_week = penalty at the end og week len_week-1 which is the last week
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
            # penalty at the beginning of the week
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


class RewardApproximation:
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
        self.controls: List[float] = []
        self.costs: List[float] = []
        self.duals: List[float] = []

    def reward_function(self) -> Callable:
        """Return a function to evaluate reward at any point based on the current approximation."""
        return lambda x: min(
            [
                -self.duals[i] * (x - self.controls[i]) + self.costs[i]
                for i in range(len(self.controls))
            ]
        )

    def update_reward_approximation(
        self, new_control: List[float], new_cost: List[float], new_dual: List[float]
    ) -> None:
        """
        Update reward approximation by adding a new cut

        Returns
        -------
        None
        """

        self.controls = self.controls + new_control
        self.costs = self.costs + new_cost
        self.duals = self.duals + new_dual


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
        for week in range(self.time_scenario_param.len_week + 1):
            for scenario in range(self.time_scenario_param.len_scenario):
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
        V_fut: Callable,
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
        penalty = pen(level_i)
        reward_fn = self.reward_fn[TimeScenarioIndex(week=week, scenario=scenario)]
        points = self.reward_approximation[
            TimeScenarioIndex(week=week, scenario=scenario)
        ].controls
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
                    if (G + V_fut(X[i_fut]) + penalty) > Vu:
                        Vu = G + V_fut(X[i_fut]) + penalty
                        xf = X[i_fut]
                        control = u

        for u in range(len(points)):
            state_fut = level_i - points[u] + stock.inflow[week, scenario]
            if 0 <= state_fut <= stock.capacity:
                G = reward_fn(points[u])
                if (G + V_fut(state_fut) + penalty) > Vu:
                    Vu = G + V_fut(state_fut) + penalty
                    xf = state_fut
                    control = points[u]
        if (
            week == self.time_scenario_param.len_week - 1
            and self.reservoir_management.final_level
        ):
            Ufinal = (
                level_i
                + stock.inflow[week, scenario]
                - self.reservoir_management.final_level
            )
            if (
                -stock.max_pumping[week] * stock.efficiency
                <= Ufinal
                <= stock.max_generating[week]
            ):
                state_fut = level_i - Ufinal + stock.inflow[week, scenario]
                if (reward_fn(Ufinal) + V_fut(state_fut) + penalty) > Vu:
                    Vu = reward_fn(Ufinal) + V_fut(state_fut) + penalty
                    xf = state_fut
                    control = Ufinal
        else:
            Umin = (
                level_i + stock.inflow[week, scenario] - stock.bottom_rule_curve[week]
            )
            if (
                -stock.max_pumping[week] * stock.efficiency
                <= Umin
                <= stock.max_generating[week]
            ):
                state_fut = level_i - Umin + stock.inflow[week, scenario]
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
    ) -> Array2D:
        """
        Calculate Bellman values for every week based on reward approximation

        Parameters
        ----------

        Returns
        -------

        """
        X = self.stock_discretization
        V = np.zeros(
            (
                len(X),
                self.time_scenario_param.len_week + 1,
                self.time_scenario_param.len_scenario,
            )
        )
        if len(final_values) == len(X):
            for scenario in range(self.time_scenario_param.len_scenario):
                V[:, self.time_scenario_param.len_week, scenario] = final_values
        else:
            for scenario in range(self.time_scenario_param.len_scenario):
                pen = self.penalty_fn[
                    TimeScenarioIndex(
                        week=self.time_scenario_param.len_week, scenario=scenario
                    )
                ]
                for i in range(len(X)):
                    V[i, self.time_scenario_param.len_week, scenario] = pen(X[i])

        for week in range(self.time_scenario_param.len_week - 1, -1, -1):

            for scenario in range(self.time_scenario_param.len_scenario):
                V_fut = interp1d(X, V[:, week + 1, scenario])
                for i in range(len(X)):

                    Vu, _, _ = self.solve_weekly_problem_with_approximation(
                        level_i=X[i],
                        V_fut=V_fut,
                        week=week,
                        scenario=scenario,
                    )

                    V[i, week, scenario] = Vu + V[i, week, scenario]

            V[:, week, :] = np.repeat(
                np.mean(V[:, week, :], axis=1, keepdims=True),
                self.time_scenario_param.len_scenario,
                axis=1,
            )
        return np.mean(V, axis=2)
