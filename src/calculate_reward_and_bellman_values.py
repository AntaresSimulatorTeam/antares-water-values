from itertools import product

import numpy as np

from estimation import PieceWiseLinearInterpolator, RewardApproximation
from read_antares_data import TimeScenarioIndex, TimeScenarioParameter
from reservoir_management import ReservoirManagement
from type_definition import Array1D, Callable, Dict, Iterable, List


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
