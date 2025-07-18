import numpy as np

from estimation import PieceWiseLinearInterpolator, RewardApproximation
from reservoir_management import ReservoirManagement
from type_definition import (
    Array1D,
    Dict,
    TimeScenarioIndex,
    TimeScenarioParameter,
    WeekIndex,
)


def solve_weekly_problem_with_approximation(
    week: int,
    scenario: int,
    level_i: float,
    V_fut: PieceWiseLinearInterpolator,
    reservoir_management: ReservoirManagement,
    param: TimeScenarioParameter,
    reward: RewardApproximation,
) -> tuple[float, float, float, float]:
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
    stock = reservoir_management.reservoir
    pen = reservoir_management.get_penalty(week=week, len_week=param.len_week)
    reward_fn = reward.reward_function()
    points = reward.breaking_point
    X = V_fut.inputs

    for i_fut in range(len(X)):
        u = -X[i_fut] + level_i + stock.inflow[week, scenario]
        if -stock.max_pumping[week] * stock.efficiency <= u:
            if reservoir_management.overflow or u <= stock.max_generating[week]:
                u = min(u, stock.max_generating[week])
                G = reward_fn(u)
                penalty = pen(X[i_fut])
                if (G + V_fut(X[i_fut]) + penalty) > Vu:
                    Vu = G + V_fut(X[i_fut]) + penalty
                    xf = X[i_fut]
                    control = u
                    cost = G

    for u in range(len(points)):
        state_fut = level_i - points[u] + stock.inflow[week, scenario]
        if 0 <= state_fut <= stock.capacity:
            penalty = pen(state_fut)
            G = reward_fn(points[u])
            if (G + V_fut(state_fut) + penalty) > Vu:
                Vu = G + V_fut(state_fut) + penalty
                xf = state_fut
                control = points[u]
                cost = G

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
            cost = reward_fn(Umin)

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
            cost = reward_fn(Umax)

    control = min(
        -(xf - level_i - stock.inflow[week, scenario]),
        stock.max_generating[week],
    )
    return (Vu, xf, control, cost)


def calculate_VU(
    stock_discretization: Array1D,
    time_scenario_param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    reward: Dict[TimeScenarioIndex, RewardApproximation],
    final_values: Array1D = np.zeros(1, dtype=np.float32),
) -> Dict[WeekIndex, PieceWiseLinearInterpolator]:
    """
    Calculate Bellman values for every week based on reward approximation

    Parameters
    ----------

    Returns
    -------

    """
    X = stock_discretization
    V = {
        week: np.zeros((len(X), time_scenario_param.len_scenario), dtype=np.float32)
        for week in range(time_scenario_param.len_week + 1)
    }
    if len(final_values) == len(X):
        for scenario in range(time_scenario_param.len_scenario):
            V[time_scenario_param.len_week][:, scenario] = final_values

    for week in range(time_scenario_param.len_week - 1, -1, -1):

        for scenario in range(time_scenario_param.len_scenario):
            V_fut = PieceWiseLinearInterpolator(X, V[week + 1][:, scenario])
            for i in range(len(X)):

                Vu, _, _, _ = solve_weekly_problem_with_approximation(
                    level_i=X[i],
                    V_fut=V_fut,
                    week=week,
                    scenario=scenario,
                    reservoir_management=reservoir_management,
                    param=time_scenario_param,
                    reward=reward[TimeScenarioIndex(week, scenario)],
                )

                V[week][i, scenario] = Vu + V[week][i, scenario]

        V[week] = np.repeat(
            np.mean(V[week], axis=1, keepdims=True),
            time_scenario_param.len_scenario,
            axis=1,
        )
    return {
        WeekIndex(week): PieceWiseLinearInterpolator(X, np.mean(v, axis=1))
        for (week, v) in V.items()
    }
