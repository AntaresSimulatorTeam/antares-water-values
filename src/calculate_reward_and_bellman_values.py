import numpy as np
from scipy.optimize import minimize

from estimation import LinearInterpolator, PieceWiseLinearInterpolator
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
    reward: LinearInterpolator,
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

    pen = reservoir_management.get_penalty(week=week, len_week=param.len_week)

    def noise_penalty(x: float) -> float:
        return -0.01 * x

    def objective(x_fut: Array1D) -> float:
        return (
            reward(
                -x_fut[0]
                + level_i
                + reservoir_management.reservoir.inflow[week, scenario]
            )
            + V_fut(x_fut[0])
            + pen(x_fut[0])
            + noise_penalty(x_fut[0])
        )

    lb = max(
        0,
        level_i
        + reservoir_management.reservoir.inflow[week, scenario]
        - reservoir_management.reservoir.max_generating[week],
    )
    ub = min(
        reservoir_management.reservoir.capacity,
        level_i
        + reservoir_management.reservoir.inflow[week, scenario]
        + reservoir_management.reservoir.max_pumping[week]
        * reservoir_management.reservoir.efficiency,
    )

    res = minimize(
        objective,
        x0=[(lb + ub) / 2],
        method="Nelder-Mead",
        bounds=[(lb, ub)],
    )
    assert res.status == 0
    xf = res.x[0]
    control = min(
        -(xf - level_i - reservoir_management.reservoir.inflow[week, scenario]),
        reservoir_management.reservoir.max_generating[week],
    )
    Vu = objective(np.array([xf]))
    Vu = Vu - noise_penalty(xf)
    cost = reward(control)

    return (Vu, xf, control, cost)


def calculate_VU(
    stock_discretization: Array1D,
    time_scenario_param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    reward: Dict[TimeScenarioIndex, LinearInterpolator],
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
