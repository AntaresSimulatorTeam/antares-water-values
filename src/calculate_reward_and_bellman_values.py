import numpy as np

from estimation import LinearCostEstimator, PieceWiseLinearInterpolator
from optimization import solve_weekly_problem_with_approximation
from reservoir_management import ReservoirManagement
from type_definition import (
    Array1D,
    Dict,
    TimeScenarioIndex,
    TimeScenarioParameter,
    WeekIndex,
)


def calculate_VU(
    stock_discretization: Array1D,
    time_scenario_param: TimeScenarioParameter,
    reservoir_management: ReservoirManagement,
    reward: LinearCostEstimator,
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
