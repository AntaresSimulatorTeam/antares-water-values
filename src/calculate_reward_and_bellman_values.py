import numpy as np

from estimation import LinearCostEstimator, PieceWiseLinearInterpolator
from optimization import WeeklyBellmanProblem
from reservoir_management import MultiStockManagement
from type_definition import (
    AreaIndex,
    Dict,
    List,
    Optional,
    ScenarioIndex,
    TimeScenarioIndex,
    TimeScenarioParameter,
    WeekIndex,
)


def calculate_VU(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    costs_approx: LinearCostEstimator,
    levels: Dict[WeekIndex, List[Dict[AreaIndex, float]]],
    final_bellman_values: Optional[PieceWiseLinearInterpolator] = None,
    name_solver: str = "CLP",
    divisor: dict[str, float] = {"euro": 1e8, "energy": 1e4},
    verbose: bool = False,
    n_cycle: int = 1,
) -> Dict[WeekIndex, PieceWiseLinearInterpolator]:
    """
    Calculate Bellman values for every week based on reward approximation

    Parameters
    ----------

    Returns
    -------

    """
    area = multi_stock_management.areas[0]

    if final_bellman_values is None:
        X = np.array([x[area] for x in levels[WeekIndex(param.len_week)]])
        final_bellman_values = PieceWiseLinearInterpolator(
            X, np.zeros(len(X), dtype=np.float32)
        )
    for i in range(n_cycle):
        bellman_values = {WeekIndex(param.len_week): final_bellman_values}

        for week in range(param.len_week - 1, -1, -1):
            week_vb = np.zeros(
                (len(levels[WeekIndex(week)]), param.len_scenario),
                dtype=np.float32,
            )
            for scenario in range(param.len_scenario):
                for i, lvl_init in enumerate(levels[WeekIndex(week)]):

                    problem = WeeklyBellmanProblem(
                        param=param,
                        multi_stock_management=multi_stock_management,
                        week_costs_estimation={
                            ScenarioIndex(scenario): costs_approx[
                                TimeScenarioIndex(week, scenario)
                            ]
                        },
                        name_solver=name_solver,
                        divisor=divisor,
                        week=week,
                    )

                    _, Vu, _, _ = problem.solve(
                        level_init=lvl_init,
                        future_costs_estimation=bellman_values[WeekIndex(week + 1)],
                    )

                    week_vb[i, scenario] = -Vu

            bellman_values[WeekIndex(week)] = PieceWiseLinearInterpolator(
                np.array([x[area] for x in levels[WeekIndex(week)]]),
                np.mean(week_vb, axis=1),
            )
        final_bellman_values = bellman_values[WeekIndex(0)]
    return bellman_values
