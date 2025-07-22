import numpy as np
import pytest

from estimation import LinearInterpolator
from functions_iterative import TimeScenarioParameter
from multi_stock_bellman_value_calculation import (
    initialize_future_costs,
    precalculated_method,
)
from optimization import WeeklyBellmanProblem
from reservoir_management import MultiStockManagement
from type_definition import (
    ScenarioIndex,
    TimeScenarioIndex,
    WeekIndex,
    time_list_area_value_to_array,
)


def test_weekly_bellman_problem(
    param: TimeScenarioParameter, multi_stock_management_two_nodes: MultiStockManagement
) -> None:
    week = param.len_week - 1
    # Initialize cost functions
    costs_approx = {
        ScenarioIndex(0): LinearInterpolator(
            controls=np.array(
                [
                    [
                        res.reservoir.max_generating[week]
                        for res in multi_stock_management_two_nodes.dict_reservoirs.values()
                    ]
                ]
            ),
            costs=np.array([1e7]),
            duals=np.array([[-100 for a in multi_stock_management_two_nodes.areas]]),
        )
    }

    problem = WeeklyBellmanProblem(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        week_costs_estimation=costs_approx,
        name_solver="CLP",
        divisor={"euro": 1e8, "energy": 1e4},
        week=week,
    )

    controls, cost, duals, levels = problem.solve(
        level_init={
            a: res.reservoir.capacity / 2
            for a, res in multi_stock_management_two_nodes.dict_reservoirs.items()
        },
        future_costs_estimation=initialize_future_costs(
            multi_stock_management_two_nodes
        ),
    )

    assert np.array(
        [controls[a][ScenarioIndex(0)] for a in multi_stock_management_two_nodes.areas]
    ) == pytest.approx(np.array([238355.804, 419664.0]))
    assert cost == pytest.approx(16858019.6)
    assert np.array(
        [duals[a] for a in multi_stock_management_two_nodes.areas]
    ) == pytest.approx(np.array([-100, 0]))
    assert np.array(
        [levels[a][ScenarioIndex(0)] for a in multi_stock_management_two_nodes.areas]
    ) == pytest.approx(np.array([159959.696, 481561.5]))


def test_bellman_value_precalculated_multi_stock(
    param: TimeScenarioParameter,
    multi_stock_management_two_nodes: MultiStockManagement,
) -> None:

    levels, _, bellman_values, _ = precalculated_method(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        output_path="test_data/two_nodes",
        len_controls=5,
        len_bellman=5,
        name_solver="CLP",
        controls_looked_up="line+diagonal",
        verbose=True,
    )

    assert time_list_area_value_to_array(
        levels, param, multi_stock_management_two_nodes.areas
    )[::-1] == pytest.approx(
        np.array(
            [
                [
                    [0.0, 628377.6569],
                    [277853.0681, 0.0],
                    [175340.436, 628377.6569],
                    [277853.0681, 396540.564],
                    [396823.092, 628377.6569],
                    [277853.0681, 897433.908],
                    [582930.046, 628377.6569],
                    [277853.0681, 1318323.454],
                    [769037.0, 628377.6569],
                    [277853.0681, 1739213.0],
                ],
                [
                    [0.0, 628377.6569],
                    [277853.0681, 0.0],
                    [177647.547, 628377.6569],
                    [277853.0681, 401758.203],
                    [486031.384, 628377.6569],
                    [277853.0681, 1099182.616],
                    [627534.192, 628377.6569],
                    [277853.0681, 1419197.808],
                    [769037.0, 628377.6569],
                    [277853.0681, 1739213.0],
                ],
                [
                    [0.0, 628377.6569],
                    [277853.0681, 0.0],
                    [180723.695, 628377.6569],
                    [277853.0681, 408715.055],
                    [488338.495, 628377.6569],
                    [277853.0681, 1104400.255],
                    [628687.7475, 628377.6569],
                    [277853.0681, 1421806.6275],
                    [769037.0, 628377.6569],
                    [277853.0681, 1739213.0],
                ],
                [
                    [0.0, 628377.6569],
                    [277853.0681, 0.0],
                    [183030.806, 628377.6569],
                    [277853.0681, 413932.694],
                    [490645.606, 628377.6569],
                    [277853.0681, 1109617.894],
                    [629841.303, 628377.6569],
                    [277853.0681, 1424415.447],
                    [769037.0, 628377.6569],
                    [277853.0681, 1739213.0],
                ],
                [
                    [0.0, 628377.6569],
                    [277853.0681, 0.0],
                    [185337.917, 628377.6569],
                    [277853.0681, 419150.333],
                    [493721.754, 628377.6569],
                    [277853.0681, 1116574.746],
                    [631379.377, 628377.6569],
                    [277853.0681, 1427893.873],
                    [769037.0, 628377.6569],
                    [277853.0681, 1739213.0],
                ],
            ]
        )
    )

    assert np.array(
        [
            bellman_values[WeekIndex(w)].true_costs
            - min(bellman_values[WeekIndex(w + 1)].true_costs)
            for w in range(param.len_week)
        ]
    )[::-1] == pytest.approx(
        np.array(
            [
                [
                    6.04613200e08,
                    1.88645524e09,
                    1.34185588e08,
                    1.56207767e08,
                    9.81340925e07,
                    7.71508621e07,
                    7.20928725e07,
                    2.48914696e08,
                    3.40939014e08,
                    1.47592387e09,
                ],
                [
                    1.56229290e09,
                    5.14536786e09,
                    3.14623812e08,
                    5.61668879e08,
                    1.31192279e08,
                    9.29706317e07,
                    1.11108924e08,
                    4.57162649e08,
                    3.30687337e08,
                    1.69639743e09,
                ],
                [
                    2.88112493e09,
                    7.44917620e09,
                    8.27702609e08,
                    2.02131251e09,
                    1.53845069e08,
                    1.09697727e08,
                    1.31937494e08,
                    6.49155773e07,
                    1.17281364e08,
                    9.79132094e08,
                ],
                [
                    4.20897115e09,
                    7.96162294e09,
                    1.64178620e09,
                    2.83801330e09,
                    1.83628949e08,
                    1.38997572e08,
                    1.60489058e08,
                    9.04475775e07,
                    1.37349165e08,
                    7.93046500e08,
                ],
                [
                    1.85048426e09,
                    4.63202915e09,
                    1.60381630e08,
                    1.82910310e08,
                    1.09338310e08,
                    6.84629012e07,
                    9.21743178e07,
                    5.42603810e07,
                    7.57435003e07,
                    7.41652683e08,
                ],
            ]
        )
    )

    assert np.array(
        [bellman_values[WeekIndex(w)].duals for w in range(param.len_week)]
    )[::-1] == pytest.approx(
        np.array(
            [
                [
                    [-3215.57, -215.58],
                    [-9769.22, -9769.22],
                    [-166.24, -166.25],
                    [-215.57, -215.58],
                    [-139.926094, -166.25],
                    [-166.24, -119.16],
                    [-139.926094, -166.25],
                    [-119.32, 2884.88],
                    [2880.680183, -115.17],
                    [-115.17, 3000.0],
                ],
                [
                    [-10000.0, -500.0],
                    [-10000.0, -12769.22],
                    [-6215.57, -354.135648],
                    [-3215.58, -3215.58],
                    [-166.24, -166.25],
                    [-166.24, -119.16],
                    [-104.426137, -166.25],
                    [-119.16, 2914.586708],
                    [2860.07, -166.25],
                    [-119.32, 5884.88],
                ],
                [
                    [-13000.0, -10000.0],
                    [-10000.0, -15769.22],
                    [-3215.58, -3215.58],
                    [-10000.0, -10000.0],
                    [-166.24, -166.25],
                    [-166.250321, -166.250321],
                    [-104.426137, -166.25],
                    [-151.933904, -108.905462],
                    [0.0, -166.25],
                    [-119.16, 5914.59],
                ],
                [
                    [-16000.0, -10000.0],
                    [-10000.0, -13333.33],
                    [-10000.0, -10000.0],
                    [-10000.0, -10000.0],
                    [-166.24, -166.25],
                    [-166.25, -166.25],
                    [-166.24, -166.25],
                    [-166.25, -138.85],
                    [0.0, -166.25],
                    [-151.93, 2891.09],
                ],
                [
                    [-19000.0, -10000.0],
                    [-10000.0, -13333.33],
                    [-166.25, -166.25],
                    [-220.43833077, -220.43833077],
                    [-124.68609673, -166.25],
                    [-166.25, -114.57485738],
                    [-124.68609673, -166.25],
                    [-166.25, 0.0],
                    [0.0, -166.25],
                    [-166.25, 2861.15],
                ],
            ]
        )
    )
