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
    ) == pytest.approx(np.array([56094.035, 127275.715]))
    assert cost == pytest.approx(64323025)
    assert np.array(
        [duals[a] for a in multi_stock_management_two_nodes.areas]
    ) == pytest.approx(np.array([-100, -100]))
    assert np.array(
        [levels[a][ScenarioIndex(0)] for a in multi_stock_management_two_nodes.areas]
    ) == pytest.approx(np.array([342221.465, 773949.785]))


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
                    2.42128078e09,
                    5.51990258e09,
                    1.07470918e09,
                    1.66543854e09,
                    1.45146339e08,
                    4.20060401e08,
                    1.21190668e08,
                    6.46747570e08,
                    4.40038471e08,
                    1.89532665e09,
                ],
                [
                    4.31543953e09,
                    8.32363876e09,
                    2.29582512e09,
                    3.37372500e09,
                    3.27972421e08,
                    5.99791688e08,
                    1.08317908e08,
                    5.67451480e08,
                    9.23825146e07,
                    1.66552746e09,
                ],
                [
                    6.83475714e09,
                    1.06710309e10,
                    4.19373077e09,
                    5.41906297e09,
                    1.58480937e09,
                    5.28360544e08,
                    7.47078107e08,
                    4.91779685e08,
                    1.89892715e08,
                    1.20257106e09,
                ],
                [
                    7.94747855e09,
                    1.12890211e10,
                    4.91455961e09,
                    6.11078671e09,
                    1.97371086e09,
                    3.63021044e08,
                    1.13853672e09,
                    3.03182855e08,
                    4.96058171e08,
                    9.99486371e08,
                ],
                [
                    5.28537372e09,
                    7.71317766e09,
                    1.91410509e09,
                    3.06404378e09,
                    7.08033144e07,
                    5.68137622e07,
                    4.54252867e07,
                    5.07124883e07,
                    2.89122467e07,
                    7.27667407e08,
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
                    [-18000.0, -6000.0],
                    [-12000.0, -12000.0],
                    [-6000.0, -215.58],
                    [-6000.0, -6000.01],
                    [-161.68, -215.58],
                    [-6000.0, -115.17],
                    [0.0, -115.17],
                    [-6000.0, 2918.44],
                    [2861.15, -115.17],
                    [-3000.0, 3000.0],
                ],
                [
                    [-17000.0, -10000.0],
                    [-10333.327222, -13333.33],
                    [-9000.0, -6000.01],
                    [-10000.0, -10000.0],
                    [-6000.0, -6000.01],
                    [-6000.0, -115.17],
                    [-161.68, -215.58],
                    [-6000.0, -48.238814],
                    [0.0, -215.58],
                    [-6000.0, 5918.44],
                ],
                [
                    [-20000.0, -10000.0],
                    [-10033.3327, -13333.33],
                    [-10000.0, -10000.0],
                    [-10000.0, -10000.0],
                    [-6000.0, -6000.01],
                    [-6000.0, -119.16],
                    [-5789.484211, -6000.01],
                    [-6000.0, -115.17],
                    [0.0, -500.0],
                    [-6000.0, 2884.83],
                ],
                [
                    [-23000.0, -10000.0],
                    [-10003.33307, -13333.34],
                    [-10000.0, -10000.0],
                    [-10000.0, -10000.0],
                    [-6000.0, -6000.01],
                    [-6000.0, -6000.004068],
                    [-6000.0, -6000.01],
                    [-6000.0, -119.16],
                    [0.0, -6000.01],
                    [-6000.0, 2884.83],
                ],
                [
                    [-26000.0, -10000.0],
                    [-10000.332996, -13333.33],
                    [-6000.0, -6000.01],
                    [-10000.0, -10000.0],
                    [-4500.00580527, -6000.01],
                    [-6000.0, -46.60474687],
                    [-138.85, -185.14],
                    [-6000.0, 0.0],
                    [0.0, -158.88],
                    [-6000.0, 2880.84],
                ],
            ]
        )
    )
