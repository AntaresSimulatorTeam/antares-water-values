import numpy as np
import pytest

from functions_iterative import ReservoirManagement, TimeScenarioParameter
from multi_stock_bellman_value_calculation import *
from read_antares_data import Reservoir
from reservoir_management import MultiStockManagement
from type_definition import (
    array_to_area_value,
    array_to_timescenario_area_value,
    array_to_timescenario_list_area_value,
    list_to_week_value,
    time_list_area_value_to_array,
    timescenario_area_value_to_array,
)

param = TimeScenarioParameter(len_week=5, len_scenario=1)

reservoir_1 = Reservoir("test_data/two_nodes", "area_1")
reservoir_management_1 = ReservoirManagement(
    reservoir=reservoir_1,
    penalty_bottom_rule_curve=3000,
    penalty_upper_rule_curve=3000,
    penalty_final_level=3000,
    force_final_level=True,
)

reservoir_2 = Reservoir("test_data/two_nodes", "area_2")
reservoir_management_2 = ReservoirManagement(
    reservoir=reservoir_2,
    penalty_bottom_rule_curve=3000,
    penalty_upper_rule_curve=3000,
    penalty_final_level=3000,
    force_final_level=True,
)

multi_management = MultiStockManagement(
    [reservoir_management_1, reservoir_management_2]
)

n_controls_init = 2
output_path = "test_data/two_nodes"
saving_dir = "dev/test"
name_solver = "CLP"
nSteps_bellman = 5
starting_pt = np.array(
    [
        mng.reservoir.bottom_rule_curve[0] * 0.7
        + mng.reservoir.upper_rule_curve[0] * 0.3
        for mng in multi_management.dict_reservoirs.values()
    ]
)
precision = 1e-3
method = "lines"
divisor = {"euro": 1e8, "energy": 1e4}

expected_controls_list = np.array(
    [
        [[[0.0, -322182.0], [306936.0, 419664.0]]],
        [[[0.0, -322182.0], [306936.0, 419664.0]]],
        [[[0.0, -322182.0], [306936.0, 419664.0]]],
        [[[0.0, -322182.0], [306936.0, 419664.0]]],
        [[[0.0, -322182.0], [306936.0, 419664.0]]],
    ]
)
expected_controls = np.array(
    [
        [[[0.0, -322182.0], [306936.0, 419664.0]]],
        [[[0.0, -322182.0], [306936.0, 419664.0]]],
        [[[0.0, -322182.0], [306936.0, 419664.0]]],
        [[[0.0, -322182.0], [306936.0, 419664.0]]],
        [[[0.0, -322182.0], [306936.0, 419664.0]]],
    ]
)
expected_costs = np.array(
    [
        [[1.26720349e09, 5.47296058e06]],
        [[5.47244322e09, 2.37623525e07]],
        [[6.90754711e09, 3.06500583e07]],
        [[6.98829385e09, 3.96365352e07]],
        [[2.40742031e09, 5.74309515e06]],
    ]
)
expected_duals = np.array(
    [
        [[[-1.00000004e04, -1.33333352e04], [4.06062600e-04, 4.05834900e-04]]],
        [[[-1.00000004e04, -1.33333352e04], [3.94735200e-04, 4.06859500e-04]]],
        [[[-1.00000004e04, -1.33333352e04], [4.01110400e-04, 4.09307100e-04]]],
        [[[-1.00000004e04, -1.33333352e04], [4.02533400e-04, 3.89669100e-04]]],
        [[[-1.00000004e04, -1.33333352e04], [4.08225600e-04, 4.09762500e-04]]],
    ]
)
expected_future_costs_approx = LinearInterpolator(
    np.array(
        [
            [277853.0681, 628377.6569],
            [277853.0681, 628377.6569],
            [277853.0681, 628377.6569],
            [277853.0681, 628377.6569],
        ]
    ),
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([[0.0, 0.0], [-0.0, 0.0], [0.0, -0.0], [0.0, 0.0]]),
)

expected_levels = np.array(
    [
        [
            [0.0, 628377.6569],
            [277853.0681, 0.0],
            [175340.436, 628377.6569],
            [277853.0681, 396540.564],
            [409896.721, 628377.6569],
            [277853.0681, 927000.529],
            [589466.8605, 628377.6569],
            [277853.0681, 1333106.7645],
            [769037.0, 628377.6569],
            [277853.0681, 1739213.0],
        ],
        [
            [0.0, 628377.6569],
            [277853.0681, 0.0],
            [178416.584, 628377.6569],
            [277853.0681, 403497.416],
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
            [491414.643, 628377.6569],
            [277853.0681, 1111357.107],
            [630225.8215, 628377.6569],
            [277853.0681, 1425285.0535],
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

expected_future_costs_approx_l = [
    LinearInterpolator(
        np.array(
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
            ]
        ),
        np.array(
            [
                8.84872182e09,
                3.98888968e09,
                3.39753723e09,
                2.47238573e09,
                3.13698928e08,
                2.47238574e09,
                0.00000000e00,
                3.40634310e09,
                1.25983374e08,
                5.78151779e09,
            ]
        ),
        np.array(
            [
                [-37000.0, 0.0],
                [-10000.0, -6000.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-1000.0, 0.0],
                [-10000.0, 3000.0],
                [3000.0, 0.0],
                [-10000.0, 9000.0],
            ]
        ),
    ),
    LinearInterpolator(
        np.array(
            [
                [0.0, 628377.6569],
                [277853.0681, 0.0],
                [183030.806, 628377.6569],
                [277853.0681, 413932.694],
                [491414.643, 628377.6569],
                [277853.0681, 1111357.107],
                [630225.8215, 628377.6569],
                [277853.0681, 1425285.0535],
                [769037.0, 628377.6569],
                [277853.0681, 1739213.0],
            ]
        ),
        np.array(
            [
                9.10018478e09,
                5.24449023e09,
                4.44184942e09,
                3.49362683e09,
                1.35801112e09,
                3.49362683e09,
                3.86332850e08,
                4.43541065e09,
                0.00000000e00,
                6.85868289e09,
            ]
        ),
        np.array(
            [
                [-31000.0, 0.0],
                [-10000.0, -6000.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-7000.0, 0.0],
                [-10000.0, 3000.0],
                [-1000.0, 0.0],
                [-10000.0, 9000.0],
            ]
        ),
    ),
    LinearInterpolator(
        np.array(
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
            ]
        ),
        np.array(
            [
                7.85219955e09,
                5.38347261e09,
                4.05859285e09,
                3.08729903e09,
                9.82444750e08,
                3.08729905e09,
                0.00000000e00,
                4.03951816e09,
                3.35066384e08,
                6.61132393e09,
            ]
        ),
        np.array(
            [
                [-25000.0, 0.0],
                [-3972.218333, -16333.34],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-7000.0, 0.0],
                [-10000.0, 3000.0],
                [3000.0, 0.0],
                [-10000.0, 12000.0],
            ]
        ),
    ),
    LinearInterpolator(
        np.array(
            [
                [0.0, 628377.6569],
                [277853.0681, 0.0],
                [178416.584, 628377.6569],
                [277853.0681, 403497.416],
                [486031.384, 628377.6569],
                [277853.0681, 1099182.616],
                [627534.192, 628377.6569],
                [277853.0681, 1419197.808],
                [769037.0, 628377.6569],
                [277853.0681, 1739213.0],
            ]
        ),
        np.array(
            [
                5.09816331e09,
                4.21709339e09,
                2.23973649e09,
                1.24537157e09,
                0.00000000e00,
                1.24537158e09,
                4.24508418e08,
                2.51768500e09,
                1.24489851e09,
                5.15616581e09,
            ]
        ),
        np.array(
            [
                [-19000.0, 0.0],
                [-3000.0, -12000.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [0.0, 0.0],
                [-10000.0, 0.0],
                [3000.0, 0.0],
                [-10000.0, 6000.0],
                [9000.0, 0.0],
                [-3000.0, 12000.0],
            ]
        ),
    ),
    LinearInterpolator(
        np.array(
            [
                [0.0, 628377.6569],
                [277853.0681, 0.0],
                [175340.436, 628377.6569],
                [277853.0681, 396540.564],
                [409896.721, 628377.6569],
                [277853.0681, 927000.529],
                [589466.8605, 628377.6569],
                [277853.0681, 1333106.7645],
                [769037.0, 628377.6569],
                [277853.0681, 1739213.0],
            ]
        ),
        np.array(
            [
                1.74152289e09,
                1.57580902e09,
                1.79389452e08,
                1.20000000e01,
                0.00000000e00,
                2.00000000e00,
                5.38710416e08,
                1.44990211e09,
                1.35386645e09,
                3.88653955e09,
            ]
        ),
        np.array(
            [
                [-13000.0, 0.0],
                [0.0, -6000.0],
                [-3000.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [3000.0, 0.0],
                [0.0, 6000.0],
                [6000.0, 0.0],
                [0.0, 6000.0],
            ]
        ),
    ),
    LinearInterpolator(
        np.array([[277853.0681, 628377.6569]]),
        np.array([0.0]),
        np.array([[0.0, 0.0]]),
    ),
]
expected_correlations = np.array([[1.0, 0.0], [0.0, 1.0]])
expected_pseudo_opt_controls = np.array(
    [
        [[45619.31803823, 246455.966]],
        [[79349.055, 36983.635]],
        [[16083.106, 36780.644]],
        [[88581.367, 38512.856]],
        [[86667.11661416, -206625.83300034]],
    ]
)
expected_controls_to_explore = np.array(
    [
        [[[-0.0, 419664.0]]],
        [[[-0.0, -322182.0]]],
        [[[-0.0, -322182.0]]],
        [[[-0.0, 419664.0]]],
        [[[306936.0, -322182.0]]],
    ]
)

costs_approx = LinearCostEstimator(
    param=param, controls=expected_controls, costs=expected_costs, duals=expected_duals
)


def test_initialize_controls() -> None:
    controls_list = initialize_controls(
        param=param,
        multi_stock_management=multi_management,
        n_controls_init=n_controls_init,
    )

    assert timescenario_list_area_value_to_array(
        controls_list, param, multi_management.areas
    ) == pytest.approx(expected_controls_list)


def test_Lget_costs() -> None:
    controls, costs, duals = Lget_costs(
        param=param,
        multi_stock_management=multi_management,
        output_path=output_path,
        saving_directory=saving_dir,
        name_solver=name_solver,
        controls_list=array_to_timescenario_list_area_value(
            expected_controls_list, param, multi_management.areas
        ),
        load_from_protos=True,
        verbose=False,
    )
    assert timescenario_list_area_value_to_array(
        controls, param, multi_management.areas
    ) == pytest.approx(expected_controls)
    assert timescenario_list_value_to_array(costs, param) == pytest.approx(
        expected_costs
    )
    assert timescenario_list_area_value_to_array(
        duals, param, multi_management.areas
    ) == pytest.approx(expected_duals)


def test_initialize_future_costs() -> None:

    # Initialize our approximation on future costs
    future_costs_approx = initialize_future_costs(
        starting_pt=array_to_area_value(starting_pt, multi_management.areas),
        multi_stock_management=multi_management,
    )

    assert future_costs_approx.inputs == pytest.approx(
        expected_future_costs_approx.inputs
    )
    assert future_costs_approx.costs == pytest.approx(
        expected_future_costs_approx.costs
    )
    assert future_costs_approx.duals == pytest.approx(
        expected_future_costs_approx.duals
    )


def test_get_correlation_matrix() -> None:
    correlation_matrix = get_correlation_matrix(
        multi_stock_management=multi_management,
        corr_type="no_corrs",
    )
    assert correlation_matrix == pytest.approx(expected_correlations)


def test_get_bellman_values_from_costs() -> None:
    trajectory = {
        TimeScenarioIndex(w, s): array_to_area_value(
            starting_pt, multi_management.areas
        )
        for w in range(param.len_week)
        for s in range(param.len_scenario)
    }

    (
        levels,
        bellman_costs,
        _,
        _,
        future_costs_approx_l,
    ) = get_bellman_values_from_costs(
        param=param,
        multi_stock_management=multi_management,
        costs_approx=costs_approx,
        future_costs_approx=expected_future_costs_approx,
        nSteps_bellman=nSteps_bellman,
        name_solver=name_solver,
        method=method,
        trajectory=trajectory,
        correlations=expected_correlations,
        divisor=divisor,
        verbose=False,
    )

    assert time_list_area_value_to_array(levels, param, multi_management.areas)[
        ::-1
    ] == pytest.approx(expected_levels)
    assert np.array(
        [[c for c in bellman_costs[WeekIndex(w)]] for w in range(param.len_week)]
    )[::-1] == pytest.approx(
        np.array(
            [
                [
                    1.00000017e16,
                    1.00000016e16,
                    1.00000002e16,
                    1.00000000e16,
                    1.00000000e16,
                    1.00000000e16,
                    1.00000005e16,
                    1.00000014e16,
                    1.00000014e16,
                    1.00000039e16,
                ],
                [
                    1.00000051e16,
                    1.00000042e16,
                    1.00000022e16,
                    1.00000012e16,
                    1.00000000e16,
                    1.00000012e16,
                    1.00000004e16,
                    1.00000025e16,
                    1.00000012e16,
                    1.00000052e16,
                ],
                [
                    1.00000084e16,
                    1.00000059e16,
                    1.00000046e16,
                    1.00000036e16,
                    1.00000015e16,
                    1.00000036e16,
                    1.00000005e16,
                    1.00000045e16,
                    1.00000008e16,
                    1.00000071e16,
                ],
                [
                    1.00000105e16,
                    1.00000067e16,
                    1.00000059e16,
                    1.00000049e16,
                    1.00000028e16,
                    1.00000049e16,
                    1.00000018e16,
                    1.00000059e16,
                    1.00000014e16,
                    1.00000083e16,
                ],
                [
                    1.00000102e16,
                    1.00000053e16,
                    1.00000047e16,
                    1.00000038e16,
                    1.00000017e16,
                    1.00000038e16,
                    1.00000013e16,
                    1.00000047e16,
                    1.00000015e16,
                    1.00000071e16,
                ],
            ]
        )
    )
    for i in range(6):
        assert future_costs_approx_l[i].inputs == pytest.approx(
            expected_future_costs_approx_l[i].inputs
        )
        assert future_costs_approx_l[i].costs == pytest.approx(
            expected_future_costs_approx_l[i].costs
        )

        assert future_costs_approx_l[i].duals == pytest.approx(
            expected_future_costs_approx_l[i].duals
        )


def test_solve_for_optimal_trajectory() -> None:
    trajectory, pseudo_opt_controls, _ = solve_for_optimal_trajectory(
        param=param,
        multi_stock_management=multi_management,
        costs_approx=costs_approx,
        future_costs_approx_l=list_to_week_value(
            expected_future_costs_approx_l, param.len_week + 1
        ),
        starting_pt=array_to_area_value(starting_pt, multi_management.areas),
        name_solver=name_solver,
        divisor=divisor,
    )

    assert timescenario_area_value_to_array(
        trajectory, param, multi_management.areas
    ) == pytest.approx(
        np.array(
            [
                # [[277853.0681, 628377.6569]],
                [[246205.75196177, 413932.694]],
                [[180723.695, 408715.055]],
                [[178416.584, 403497.416]],
                [[103611.213, 396540.564]],
                [[30741.09338584, 634785.39300033]],
            ]
        )
    )
    assert timescenario_area_value_to_array(
        pseudo_opt_controls, param, multi_management.areas
    ) == pytest.approx(expected_pseudo_opt_controls)


def test_select_controls_to_explore() -> None:
    controls_list = select_controls_to_explore(
        param=param,
        multi_stock_management=multi_management,
        pseudo_opt_controls=array_to_timescenario_area_value(
            expected_pseudo_opt_controls, param, multi_management.areas
        ),
        costs_approx=costs_approx,
    )
    assert timescenario_list_area_value_to_array(
        controls_list, param, multi_management.areas
    ) == pytest.approx(expected_controls_to_explore)


def test_get_opt_gap() -> None:

    controls, costs, _ = Lget_costs(
        param=param,
        multi_stock_management=multi_management,
        controls_list=array_to_timescenario_list_area_value(
            expected_controls_to_explore, param, multi_management.areas
        ),
        saving_directory=saving_dir,
        output_path=output_path,
        name_solver=name_solver,
        verbose=False,
        load_from_protos=True,
        prefix=f"test_get_opt_gap",
    )

    assert timescenario_list_area_value_to_array(
        controls, param, multi_management.areas
    ) == pytest.approx(
        np.array(
            [
                [[[-0.0, 419664.0]]],
                [[[-0.0, -322182.0]]],
                [[[-0.0, -322182.0]]],
                [[[-0.0, 419664.0]]],
                [[[306936.0, -322182.0]]],
            ]
        )
    )

    assert timescenario_list_value_to_array(costs, param) == pytest.approx(
        np.array(
            [
                [[8.71941839e06]],
                [[5.47244322e09]],
                [[6.90754711e09]],
                [[5.79023676e08]],
                [[6.30109715e08]],
            ]
        )
    )

    max_gap = np.mean(
        np.max(expected_costs, axis=2) - np.min(expected_costs, axis=2), axis=1
    )

    assert max_gap == pytest.approx(
        np.array(
            [1.26173053e09, 5.44868087e09, 6.87689705e09, 6.94865731e09, 2.40167721e09]
        )
    )

    opt_gap = get_opt_gap(
        param=param,
        costs=costs,
        costs_approx=costs_approx,
        controls_list=controls,
        opt_gap=1,
        max_gap={WeekIndex(w): max_gap[w] for w in range(param.len_week)},
    )

    assert opt_gap == pytest.approx(3.1317827969145355e-10)
