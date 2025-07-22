import numpy as np
import pytest

from functions_iterative import TimeScenarioParameter
from multi_stock_bellman_value_calculation import *
from reservoir_management import MultiStockManagement
from type_definition import (
    array_to_timescenario_area_value,
    array_to_timescenario_list_area_value,
    array_to_timescenario_list_value,
    list_to_week_value,
    time_list_area_value_to_array,
    timescenario_area_value_to_array,
)

n_controls_init = 2
output_path = "test_data/two_nodes"
saving_dir = "dev/test"
name_solver = "CLP"
nSteps_bellman = 5
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
    np.array([[769037.000000, 1739213.000000]]),
    np.array([0.0]),
    np.array([[0.0, 0.0]]),
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
                8.17125120e09,
                4.32746725e09,
                4.89092244e09,
                3.96577094e09,
                1.80708414e09,
                3.96577094e09,
                7.57101983e08,
                3.96577094e09,
                0.00000000e00,
                4.68637962e09,
            ]
        ),
        np.array(
            [
                [-22000.0, 0.0],
                [-10000.0, -3000.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-7000.0, 0.0],
                [-10000.0, 0.0],
                [-4000.0, 0.0],
                [-10000.0, 3000.0],
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
                8.43608972e09,
                7.22886200e09,
                5.56026045e09,
                4.61203785e09,
                2.47642215e09,
                4.61203785e09,
                1.08831035e09,
                4.61203785e09,
                0.00000000e00,
                5.35278208e09,
            ]
        ),
        np.array(
            [
                [-19000.0, 0.0],
                [-10000.0, -9000.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-7000.0, 0.0],
                [-10000.0, 3000.0],
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
                [277853.0681, 1739213.0],
            ]
        ),
        np.array(
            [
                6.80374086e09,
                5.60917357e09,
                4.36835083e09,
                3.39705703e09,
                1.29220273e09,
                3.39705703e09,
                0.00000000e00,
                3.39705703e09,
                4.26085554e09,
            ]
        ),
        np.array(
            [
                [-16000.0, 0.0],
                [-10000.0, -6000.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [0.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 6000.0],
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
                [277853.0681, 1419197.808],
                [769037.0, 628377.6569],
                [277853.0681, 1739213.0],
            ]
        ),
        np.array(
            [
                4.03284063e09,
                2.21347210e09,
                2.07311540e09,
                1.07875050e09,
                0.00000000e00,
                1.07875050e09,
                1.39101834e09,
                1.97940837e08,
                2.35106391e09,
            ]
        ),
        np.array(
            [
                [-13000.0, 0.0],
                [-10000.0, -3000.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [0.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 3000.0],
                [3000.0, 0.0],
                [-10000.0, 3000.0],
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
                [277853.0681, 1333106.7645],
                [769037.0, 628377.6569],
                [277853.0681, 1739213.0],
            ]
        ),
        np.array(
            [
                1.20975849e09,
                3.84895134e08,
                1.77666522e08,
                0.00000000e00,
                2.31583419e08,
                2.76445611e08,
                1.44990214e09,
            ]
        ),
        np.array(
            [
                [-10000.0, 0.0],
                [0.0, -3000.0],
                [-3000.0, 0.0],
                [0.0, 0.0],
                [0.0, 3000.0],
                [3000.0, 0.0],
                [0.0, 3000.0],
            ]
        ),
    ),
    LinearInterpolator(
        np.array([[769037.000000, 1739213.000000]]),
        np.array([0.0]),
        np.array([[0.0, 0.0]]),
    ),
]
expected_correlations = np.array([[1.0, 0.0], [0.0, 1.0]])
expected_pseudo_opt_controls = np.array(
    [
        [[45072.021918, 246455.966]],
        [[79896.355, 36983.635]],
        [[16083.106, 36780.644]],
        [[119896.77142857, 38512.856]],
        [[86092.807085, -206625.87197806]],
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


@pytest.fixture
def costs_approx(
    param: TimeScenarioParameter, multi_stock_management_two_nodes: MultiStockManagement
) -> LinearCostEstimator:
    return LinearCostEstimator(
        param=param,
        controls=array_to_timescenario_list_area_value(
            expected_controls, param, multi_stock_management_two_nodes.areas
        ),
        costs=array_to_timescenario_list_value(expected_costs, param),
        duals=array_to_timescenario_list_area_value(
            expected_duals, param, multi_stock_management_two_nodes.areas
        ),
        type_estimator="LinearDecomposer",
    )


def test_initialize_controls(
    param: TimeScenarioParameter,
    multi_stock_management_two_nodes: MultiStockManagement,
) -> None:
    controls_list = initialize_controls(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        n_controls_init=n_controls_init,
    )

    assert timescenario_list_area_value_to_array(
        controls_list, param, multi_stock_management_two_nodes.areas
    ) == pytest.approx(expected_controls_list)


def test_Lget_costs(
    param: TimeScenarioParameter,
    multi_stock_management_two_nodes: MultiStockManagement,
) -> None:
    controls, costs, duals = Lget_costs(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        output_path=output_path,
        saving_directory=saving_dir,
        name_solver=name_solver,
        controls_list=array_to_timescenario_list_area_value(
            expected_controls_list, param, multi_stock_management_two_nodes.areas
        ),
        save_protos=True,
        verbose=False,
    )
    assert timescenario_list_area_value_to_array(
        controls, param, multi_stock_management_two_nodes.areas
    ) == pytest.approx(expected_controls)
    assert timescenario_list_value_to_array(costs, param) == pytest.approx(
        expected_costs
    )
    assert timescenario_list_area_value_to_array(
        duals, param, multi_stock_management_two_nodes.areas
    ) == pytest.approx(expected_duals)


def test_initialize_future_costs(
    multi_stock_management_two_nodes: MultiStockManagement,
) -> None:

    # Initialize our approximation on future costs
    future_costs_approx = initialize_future_costs(
        multi_stock_management=multi_stock_management_two_nodes,
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


def test_get_correlation_matrix(
    multi_stock_management_two_nodes: MultiStockManagement,
) -> None:
    correlation_matrix = get_correlation_matrix(
        multi_stock_management=multi_stock_management_two_nodes,
        corr_type="no_corrs",
    )
    assert correlation_matrix == pytest.approx(expected_correlations)


def test_get_bellman_values_from_costs(
    param: TimeScenarioParameter,
    costs_approx: LinearCostEstimator,
    multi_stock_management_two_nodes: MultiStockManagement,
    starting_pt: Dict[AreaIndex, float],
) -> None:
    trajectory = {
        TimeScenarioIndex(w, s): starting_pt
        for w in range(param.len_week)
        for s in range(param.len_scenario)
    }

    levels = multi_stock_management_two_nodes.get_disc(
        param=param,
        xNsteps=nSteps_bellman,
        trajectory=trajectory,
        correlation_matrix=expected_correlations,
        method=method,
    )

    bellman_values = get_bellman_values_from_costs(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        costs_approx=costs_approx,
        final_bellman_values=expected_future_costs_approx,
        name_solver=name_solver,
        divisor=divisor,
        verbose=False,
        levels=levels,
    )

    assert time_list_area_value_to_array(
        levels, param, multi_stock_management_two_nodes.areas
    )[::-1] == pytest.approx(expected_levels)

    for i in range(5, -1, -1):
        assert bellman_values[WeekIndex(i)].inputs == pytest.approx(
            expected_future_costs_approx_l[i].inputs
        )
        assert bellman_values[WeekIndex(i)].costs - min(
            bellman_values[WeekIndex(i)].costs
        ) == pytest.approx(expected_future_costs_approx_l[i].costs)

        assert bellman_values[WeekIndex(i)].duals == pytest.approx(
            expected_future_costs_approx_l[i].duals
        )


def test_solve_for_optimal_trajectory(
    param: TimeScenarioParameter,
    costs_approx: LinearCostEstimator,
    multi_stock_management_two_nodes: MultiStockManagement,
    starting_pt: Dict[AreaIndex, float],
) -> None:
    trajectory, pseudo_opt_controls, _ = solve_for_optimal_trajectory(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        costs_approx=costs_approx,
        future_costs_approx_l=list_to_week_value(
            expected_future_costs_approx_l, param.len_week + 1
        ),
        starting_pt=starting_pt,
        name_solver=name_solver,
        divisor=divisor,
    )

    assert timescenario_area_value_to_array(
        trajectory, param, multi_stock_management_two_nodes.areas
    ) == pytest.approx(
        np.array(
            [
                # [[277853.0681, 628377.6569]],
                [[2.46753048e05, 4.13932694e05]],
                [[1.80723695e05, 4.08715055e05]],
                [[1.78416584e05, 4.03497416e05]],
                [[7.22958086e04, 3.96540564e05]],
                [[2.91500001e-03, 6.34785432e05]],
            ]
        )
    )
    assert timescenario_area_value_to_array(
        pseudo_opt_controls, param, multi_stock_management_two_nodes.areas
    ) == pytest.approx(expected_pseudo_opt_controls)


def test_select_controls_to_explore(
    param: TimeScenarioParameter,
    costs_approx: LinearCostEstimator,
    multi_stock_management_two_nodes: MultiStockManagement,
) -> None:
    controls_list = select_controls_to_explore(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        pseudo_opt_controls=array_to_timescenario_area_value(
            expected_pseudo_opt_controls, param, multi_stock_management_two_nodes.areas
        ),
        costs_approx=costs_approx,
    )
    assert timescenario_list_area_value_to_array(
        controls_list, param, multi_stock_management_two_nodes.areas
    ) == pytest.approx(expected_controls_to_explore)


def test_get_opt_gap(
    param: TimeScenarioParameter,
    costs_approx: LinearCostEstimator,
    multi_stock_management_two_nodes: MultiStockManagement,
) -> None:

    controls, costs, _ = Lget_costs(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        controls_list=array_to_timescenario_list_area_value(
            expected_controls_to_explore, param, multi_stock_management_two_nodes.areas
        ),
        saving_directory=saving_dir,
        output_path=output_path,
        name_solver=name_solver,
        verbose=False,
        save_protos=True,
        prefix=f"test_get_opt_gap",
    )

    assert timescenario_list_area_value_to_array(
        controls, param, multi_stock_management_two_nodes.areas
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
