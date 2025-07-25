import numpy as np
import pytest

from calculate_reward_and_bellman_values import (
    get_bellman_values_from_approximate_costs,
    get_default_linear_interpolator,
    get_optimal_trajectory_from_approximate_costs,
)
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
                1.03939988e10,
                6.54510369e09,
                7.09290596e09,
                6.16775446e09,
                4.00906766e09,
                6.16775446e09,
                2.95216417e09,
                6.16775446e09,
                2.00114224e09,
                6.16775446e09,
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
                [0.0, 0.0],
                [-10000.0, 0.0],
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
                [490645.606, 628377.6569],
                [277853.0681, 1109617.894],
                [629841.303, 628377.6569],
                [277853.0681, 1424415.447],
                [769037.0, 628377.6569],
                [277853.0681, 1739213.0],
            ]
        ),
        np.array(
            [
                9.68917584e09,
                8.99575024e09,
                6.79950385e09,
                5.85128125e09,
                3.72335585e09,
                5.85128125e09,
                2.33139895e09,
                5.85128125e09,
                1.26841564e09,
                5.85128125e09,
            ]
        ),
        np.array(
            [
                [-19000.0, 0.0],
                [-10000.0, -12000.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [0.0, 0.0],
                [-10000.0, 0.0],
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
                7.43347719e09,
                7.11261520e09,
                4.99116577e09,
                4.01987197e09,
                1.91501767e09,
                4.01987197e09,
                8.30526866e08,
                4.01987197e09,
                5.56082686e08,
                4.01987197e09,
            ]
        ),
        np.array(
            [
                [-16000.0, 0.0],
                [-9000.0, -9000.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-10000.0, 0.0],
                [-3000.0, 0.0],
                [-10000.0, 0.0],
                [0.0, 0.0],
                [-10000.0, 0.0],
            ]
        ),
    ),
    LinearInterpolator(
        np.array(
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
            ]
        ),
        np.array(
            [
                4.62500557e09,
                4.26037813e09,
                2.67297065e09,
                1.90472902e09,
                5.32353961e08,
                1.67091545e09,
                1.07845531e08,
                1.67091545e09,
                4.53796304e07,
                1.67091545e09,
            ]
        ),
        np.array(
            [
                [-13000.0, 0.0],
                [-6000.0, -6000.0],
                [-10000.0, 0.0],
                [-6000.0, -3000.0],
                [-3000.0, 0.0],
                [-10000.0, 0.0],
                [-3000.0, 0.0],
                [-10000.0, 0.0],
                [0.0, 0.0],
                [-10000.0, 0.0],
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
                [396823.092, 628377.6569],
                [277853.0681, 897433.908],
                [582930.046, 628377.6569],
                [277853.0681, 1318323.454],
                [769037.0, 628377.6569],
                [277853.0681, 1739213.0],
            ]
        ),
        np.array(
            [
                1.75536556e09,
                2.02285044e09,
                7.23273591e08,
                8.33228760e08,
                5.88256414e07,
                4.15735701e08,
                5.74309515e06,
                4.15735701e08,
                5.74309515e06,
                4.15735701e08,
            ]
        ),
        np.array(
            [
                [-10000.0, 0.0],
                [-3000.0, -3000.0],
                [-3000.0, 0.0],
                [-3000.0, -3000.0],
                [-3000.0, 0.0],
                [-3000.0, 0.0],
                [0.0, 0.0],
                [-3000.0, 0.0],
                [0.0, 0.0],
                [-3000.0, 0.0],
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
        [[45072.021918, -261356.25714487]],
        [[77589.244, -59504.83195583]],
        [[0.0, 9380.12683169]],
        [[184868.35609133, 13255.96877977]],
        [[39511.45, -206625.87197806]],
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
    costs, duals, _, _ = get_antares_costs(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        output_path=output_path,
        saving_dir=saving_dir,
        name_solver=name_solver,
        controls=array_to_timescenario_list_area_value(
            expected_controls_list, param, multi_stock_management_two_nodes.areas
        ),
        save_protos=True,
        verbose=False,
    )
    assert timescenario_list_area_value_to_array(
        array_to_timescenario_list_area_value(
            expected_controls_list, param, multi_stock_management_two_nodes.areas
        ),
        param,
        multi_stock_management_two_nodes.areas,
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
    future_costs_approx = get_default_linear_interpolator(
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

    bellman_values = get_bellman_values_from_approximate_costs(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        costs_approx=costs_approx,
        final_bellman_values=expected_future_costs_approx,
        name_solver=name_solver,
        divisor=divisor,
        verbose=False,
        levels=levels,
        piecewiselinear=False,
    )

    assert time_list_area_value_to_array(
        levels, param, multi_stock_management_two_nodes.areas
    )[::-1] == pytest.approx(expected_levels)

    for i in range(5, -1, -1):
        assert bellman_values[WeekIndex(i)].get_true_inputs() == pytest.approx(
            expected_future_costs_approx_l[i].inputs
        )
        assert bellman_values[WeekIndex(i)].get_true_costs() == pytest.approx(
            expected_future_costs_approx_l[i].costs
        )

        assert bellman_values[WeekIndex(i)].get_true_duals() == pytest.approx(
            expected_future_costs_approx_l[i].duals
        )


def test_solve_for_optimal_trajectory(
    param: TimeScenarioParameter,
    costs_approx: LinearCostEstimator,
    multi_stock_management_two_nodes: MultiStockManagement,
    starting_pt: Dict[AreaIndex, float],
) -> None:
    trajectory, pseudo_opt_controls, _ = get_optimal_trajectory_from_approximate_costs(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        costs_approx=costs_approx,
        bellman_values=list_to_week_value(
            expected_future_costs_approx_l, param.len_week + 1
        ),
        level_init=starting_pt,
        name_solver=name_solver,
        divisor=divisor,
    )

    assert timescenario_area_value_to_array(trajectory, param) == pytest.approx(
        np.array(
            [
                [[246753.048082, 419150.333]],
                [[183030.806, 413932.694]],
                [[196806.81, 408715.055]],
                [[25714.45462296, 427015.09122023]],
                [[0.0, 665259.96197806]],
            ]
        )
    )
    assert timescenario_area_value_to_array(
        pseudo_opt_controls, param
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

    costs, _, _, _ = get_antares_costs(
        param=param,
        multi_stock_management=multi_stock_management_two_nodes,
        controls=array_to_timescenario_list_area_value(
            expected_controls_to_explore, param, multi_stock_management_two_nodes.areas
        ),
        saving_dir=saving_dir,
        output_path=output_path,
        name_solver=name_solver,
        verbose=False,
        save_protos=True,
        keep_intermed_res=True,
        prefix=f"test_get_opt_gap",
    )

    assert timescenario_list_area_value_to_array(
        array_to_timescenario_list_area_value(
            expected_controls_to_explore, param, multi_stock_management_two_nodes.areas
        ),
        param,
        multi_stock_management_two_nodes.areas,
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
        controls_list=array_to_timescenario_list_area_value(
            expected_controls_to_explore, param, multi_stock_management_two_nodes.areas
        ),
        opt_gap=1,
        max_gap={WeekIndex(w): max_gap[w] for w in range(param.len_week)},
    )

    assert opt_gap == pytest.approx(3.1317827969145355e-10)
