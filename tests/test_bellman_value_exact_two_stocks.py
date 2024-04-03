from functions_iterative import (
    TimeScenarioParameter,
    ReservoirManagement,
    TimeScenarioIndex,
)
from multi_stock_bellman_value_calculation import calculate_bellman_value_multi_stock
from calculate_reward_and_bellman_values import (
    MultiStockManagement,
    BellmanValueCalculation,
    MultiStockBellmanValueCalculation,
    RewardApproximation,
)
from read_antares_data import Reservoir
import pytest
import numpy as np
from itertools import product


def test_iterate_over_stock_discretization() -> None:
    param = TimeScenarioParameter(len_week=1, len_scenario=1)

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

    x_1 = np.linspace(0, reservoir_1.capacity, num=20)
    x_2 = np.linspace(0, reservoir_2.capacity, num=20)

    calculation_1 = BellmanValueCalculation(
        param=param,
        reward={
            TimeScenarioIndex(0, 0): RewardApproximation(
                lb_control=-reservoir_1.max_pumping[0],
                ub_control=reservoir_1.max_generating[0],
                ub_reward=0,
            )
        },
        reservoir_management=reservoir_management_1,
        stock_discretization=x_1,
    )
    calculation_2 = BellmanValueCalculation(
        param=param,
        reward={
            TimeScenarioIndex(0, 0): RewardApproximation(
                lb_control=-reservoir_2.max_pumping[0],
                ub_control=reservoir_2.max_generating[0],
                ub_reward=0,
            )
        },
        reservoir_management=reservoir_management_2,
        stock_discretization=x_2,
    )

    all_reservoirs = MultiStockBellmanValueCalculation([calculation_1, calculation_2])

    assert [idx for idx in all_reservoirs.get_product_stock_discretization()][0] == [
        i for i in product([i for i in range(20)], [i for i in range(20)])
    ][0]
    assert [idx for idx in all_reservoirs.get_product_stock_discretization()] == [
        i for i in product([i for i in range(20)], [i for i in range(20)])
    ]


def test_bellman_value_exact_calculation_multi_stock() -> None:

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

    all_reservoirs = MultiStockManagement(
        [reservoir_management_1, reservoir_management_2]
    )

    x_1 = np.linspace(0, reservoir_1.capacity, num=5)
    x_2 = np.linspace(0, reservoir_2.capacity, num=5)

    vb = calculate_bellman_value_multi_stock(
        param=param,
        multi_stock_management=all_reservoirs,
        output_path="test_data/two_nodes",
        X={"area_1": x_1, "area_2": x_2},
    )

    assert vb[0]["intercept"] == pytest.approx(
        np.array(
            [
                [2.7090407e10, 2.6058617e10, 2.6046816e10, 2.6046816e10, 2.6046816e10],
                [7.7349028e09, 6.7031137e09, 6.6913116e09, 6.6913116e09, 6.6913116e09],
                [9.2355341e08, 5.8537856e08, 5.8481184e08, 5.8481184e08, 5.8481184e08],
                [9.2291040e08, 5.8490163e08, 5.8481184e08, 5.8481184e08, 5.8481184e08],
                [9.2291040e08, 5.8490163e08, 5.8481184e08, 5.8481184e08, 5.8481184e08],
            ]
        )
    )
    assert vb[0]["slope_area_1"] == pytest.approx(
        np.array(
            [
                [
                    -1.0100000e05,
                    -1.0100000e05,
                    -1.0100000e05,
                    -1.0100000e05,
                    -1.0100000e05,
                ],
                [
                    -9.8000000e04,
                    -9.8000000e04,
                    -9.8000000e04,
                    -9.8000000e04,
                    -9.8000000e04,
                ],
                [
                    -1.1916082e02,
                    -8.6371864e01,
                    0.0000000e00,
                    0.0000000e00,
                    0.0000000e00,
                ],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00],
            ]
        )
    )
    assert vb[0]["slope_area_2"] == pytest.approx(
        np.array(
            [
                [-13333.335, -115.168045, 0.0, 0.0, 0.0],
                [-13333.335, -115.168045, 0.0, 0.0, 0.0],
                [-3000.0, -115.16554, 0.0, 0.0, 0.0],
                [-3000.0, -115.16548, 0.0, 0.0, 0.0],
                [-3000.0, -115.16548, 0.0, 0.0, 0.0],
            ]
        )
    )
