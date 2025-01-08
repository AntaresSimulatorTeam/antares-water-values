from itertools import product

import numpy as np
import pytest

from estimation import BellmanValueEstimation
from functions_iterative import ReservoirManagement, TimeScenarioParameter
from multi_stock_bellman_value_calculation import AntaresProblem
from read_antares_data import Reservoir
from reservoir_management import MultiStockManagement
from simple_bellman_value_calculation import calculate_bellman_value_directly
from stock_discretization import StockDiscretization


def test_iterate_over_stock_discretization() -> None:

    reservoir_1 = Reservoir("test_data/two_nodes", "area_1")

    reservoir_2 = Reservoir("test_data/two_nodes", "area_2")

    x_1 = np.linspace(0, reservoir_1.capacity, num=20)
    x_2 = np.linspace(0, reservoir_2.capacity, num=20)

    stock_discretization = StockDiscretization(
        {reservoir_1.area: x_1, reservoir_2.area: x_2}
    )

    assert [idx for idx in stock_discretization.get_product_stock_discretization()][
        0
    ] == [i for i in product([i for i in range(20)], [i for i in range(20)])][0]
    assert [idx for idx in stock_discretization.get_product_stock_discretization()] == [
        i for i in product([i for i in range(20)], [i for i in range(20)])
    ]


def test_solve_with_bellman_multi_stock() -> None:

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

    multi_stock_management = MultiStockManagement(
        [reservoir_management_1, reservoir_management_2]
    )

    x_1 = np.linspace(0, reservoir_1.capacity, num=5)
    x_2 = np.linspace(0, reservoir_2.capacity, num=5)
    X = {"area_1": x_1, "area_2": x_2}

    m = AntaresProblem(
        scenario=0,
        week=0,
        path="test_data/two_nodes",
        itr=1,
    )
    m.create_weekly_problem_itr(
        param=param,
        multi_stock_management=multi_stock_management,
    )

    V = {
        "intercept": np.array(
            [
                [2.7942497e09, 1.3248531e09, 1.2739775e09, 1.2459432e09, 1.2459430e09],
                [1.3137651e09, 7.6301440e07, 3.7071564e07, 3.1171368e07, 3.1171356e07],
                [4.1457952e08, 5.7581808e07, 3.0331620e07, 3.0331258e07, 3.0331248e07],
                [4.0567504e08, 5.7574440e07, 3.0331604e07, 3.0331258e07, 3.0331248e07],
                [4.0567504e08, 5.7574440e07, 3.0331604e07, 3.0331258e07, 3.0331248e07],
            ]
        ),
        "slope_area_1": np.array(
            [
                [
                    -8.0000000e03,
                    -8.0000000e03,
                    -8.0000000e03,
                    -8.0000000e03,
                    -8.0000000e03,
                ],
                [
                    -5.0000000e03,
                    -1.5355737e02,
                    -1.1516584e02,
                    -1.1516584e02,
                    -1.1516584e02,
                ],
                [
                    -5.0000000e02,
                    -1.3635868e01,
                    -3.0350534e-04,
                    0.0000000e00,
                    0.0000000e00,
                ],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00],
            ]
        ),
        "slope_area_2": np.array(
            [
                [
                    -5.81250000e03,
                    -1.85135559e02,
                    -7.67796326e01,
                    -1.42892147e-03,
                    0.00000000e00,
                ],
                [
                    -5.81250000e03,
                    -1.53558563e02,
                    -4.37480087e01,
                    -1.33653477e-04,
                    0.00000000e00,
                ],
                [
                    -3.37826025e03,
                    -1.15165504e02,
                    -1.46546517e-03,
                    -1.33653477e-04,
                    0.00000000e00,
                ],
                [
                    -3.10778931e03,
                    -1.15165482e02,
                    -1.22650794e-03,
                    -1.33653477e-04,
                    0.00000000e00,
                ],
                [
                    -3.10778931e03,
                    -1.15165482e02,
                    -1.22650794e-03,
                    -1.33653477e-04,
                    0.00000000e00,
                ],
            ]
        ),
    }

    _, _, Vu, slope, _, xf, _ = m.solve_problem_with_bellman_values(
        multi_stock_management=multi_stock_management,
        stock_discretization=StockDiscretization(
            {area.area: X[area.area] for area in m.range_reservoir}
        ),
        V=BellmanValueEstimation(V, StockDiscretization(X)),
        level_i={
            area.area: multi_stock_management.dict_reservoirs[
                area
            ].reservoir.initial_level
            for i, area in enumerate(m.range_reservoir)
        },
        take_into_account_z_and_y=True,
    )

    assert Vu == pytest.approx(37137635.6914607)

    assert slope == pytest.approx(
        {"area_1": -73.24037883117816, "area_2": -73.23892027017814}
    )

    assert xf == pytest.approx(
        {"area_1": 315840.9116434012, "area_2": 668042.0658865988}
    )


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

    vb, lb, ub = calculate_bellman_value_directly(
        param=param,
        multi_stock_management=all_reservoirs,
        output_path="test_data/two_nodes",
        X={"area_1": x_1, "area_2": x_2},
        univariate=False,
    )
    assert lb == pytest.approx(419088906.63159156)

    assert ub == pytest.approx(798837417.3288709)

    assert type(vb[0]) is BellmanValueEstimation
    assert vb[0]["intercept"] == pytest.approx(
        np.array(
            [
                [9.9394519e09, 5.4620790e09, 3.9168504e09, 3.8727749e09, 3.8727749e09],
                [7.1669325e09, 2.6933955e09, 1.1481676e09, 1.1040920e09, 1.1040919e09],
                [5.2626033e09, 1.2132852e09, 3.5723610e08, 3.0289034e08, 3.0289021e08],
                [3.4945702e09, 7.5961869e08, 3.2175792e08, 2.8026576e08, 2.8026566e08],
                [2.8697385e09, 7.2652160e08, 2.9893891e08, 2.8026563e08, 2.8026560e08],
            ]
        )
    )
    assert vb[0]["slope_area_1"] == pytest.approx(
        np.array(
            [
                [
                    -1.8733332e04,
                    -1.9000000e04,
                    -1.9000000e04,
                    -1.9000000e04,
                    -1.9000000e04,
                ],
                [
                    -1.0000000e04,
                    -9.9999990e03,
                    -9.9999990e03,
                    -9.9999990e03,
                    -9.9999990e03,
                ],
                [
                    -9.8999990e03,
                    -2.6754175e03,
                    -2.1557803e02,
                    -2.1557803e02,
                    -2.1557803e02,
                ],
                [
                    -8.9999971e03,
                    -2.0065604e03,
                    -1.3885049e02,
                    -1.4066651e-03,
                    -1.3115490e-03,
                ],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00],
            ]
        )
    )
    assert vb[0]["slope_area_2"] == pytest.approx(
        np.array(
            [
                [
                    -1.3333334e04,
                    -6.0000000e03,
                    -1.5355759e02,
                    -1.4289215e-03,
                    0.0000000e00,
                ],
                [
                    -1.2999999e04,
                    -6.0000000e03,
                    -1.5355759e02,
                    -1.4289215e-03,
                    0.0000000e00,
                ],
                [
                    -1.3200002e04,
                    -2.6754175e03,
                    -2.1557808e02,
                    -1.4289215e-03,
                    0.0000000e00,
                ],
                [
                    -1.2000000e04,
                    -2.6754175e03,
                    -1.8513564e02,
                    -1.2703377e-03,
                    0.0000000e00,
                ],
                [
                    -1.1000001e04,
                    -2.6754175e03,
                    -1.5355794e02,
                    -1.1774984e-03,
                    0.0000000e00,
                ],
            ]
        )
    )


def test_bellman_value_exact_calculation_univariate() -> None:

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

    try:

        vb, lb, ub = calculate_bellman_value_directly(
            param=param,
            multi_stock_management=all_reservoirs,
            output_path="test_data/two_nodes",
            X={"area_1": x_1, "area_2": x_2},
            univariate=True,
        )
        assert False
    except AssertionError:
        assert True
