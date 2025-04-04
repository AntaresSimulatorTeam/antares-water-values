import numpy as np
import pytest

from functions_iterative import ReservoirManagement, TimeScenarioParameter
from multi_stock_bellman_value_calculation import generate_fast_uvs_v2
from read_antares_data import Reservoir
from reservoir_management import MultiStockManagement
from type_definition import AreaIndex, WeekIndex, area_list_value_to_array


def test_fast_usage_values_multi_stock(param: TimeScenarioParameter) -> None:

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

    mrg_prices = {
        AreaIndex("area_1"): dict(mean=42.77, std=31.80),
        AreaIndex("area_2"): dict(mean=41.71, std=3.53),
    }

    uvs = generate_fast_uvs_v2(
        param=param, multi_stock_management=multi_management, mrg_prices=mrg_prices
    )

    assert np.transpose(
        area_list_value_to_array({a: x[WeekIndex(0)] for a, x in uvs.items()})
    ) == pytest.approx(
        np.array(
            [
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [74.57, 45.24],
                [72.93923077, 45.05897436],
                [71.30846154, 44.87794872],
                [69.67769231, 44.69692308],
                [68.04692308, 44.51589744],
                [66.41615385, 44.33487179],
                [64.78538462, 44.15384615],
                [63.15461538, 43.97282051],
                [61.52384615, 43.79179487],
                [59.89307692, 43.61076923],
                [58.26230769, 43.42974359],
                [56.63153846, 43.24871795],
                [55.00076923, 43.06769231],
                [53.37, 42.88666667],
                [51.73923077, 42.70564103],
                [50.10846154, 42.52461538],
                [48.47769231, 42.34358974],
                [46.84692308, 42.1625641],
                [45.21615385, 41.98153846],
                [43.58538462, 41.80051282],
                [41.95461538, 41.61948718],
                [40.32384615, 41.43846154],
                [38.69307692, 41.2574359],
                [37.06230769, 41.07641026],
                [35.43153846, 40.89538462],
                [33.80076923, 40.71435897],
                [32.17, 40.53333333],
                [30.53923077, 40.35230769],
                [28.90846154, 40.17128205],
                [27.27769231, 39.99025641],
                [25.64692308, 39.80923077],
                [24.01615385, 39.62820513],
                [22.38538462, 39.44717949],
                [20.75461538, 39.26615385],
                [19.12384615, 39.08512821],
                [17.49307692, 38.90410256],
                [15.86230769, 38.72307692],
                [14.23153846, 38.54205128],
                [12.60076923, 38.36102564],
                [10.97, 38.18],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
    )
