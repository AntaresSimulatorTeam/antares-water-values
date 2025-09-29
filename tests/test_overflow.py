import numpy as np
import pytest

from functions_iterative import ReservoirManagement, TimeScenarioParameter
from multi_stock_bellman_value_calculation import MultiStockManagement
from read_antares_data import Reservoir
from simple_bellman_value_calculation import (
    calculate_bellman_value_with_precalculated_cost,
)
from type_definition import AreaIndex, WeekIndex


def test_bellman_value_precalculated_reward_overflow(
    param: TimeScenarioParameter,
    reservoir_one_node: Reservoir,
) -> None:
    reservoir_one_node.initial_level = reservoir_one_node.capacity
    reservoir_management = ReservoirManagement(
        reservoir=reservoir_one_node,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=False,
        overflow=True,
    )
    a = AreaIndex("area")
    levels = {
        WeekIndex(w): [
            {a: x}
            for x in np.linspace(
                0,
                reservoir_management.reservoir.capacity,
                20,
            )
        ]
        for w in range(param.len_week + 1)
    }

    vb, _, V0, _ = calculate_bellman_value_with_precalculated_cost(
        len_controls=20,
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        output_path="test_data/one_node",
        levels=levels,
        piecewiselinear=True,
        type_estimator="LinearInterpolator",
    )

    assert float(V0) == pytest.approx(-3546553410.818109, rel=1e-4)

    reservoir_management = ReservoirManagement(
        reservoir=reservoir_one_node,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=False,
        overflow=False,
    )

    vb, _, V0, _ = calculate_bellman_value_with_precalculated_cost(
        len_controls=20,
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        output_path="test_data/one_node",
        levels=levels,
        piecewiselinear=True,
        type_estimator="LinearInterpolator",
    )

    assert V0 == pytest.approx(-3546553410.818109, rel=1e-4)
