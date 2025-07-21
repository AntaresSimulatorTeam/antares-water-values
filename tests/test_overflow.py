import numpy as np
import pytest
from scipy.interpolate import interp1d

from functions_iterative import ReservoirManagement, TimeScenarioParameter
from multi_stock_bellman_value_calculation import MultiStockManagement
from read_antares_data import Reservoir
from simple_bellman_value_calculation import (
    calculate_bellman_value_with_precalculated_reward,
)


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
    xNsteps = 20
    X = np.linspace(0, reservoir_one_node.capacity, num=xNsteps)

    vb, _ = calculate_bellman_value_with_precalculated_reward(
        len_controls=20,
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        output_path="test_data/one_node",
        len_bellman=len(X),
    )

    V_fut = interp1d(X, vb[:, 0])
    V0 = V_fut(reservoir_one_node.initial_level)

    assert float(V0) == pytest.approx(-3546553410.818109)

    reservoir_management = ReservoirManagement(
        reservoir=reservoir_one_node,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=False,
        overflow=False,
    )

    vb, _ = calculate_bellman_value_with_precalculated_reward(
        len_controls=20,
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        output_path="test_data/one_node",
        len_bellman=len(X),
    )

    V_fut = interp1d(X, vb[:, 0])
    V0 = V_fut(reservoir_one_node.initial_level)

    assert V0 == pytest.approx(-3546553410.818109)
