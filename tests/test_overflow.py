from functions_iterative import (
    TimeScenarioParameter,
    ReservoirManagement,
)
from simple_bellman_value_calculation import (
    calculate_bellman_value_with_precalculated_reward,
)
from read_antares_data import Reservoir
import numpy as np
from scipy.interpolate import interp1d


def test_bellman_value_precalculated_reward_overflow() -> None:

    param = TimeScenarioParameter(len_week=5, len_scenario=1)
    reservoir = Reservoir("test_data/one_node", "area")
    reservoir.initial_level = reservoir.capacity
    reservoir_management = ReservoirManagement(
        reservoir=reservoir,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=False,
        overflow=True,
    )
    xNsteps = 20
    X = np.linspace(0, reservoir.capacity, num=xNsteps)

    vb, _ = calculate_bellman_value_with_precalculated_reward(
        len_controls=20,
        param=param,
        reservoir_management=reservoir_management,
        output_path="test_data/one_node",
        X=X,
    )

    V_fut = interp1d(X, vb[:, 0])
    V0 = V_fut(reservoir_management.reservoir.initial_level)

    assert float(V0) == -3546553410.818109

    reservoir_management = ReservoirManagement(
        reservoir=reservoir,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=False,
        overflow=False,
    )

    vb, _ = calculate_bellman_value_with_precalculated_reward(
        len_controls=20,
        param=param,
        reservoir_management=reservoir_management,
        output_path="test_data/one_node",
        X=X,
    )

    V_fut = interp1d(X, vb[:, 0])
    V0 = V_fut(reservoir_management.reservoir.initial_level)

    assert V0 == -3546553410.818109
