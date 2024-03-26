from functions_iterative import (
    TimeScenarioParameter,
    compute_upper_bound,
    RewardApproximation,
    TimeScenarioIndex,
    ReservoirManagement,
    BellmanValueCalculation,
)
from optimization import AntaresProblem
from read_antares_data import Reservoir
import pytest
import numpy as np


def test_upper_bound() -> None:
    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
    param = TimeScenarioParameter(len_week=1, len_scenario=1)
    reservoir = Reservoir("test_data/one_node", "area")
    reservoir_management = ReservoirManagement(
        reservoir=reservoir,
        penalty_bottom_rule_curve=0,
        penalty_upper_rule_curve=0,
        penalty_final_level=0,
        force_final_level=True,
    )
    problem.create_weekly_problem_itr(
        param=param, reservoir_management=reservoir_management
    )
    bellman_value_calculation = BellmanValueCalculation(
        param=param,
        reward={
            TimeScenarioIndex(0, 0): RewardApproximation(
                lb_control=-reservoir.max_pumping[0],
                ub_control=reservoir.max_generating[0],
                ub_reward=0,
            )
        },
        reservoir_management=reservoir_management,
        stock_discretization=np.linspace(0, reservoir.capacity, num=20),
    )

    upper_bound, controls, _ = compute_upper_bound(
        bellman_value_calculation=bellman_value_calculation,
        list_models={TimeScenarioIndex(0, 0): problem},
        V=np.zeros((20, 2), dtype=np.float32),
    )

    assert upper_bound == pytest.approx(380492940.000565)
    assert controls[0, 0] == pytest.approx(4482011.0)
