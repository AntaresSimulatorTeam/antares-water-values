from functions_iterative import (
    AntaresParameter,
    Reservoir,
    compute_upper_bound,
    RewardApproximation,
    TimeScenarioIndex,
    ReservoirManagement,
)
from optimization import AntaresProblem
import pytest
import numpy as np


def test_upper_bound() -> None:
    problem = AntaresProblem(year=0, week=0, path="test_data/one_node", itr=1)
    param = AntaresParameter(S=1, NTrain=1)
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

    upper_bound, controls, _ = compute_upper_bound(
        param=param,
        reservoir_management=reservoir_management,
        list_models={TimeScenarioIndex(0, 0): problem},
        X=np.linspace(0, reservoir.capacity, num=20),
        V=np.zeros((20, 2), dtype=np.float32),
        G=[
            [
                RewardApproximation(
                    lb_control=-reservoir.max_pumping[0],
                    ub_control=reservoir.max_generating[0],
                    ub_reward=0,
                )
            ]
        ],
    )

    assert upper_bound == pytest.approx(380492940.000565)
    assert controls[0, 0] == pytest.approx(4482011.0)
