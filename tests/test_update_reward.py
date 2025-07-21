import numpy as np
from pytest import approx

from calculate_reward_and_bellman_values import LinearInterpolator


def test_init_reward() -> None:
    initial_reward = LinearInterpolator(
        duals=np.array([0]), controls=np.array([10]), costs=np.array([10])
    )

    assert initial_reward(np.array([0])) == 10


def test_update_reward() -> None:
    initial_reward = LinearInterpolator(
        duals=np.array([0]), controls=np.array([10]), costs=np.array([10])
    )

    initial_reward.update(
        controls=np.array([0]), duals=np.array([-5]), costs=np.array([30])
    )
    assert initial_reward(np.array([0])) == 30
    assert initial_reward(np.array([4])) == 10
    assert initial_reward(np.array([10])) == 10
