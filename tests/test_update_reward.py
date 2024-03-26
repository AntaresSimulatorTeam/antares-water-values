from calculate_reward_and_bellman_values import RewardApproximation
from pytest import approx


def test_construct_initial_reward_approximation() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    assert initial_reward.breaking_point == [0, 10]
    assert initial_reward.list_cut == [(1, 0)]
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)
    assert initial_reward.breaking_point == [0, 5, 10]
    assert initial_reward.list_cut == [(1, 0), (-1, 10)]


def test_add_cut_above() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=0.4, intercept_new_cut=10)
    assert initial_reward.breaking_point == [0, 5, 10]
    assert initial_reward.list_cut == [(1, 0), (-1, 10)]


def test_add_cut_below() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=0.4, intercept_new_cut=-10)
    assert initial_reward.breaking_point == [0, 10]
    assert initial_reward.list_cut == [(0.4, -10)]


def test_add_cut_intercepting_both_cuts() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=0.5, intercept_new_cut=1)
    assert initial_reward.breaking_point == [0, 2, 6, 10]
    assert initial_reward.list_cut == [(1, 0), (0.5, 1), (-1, 10)]


def test_add_cut_removing_first_cut() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=0.5, intercept_new_cut=-2)
    assert initial_reward.breaking_point == [0, 8, 10]
    assert initial_reward.list_cut == [(0.5, -2), (-1, 10)]


def test_add_cut_intercepting_first_cut() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=2, intercept_new_cut=-2)
    assert initial_reward.breaking_point == [0, 2, 5, 10]
    assert initial_reward.list_cut == [(2, -2), (1, 0), (-1, 10)]


def test_add_cut_intercepting_second_cut() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=-2, intercept_new_cut=17)
    assert initial_reward.breaking_point == [0, 5, 7, 10]
    assert initial_reward.list_cut == [(1, 0), (-1, 10), (-2, 17)]


def test_add_cut_with_same_intercept_as_first_cut() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=2, intercept_new_cut=0)
    assert initial_reward.breaking_point == [0, 5, 10]
    assert initial_reward.list_cut == [(1, 0), (-1, 10)]

    initial_reward.update_reward_approximation(slope_new_cut=0.5, intercept_new_cut=0)
    assert initial_reward.breaking_point == approx([0, 20 / 3, 10])
    assert initial_reward.list_cut == [(0.5, 0), (-1, 10)]


def test_add_cut_intercepting_second_cut_at_domain_limit() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=-2, intercept_new_cut=20)
    assert initial_reward.breaking_point == [0, 5, 10]
    assert initial_reward.list_cut == [(1, 0), (-1, 10)]

    initial_reward.update_reward_approximation(slope_new_cut=-0.5, intercept_new_cut=3)
    assert initial_reward.breaking_point == approx([0, 2, 10])
    assert initial_reward.list_cut == [(1, 0), (-0.5, 3)]


def test_add_cut_identical_to_first_cut() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    assert initial_reward.breaking_point == [0, 5, 10]
    assert initial_reward.list_cut == [(1, 0), (-1, 10)]


def test_add_cut_identical_to_second_cut() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)
    assert initial_reward.breaking_point == [0, 5, 10]
    assert initial_reward.list_cut == [(1, 0), (-1, 10)]


def test_add_cut_intercepting_both_cuts_at_the_same_point() -> None:
    initial_reward = RewardApproximation(lb_control=0, ub_control=10, ub_reward=10)

    initial_reward.update_reward_approximation(slope_new_cut=1, intercept_new_cut=0)
    initial_reward.update_reward_approximation(slope_new_cut=-1, intercept_new_cut=10)

    initial_reward.update_reward_approximation(slope_new_cut=0, intercept_new_cut=5)
    assert initial_reward.breaking_point == [0, 5, 10]
    assert initial_reward.list_cut == [(1, 0), (-1, 10)]

    initial_reward.update_reward_approximation(slope_new_cut=2, intercept_new_cut=-5)
    assert initial_reward.breaking_point == approx([0, 5, 10])
    assert initial_reward.list_cut == [(2, -5), (-1, 10)]

    initial_reward.update_reward_approximation(slope_new_cut=-2, intercept_new_cut=15)
    assert initial_reward.breaking_point == approx([0, 5, 10])
    assert initial_reward.list_cut == [(2, -5), (-2, 15)]
