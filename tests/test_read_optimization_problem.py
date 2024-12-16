import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
import pytest

from calculate_reward_and_bellman_values import (
    BellmanValueCalculation,
    MultiStockBellmanValueCalculation,
    MultiStockManagement,
    RewardApproximation,
)
from estimation import PieceWiseLinearInterpolator, UniVariateEstimator
from functions_iterative import (
    ReservoirManagement,
    TimeScenarioIndex,
    TimeScenarioParameter,
)
from optimization import AntaresProblem, Basis
from read_antares_data import Reservoir


def test_create_and_modify_weekly_problem() -> None:
    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
    param = TimeScenarioParameter(len_week=52, len_scenario=1)
    reservoir = Reservoir("test_data/one_node", "area")
    reservoir_management = MultiStockManagement(
        [
            ReservoirManagement(
                reservoir=reservoir,
                penalty_bottom_rule_curve=0,
                penalty_upper_rule_curve=0,
                penalty_final_level=0,
                force_final_level=True,
            )
        ]
    )
    problem.create_weekly_problem_itr(
        param=param, multi_stock_management=reservoir_management
    )

    beta, lamb, _, _ = problem.solve_with_predefined_controls(
        control={"area": 0}, prev_basis=Basis([], [])
    )
    assert beta == pytest.approx(943484691.8759749)
    assert lamb["area"] == pytest.approx(-200.08020911704824)

    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
    problem.create_weekly_problem_itr(
        param=param, multi_stock_management=reservoir_management
    )
    beta, lamb, _, _ = problem.solve_with_predefined_controls(
        control={"area": 8400000}, prev_basis=Basis([], [])
    )
    assert beta == pytest.approx(38709056.48535345)
    assert lamb["area"] == pytest.approx(0.0004060626000000001)

    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
    problem.create_weekly_problem_itr(
        param=param, multi_stock_management=reservoir_management
    )
    beta, lamb, _, _ = problem.solve_with_predefined_controls(
        control={"area": -8400000}, prev_basis=Basis([], [])
    )
    assert beta == pytest.approx(20073124196.898315)
    assert lamb["area"] == pytest.approx(-3000.0013996873)


def test_create_and_modify_weekly_problem_with_xpress() -> None:

    solver = pywraplp.Solver.CreateSolver("XPRESS_LP")
    if solver:

        problem = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/one_node",
            itr=1,
            name_solver="XPRESS_LP",
        )
        param = TimeScenarioParameter(len_week=52, len_scenario=1)
        reservoir = Reservoir("test_data/one_node", "area")
        reservoir_management = MultiStockManagement(
            [
                ReservoirManagement(
                    reservoir=reservoir,
                    penalty_bottom_rule_curve=0,
                    penalty_upper_rule_curve=0,
                    penalty_final_level=0,
                    force_final_level=True,
                )
            ]
        )
        problem.create_weekly_problem_itr(
            param=param, multi_stock_management=reservoir_management
        )

        beta, lamb, _, _ = problem.solve_with_predefined_controls(
            control={"area": 0}, prev_basis=Basis([], [])
        )
        assert beta == pytest.approx(943484691.8759749)
        assert lamb["area"] == pytest.approx(-200.08020911704824)

        problem = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/one_node",
            itr=1,
            name_solver="XPRESS_LP",
        )
        problem.create_weekly_problem_itr(
            param=param, multi_stock_management=reservoir_management
        )
        beta, lamb, _, _ = problem.solve_with_predefined_controls(
            control={"area": 8400000}, prev_basis=Basis([], [])
        )
        assert beta == pytest.approx(38709056.48535345)
        assert lamb["area"] == pytest.approx(0.0004060626000000001)

        problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
        problem.create_weekly_problem_itr(
            param=param, multi_stock_management=reservoir_management
        )
        beta, lamb, _, _ = problem.solve_with_predefined_controls(
            control={"area": -8400000}, prev_basis=Basis([], [])
        )
        assert beta == pytest.approx(20073124196.898315)
        assert lamb["area"] == pytest.approx(-3000.0013996873)
    else:
        print("Ignore test, xpress not available")


def test_create_and_modify_weekly_problem_with_bellman_values() -> None:
    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
    param = TimeScenarioParameter(len_week=1, len_scenario=1)
    reservoir = Reservoir("test_data/one_node", "area")
    res_management = ReservoirManagement(
        reservoir=reservoir,
        penalty_bottom_rule_curve=1000,
        penalty_upper_rule_curve=1000,
        penalty_final_level=0,
        force_final_level=False,
    )

    reservoir_management = MultiStockManagement([res_management])
    problem.create_weekly_problem_itr(
        param=param, multi_stock_management=reservoir_management
    )

    X = np.linspace(0, reservoir.capacity, num=20)

    bellman_value_calculation = MultiStockBellmanValueCalculation(
        [
            BellmanValueCalculation(
                param=param,
                reward={
                    TimeScenarioIndex(0, 0): RewardApproximation(
                        lb_control=-reservoir.max_pumping[0],
                        ub_control=reservoir.max_generating[0],
                        ub_reward=0,
                    )
                },
                reservoir_management=res_management,
                stock_discretization=X,
            )
        ]
    )

    V = {
        week: PieceWiseLinearInterpolator(X, np.linspace(-5e9, -3e9, num=20))
        for week in range(2)
    }

    problem.set_constraints_initial_level_and_bellman_values(
        UniVariateEstimator({"area": V[1]}),
        {"area": reservoir.initial_level},
        bellman_value_calculation,
    )

    lp = problem.solver.ExportModelAsLpFormat(False)

    with open("test_data/one_node/lp_problem.txt", "r") as file:
        assert lp == file.read()

    _, _, cout, _, optimal_controls, _, _ = problem.solve_problem_with_bellman_values(
        bellman_value_calculation,
        UniVariateEstimator({"area": V[1]}),
        {"area": reservoir.initial_level},
        True,
        False,
    )

    assert cout == pytest.approx(5046990806.783945)
    assert optimal_controls["area"] == pytest.approx(1146984.0000000005)
