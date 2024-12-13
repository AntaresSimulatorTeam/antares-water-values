import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
import pytest

from calculate_reward_and_bellman_values import MultiStockManagement
from estimation import PieceWiseLinearInterpolator
from functions_iterative import (
    BellmanValueCalculation,
    ReservoirManagement,
    RewardApproximation,
    TimeScenarioIndex,
    TimeScenarioParameter,
    compute_upper_bound,
)
from optimization import AntaresProblem, Basis
from read_antares_data import Reservoir


def test_basis_with_xpress() -> None:
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

        beta_1, _, _, _ = problem.solve_with_predefined_controls(
            control={"area": 8400000}, prev_basis=Basis([], [])
        )

        problem_2 = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/one_node",
            itr=1,
            name_solver="XPRESS_LP",
        )
        problem_2.create_weekly_problem_itr(
            param=param, multi_stock_management=reservoir_management
        )
        beta_2, _, itr_with_basis, _ = problem_2.solve_with_predefined_controls(
            control={"area": 8400000}, prev_basis=problem.basis[-1]
        )

        assert itr_with_basis == 0
        assert beta_1 == pytest.approx(beta_2)


def test_basis_with_upper_bound() -> None:
    solver = pywraplp.Solver.CreateSolver("XPRESS_LP")
    if solver:
        problem = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/one_node",
            itr=1,
            name_solver="XPRESS_LP",
        )
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
            param=param,
            multi_stock_management=MultiStockManagement([reservoir_management]),
        )

        list_models = {TimeScenarioIndex(0, 0): problem}
        X = np.linspace(0, reservoir.capacity, num=20)
        V = {
            week: PieceWiseLinearInterpolator(X, np.zeros(20, dtype=np.float32))
            for week in range(2)
        }

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
            stock_discretization=X,
        )

        _, _, _, _ = problem.solve_with_predefined_controls(
            control={"area": 0}, prev_basis=Basis([], [])
        )

        upper_bound_1, _, _ = compute_upper_bound(
            bellman_value_calculation=bellman_value_calculation,
            list_models=list_models,
            V=V,
        )

        _, _, _, _ = problem.solve_with_predefined_controls(
            control={"area": 8400000}, prev_basis=Basis([], [])
        )

        upper_bound_2, _, itr_with_basis = compute_upper_bound(
            bellman_value_calculation=bellman_value_calculation,
            list_models=list_models,
            V=V,
        )
        assert upper_bound_2 == pytest.approx(upper_bound_1)
        assert itr_with_basis[0, 0, 0] == 0
