from functions_iterative import (
    TimeScenarioParameter,
    ReservoirManagement,
    TimeScenarioIndex,
    compute_upper_bound,
    BellmanValueCalculation,
    RewardApproximation,
)
from read_antares_data import Reservoir
from optimization import AntaresProblem, Basis
import pytest
import ortools.linear_solver.pywraplp as pywraplp
import numpy as np


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

        beta_1, _, itr_without_basis, basis, _ = problem.modify_weekly_problem_itr(
            control=8400000, i=0, prev_basis=Basis()
        )

        problem = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/one_node",
            itr=1,
            name_solver="XPRESS_LP",
        )
        problem.create_weekly_problem_itr(
            param=param, reservoir_management=reservoir_management
        )
        beta_2, _, itr_with_basis, _, _ = problem.modify_weekly_problem_itr(
            control=8400000, i=0, prev_basis=basis
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
            param=param, reservoir_management=reservoir_management
        )

        list_models = {TimeScenarioIndex(0, 0): problem}
        V = np.zeros((20, 2), dtype=np.float32)

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

        _, _, _, _, _ = problem.modify_weekly_problem_itr(
            control=0, i=0, prev_basis=Basis()
        )

        upper_bound_1, _, itr_without_basis = compute_upper_bound(
            bellman_value_calculation=bellman_value_calculation,
            list_models=list_models,
            V=V,
        )

        _, _, _, _, _ = problem.modify_weekly_problem_itr(
            control=8400000, i=0, prev_basis=Basis()
        )

        upper_bound_2, _, itr_with_basis = compute_upper_bound(
            bellman_value_calculation=bellman_value_calculation,
            list_models=list_models,
            V=V,
        )
        assert upper_bound_2 == pytest.approx(upper_bound_1)
        assert itr_with_basis[0, 0, 0] == 0
