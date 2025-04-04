import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
import pytest

from estimation import PieceWiseLinearInterpolator, UniVariateEstimator
from functions_iterative import (
    TimeScenarioIndex,
    TimeScenarioParameter,
    compute_upper_bound,
)
from optimization import AntaresProblem, Basis
from reservoir_management import MultiStockManagement
from stock_discretization import StockDiscretization
from type_definition import AreaIndex, Array1D, Dict, WeekIndex


def test_basis_with_xpress(
    param_one_week: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
) -> None:
    solver = pywraplp.Solver.CreateSolver("XPRESS_LP")
    if solver:

        problem = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/one_node",
            itr=1,
            name_solver="XPRESS_LP",
        )

        problem.create_weekly_problem_itr(
            param=param_one_week, multi_stock_management=multi_stock_management_one_node
        )

        beta_1, _, _, _ = problem.solve_with_predefined_controls(
            control={AreaIndex("area"): 8400000}, prev_basis=Basis([], [])
        )

        problem_2 = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/one_node",
            itr=1,
            name_solver="XPRESS_LP",
        )
        problem_2.create_weekly_problem_itr(
            param=param_one_week, multi_stock_management=multi_stock_management_one_node
        )
        beta_2, _, itr_with_basis, _ = problem_2.solve_with_predefined_controls(
            control={AreaIndex("area"): 8400000}, prev_basis=problem.basis[-1]
        )

        assert itr_with_basis == 0
        assert beta_1 == pytest.approx(beta_2)


def test_basis_with_upper_bound(
    param_one_week: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
    discretization_one_node: Dict[AreaIndex, Array1D],
) -> None:
    solver = pywraplp.Solver.CreateSolver("XPRESS_LP")
    if solver:
        problem = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/one_node",
            itr=1,
            name_solver="XPRESS_LP",
        )

        problem.create_weekly_problem_itr(
            param=param_one_week,
            multi_stock_management=multi_stock_management_one_node,
        )

        list_models = {TimeScenarioIndex(0, 0): problem}

        V = {
            area.area: PieceWiseLinearInterpolator(
                discretization_one_node[area], np.zeros(20, dtype=np.float32)
            )
            for area in multi_stock_management_one_node.areas
        }

        _, _, _, _ = problem.solve_with_predefined_controls(
            control={AreaIndex("area"): 0}, prev_basis=Basis([], [])
        )

        upper_bound_1, _, _, _ = compute_upper_bound(
            param=param_one_week,
            multi_stock_management=multi_stock_management_one_node,
            stock_discretization=StockDiscretization(discretization_one_node),
            list_models=list_models,
            V={
                WeekIndex(week): UniVariateEstimator(V)
                for week in range(param_one_week.len_week + 1)
            },
        )

        _, _, _, _ = problem.solve_with_predefined_controls(
            control={AreaIndex("area"): 8400000}, prev_basis=Basis([], [])
        )

        upper_bound_2, _, itr_with_basis, _ = compute_upper_bound(
            param=param_one_week,
            multi_stock_management=multi_stock_management_one_node,
            stock_discretization=StockDiscretization(discretization_one_node),
            list_models=list_models,
            V={
                WeekIndex(week): UniVariateEstimator(V)
                for week in range(param_one_week.len_week + 1)
            },
        )
        assert upper_bound_2 == pytest.approx(upper_bound_1)
        assert itr_with_basis[TimeScenarioIndex(0, 0)] == 0
