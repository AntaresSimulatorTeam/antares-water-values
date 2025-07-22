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
from type_definition import AreaIndex, Array1D, Dict, WeekIndex


def test_basis_with_xpress(
    param_one_week: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
    antares_problem_one_node_xpress: AntaresProblem,
) -> None:
    solver = pywraplp.Solver.CreateSolver("XPRESS_LP")
    if solver:

        beta_1, _, _, _ = (
            antares_problem_one_node_xpress.solve_with_predefined_controls(
                control={AreaIndex("area"): 8400000}, prev_basis=Basis([], [])
            )
        )

        problem_2 = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/one_node",
            name_solver="XPRESS_LP",
            param=param_one_week,
            multi_stock_management=multi_stock_management_one_node,
        )
        beta_2, _, itr_with_basis, _ = problem_2.solve_with_predefined_controls(
            control={AreaIndex("area"): 8400000},
            prev_basis=antares_problem_one_node_xpress.basis[-1],
        )

        assert itr_with_basis == 0
        assert beta_1 == pytest.approx(beta_2)


def test_basis_with_upper_bound(
    param_one_week: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
    discretization_one_node: Dict[AreaIndex, Array1D],
    antares_problem_one_node_xpress: AntaresProblem,
) -> None:
    solver = pywraplp.Solver.CreateSolver("XPRESS_LP")
    if solver:
        list_models = {TimeScenarioIndex(0, 0): antares_problem_one_node_xpress}

        V = {
            area.area: PieceWiseLinearInterpolator(
                discretization_one_node[area], np.zeros(20, dtype=np.float32)
            )
            for area in multi_stock_management_one_node.areas
        }

        _, _, _, _ = antares_problem_one_node_xpress.solve_with_predefined_controls(
            control={AreaIndex("area"): 0}, prev_basis=Basis([], [])
        )

        upper_bound_1, _, _, _ = compute_upper_bound(
            param=param_one_week,
            multi_stock_management=multi_stock_management_one_node,
            list_models=list_models,
            V={
                WeekIndex(week): UniVariateEstimator(V)
                for week in range(param_one_week.len_week + 1)
            },
        )

        _, _, _, _ = antares_problem_one_node_xpress.solve_with_predefined_controls(
            control={AreaIndex("area"): 8400000}, prev_basis=Basis([], [])
        )

        upper_bound_2, _, itr_with_basis, _ = compute_upper_bound(
            param=param_one_week,
            multi_stock_management=multi_stock_management_one_node,
            list_models=list_models,
            V={
                WeekIndex(week): UniVariateEstimator(V)
                for week in range(param_one_week.len_week + 1)
            },
        )
        assert upper_bound_2 == pytest.approx(upper_bound_1)
        assert itr_with_basis[TimeScenarioIndex(0, 0)] == 0
