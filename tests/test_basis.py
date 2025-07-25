import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
import pytest

from estimation import LinearCostEstimator, PieceWiseLinearInterpolator
from functions_iterative import (
    TimeScenarioIndex,
    TimeScenarioParameter,
    compute_upper_bound,
)
from optimization import AntaresProblem, Basis
from reservoir_management import MultiStockManagement
from type_definition import AreaIndex, Array1D, Dict, List, WeekIndex


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
    param: TimeScenarioParameter,
    controls_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    costs_precalculated_one_node_10: Dict[TimeScenarioIndex, List[float]],
    slopes_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
) -> None:
    solver = pywraplp.Solver.CreateSolver("XPRESS_LP")
    if solver:
        list_models = {TimeScenarioIndex(0, 0): antares_problem_one_node_xpress}

        reward = LinearCostEstimator(
            controls=controls_precalculated_one_node_10,
            param=param,
            costs=costs_precalculated_one_node_10,
            duals=slopes_precalculated_one_node_10,
            type_estimator="LinearInterpolator",
        )

        V = PieceWiseLinearInterpolator(
            discretization_one_node[AreaIndex("area")], np.zeros(20, dtype=np.float32)
        )

        _, _, _, _ = antares_problem_one_node_xpress.solve_with_predefined_controls(
            control={AreaIndex("area"): 0}, prev_basis=Basis([], [])
        )

        upper_bound_1, _, _, _ = compute_upper_bound(
            param=param_one_week,
            multi_stock_management=multi_stock_management_one_node,
            list_models=list_models,
            V={WeekIndex(week): V for week in range(param_one_week.len_week + 1)},
        )

        _, _, _, _ = antares_problem_one_node_xpress.solve_with_predefined_controls(
            control={AreaIndex("area"): 8400000}, prev_basis=Basis([], [])
        )

        upper_bound_2, _, itr_with_basis, _ = compute_upper_bound(
            param=param_one_week,
            multi_stock_management=multi_stock_management_one_node,
            list_models=list_models,
            V={WeekIndex(week): V for week in range(param_one_week.len_week + 1)},
            reward_approximation=reward,
        )
        assert upper_bound_2 == pytest.approx(upper_bound_1)
        assert itr_with_basis[TimeScenarioIndex(0, 0)] == 0
