import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
import pytest

from estimation import PieceWiseLinearInterpolator, UniVariateEstimator
from functions_iterative import TimeScenarioParameter
from optimization import AntaresProblem, Basis
from reservoir_management import MultiStockManagement
from stock_discretization import StockDiscretization
from type_definition import AreaIndex, Array1D, Dict


def test_create_and_modify_weekly_problem(
    param: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
) -> None:
    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)

    problem.create_weekly_problem_itr(
        param=param, multi_stock_management=multi_stock_management_one_node
    )

    beta, lamb, _, _ = problem.solve_with_predefined_controls(
        control={AreaIndex("area"): 0}, prev_basis=Basis([], [])
    )
    assert beta == pytest.approx(943484691.8759749)
    assert lamb[AreaIndex("area")] == pytest.approx(-200.08020911704824)

    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
    problem.create_weekly_problem_itr(
        param=param, multi_stock_management=multi_stock_management_one_node
    )
    beta, lamb, _, _ = problem.solve_with_predefined_controls(
        control={AreaIndex("area"): 8400000}, prev_basis=Basis([], [])
    )
    assert beta == pytest.approx(38709056.48535345)
    assert lamb[AreaIndex("area")] == pytest.approx(0.0004060626000000001)

    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
    problem.create_weekly_problem_itr(
        param=param, multi_stock_management=multi_stock_management_one_node
    )
    beta, lamb, _, _ = problem.solve_with_predefined_controls(
        control={AreaIndex("area"): -8400000}, prev_basis=Basis([], [])
    )
    assert beta == pytest.approx(20073124196.898315)
    assert lamb[AreaIndex("area")] == pytest.approx(-3000.0013996873)


def test_create_and_modify_weekly_problem_with_xpress(
    param: TimeScenarioParameter,
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
            param=param, multi_stock_management=multi_stock_management_one_node
        )

        beta, lamb, _, _ = problem.solve_with_predefined_controls(
            control={AreaIndex("area"): 0}, prev_basis=Basis([], [])
        )
        assert beta == pytest.approx(943484691.8759749)
        assert lamb[AreaIndex("area")] == pytest.approx(-200.08020911704824)

        problem = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/one_node",
            itr=1,
            name_solver="XPRESS_LP",
        )
        problem.create_weekly_problem_itr(
            param=param, multi_stock_management=multi_stock_management_one_node
        )
        beta, lamb, _, _ = problem.solve_with_predefined_controls(
            control={AreaIndex("area"): 8400000}, prev_basis=Basis([], [])
        )
        assert beta == pytest.approx(38709056.48535345)
        assert lamb[AreaIndex("area")] == pytest.approx(0.0004060626000000001)

        problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
        problem.create_weekly_problem_itr(
            param=param, multi_stock_management=multi_stock_management_one_node
        )
        beta, lamb, _, _ = problem.solve_with_predefined_controls(
            control={AreaIndex("area"): -8400000}, prev_basis=Basis([], [])
        )
        assert beta == pytest.approx(20073124196.898315)
        assert lamb[AreaIndex("area")] == pytest.approx(-3000.0013996873)
    else:
        print("Ignore test, xpress not available")


def test_create_and_modify_weekly_problem_with_bellman_values(
    param: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
    discretization_one_node: Dict[AreaIndex, Array1D],
) -> None:
    multi_stock_management_one_node.dict_reservoirs[
        AreaIndex("area")
    ].penalty_bottom_rule_curve = 1000
    multi_stock_management_one_node.dict_reservoirs[
        AreaIndex("area")
    ].penalty_upper_rule_curve = 1000
    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)

    problem.create_weekly_problem_itr(
        param=param, multi_stock_management=multi_stock_management_one_node
    )
    V = {
        week: PieceWiseLinearInterpolator(
            discretization_one_node[area], np.linspace(-5e9, -3e9, num=20)
        )
        for week in range(2)
        for area in multi_stock_management_one_node.areas
    }

    problem.set_constraints_initial_level_and_bellman_values(
        UniVariateEstimator({"area": V[1]}),
        multi_stock_management_one_node.get_initial_level(),
        StockDiscretization(discretization_one_node),
    )

    lp = problem.solver.ExportModelAsLpFormat(False)

    with open("test_data/one_node/lp_problem.txt", "r") as file:
        assert lp == file.read()

    _, _, cout, _, optimal_controls, _, _ = problem.solve_problem_with_bellman_values(
        multi_stock_management_one_node,
        StockDiscretization(discretization_one_node),
        UniVariateEstimator({"area": V[1]}),
        multi_stock_management_one_node.get_initial_level(),
        True,
        False,
    )

    assert cout == pytest.approx(5046990806.783945)
    assert optimal_controls[AreaIndex("area")] == pytest.approx(1146984.0000000005)
