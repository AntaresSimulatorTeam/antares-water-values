import ortools.linear_solver.pywraplp as pywraplp
import pytest

from functions_iterative import ReservoirManagement, TimeScenarioParameter
from optimization import AntaresProblem, Basis
from read_antares_data import Reservoir
from reservoir_management import MultiStockManagement
from type_definition import AreaIndex


def test_create_weekly_problem_with_two_stocks(
    param: TimeScenarioParameter,
    multi_stock_management_two_nodes: MultiStockManagement,
) -> None:
    problem = AntaresProblem(scenario=0, week=0, path="test_data/two_nodes", itr=1)

    problem.create_weekly_problem_itr(
        param=param, multi_stock_management=multi_stock_management_two_nodes
    )

    beta, lamb, _, _ = problem.solve_with_predefined_controls(
        control={AreaIndex("area_1"): 0, AreaIndex("area_2"): 0},
        prev_basis=Basis([], []),
    )
    assert beta == pytest.approx(26357713.948390935)
    assert lamb[AreaIndex("area_1")] == pytest.approx(-138.85274530229998)
    assert lamb[AreaIndex("area_2")] == pytest.approx(-138.85108188019998)


def test_create_weekly_problem_with_two_stocks_with_xpress(
    param: TimeScenarioParameter,
    multi_stock_management_two_nodes: MultiStockManagement,
) -> None:

    solver = pywraplp.Solver.CreateSolver("XPRESS_LP")
    if solver:

        problem = AntaresProblem(
            scenario=0,
            week=0,
            path="test_data/two_nodes",
            itr=1,
            name_solver="XPRESS_LP",
        )

        problem.create_weekly_problem_itr(
            param=param, multi_stock_management=multi_stock_management_two_nodes
        )

        beta, lamb, _, _ = problem.solve_with_predefined_controls(
            control={AreaIndex("area_1"): 0, AreaIndex("area_2"): 0},
            prev_basis=Basis([], []),
        )
        assert beta == pytest.approx(26357713.948390935)
        assert lamb[AreaIndex("area_1")] == pytest.approx(-138.85274530229998)
        assert lamb[AreaIndex("area_2")] == pytest.approx(-138.85108188019998)
    else:
        print("Ignore test, xpress not available")
