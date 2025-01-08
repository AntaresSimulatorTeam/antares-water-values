import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
import pytest

from estimation import PieceWiseLinearInterpolator, UniVariateEstimator
from functions_iterative import (
    ReservoirManagement,
    TimeScenarioIndex,
    TimeScenarioParameter,
    compute_upper_bound,
)
from optimization import AntaresProblem
from read_antares_data import Reservoir
from reservoir_management import MultiStockManagement
from stock_discretization import StockDiscretization
from type_definition import AreaIndex

bellman_values = np.array(
    [
        [
            -5.44276365e09,
            -4.36329798e09,
            -4.30354500e09,
            -2.41781989e09,
            -1.41870571e09,
            0.00000000e00,
        ],
        [
            -5.28486774e09,
            -4.25799257e09,
            -3.38627742e09,
            -2.31251448e09,
            -1.31340029e09,
            0.00000000e00,
        ],
        [
            -5.12697182e09,
            -4.15268716e09,
            -3.22838136e09,
            -2.20720907e09,
            -1.20809488e09,
            0.00000000e00,
        ],
        [
            -4.98446159e09,
            -4.04738174e09,
            -3.07425277e09,
            -2.10190367e09,
            -1.10278947e09,
            0.00000000e00,
        ],
        [
            -4.87915618e09,
            -3.94207633e09,
            -2.96894737e09,
            -1.99659826e09,
            -9.97484054e08,
            0.00000000e00,
        ],
        [
            -4.77385077e09,
            -3.83677092e09,
            -2.86364196e09,
            -1.89129286e09,
            -8.92178642e08,
            0.00000000e00,
        ],
        [
            -4.66854538e09,
            -3.73146551e09,
            -2.75833659e09,
            -1.78598745e09,
            -7.86873443e08,
            0.00000000e00,
        ],
        [
            -4.56324001e09,
            -3.62616010e09,
            -2.65303124e09,
            -1.68068205e09,
            -6.97788201e08,
            0.00000000e00,
        ],
        [
            -4.45793464e09,
            -3.52085471e09,
            -2.54772590e09,
            -1.57537670e09,
            -6.45156592e08,
            0.00000000e00,
        ],
        [
            -4.35262927e09,
            -3.41554936e09,
            -2.44242055e09,
            -1.47481530e09,
            -5.92524983e08,
            0.00000000e00,
        ],
        [
            -4.24732390e09,
            -3.31024401e09,
            -2.33711521e09,
            -1.39639147e09,
            -5.39893375e08,
            0.00000000e00,
        ],
        [
            -4.14201854e09,
            -3.20493866e09,
            -2.23180986e09,
            -1.34375968e09,
            -4.87261766e08,
            0.00000000e00,
        ],
        [
            -4.03671319e09,
            -3.09963332e09,
            -2.12650452e09,
            -1.29112789e09,
            -4.34630157e08,
            0.00000000e00,
        ],
        [
            -3.93140785e09,
            -2.99432797e09,
            -2.02119917e09,
            -1.23849610e09,
            -3.81998548e08,
            0.00000000e00,
        ],
        [
            -3.82610250e09,
            -2.88902263e09,
            -1.91589383e09,
            -1.18586431e09,
            -3.29366940e08,
            0.00000000e00,
        ],
        [
            -3.75680250e09,
            -2.80469325e09,
            -1.81058848e09,
            -1.13323252e09,
            -2.76735331e08,
            0.00000000e00,
        ],
        [
            -3.70417075e09,
            -2.75206123e09,
            -1.70528314e09,
            -1.08060073e09,
            -2.24103722e08,
            0.00000000e00,
        ],
        [
            -3.65153900e09,
            -2.69942962e09,
            -1.59997779e09,
            -1.02796894e09,
            -1.71472114e08,
            0.00000000e00,
        ],
        [
            -3.59890725e09,
            -2.64679800e09,
            -1.49830812e09,
            -9.75337146e08,
            -1.18840505e08,
            0.00000000e00,
        ],
        [
            -3.54627549e09,
            -2.59416639e09,
            -1.41471259e09,
            -9.22705355e08,
            -6.62088960e07,
            0.00000000e00,
        ],
    ]
)


def test_upper_bound() -> None:
    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
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
    assert len(problem.solver.constraints()) == 3535
    assert len(problem.solver.variables()) == 3533

    upper_bound, controls, _ = compute_upper_bound(
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        stock_discretization=StockDiscretization({AreaIndex(reservoir.area): X}),
        list_models=list_models,
        V={
            week: UniVariateEstimator({reservoir.area: V[week]})
            for week in range(param.len_week + 1)
        },
    )

    assert upper_bound == pytest.approx(380492940.000565)
    assert controls[TimeScenarioIndex(0, 0)][reservoir.area] == pytest.approx(4482011.0)
    assert len(problem.solver.constraints()) == 3555
    assert len(problem.solver.variables()) == 3533

    V[1].costs = np.linspace(-5e9, -3e9, num=20)
    upper_bound, controls, _ = compute_upper_bound(
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        stock_discretization=StockDiscretization({AreaIndex(reservoir.area): X}),
        list_models=list_models,
        V={
            week: UniVariateEstimator({reservoir.area: V[week]})
            for week in range(param.len_week + 1)
        },
    )

    assert upper_bound == pytest.approx(5046992854.133574)
    assert controls[TimeScenarioIndex(0, 0)][reservoir.area] == pytest.approx(1146984.0)
    assert len(problem.solver.constraints()) == 3555
    assert len(problem.solver.variables()) == 3533


def test_upper_bound_with_bellman_values() -> None:

    problem = AntaresProblem(scenario=0, week=0, path="test_data/one_node", itr=1)
    param = TimeScenarioParameter(len_week=1, len_scenario=1)
    reservoir = Reservoir("test_data/one_node", "area")

    reservoir_management = ReservoirManagement(
        reservoir=reservoir,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=False,
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

    V[1].costs = bellman_values[:, 1]
    upper_bound, controls, _ = compute_upper_bound(
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        stock_discretization=StockDiscretization({AreaIndex(reservoir.area): X}),
        list_models=list_models,
        V={
            week: UniVariateEstimator({reservoir.area: V[week]})
            for week in range(param.len_week + 1)
        },
    )

    assert controls[TimeScenarioIndex(0, 0)][reservoir.area] == pytest.approx(133776)

    control = 123864.0
    vb = V[1](reservoir.initial_level + reservoir.inflow[0, 0] - control)
    cost, _, _, _ = problem.solve_with_predefined_controls({"area": control})
    assert cost - vb == pytest.approx(upper_bound)


def test_upper_bound_with_xpress() -> None:
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
        assert len(problem.solver.constraints()) == 3535
        assert len(problem.solver.variables()) == 3533

        upper_bound, controls, _ = compute_upper_bound(
            param=param,
            multi_stock_management=MultiStockManagement([reservoir_management]),
            stock_discretization=StockDiscretization({AreaIndex(reservoir.area): X}),
            list_models=list_models,
            V={
                week: UniVariateEstimator({reservoir.area: V[week]})
                for week in range(param.len_week + 1)
            },
        )

        assert upper_bound == pytest.approx(380492940.000565)
        assert controls[TimeScenarioIndex(0, 0)][reservoir.area] == pytest.approx(
            4482011.0
        )
        assert len(problem.solver.constraints()) == 3555
        assert len(problem.solver.variables()) == 3533

        V[1].costs = np.linspace(-5e9, -3e9, num=20)
        upper_bound, controls, _ = compute_upper_bound(
            param=param,
            multi_stock_management=MultiStockManagement([reservoir_management]),
            stock_discretization=StockDiscretization({AreaIndex(reservoir.area): X}),
            list_models=list_models,
            V={
                week: UniVariateEstimator({reservoir.area: V[week]})
                for week in range(param.len_week + 1)
            },
        )

        assert upper_bound == pytest.approx(5046992854.133574)
        assert controls[TimeScenarioIndex(0, 0)][reservoir.area] == pytest.approx(
            1146984.0
        )
        assert len(problem.solver.constraints()) == 3555
        assert len(problem.solver.variables()) == 3533
    else:
        print("Ignore test, xpress not available")
