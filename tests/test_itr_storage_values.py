import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
import pytest

from functions_iterative import (
    ReservoirManagement,
    TimeScenarioIndex,
    TimeScenarioParameter,
    itr_control,
)
from read_antares_data import Reservoir

expected_vb = np.array(
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
            -2.80569404e09,
            -1.81058848e09,
            -1.13323252e09,
            -2.76735331e08,
            0.00000000e00,
        ],
        [
            -3.70417075e09,
            -2.75306203e09,
            -1.70528314e09,
            -1.08060073e09,
            -2.24103722e08,
            0.00000000e00,
        ],
        [
            -3.65153900e09,
            -2.70043042e09,
            -1.59997779e09,
            -1.02796894e09,
            -1.71472114e08,
            0.00000000e00,
        ],
        [
            -3.59890725e09,
            -2.64779880e09,
            -1.49830812e09,
            -9.75337146e08,
            -1.18840505e08,
            0.00000000e00,
        ],
        [
            -3.54627549e09,
            -2.59516719e09,
            -1.41471259e09,
            -9.22705355e08,
            -6.62088960e07,
            0.00000000e00,
        ],
    ]
)


def test_itr_control() -> None:

    param = TimeScenarioParameter(len_week=5, len_scenario=1)
    reservoir = Reservoir("test_data/one_node", "area")
    reservoir_management = ReservoirManagement(
        reservoir=reservoir,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=False,
    )
    xNsteps = 20
    X = np.linspace(0, reservoir.capacity, num=xNsteps)

    vb, G, _, _, controls_upper, traj = itr_control(
        param=param,
        reservoir_management=reservoir_management,
        output_path="test_data/one_node",
        X=X,
        N=3,
        tol_gap=1e-4,
    )

    assert G[TimeScenarioIndex(0, 0)].list_cut[0] == pytest.approx(
        (300.0022431781, -848257117.7874993)
    )
    assert G[TimeScenarioIndex(0, 0)].list_cut[1] == pytest.approx(
        (200.08020216786073, -943484691.5152471)
    )
    assert G[TimeScenarioIndex(0, 0)].list_cut[2] == pytest.approx(
        (100.0003310016, -828694927.2829424)
    )
    assert G[TimeScenarioIndex(0, 0)].list_cut[3] == pytest.approx((0.0, 0.0))

    assert G[TimeScenarioIndex(0, 0)].breaking_point == pytest.approx(
        np.array(
            [
                -8400000.0,
                -953018.7010290311,
                1146981.5347944114,
                8286921.842985533,
                8400000.0,
            ]
        )
    )

    assert controls_upper[-1] == pytest.approx(
        np.array([[123864.0], [255912.0], [34924.0], [1139897.0], [773918.0]])
    )

    assert traj[1] == pytest.approx(
        np.array(
            [
                [4450000.0],
                [6420000.0],
                [2380000.0],
                [6350000.0],
                [6320000.0],
                [2280000.0],
            ]
        )
    )

    assert vb == pytest.approx(expected_vb)


def test_itr_control_with_xpress() -> None:

    solver = pywraplp.Solver.CreateSolver("XPRESS_LP")
    if solver:

        param = TimeScenarioParameter(len_week=5, len_scenario=1)
        reservoir = Reservoir("test_data/one_node", "area")
        reservoir_management = ReservoirManagement(
            reservoir=reservoir,
            penalty_bottom_rule_curve=3000,
            penalty_upper_rule_curve=3000,
            penalty_final_level=3000,
            force_final_level=False,
        )
        xNsteps = 20
        X = np.linspace(0, reservoir.capacity, num=xNsteps)

        vb, G, _, _, controls_upper, traj = itr_control(
            param=param,
            reservoir_management=reservoir_management,
            output_path="test_data/one_node",
            X=X,
            N=3,
            tol_gap=1e-4,
            solver="XPRESS_LP",
        )

        assert G[TimeScenarioIndex(0, 0)].list_cut[0] == pytest.approx(
            (300.0022431781, -848257117.7874993)
        )
        assert G[TimeScenarioIndex(0, 0)].list_cut[1] == pytest.approx(
            (200.08020216786073, -943484691.5152471)
        )
        assert G[TimeScenarioIndex(0, 0)].list_cut[2] == pytest.approx(
            (100.0003310016, -828694927.2829424)
        )
        assert G[TimeScenarioIndex(0, 0)].list_cut[3] == pytest.approx((0.0, 0.0))

        assert G[TimeScenarioIndex(0, 0)].breaking_point == pytest.approx(
            np.array(
                [
                    -8400000.0,
                    -953018.7010290311,
                    1146981.5347944114,
                    8286921.842985533,
                    8400000.0,
                ]
            )
        )

        assert controls_upper[-1] == pytest.approx(
            np.array([[123864.0], [255912.0], [34924.0], [1139897.0], [773918.0]])
        )

        assert traj[1] == pytest.approx(
            np.array(
                [
                    [4450000.0],
                    [6420000.0],
                    [2380000.0],
                    [6350000.0],
                    [6320000.0],
                    [2280000.0],
                ]
            )
        )

        assert vb == pytest.approx(expected_vb)
    else:
        print("Ignore test, xpress not available")
