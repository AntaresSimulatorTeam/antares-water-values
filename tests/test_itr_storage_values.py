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

expected_traj = np.array(
    [
        [4450000.0],
        [2410000.0],
        [6390000.0],
        [6350000.0],
        [6320000.0],
        [2280000.0],
    ]
)

expected_vb = np.array(
    [
        [
            -5.44276378e09,
            -5.37158298e09,
            -3.39016883e09,
            -2.41782016e09,
            -1.41870451e09,
            0.00000000e00,
        ],
        [
            -5.28486810e09,
            -4.36279347e09,
            -3.28486349e09,
            -2.31251482e09,
            -1.31339930e09,
            0.00000000e00,
        ],
        [
            -5.12697190e09,
            -4.20489754e09,
            -3.17955814e09,
            -2.20720922e09,
            -1.20809421e09,
            0.00000000e00,
        ],
        [
            -4.98446182e09,
            -4.04738176e09,
            -3.07425280e09,
            -2.10190387e09,
            -1.10278899e09,
            0.00000000e00,
        ],
        [
            -4.87915622e09,
            -3.94207642e09,
            -2.96894746e09,
            -1.99659853e09,
            -9.97483776e08,
            0.00000000e00,
        ],
        [
            -4.77385114e09,
            -3.83677107e09,
            -2.86364211e09,
            -1.89129306e09,
            -8.92178624e08,
            0.00000000e00,
        ],
        [
            -4.66854554e09,
            -3.73146573e09,
            -2.75833677e09,
            -1.78598771e09,
            -7.86873472e08,
            0.00000000e00,
        ],
        [
            -4.56324045e09,
            -3.62616038e09,
            -2.65303142e09,
            -1.68068224e09,
            -6.97788224e08,
            0.00000000e00,
        ],
        [
            -4.45793536e09,
            -3.52085504e09,
            -2.54772608e09,
            -1.57537690e09,
            -6.45156608e08,
            0.00000000e00,
        ],
        [
            -4.35262976e09,
            -3.41554970e09,
            -2.44242074e09,
            -1.47481523e09,
            -5.92524992e08,
            0.00000000e00,
        ],
        [
            -4.24732467e09,
            -3.31024461e09,
            -2.33711539e09,
            -1.39639142e09,
            -5.39893376e08,
            0.00000000e00,
        ],
        [
            -4.14201907e09,
            -3.20493926e09,
            -2.23181005e09,
            -1.34375974e09,
            -4.87261760e08,
            0.00000000e00,
        ],
        [
            -4.03671373e09,
            -3.09963392e09,
            -2.12650483e09,
            -1.29112819e09,
            -4.34630144e08,
            0.00000000e00,
        ],
        [
            -3.93140838e09,
            -2.99432832e09,
            -2.02119949e09,
            -1.23849651e09,
            -3.81998560e08,
            0.00000000e00,
        ],
        [
            -3.82610304e09,
            -2.88902298e09,
            -1.91589414e09,
            -1.18586496e09,
            -3.29366944e08,
            0.00000000e00,
        ],
        [
            -3.75680307e09,
            -2.78371763e09,
            -1.81058880e09,
            -1.13323328e09,
            -2.76735328e08,
            0.00000000e00,
        ],
        [
            -3.70417126e09,
            -2.67841229e09,
            -1.70528346e09,
            -1.08060160e09,
            -2.24103728e08,
            0.00000000e00,
        ],
        [
            -3.65153946e09,
            -2.57310669e09,
            -1.59997811e09,
            -1.02796986e09,
            -1.71472112e08,
            0.00000000e00,
        ],
        [
            -3.59890790e09,
            -2.46780134e09,
            -1.49830822e09,
            -9.75338112e08,
            -1.18840504e08,
            0.00000000e00,
        ],
        [
            -3.54627610e09,
            -2.36249600e09,
            -1.41471270e09,
            -9.22706368e08,
            -6.62088960e07,
            0.00000000e00,
        ],
    ]
)


def test_itr_control(param: TimeScenarioParameter) -> None:
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

    vb, G, _, _, controls_upper, traj, lb, ub = itr_control(
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

    assert -lb == pytest.approx(4410020694.484235)
    assert ub == pytest.approx(4410021160.928989)

    assert np.array(
        [
            [traj[1][TimeScenarioIndex(w, s)] for s in range(param.len_scenario)]
            for w in range(param.len_week + 1)
        ]
    ) == pytest.approx(expected_traj)

    assert np.transpose([x for x in vb.values()]) == pytest.approx(expected_vb)


def test_itr_control_with_xpress(param: TimeScenarioParameter) -> None:

    solver = pywraplp.Solver.CreateSolver("XPRESS_LP")
    if solver:
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

        vb, G, _, _, controls_upper, traj, lb, ub = itr_control(
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

        assert -lb == pytest.approx(4410020694.484235)
        assert ub == pytest.approx(4410021160.928989)

        assert np.array(
            [
                [traj[1][TimeScenarioIndex(w, s)] for s in range(param.len_scenario)]
                for w in range(param.len_week + 1)
            ]
        ) == pytest.approx(expected_traj)

        assert np.transpose([x for x in vb.values()]) == pytest.approx(expected_vb)
    else:
        print("Ignore test, xpress not available")
