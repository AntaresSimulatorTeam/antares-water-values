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
from type_definition import timescenario_area_value_to_array

expected_traj = np.array(
    [[[4450000.0]], [[4450000.0]], [[4450000.0]], [[4450000.0]], [[4450000.0]]]
)

expected_vb = np.array(
    [
        [
            -5.3003807e09,
            -4.3632993e09,
            -3.3901696e09,
            -2.4178199e09,
            -1.4187050e09,
            0.0000000e00,
        ],
        [
            -5.1950756e09,
            -4.2579940e09,
            -3.2848643e09,
            -2.3125146e09,
            -1.3133998e09,
            0.0000000e00,
        ],
        [
            -5.0897705e09,
            -4.1526886e09,
            -3.1795589e09,
            -2.2072092e09,
            -1.2080946e09,
            0.0000000e00,
        ],
        [
            -4.9844649e09,
            -4.0473836e09,
            -3.0742538e09,
            -2.1019040e09,
            -1.1027892e09,
            0.0000000e00,
        ],
        [
            -4.8791598e09,
            -3.9420782e09,
            -2.9689485e09,
            -1.9965988e09,
            -9.9748403e08,
            0.0000000e00,
        ],
        [
            -4.7738547e09,
            -3.8367729e09,
            -2.8636431e09,
            -1.8912934e09,
            -8.9217875e08,
            0.0000000e00,
        ],
        [
            -4.6685491e09,
            -3.7314675e09,
            -2.7583380e09,
            -1.7859882e09,
            -7.8687347e08,
            0.0000000e00,
        ],
        [
            -4.5632440e09,
            -3.6261624e09,
            -2.6530327e09,
            -1.6806830e09,
            -6.8156819e08,
            0.0000000e00,
        ],
        [
            -4.4579389e09,
            -3.5208571e09,
            -2.5477274e09,
            -1.5753777e09,
            -5.7626298e08,
            0.0000000e00,
        ],
        [
            -4.3526333e09,
            -3.4155517e09,
            -2.4424220e09,
            -1.4700724e09,
            -4.7095770e08,
            0.0000000e00,
        ],
        [
            -4.2473283e09,
            -3.3102467e09,
            -2.3371169e09,
            -1.3647672e09,
            -3.6565242e08,
            0.0000000e00,
        ],
        [
            -4.1420229e09,
            -3.2049413e09,
            -2.2318116e09,
            -1.2594619e09,
            -2.6034717e08,
            0.0000000e00,
        ],
        [
            -4.0367178e09,
            -3.0996360e09,
            -2.1265064e09,
            -1.1541567e09,
            -1.5504190e08,
            0.0000000e00,
        ],
        [
            -3.9314125e09,
            -2.9943309e09,
            -2.0212012e09,
            -1.0488514e09,
            -4.9736636e07,
            0.0000000e00,
        ],
        [
            -3.8261071e09,
            -2.8890255e09,
            -1.9158958e09,
            -9.4354611e08,
            0.0000000e00,
            0.0000000e00,
        ],
        [
            -3.7208020e09,
            -2.7837202e09,
            -1.8105905e09,
            -8.3824083e08,
            0.0000000e00,
            0.0000000e00,
        ],
        [
            -3.6154967e09,
            -2.6784151e09,
            -1.7052851e09,
            -7.3293562e08,
            0.0000000e00,
            0.0000000e00,
        ],
        [
            -3.5101914e09,
            -2.5731100e09,
            -1.5999798e09,
            -6.2763034e08,
            0.0000000e00,
            0.0000000e00,
        ],
        [
            -3.4048860e09,
            -2.4678049e09,
            -1.4946746e09,
            -5.2232506e08,
            0.0000000e00,
            0.0000000e00,
        ],
        [
            -3.2995809e09,
            -2.3625001e09,
            -1.3893695e09,
            -4.1701981e08,
            0.0000000e00,
            0.0000000e00,
        ],
    ]
)
true_list_cut = [
    (0.0, 0.0),
    (200.08020216786073, -943484691.5152471),
]


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

    for i, cut in enumerate(true_list_cut):
        assert -G[TimeScenarioIndex(0, 0)].costs[i] + G[TimeScenarioIndex(0, 0)].duals[
            i
        ] * G[TimeScenarioIndex(0, 0)].inputs[i] == pytest.approx(cut[1])
        assert G[TimeScenarioIndex(0, 0)].duals[i] == pytest.approx(-cut[0])

    assert lb == pytest.approx(-4410024896.0)
    assert ub == pytest.approx(4410021272)

    assert timescenario_area_value_to_array(traj[0], param) == pytest.approx(
        expected_traj
    )

    assert np.transpose([x for x in vb.values()]) == pytest.approx(
        expected_vb, rel=1e-5
    )


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

        for i, cut in enumerate(true_list_cut):
            assert -G[TimeScenarioIndex(0, 0)].costs[i] + G[
                TimeScenarioIndex(0, 0)
            ].duals[i] * G[TimeScenarioIndex(0, 0)].inputs[i] == pytest.approx(cut[1])
            assert G[TimeScenarioIndex(0, 0)].duals[i] == pytest.approx(-cut[0])

        assert lb == pytest.approx(-4410024896.0)
        assert ub == pytest.approx(4410021272)

        assert timescenario_area_value_to_array(traj[0], param) == pytest.approx(
            expected_traj
        )

        assert np.transpose([x for x in vb.values()]) == pytest.approx(
            expected_vb, rel=1e-5
        )
    else:
        print("Ignore test, xpress not available")
