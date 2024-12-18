import numpy as np
import pytest

from estimation import BellmanValueEstimation
from functions_iterative import ReservoirManagement, TimeScenarioParameter
from multi_stock_bellman_value_calculation import MultiStockManagement
from read_antares_data import Reservoir
from simple_bellman_value_calculation import calculate_bellman_value_directly

expected_vb = -np.array(
    [
        [
            -5.8881910e09,
            -5.3715830e09,
            -4.3035453e09,
            -3.6217492e09,
            -1.9985748e09,
            0.0000000e00,
        ],
        [
            -5.2848681e09,
            -4.3627935e09,
            -3.3862774e09,
            -2.4242972e09,
            -1.4029120e09,
            0.0000000e00,
        ],
        [
            -5.1269724e09,
            -4.2048975e09,
            -3.2283817e09,
            -2.2664013e09,
            -1.2450163e09,
            0.0000000e00,
        ],
        [
            -4.9844623e09,
            -4.0473820e09,
            -3.0742531e09,
            -2.1085059e09,
            -1.1027895e09,
            0.0000000e00,
        ],
        [
            -4.8791567e09,
            -3.9420764e09,
            -2.9689475e09,
            -1.9965983e09,
            -9.9748403e08,
            0.0000000e00,
        ],
        [
            -4.7738511e09,
            -3.8367711e09,
            -2.8636421e09,
            -1.8912929e09,
            -8.9217869e08,
            0.0000000e00,
        ],
        [
            -4.6685460e09,
            -3.7314657e09,
            -2.7583368e09,
            -1.7859876e09,
            -7.8687347e08,
            0.0000000e00,
        ],
        [
            -4.5632404e09,
            -3.6261604e09,
            -2.6530314e09,
            -1.6806822e09,
            -6.9778861e08,
            0.0000000e00,
        ],
        [
            -4.4579354e09,
            -3.5208550e09,
            -2.5477261e09,
            -1.5753770e09,
            -6.4515680e08,
            0.0000000e00,
        ],
        [
            -4.3526298e09,
            -3.4155497e09,
            -2.4424210e09,
            -1.4748157e09,
            -5.9252512e08,
            0.0000000e00,
        ],
        [
            -4.2473244e09,
            -3.3102446e09,
            -2.3371159e09,
            -1.3963919e09,
            -5.3989344e08,
            0.0000000e00,
        ],
        [
            -4.1420193e09,
            -3.2049393e09,
            -2.2344038e09,
            -1.3437601e09,
            -4.8726179e08,
            0.0000000e00,
        ],
        [
            -4.0367140e09,
            -3.0996342e09,
            -2.1459436e09,
            -1.2911283e09,
            -4.3463014e08,
            0.0000000e00,
        ],
        [
            -3.9314089e09,
            -2.9957274e09,
            -2.0816184e09,
            -1.2384965e09,
            -3.8199856e08,
            0.0000000e00,
        ],
        [
            -3.8261038e09,
            -2.9007002e09,
            -2.0289866e09,
            -1.1858648e09,
            -3.2936710e08,
            0.0000000e00,
        ],
        [
            -3.7570803e09,
            -2.8299108e09,
            -1.9763548e09,
            -1.1332330e09,
            -2.7673613e08,
            0.0000000e00,
        ],
        [
            -3.7044485e09,
            -2.7772790e09,
            -1.9237230e09,
            -1.0806013e09,
            -2.2410518e08,
            0.0000000e00,
        ],
        [
            -3.6518167e09,
            -2.7246472e09,
            -1.8710912e09,
            -1.0279697e09,
            -1.7147429e08,
            0.0000000e00,
        ],
        [
            -3.5991852e09,
            -2.6720154e09,
            -1.8184595e09,
            -9.7533811e08,
            -1.1884343e08,
            0.0000000e00,
        ],
        [
            -3.5465536e09,
            -2.6193836e09,
            -1.7658280e09,
            -9.2270643e08,
            -6.6212612e07,
            0.0000000e00,
        ],
    ],
)

expected_vb_lower_approximation = np.array(
    [
        [
            5.88819046e09,
            5.37158298e09,
            4.30354534e09,
            3.62174925e09,
            1.99857485e09,
            0.00000000e00,
        ],
        [
            5.28486810e09,
            4.36279347e09,
            3.38627744e09,
            2.42429722e09,
            1.40291200e09,
            0.00000000e00,
        ],
        [
            5.12697223e09,
            4.20489754e09,
            3.22838170e09,
            2.26640128e09,
            1.24501632e09,
            0.00000000e00,
        ],
        [
            4.98446182e09,
            4.04738202e09,
            3.07425285e09,
            2.10850586e09,
            1.10278950e09,
            0.00000000e00,
        ],
        [
            4.87915648e09,
            3.94207644e09,
            2.96894746e09,
            1.99659827e09,
            9.97484032e08,
            0.00000000e00,
        ],
        [
            4.77385114e09,
            3.83677107e09,
            2.86364211e09,
            1.89129290e09,
            8.92178688e08,
            0.00000000e00,
        ],
        [
            4.66854579e09,
            3.73146573e09,
            2.75833680e09,
            1.78598758e09,
            7.86873472e08,
            0.00000000e00,
        ],
        [
            4.56324045e09,
            3.62616042e09,
            2.65303149e09,
            1.68068227e09,
            6.97788608e08,
            0.00000000e00,
        ],
        [
            4.45793514e09,
            3.52085510e09,
            2.54772621e09,
            1.57537709e09,
            6.45156815e08,
            0.00000000e00,
        ],
        [
            4.35262982e09,
            3.41554982e09,
            2.44242106e09,
            1.47007194e09,
            5.92525120e08,
            0.00000000e00,
        ],
        [
            4.24732451e09,
            3.31024467e09,
            2.33711593e09,
            1.39639194e09,
            5.39893440e08,
            0.00000000e00,
        ],
        [
            4.14201938e09,
            3.20493954e09,
            2.23181082e09,
            1.34376014e09,
            4.87261792e08,
            0.00000000e00,
        ],
        [
            4.03671424e09,
            3.09963443e09,
            2.13425011e09,
            1.29112834e09,
            4.34630153e08,
            0.00000000e00,
        ],
        [
            3.93140910e09,
            2.99432932e09,
            2.08161830e09,
            1.23849656e09,
            3.81998560e08,
            0.00000000e00,
        ],
        [
            3.82610397e09,
            2.88902421e09,
            2.02898650e09,
            1.18586485e09,
            3.29367104e08,
            0.00000000e00,
        ],
        [
            3.75680376e09,
            2.81986609e09,
            1.97635470e09,
            1.13323315e09,
            2.76736128e08,
            0.00000000e00,
        ],
        [
            3.70417203e09,
            2.76723430e09,
            1.92372291e09,
            1.08060146e09,
            2.24105184e08,
            0.00000000e00,
        ],
        [
            3.65154032e09,
            2.71460252e09,
            1.87109120e09,
            1.02796976e09,
            1.71474288e08,
            0.00000000e00,
        ],
        [
            3.59890867e09,
            2.66197074e09,
            1.81845952e09,
            9.75338066e08,
            1.18843432e08,
            0.00000000e00,
        ],
        [
            3.54627702e09,
            2.60933898e09,
            1.76582784e09,
            9.22706432e08,
            6.62126120e07,
            0.00000000e00,
        ],
    ]
)


def test_bellman_value_exact() -> None:

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

    vb, lb, ub = calculate_bellman_value_directly(
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        output_path="test_data/one_node",
        X={reservoir.area: X},
        univariate=True,
    )

    assert lb == pytest.approx(4410021312)

    assert ub == pytest.approx(4410021125.56477)

    assert np.transpose(
        [
            [vb[week].get_value({reservoir.area: x}) for x in X]
            for week in range(param.len_week + 1)
        ]
    ) == pytest.approx(expected_vb)


def test_bellman_value_exact_with_multi_stock() -> None:

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

    vb, lb, ub = calculate_bellman_value_directly(
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        output_path="test_data/one_node",
        X={"area": X},
        univariate=False,
    )

    assert lb == pytest.approx(4410021218.59294)

    assert ub == pytest.approx(4410021094.449415)

    computed_vb = np.transpose(
        [
            [vb[week].get_value({reservoir.area: x}) for x in X]
            for week in range(param.len_week + 1)
        ]
    )

    for week in range(param.len_week):
        assert computed_vb[:, param.len_week - week - 1] == pytest.approx(
            expected_vb_lower_approximation[:, param.len_week - week - 1]
        )
