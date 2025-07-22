import numpy as np
import pytest

from functions_iterative import TimeScenarioParameter
from multi_stock_bellman_value_calculation import MultiStockManagement
from simple_bellman_value_calculation import calculate_bellman_value_directly
from type_definition import AreaIndex, Array1D, Dict

expected_vb = np.array(
    [
        [
            5.88819098e09,
            5.37158298e09,
            4.30354534e09,
            3.59375002e09,
            1.99857485e09,
            -0.00000000e00,
        ],
        [
            5.28486810e09,
            4.36279347e09,
            3.38627738e09,
            2.42329805e09,
            1.40291200e09,
            -0.00000000e00,
        ],
        [
            5.12697242e09,
            4.20489754e09,
            3.22838170e09,
            2.26540211e09,
            1.24501632e09,
            -0.00000000e00,
        ],
        [
            4.98446234e09,
            4.04738202e09,
            3.07425306e09,
            2.10750656e09,
            1.10278950e09,
            -0.00000000e00,
        ],
        [
            4.87915674e09,
            3.94207642e09,
            2.96894746e09,
            1.99659827e09,
            9.97484032e08,
            -0.00000000e00,
        ],
        [
            4.77385114e09,
            3.83677107e09,
            2.86364211e09,
            1.89129293e09,
            8.92178688e08,
            -0.00000000e00,
        ],
        [
            4.66854605e09,
            3.73146573e09,
            2.75833677e09,
            1.78598758e09,
            7.86873472e08,
            -0.00000000e00,
        ],
        [
            4.56324045e09,
            3.62616038e09,
            2.65303142e09,
            1.68068224e09,
            6.97788608e08,
            -0.00000000e00,
        ],
        [
            4.45793536e09,
            3.52085504e09,
            2.54772608e09,
            1.57537702e09,
            6.45156800e08,
            -0.00000000e00,
        ],
        [
            4.35262976e09,
            3.41554970e09,
            2.44242099e09,
            1.47481574e09,
            5.92525120e08,
            -0.00000000e00,
        ],
        [
            4.24732442e09,
            3.31024461e09,
            2.33711590e09,
            1.39639194e09,
            5.39893440e08,
            -0.00000000e00,
        ],
        [
            4.14201933e09,
            3.20493926e09,
            2.23440384e09,
            1.34376013e09,
            4.87261792e08,
            -0.00000000e00,
        ],
        [
            4.03671398e09,
            3.09963418e09,
            2.14594355e09,
            1.29112832e09,
            4.34630144e08,
            -0.00000000e00,
        ],
        [
            3.93140890e09,
            2.99572736e09,
            2.08161843e09,
            1.23849651e09,
            3.81998560e08,
            -0.00000000e00,
        ],
        [
            3.82610381e09,
            2.90070016e09,
            2.02898662e09,
            1.18586483e09,
            3.29367104e08,
            -0.00000000e00,
        ],
        [
            3.75708032e09,
            2.83013274e09,
            1.97635482e09,
            1.13323302e09,
            2.76736128e08,
            -0.00000000e00,
        ],
        [
            3.70444851e09,
            2.77750093e09,
            1.92372301e09,
            1.08060134e09,
            2.24105184e08,
            -0.00000000e00,
        ],
        [
            3.65181670e09,
            2.72486938e09,
            1.87109120e09,
            1.02796973e09,
            1.71474288e08,
            -0.00000000e00,
        ],
        [
            3.59918515e09,
            2.67223757e09,
            1.81845952e09,
            9.75338048e08,
            1.18843432e08,
            -0.00000000e00,
        ],
        [
            3.54655360e09,
            2.61960576e09,
            1.76582797e09,
            9.22706432e08,
            6.62126120e07,
            -0.00000000e00,
        ],
    ]
)

expected_vb_lower_approximation = np.array(
    [
        [
            5.88819046e09,
            5.37158298e09,
            4.30354534e09,
            3.59375002e09,
            1.99857485e09,
            0.00000000e00,
        ],
        [
            5.28486810e09,
            4.36279347e09,
            3.38627744e09,
            2.42329805e09,
            1.40291200e09,
            0.00000000e00,
        ],
        [
            5.12697223e09,
            4.20489754e09,
            3.22838170e09,
            2.26540211e09,
            1.24501632e09,
            0.00000000e00,
        ],
        [
            4.98446182e09,
            4.04738202e09,
            3.07425286e09,
            2.10750656e09,
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


def test_bellman_value_exact(
    param: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
    discretization_one_node: Dict[AreaIndex, Array1D],
) -> None:
    vb, lb, ub = calculate_bellman_value_directly(
        param=param,
        multi_stock_management=multi_stock_management_one_node,
        output_path="test_data/one_node",
        X=discretization_one_node,
        univariate=True,
    )

    assert lb == pytest.approx(4410021312)

    assert ub == pytest.approx(4410021125.56477)

    assert np.transpose(
        [
            [
                vb[week].get_value({area.area: x})
                for area in multi_stock_management_one_node.areas
                for x in discretization_one_node[area]
            ]
            for week in range(param.len_week + 1)
        ]
    ) == pytest.approx(expected_vb)


def test_bellman_value_exact_with_multi_stock(
    param: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
    discretization_one_node: Dict[AreaIndex, Array1D],
) -> None:

    vb, lb, ub = calculate_bellman_value_directly(
        param=param,
        multi_stock_management=multi_stock_management_one_node,
        output_path="test_data/one_node",
        X=discretization_one_node,
        univariate=False,
    )

    assert lb == pytest.approx(4410021218.59294)

    assert ub == pytest.approx(4410021094.449415)

    computed_vb = np.transpose(
        [
            [
                vb[week].get_value({area.area: x})
                for area in multi_stock_management_one_node.areas
                for x in discretization_one_node[area]
            ]
            for week in range(param.len_week + 1)
        ]
    )

    for week in range(param.len_week):
        assert computed_vb[:, param.len_week - week - 1] == pytest.approx(
            expected_vb_lower_approximation[:, param.len_week - week - 1]
        )
