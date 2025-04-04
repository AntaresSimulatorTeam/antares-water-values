import numpy as np
import pytest

from functions_iterative import ReservoirManagement, TimeScenarioParameter
from multi_stock_bellman_value_calculation import (
    MultiStockManagement,
    precalculated_method,
)
from read_antares_data import Reservoir
from simple_bellman_value_calculation import (
    calculate_bellman_value_with_precalculated_reward,
)
from type_definition import AreaIndex, Dict, List, TimeScenarioIndex, WeekIndex

expected_vb = np.array(
    [
        [
            -5.88819050e09,
            -5.37158308e09,
            -4.30354519e09,
            -3.62174927e09,
            -1.99857483e09,
            0.00000000e00,
        ],
        [
            -5.28486770e09,
            -4.36279337e09,
            -3.38627736e09,
            -2.42429723e09,
            -1.40291199e09,
            0.00000000e00,
        ],
        [
            -5.12697202e09,
            -4.20489758e09,
            -3.22838159e09,
            -2.26640141e09,
            -1.24501626e09,
            0.00000000e00,
        ],
        [
            -4.98446177e09,
            -4.04738189e09,
            -3.07425291e09,
            -2.10850558e09,
            -1.10278955e09,
            0.00000000e00,
        ],
        [
            -4.87915628e09,
            -3.94207640e09,
            -2.96894743e09,
            -1.99659827e09,
            -9.97484041e08,
            0.00000000e00,
        ],
        [
            -4.77385081e09,
            -3.83677093e09,
            -2.86364201e09,
            -1.89129286e09,
            -8.92178681e08,
            0.00000000e00,
        ],
        [
            -4.66854548e09,
            -3.73146560e09,
            -2.75833669e09,
            -1.78598754e09,
            -7.86873403e08,
            0.00000000e00,
        ],
        [
            -4.56324018e09,
            -3.62616030e09,
            -2.65303139e09,
            -1.68068223e09,
            -6.97788603e08,
            0.00000000e00,
        ],
        [
            -4.45793490e09,
            -3.52085502e09,
            -2.54772616e09,
            -1.57537699e09,
            -6.45156824e08,
            0.00000000e00,
        ],
        [
            -4.35262964e09,
            -3.41554976e09,
            -2.44242093e09,
            -1.47481564e09,
            -5.92525097e08,
            0.00000000e00,
        ],
        [
            -4.24732437e09,
            -3.31024452e09,
            -2.33711570e09,
            -1.39639186e09,
            -5.39893426e08,
            0.00000000e00,
        ],
        [
            -4.14201910e09,
            -3.20493930e09,
            -2.23440362e09,
            -1.34376008e09,
            -4.87261771e08,
            0.00000000e00,
        ],
        [
            -4.03671387e09,
            -3.09963410e09,
            -2.14594333e09,
            -1.29112830e09,
            -4.34630157e08,
            0.00000000e00,
        ],
        [
            -3.93140864e09,
            -2.99572720e09,
            -2.08161815e09,
            -1.23849653e09,
            -3.81998560e08,
            0.00000000e00,
        ],
        [
            -3.82610344e09,
            -2.90069998e09,
            -2.02898637e09,
            -1.18586480e09,
            -3.29367085e08,
            0.00000000e00,
        ],
        [
            -3.75708018e09,
            -2.82991041e09,
            -1.97635460e09,
            -1.13323307e09,
            -2.76736122e08,
            0.00000000e00,
        ],
        [
            -3.70444843e09,
            -2.77727863e09,
            -1.92372285e09,
            -1.08060136e09,
            -2.24105162e08,
            0.00000000e00,
        ],
        [
            -3.65181671e09,
            -2.72464685e09,
            -1.87109111e09,
            -1.02796969e09,
            -1.71474287e08,
            0.00000000e00,
        ],
        [
            -3.59918506e09,
            -2.67201512e09,
            -1.81845943e09,
            -9.75338039e08,
            -1.18843427e08,
            0.00000000e00,
        ],
        [
            -3.54655341e09,
            -2.61938341e09,
            -1.76582776e09,
            -9.22706398e08,
            -6.62126083e07,
            0.00000000e00,
        ],
    ]
)


def test_bellman_value_precalculated_reward(param: TimeScenarioParameter) -> None:

    reservoir = Reservoir("test_data/one_node", "area")
    reservoir_management = ReservoirManagement(
        reservoir=reservoir,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=False,
    )
    xNsteps = 20

    vb, G = calculate_bellman_value_with_precalculated_reward(
        len_controls=20,
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        output_path="test_data/one_node",
        len_bellman=xNsteps,
    )

    true_list_cut = [
        (3000.0013996873, 5126887560.475002),
        (3000.0012436081, 5126886301.973297),
        (3000.0010484806, 5126884930.052198),
        (3000.0008855708, 5126883920.159666),
        (3000.0007507801997, 5126883207.578199),
        (3000.0006388722, 5126882714.387037),
        (3000.0004975926, 5126882214.482481),
        (300.00255761340003, -848256469.2759233),
        (300.001725077, -848257974.2313964),
        (200.0804368654155, -943484638.5069666),
        (200.08000954152382, -943484644.1886383),
        (100.000405626, -828695043.6127033),
        (100.00025922340001, -828694783.0604204),
        (100.00013336960001, -828694458.0135351),
        (100.0000155417, -828694033.2488956),
        (99.9989328345, -828689254.4939792),
        (99.9988287817, -828688697.2376316),
        (99.9986947312, -828687869.8481407),
        (99.998518274, -828686633.7743558),
        (-0.0004060626000000001, -38705645.55951345),
    ]
    for i, cut in enumerate(true_list_cut):
        assert G[reservoir.area][TimeScenarioIndex(0, 0)].list_cut[i] == pytest.approx(
            cut
        )

    true_breaking_point = [
        -8400000.0,
        -8063224.997515362,
        -7030895.6782807605,
        -6199090.137508901,
        -5286581.292333476,
        -4407112.655717107,
        -3538405.80076642,
        -2213016.0157842324,
        -1807675.2837932073,
        -953016.7793065935,
        -13295.937459539502,
        1146982.96241088,
        1779697.1019513493,
        2582733.975863404,
        3604958.074023681,
        4413709.3726140605,
        5355515.15835384,
        6172222.34066152,
        7004949.556935257,
        7899894.858426053,
        8400000.0,
    ]
    for i, pt in enumerate(true_breaking_point):
        assert G[reservoir.area][TimeScenarioIndex(0, 0)].breaking_point[
            i
        ] == pytest.approx(pt, 1e-5)

    assert vb == pytest.approx(expected_vb)


def test_bellman_value_precalculated_reward_with_multi_stock(
    param: TimeScenarioParameter,
) -> None:

    reservoir = Reservoir("test_data/one_node", "area")
    reservoir_management = ReservoirManagement(
        reservoir=reservoir,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=False,
    )
    xNsteps = 20

    _, _, bellman_costs, _, _, _ = precalculated_method(
        len_controls=20,
        param=param,
        multi_stock_management=MultiStockManagement([reservoir_management]),
        output_path="test_data/one_node",
        len_bellman=xNsteps,
    )

    # assert np.transpose(bellman_costs) == pytest.approx(
    #     expected_vb[:, : param.len_week]
    # )
