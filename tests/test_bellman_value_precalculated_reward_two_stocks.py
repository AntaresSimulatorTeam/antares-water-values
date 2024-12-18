import numpy as np
import pytest

from functions_iterative import ReservoirManagement, TimeScenarioParameter
from multi_stock_bellman_value_calculation import *
from read_antares_data import Reservoir
from reservoir_management import MultiStockManagement


def test_bellman_value_precalculated_multi_stock() -> None:

    param = TimeScenarioParameter(len_week=5, len_scenario=1)

    reservoir_1 = Reservoir("test_data/two_nodes", "area_1")
    reservoir_management_1 = ReservoirManagement(
        reservoir=reservoir_1,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=True,
    )

    reservoir_2 = Reservoir("test_data/two_nodes", "area_2")
    reservoir_management_2 = ReservoirManagement(
        reservoir=reservoir_2,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=True,
    )

    multi_management = MultiStockManagement(
        [reservoir_management_1, reservoir_management_2]
    )

    levels, _, bellman_costs, bellman_controls, slopes, _ = precalculated_method(
        param=param,
        multi_stock_management=multi_management,
        output_path="test_data/two_nodes",
        xNsteps=5,
        Nsteps_bellman=5,
        name_solver="CLP",
        controls_looked_up="line+diagonal",
        verbose=True,
    )

    assert levels == pytest.approx(
        np.array(
            [
                [
                    [0.0, 628377.6569],
                    [277853.0681, 0.0],
                    [175340.436, 628377.6569],
                    [277853.0681, 396540.564],
                    [409896.721, 628377.6569],
                    [277853.0681, 927000.529],
                    [589466.8605, 628377.6569],
                    [277853.0681, 1333106.7645],
                    [769037.0, 628377.6569],
                    [277853.0681, 1739213.0],
                ],
                [
                    [0.0, 628377.6569],
                    [277853.0681, 0.0],
                    [178416.584, 628377.6569],
                    [277853.0681, 403497.416],
                    [486031.384, 628377.6569],
                    [277853.0681, 1099182.616],
                    [627534.192, 628377.6569],
                    [277853.0681, 1419197.808],
                    [769037.0, 628377.6569],
                    [277853.0681, 1739213.0],
                ],
                [
                    [0.0, 628377.6569],
                    [277853.0681, 0.0],
                    [180723.695, 628377.6569],
                    [277853.0681, 408715.055],
                    [488338.495, 628377.6569],
                    [277853.0681, 1104400.255],
                    [628687.7475, 628377.6569],
                    [277853.0681, 1421806.6275],
                    [769037.0, 628377.6569],
                    [277853.0681, 1739213.0],
                ],
                [
                    [0.0, 628377.6569],
                    [277853.0681, 0.0],
                    [183030.806, 628377.6569],
                    [277853.0681, 413932.694],
                    [491414.643, 628377.6569],
                    [277853.0681, 1111357.107],
                    [630225.8215, 628377.6569],
                    [277853.0681, 1425285.0535],
                    [769037.0, 628377.6569],
                    [277853.0681, 1739213.0],
                ],
                [
                    [0.0, 628377.6569],
                    [277853.0681, 0.0],
                    [185337.917, 628377.6569],
                    [277853.0681, 419150.333],
                    [493721.754, 628377.6569],
                    [277853.0681, 1116574.746],
                    [631379.377, 628377.6569],
                    [277853.0681, 1427893.873],
                    [769037.0, 628377.6569],
                    [277853.0681, 1739213.0],
                ],
            ]
        )
    )

    assert bellman_controls == pytest.approx(
        np.array(
            [
                [
                    [[0.0], [181530.9463735]],
                    [[129383.263], [-304244.94783314]],
                    [[26870.633], [181530.95748161]],
                    [[129383.263], [61185.617]],
                    [[160609.60919247], [124748.89981205]],
                    [[129383.263], [155975.24600452]],
                    [[219514.397], [124748.89981205]],
                    [[103827.55582937], [419664.0]],
                    [[306936.0], [124748.89981205]],
                    [[103827.55582937], [419664.0]],
                ],
                [
                    [[13776.0], [263393.096]],
                    [[217424.96382709], [-28367.50984833]],
                    [[52773.40570775], [144306.62825704]],
                    [[116288.634], [78091.78486466]],
                    [[193278.43809968], [197279.2502553]],
                    [[116288.634], [203738.091]],
                    [[259392.28273827], [263393.096]],
                    [[116288.634], [419664.0]],
                    [[306936.0], [263393.096]],
                    [[116288.634], [419664.0]],
                ],
                [
                    [[0.0], [256443.244]],
                    [[144342.19194326], [-294442.21953368]],
                    [[11011.94606618], [184605.26872985]],
                    [[146538.83712493], [41945.38354514]],
                    [[175156.44001992], [246067.35263216]],
                    [[86019.81249675], [419664.0]],
                    [[206364.00587073], [256443.244]],
                    [[86019.81249675], [419664.0]],
                    [[298585.86061946], [256443.244]],
                    [[113212.486], [419664.0]],
                ],
                [
                    [[0.0], [251428.605]],
                    [[138754.63237131], [-310705.16724054]],
                    [[16174.115], [79192.47066511]],
                    [[110996.375], [-16964.93476829]],
                    [[19371.83979269], [85278.13317882]],
                    [[0.0], [419664.0]],
                    [[158812.70351082], [95447.16468272]],
                    [[0.0], [419664.0]],
                    [[297623.88351082], [95447.16468272]],
                    [[0.0], [419664.0]],
                ],
                [
                    [[0.0], [-272320.61380624]],
                    [[34066.10375969], [-322182.0]],
                    [[14066.07670537], [-77354.29218551]],
                    [[12630.48903407], [-200568.15764549]],
                    [[153336.66361319], [31852.9091698]],
                    [[15945.4810059], [419664.0]],
                    [[290994.29361319], [31852.9091698]],
                    [[15945.4810059], [419664.0]],
                    [[306936.0], [31993.30673435]],
                    [[15945.4810059], [419664.0]],
                ],
            ]
        )
    )

    assert bellman_costs == pytest.approx(
        np.array(
            [
                [
                    1.00000010e16,
                    1.00000013e16,
                    1.00000000e16,
                    1.00000000e16,
                    1.00000000e16,
                    1.00000000e16,
                    1.00000005e16,
                    1.00000014e16,
                    1.00000014e16,
                    1.00000039e16,
                ],
                [
                    1.00000025e16,
                    1.00000041e16,
                    1.00000003e16,
                    1.00000002e16,
                    1.00000000e16,
                    1.00000001e16,
                    1.00000005e16,
                    1.00000013e16,
                    1.00000013e16,
                    1.00000048e16,
                ],
                [
                    1.00000034e16,
                    1.00000059e16,
                    1.00000003e16,
                    1.00000008e16,
                    1.00000001e16,
                    1.00000001e16,
                    1.00000005e16,
                    1.00000010e16,
                    1.00000009e16,
                    1.00000036e16,
                ],
                [
                    1.00000042e16,
                    1.00000061e16,
                    1.00000004e16,
                    1.00000011e16,
                    1.00000001e16,
                    1.00000001e16,
                    1.00000005e16,
                    1.00000010e16,
                    1.00000009e16,
                    1.00000034e16,
                ],
                [
                    1.00000050e16,
                    1.00000035e16,
                    1.00000000e16,
                    1.00000000e16,
                    1.00000000e16,
                    1.00000000e16,
                    1.00000004e16,
                    1.00000009e16,
                    1.00000008e16,
                    1.00000033e16,
                ],
            ]
        )
    )

    assert slopes == pytest.approx(
        np.array(
            [
                [
                    [-6.00000000e03, 0.00000000e00],
                    [-2.11898256e03, -6.00000000e03],
                    [-1.15170000e02, 0.00000000e00],
                    [-1.15170000e02, -1.15170000e02],
                    [0.00000000e00, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00],
                    [3.00000000e03, 0.00000000e00],
                    [0.00000000e00, 6.00000000e03],
                    [6.00000000e03, 0.00000000e00],
                    [0.00000000e00, 6.00000000e03],
                ],
                [
                    [-1.30000000e04, 0.00000000e00],
                    [-5.11898000e03, -1.20000000e04],
                    [-9.00000000e03, 0.00000000e00],
                    [-2.75281053e03, -3.11517000e03],
                    [-5.95800000e01, -5.95800000e01],
                    [-1.19160000e02, 0.00000000e00],
                    [3.00000000e03, -1.19160000e02],
                    [-1.19160000e02, 6.00000000e03],
                    [9.00000000e03, -1.19160000e02],
                    [-1.19160000e02, 1.20000000e04],
                ],
                [
                    [-1.90000000e04, 0.00000000e00],
                    [-7.04933707e03, -1.39835533e04],
                    [-2.75281035e03, -3.11516983e03],
                    [-5.85635493e03, -6.06353718e03],
                    [-1.22642731e02, -1.31127492e02],
                    [-1.66240000e02, 0.00000000e00],
                    [2.94042000e03, -3.20801251e02],
                    [-1.66240000e02, 3.00000000e03],
                    [3.00000000e03, -5.00000000e02],
                    [-1.66240000e02, 1.20000000e04],
                ],
                [
                    [-2.50000000e04, 0.00000000e00],
                    [-9.04534212e03, -1.33182190e04],
                    [-3.01822810e03, -3.11517000e03],
                    [-6.00811264e03, -6.06354000e03],
                    [-1.58015064e-01, -1.38714297e02],
                    [-1.66240000e02, 0.00000000e00],
                    [3.00000000e03, -1.38850000e02],
                    [-1.66240000e02, 3.00000000e03],
                    [3.00000000e03, -1.38850000e02],
                    [-1.66240000e02, 9.00000000e03],
                ],
                [
                    [-3.10000000e04, 0.00000000e00],
                    [-9.04534000e03, -1.93182200e04],
                    [-2.70028202e02, -1.13365718e02],
                    [-2.70028202e02, -1.13365718e02],
                    [0.00000000e00, 0.00000000e00],
                    [-1.66240000e02, 0.00000000e00],
                    [3.00000000e03, 0.00000000e00],
                    [-1.66240000e02, 3.00000000e03],
                    [3.00000000e03, 0.00000000e00],
                    [-1.66240000e02, 9.00000000e03],
                ],
            ]
        )
    )
