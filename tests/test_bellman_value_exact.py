import numpy as np
import pytest

from functions_iterative import (
    ReservoirManagement,
    TimeScenarioIndex,
    TimeScenarioParameter,
)
from read_antares_data import Reservoir
from simple_bellman_value_calculation import calculate_bellman_value_directly


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

    vb = calculate_bellman_value_directly(
        param=param,
        reservoir_management=reservoir_management,
        output_path="test_data/one_node",
        X=X,
    )

    assert vb == pytest.approx(
        np.array(
            [
                [
                    -5.8881910e09,
                    -5.3715830e09,
                    -4.3035453e09,
                    -3.5937500e09,
                    -1.9985748e09,
                    0.0000000e00,
                ],
                [
                    -5.2848681e09,
                    -4.3627935e09,
                    -3.3862774e09,
                    -2.4232980e09,
                    -1.4029120e09,
                    0.0000000e00,
                ],
                [
                    -5.1269724e09,
                    -4.2048975e09,
                    -3.2283817e09,
                    -2.2654021e09,
                    -1.2450163e09,
                    0.0000000e00,
                ],
                [
                    -4.9844623e09,
                    -4.0473820e09,
                    -3.0742531e09,
                    -2.1075066e09,
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
                    -2.8301327e09,
                    -1.9763548e09,
                    -1.1332330e09,
                    -2.7673613e08,
                    0.0000000e00,
                ],
                [
                    -3.7044485e09,
                    -2.7775009e09,
                    -1.9237230e09,
                    -1.0806013e09,
                    -2.2410518e08,
                    0.0000000e00,
                ],
                [
                    -3.6518167e09,
                    -2.7248694e09,
                    -1.8710912e09,
                    -1.0279697e09,
                    -1.7147429e08,
                    0.0000000e00,
                ],
                [
                    -3.5991852e09,
                    -2.6722376e09,
                    -1.8184595e09,
                    -9.7533805e08,
                    -1.1884343e08,
                    0.0000000e00,
                ],
                [
                    -3.5465536e09,
                    -2.6196058e09,
                    -1.7658280e09,
                    -9.2270643e08,
                    -6.6212612e07,
                    0.0000000e00,
                ],
            ]
        )
    )
