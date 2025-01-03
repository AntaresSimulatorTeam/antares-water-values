import numpy as np
from pytest import approx

from hyperplane_decomposition import decompose_hyperplanes


def test_decompose_hyperplanes() -> None:

    controls = np.array([[0, 0], [1, 1], [0, 1]])
    costs = np.array([0, 10, 7])
    duals = np.array([[1, 1], [5, 5], [2, 3]])
    correlations = np.eye(2)

    inputs_decomp, costs_decomp, duals_decomp = decompose_hyperplanes(
        inputs=controls,
        costs=costs,
        slopes=duals,
        correlations=correlations,
    )

    assert inputs_decomp.shape == (2, 3, 2)
    assert costs_decomp.shape == (3,)
    assert duals_decomp.shape == (2, 3, 2)

    assert inputs_decomp == approx(
        np.array(
            [
                [
                    [0.00000000e00, 0.00000000e00],
                    [1.11022302e-16, 1.11022302e-16],
                    [-1.07692308e00, -6.15384615e-01],
                ],
                [
                    [0.00000000e00, 0.00000000e00],
                    [1.11022302e-16, 1.11022302e-16],
                    [-1.07692308e00, -6.15384615e-01],
                ],
            ]
        )
    )
    assert costs_decomp == approx(np.array([0.0, 0.0, 0.0]))
    assert duals_decomp == approx(
        np.array(
            [[[1.0, 0.0], [5.0, 0.0], [2.0, 0.0]], [[0.0, 1.0], [0.0, 5.0], [0.0, 3.0]]]
        )
    )
