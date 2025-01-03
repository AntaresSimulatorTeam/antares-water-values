import numpy as np
from pytest import approx

from estimation import LinearDecomposer, LinearInterpolator, decompose_hyperplanes


def test_init() -> None:

    controls = np.array([[0, 0], [1, 1], [0, 1]])
    costs = np.array([0, 10, 7])
    duals = np.array([[1, 1], [5, 5], [2, 3]])

    linear_decomposer = LinearDecomposer(controls, costs, duals)

    assert len(linear_decomposer.layers) == 2

    assert linear_decomposer.inputs == approx(
        np.array(
            [
                [0.00000000e00, 0.00000000e00],
                [1.00000000e00, 1.00000000e00],
                [0.00000000e00, 1.00000000e00],
                [0.00000000e00, 0.00000000e00],
                [1.11022302e-16, 1.11022302e-16],
                [-1.07692308e00, -6.15384615e-01],
                [0.00000000e00, 0.00000000e00],
                [1.11022302e-16, 1.11022302e-16],
                [-1.07692308e00, -6.15384615e-01],
            ]
        )
    )
    assert linear_decomposer.costs == approx(
        np.array([0.0, 10.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert linear_decomposer.duals == approx(
        np.array(
            [
                [1.0, 1.0],
                [5.0, 5.0],
                [2.0, 3.0],
                [1.0, 0.0],
                [5.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [0.0, 5.0],
                [0.0, 3.0],
            ]
        )
    )

    assert linear_decomposer(np.array([1, 0])) == 6


def test_remove_inconsistence() -> None:

    controls = np.array([[0, 0], [1, 1], [0, 1]])
    costs = np.array([0, 10, 7])
    duals = np.array([[1, 1], [5, 5], [2, 3]])

    inputs_decomp, _, duals_decomp = decompose_hyperplanes(
        inputs=controls, costs=costs, slopes=duals, correlations=np.eye(2)
    )

    layers: list[LinearInterpolator] = [
        LinearInterpolator(inp, np.zeros(inp.shape[0]), slp)
        for inp, slp in zip(inputs_decomp, duals_decomp)
    ]
    assert len(layers) == 2
    for i in range(2):
        assert layers[i].inputs == approx(
            np.array(
                [
                    [0.00000000e00, 0.00000000e00],
                    [1.11022302e-16, 1.11022302e-16],
                    [-1.07692308e00, -6.15384615e-01],
                ]
            )
        )

    tolerance = 0.01

    expected_bad_guesses = [
        np.array([True, False, True]),
        np.array([True, False, True]),
    ]
    expected_first_pb_inp = [np.array([0, 0]), np.array([0, 0])]

    assert all(costs + tolerance > 0)
    guesses = np.array([layer(controls) for layer in layers])  # N_res * N_inp
    assert guesses == approx(
        np.array([[2.15384615, 5.0, 2.15384615], [1.84615385, 5.0, 5.0]])
    )
    for i in range(10):
        if not any(np.sum(guesses, axis=0) > costs + tolerance):
            break
        # Identify likely source of error
        bad_guesses = np.sum(guesses, axis=0) > costs + tolerance
        assert bad_guesses == approx(expected_bad_guesses[i])
        # Removing first potential source of pb
        first_pb_inp = controls[bad_guesses][0]
        assert first_pb_inp == approx(expected_first_pb_inp[i])
        bad_lay = [layer for layer in layers if layer(first_pb_inp) > 0][-1]
        bad_lay.remove(bad_lay.get_owner(first_pb_inp))
        guesses = np.array([layer(controls) for layer in layers])  # N_res * N_inp

    assert len(layers) == 2
    assert layers[0].inputs == approx(
        np.array(
            [
                [0.00000000e00, 0.00000000e00],
                [1.11022302e-16, 1.11022302e-16],
            ]
        )
    )
    assert layers[1].inputs == approx(
        np.array(
            [
                [0.00000000e00, 0.00000000e00],
                [1.11022302e-16, 1.11022302e-16],
            ]
        )
    )


def test_lower_bound() -> None:

    controls = np.array([[0, 0], [1, 1], [0, 1]])
    costs = np.array([0, 10, 7])
    duals = np.array([[1, 1], [5, 5], [2, 3]])

    lower_bound = LinearInterpolator(controls=controls, costs=costs, duals=duals)

    assert lower_bound(np.array([0, 0])) == 4
    assert lower_bound(np.array([1, 0])) == 6
