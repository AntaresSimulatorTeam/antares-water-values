import numpy as np
from numpy.linalg import norm
from type_definition import Array1D


def decompose_hyperplanes(
    inputs: np.ndarray, costs: Array1D, slopes: np.ndarray, correlations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Getting the orthogonal proj of input on intersection of hyperp. with 0
    n_inps, n_reservoirs = inputs.shape[0], inputs.shape[-1]
    xmid = (
        inputs
        - costs[:, None]
        * slopes
        / np.power(np.maximum(1e-4, norm(slopes, axis=1)), 2)[:, None]
    )  # N_inputs * R
    decomp_slopes = (
        slopes[:, None, :] * correlations / np.sum(correlations, axis=0)[None, None, :]
    )  # N_inputs x R x R
    xmids = np.tile(xmid, n_reservoirs).reshape(
        n_inps, n_reservoirs, n_reservoirs
    )  # N_inputs x R x R
    return xmids, np.zeros(inputs.shape[:-1]), decomp_slopes
