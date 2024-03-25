from functions_iterative import Reservoir, AntaresParameter
import pytest


def test_create_reservoir() -> None:
    reservoir = Reservoir(dir_study="test_data/one_node", name_area="area")

    assert reservoir.capacity == 1e7
    assert reservoir.efficiency == 1.0
    assert reservoir.initial_level == 0.445 * 1e7
    assert reservoir.inflow[2, 0] == 4509 / 24
