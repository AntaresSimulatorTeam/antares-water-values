from read_antares_data import Reservoir


def test_create_reservoir() -> None:
    reservoir = Reservoir(dir_study="test_data/one_node", name_area="area")

    assert reservoir.capacity == 1e7
    assert reservoir.efficiency == 1.0
    assert reservoir.initial_level == 0.445 * 1e7
    assert reservoir.inflow[2, 0] == 4509 * 7
    assert len(reservoir.max_generating) == 52
    assert reservoir.max_generating[0] == 8400000
    assert reservoir.max_pumping[0] == 8400000
