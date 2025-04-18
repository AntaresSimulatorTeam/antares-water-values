from read_antares_data import ResidualLoad

def test_residual_load() -> None:
    """
    Test the Residual_load class.
    """
    dir_study = "test_data/one_node_(1)"
    area = "area"

    residual_load=ResidualLoad(dir_study=dir_study, name_area=area)

    # check time window (1 year + 2 months = 10296 hours)
    assert residual_load.residual_load.shape[0] == 10296
    # check number of scenarios
    assert residual_load.residual_load.shape[1] == 10

