from read_antares_data import NetLoad

def test_net_load() -> None:
    dir_study = "test_data/one_node_(1)"
    area = "area"

    net_load=NetLoad(dir_study=dir_study, name_area=area)

    # check time window (1 year + 2 months = 10296 hours)
    assert net_load.net_load.shape[0] == 10296
    # check number of scenarios
    assert net_load.net_load.shape[1] == 10

