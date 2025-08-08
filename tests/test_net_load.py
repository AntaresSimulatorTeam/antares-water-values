from read_antares_data import NetLoad

def test_net_load() -> None:
    dir_study = "test_data/one_node_(1)"
    area = "area"

    net_load=NetLoad(dir_study=dir_study, name_area=area)

    assert net_load.compute_net_load().shape[0] == 8760
    # check number of scenarios
    assert net_load.compute_net_load().shape[1] == 200

