from read_antares_data import ResidualLoad
from gain_function_tempo import GainFunctionTempo
from bellman_and_usage_values_tempo import BellmanValuesTempo
import pytest

def test_bellman_values() -> None:
    """
    Test the Bellman_values class.
    """

    dir_study = "test_data/one_node_(1)"
    area = "area"

    residual_load = ResidualLoad(dir_study=dir_study, name_area=area)

    # test with tempo red
    gain_function_tempo_r = GainFunctionTempo(residual_load=residual_load, max_control=5)
    bellman_values_r = BellmanValuesTempo(gain_function=gain_function_tempo_r, capacity=22, nb_week=22, start_week=18)

    assert bellman_values_r.bv[18,0,0]== pytest.approx(0,0)
    assert bellman_values_r.bv[18,5,4]== pytest.approx(8064074,1)
    assert bellman_values_r.bv[38,22,9]==pytest.approx(5313571,673)
    assert bellman_values_r.mean_bv[21,4]==pytest.approx(6511952,507)

    # test with tempo white and red
    gain_function_tempo_wr = GainFunctionTempo(residual_load=residual_load, max_control=6)
    bellman_values_wr = BellmanValuesTempo(gain_function=gain_function_tempo_wr, capacity=65, nb_week=53, start_week=9)
    assert bellman_values_wr.bv[9,0,0]== pytest.approx(0,0)
    assert bellman_values_wr.bv[9,5,4]== pytest.approx(8090482,629)
    assert bellman_values_wr.bv[60,1,8]== pytest.approx(687190,6525)
    assert bellman_values_wr.mean_bv[37,12]==pytest.approx(11625650,36)