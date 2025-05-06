from read_antares_data import ResidualLoad
from gain_function_tempo import GainFunctionTempo
from bellman_and_usage_values_tempo import BellmanValuesTempo
import pytest

def test_usage_values() -> None:
    """
    Test the UV_tempo class.
    """

    dir_study = "test_data/one_node_(1)"
    area = "area"

    residual_load = ResidualLoad(dir_study=dir_study, name_area=area)

    # test with tempo red
    gain_function_tempo_r = GainFunctionTempo(residual_load=residual_load, max_control=5)
    bellman_values_r = BellmanValuesTempo(gain_function=gain_function_tempo_r, capacity=22, nb_week=22, start_week=18)

    assert bellman_values_r.usage_values[24,2]== pytest.approx(1634285,504)

    # test with tempo white and red
    gain_function_tempo_wr = GainFunctionTempo(residual_load=residual_load, max_control=6)
    bellman_values_wr = BellmanValuesTempo(gain_function=gain_function_tempo_wr, capacity=65, nb_week=53, start_week=9)
    
    assert bellman_values_wr.usage_values[35,11]== pytest.approx(959545,2878)