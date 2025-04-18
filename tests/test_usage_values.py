from read_antares_data import Residual_load
from gain_function_tempo import GainFunctionTEMPO
from bellman_values import Bellman_values
from usage_values import UV_tempo
import pytest

def test_usage_values() -> None:
    """
    Test the UV_tempo class.
    """

    dir_study = "test_data/one_node(1)"
    area = "area"

    residual_load = Residual_load(dir_study=dir_study, name_area=area)

    # test with tempo red
    gain_function_tempo_r = GainFunctionTEMPO(residual_load=residual_load, max_control=5)
    bellman_values_r = Bellman_values(gain_function=gain_function_tempo_r, capacity=22, nb_week=22, start_week=18)
    usage_values_r = UV_tempo(bellman_values=bellman_values_r)

    assert usage_values_r.usage_values[24,2]== pytest.approx(1634285,504)

    # test with tempo white and red
    gain_function_tempo_wr = GainFunctionTEMPO(residual_load=residual_load, max_control=6)
    bellman_values_wr = Bellman_values(gain_function=gain_function_tempo_wr, capacity=65, nb_week=53, start_week=9)
    usage_values_wr = UV_tempo(bellman_values=bellman_values_wr)
    
    assert usage_values_wr.usage_values[35,11]== pytest.approx(959545,2878)