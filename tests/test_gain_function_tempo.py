from read_antares_data import ResidualLoad
from gain_function_tempo import GainFunctionTempo
import pytest

def test_gain_gunction_tempo() -> None:
    """
    Test the GainFunctionTEMPO class.
    """

    dir_study = "test_data/one_node(1)"
    area = "area"

    residual_load = ResidualLoad(dir_study=dir_study, name_area=area)

    # Test with max_control = 5
    gain_function_tempo_r = GainFunctionTempo(residual_load=residual_load, max_control=5)
    assert gain_function_tempo_r.gain_for_week_control_and_scenario(18,5,0) == pytest.approx(4035975,878)
    assert gain_function_tempo_r.gain_for_week_control_and_scenario(22,3,1) == pytest.approx(3017476,155)

    # Test with max_control = 6
    gain_function_tempo_wr = GainFunctionTempo(residual_load=residual_load, max_control=6)
    assert gain_function_tempo_wr.gain_for_week_control_and_scenario(9,6,0) == pytest.approx(3871795,528)
    assert gain_function_tempo_wr.gain_for_week_control_and_scenario(18,5,0) == pytest.approx(4316780,39)
    assert gain_function_tempo_wr.gain_for_week_control_and_scenario(27,3,4) == pytest.approx(4274167,815)
