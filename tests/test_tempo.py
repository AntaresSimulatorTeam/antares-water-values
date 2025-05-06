from read_antares_data import NetLoad
from tempo import GainFunctionTempo
from tempo import BellmanValuesTempo
from tempo import TrajectoriesTempo
import numpy as np
import pytest

dir_study = "test_data/one_node_(1)"
area = "area"
net_load = NetLoad(dir_study=dir_study, name_area=area)
gain_function_tempo_r = GainFunctionTempo(net_load=net_load, max_control=5)
bellman_values_r = BellmanValuesTempo(gain_function=gain_function_tempo_r, capacity=22, nb_week=22, start_week=18)
gain_function_tempo_wr = GainFunctionTempo(net_load=net_load, max_control=6)
bellman_values_wr = BellmanValuesTempo(gain_function=gain_function_tempo_wr, capacity=65, nb_week=53, start_week=9)
trajectories_r=TrajectoriesTempo(bv=bellman_values_r)
trajectories_white_and_red=TrajectoriesTempo(bv=bellman_values_wr,trajectories_red=trajectories_r.trajectories)
    
# white trajectories is calculated from trajectories_white_and_red
trajectories_white=trajectories_white_and_red.trajectories_white

def test_gain_gunction_tempo() -> None:
    # Test with max_control = 5
    gain_function_tempo_r = GainFunctionTempo(net_load=net_load, max_control=5)
    assert gain_function_tempo_r.gain_for_week_control_and_scenario(18,5,0) == pytest.approx(4035975,878)
    assert gain_function_tempo_r.gain_for_week_control_and_scenario(22,3,1) == pytest.approx(3017476,155)

    # Test with max_control = 6
    assert gain_function_tempo_wr.gain_for_week_control_and_scenario(9,6,0) == pytest.approx(3871795,528)
    assert gain_function_tempo_wr.gain_for_week_control_and_scenario(18,5,0) == pytest.approx(4316780,39)
    assert gain_function_tempo_wr.gain_for_week_control_and_scenario(27,3,4) == pytest.approx(4274167,815)

def test_bellman_values() -> None:

    # test with tempo red

    assert bellman_values_r.bv[18,0,0]== pytest.approx(0,0)
    assert bellman_values_r.bv[18,5,4]== pytest.approx(8064074,1)
    assert bellman_values_r.bv[38,22,9]==pytest.approx(5313571,673)
    assert bellman_values_r.mean_bv[21,4]==pytest.approx(6511952,507)

    # test with tempo white and red
    
    assert bellman_values_wr.bv[9,0,0]== pytest.approx(0,0)
    assert bellman_values_wr.bv[9,5,4]== pytest.approx(8090482,629)
    assert bellman_values_wr.bv[60,1,8]== pytest.approx(687190,6525)
    assert bellman_values_wr.mean_bv[37,12]==pytest.approx(11625650,36)

def test_usage_values() -> None:
    assert bellman_values_r.usage_values[24,2]== pytest.approx(1634285,504) 
    assert bellman_values_wr.usage_values[35,11]== pytest.approx(959545,2878)

def test_trajectories_tempo() -> None:

    assert np.array_equal(trajectories_r.trajectory_for_scenario(4), np.array([0,0,1,4,4,1,1,0,0,2,0,0,0,0,2,1,0,0,2,2,2]).astype(float))
    assert np.array_equal(trajectories_white_and_red.white_trajectory_for_scenario(4), np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 4, 3, 1, 0, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 0, 1, 2]
).astype(float))

    assert np.array_equal(trajectories_r.stock_for_scenario(4), np.array([22,22,22,21,17,13,12,11,11,11,9,9,9,9,9,7,6,6,6,4,2,0]).astype(float))
    assert np.array_equal(trajectories_white_and_red.stock_for_scenario(4), np.array([65,65,65,65,65,65,65,65,65,64,63,61,58,52,46,41,37,36,36,32,30,30,30,30,27,25,25,25,23,21,18,18,17,17,12,11,11,11,11,11,11,11,11,11,11,7,5,5,5,3,3,2,0
]).astype(float))
    assert np.array_equal(trajectories_white_and_red.white_stock_for_scenario(4), np.array([43,43,43,43,43,43,43,43,43,42,41,39,37,35,33,29,26,25,25,23,21,21,21,21,20,19,19,19,19,19,18,18,17,17,12,11,11,11,11,11,11,11,11,11,11,7,5,5,5,3,3,2,0
]).astype(float))
    