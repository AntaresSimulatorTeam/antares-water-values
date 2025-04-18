from read_antares_data import ResidualLoad
import numpy as np
from gain_function import GainFunctionTempo
from bellman_values import BellmanValues
from usage_values import UsageValuesTempo
from trajectories import TrajectoriesTempo
import plotly.graph_objects as go

def test_trajectories_tempo() -> None:

    dir_study = "test_data/one_node(1)"
    area = "area"

    residual_load=ResidualLoad(dir_study=dir_study,name_area=area)

    gain_function_tempo_r=GainFunctionTempo(residual_load=residual_load,max_control=5)
    gain_function_tempo_wr=GainFunctionTempo(residual_load=residual_load,max_control=6)

    bellman_values_r=BellmanValues(gain_function=gain_function_tempo_r,capacity=22,nb_week=22,start_week=18)
    bellman_values_wr=BellmanValues(gain_function=gain_function_tempo_wr,capacity=65,nb_week=53,start_week=9)

    usage_values_r=UsageValuesTempo(bellman_values=bellman_values_r)
    usage_values_wr=UsageValuesTempo(bellman_values=bellman_values_wr)

    trajectories_r=TrajectoriesTempo(usage_values=usage_values_r)
    trajectories_white_and_red=TrajectoriesTempo(usage_values=usage_values_wr,trajectories_red=trajectories_r.trajectories)
    
    # white trajectories is calculated from trajectories_white_and_red
    trajectories_white=trajectories_white_and_red.trajectories_white


    assert np.array_equal(trajectories_r.trajectory_for_scenario(4), np.array([[0,0,1,4,4,1,1,0,0,2,0,0,0,0,2,1,0,0,2,2,2,0]]))
    assert np.array_equal(trajectories_white_and_red.white_trajectory_for_scenario(4), np.array([[0,0,0,0,0,0,0,0,1,1,2,2,2,2,4,3,1,0,2,2,0,0,0,1,1,0,0,0,0,1,0,1,0,5,1,0,0,0,0,0,0,0,0,0,4,2,0,0,2,0,1,2,0]]))

    assert np.array_equal(trajectories_r.stock_for_scenario(4), np.array([[22,22,22,21,17,13,12,11,11,11,9,9,9,9,9,7,6,6,6,4,2,0]]))
    assert np.array_equal(trajectories_white_and_red.stock_for_scenario(4), np.array([[65,65,65,65,65,65,65,65,65,64,63,61,58,52,46,41,37,36,36,32,30,30,30,30,27,25,25,25,23,21,18,18,17,17,12,11,11,11,11,11,11,11,11,11,11,7,5,5,5,3,3,2,0
]]))
    assert np.array_equal(trajectories_white_and_red.white_stock_for_scenario(4), np.array([[43,43,43,43,43,43,43,43,43,42,41,39,37,35,33,29,26,25,25,23,21,21,21,21,20,19,19,19,19,19,18,18,17,17,12,11,11,11,11,11,11,11,11,11,11,7,5,5,5,3,3,2,0
]]))
    