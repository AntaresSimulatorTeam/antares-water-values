import pytest
import numpy as np
from read_antares_data import Reservoir,NetLoad
from proxy_tempo import GainFunctionTempo,BellmanValuesTempo,TrajectoriesTempo

dir_study = "test_data/two_nodes"
area = "area1"

net_load = NetLoad(reservoir=Reservoir(dir_study=dir_study,name_area=area), dir_study=dir_study, name_area=area)
gain_function = GainFunctionTempo(net_load=net_load)
bellman_values_red = BellmanValuesTempo(gain_function=gain_function,
                                    capacity=22,
                                    start_week=18,
                                    end_week=38,
                                    max_control=5)
trajectories_red = TrajectoriesTempo(bv=bellman_values_red)

bellman_values_white_and_red = BellmanValuesTempo(gain_function=gain_function,
                                                  capacity=65,
                                                  start_week=9,
                                                  end_week=60,
                                                  max_control=6)
trajectories_white_and_red = TrajectoriesTempo(bv=bellman_values_white_and_red,
                                               stock_trajectories_red=trajectories_red.stock_trajectories)

def test_gain_function_tempo()->None:
    assert gain_function.gain_for_week_control_and_scenario(week_index=18,control=5,scenario=0,max_control=5)==pytest.approx(4559276.533)
    assert gain_function.gain_for_week_control_and_scenario(week_index=28,control=3,scenario=5,max_control=5)==pytest.approx(2493107.2824999997)
    assert gain_function.gain_for_week_control_and_scenario(week_index=30,control=0,scenario=9,max_control=5)==pytest.approx(0)
    assert gain_function.gain_for_week_control_and_scenario(week_index=10,control=6,scenario=0,max_control=6)==pytest.approx(4480186.672499999)

def test_bellman_values_tempo()->None:
    assert bellman_values_red.mean_bv.shape==(61,23) 
    assert bellman_values_red.mean_bv[25,10]==pytest.approx(14886250.286077848)
    assert bellman_values_red.mean_bv[38]==pytest.approx(np.zeros(23))
    assert bellman_values_red.mean_bv[18]==pytest.approx(np.array([
            0,
            1635463.0805157232,
            3228041.4476605295,
            4791119.439975521,
            6329318.290087402,
            7848022.742237193,
            9343043.539659884,
            10814433.293034298,
            12269016.144185368,
            13705500.936216895,
            15123858.057936007,
            16524442.690887073,
            17911563.972539164,
            19284557.55051198,
            20643440.72810308,
            21988573.63809376,
            23319167.41347275,
            24636573.917794164,
            25942181.54194791,
            27236704.665245306,
            28520379.91386708,
            29792372.27432084,
            31052861.257673346]
            ))

def test_optimal_trajectories()->None:
    assert trajectories_red.stock_trajectories.shape==(10,61)
    assert trajectories_red.control_trajectories.shape==(10,61)
    assert trajectories_red.stock_trajectories[0,:18]==pytest.approx(np.repeat(22,18))
    expected_stock_trajectory = np.array([
            22.0,     
            21.0,  
            17.0,   
            14.0,     
            14.0,     
            14.0,     
            11.0,     
            9.0 ,     
            5.0 ,     
            5.0 ,     
            2.0 ,     
            1.0 ,     
            1.0 ,     
            1.0 ,
            0,
            0,
            0,
            0,
            0,
            0,
            0
    ])
    assert trajectories_red.stock_trajectories[0,18:39]==pytest.approx(expected_stock_trajectory)
    assert trajectories_red.stock_trajectories[0]==pytest.approx(
        trajectories_red.stock_trajectory_for_scenario(0)
    )
    assert trajectories_red.stock_trajectories[0,39:]==pytest.approx(np.repeat(0,22))

    expected_control_trajectory = np.array([
            0.0,
            1.0,
            4.0,
            3.0,
            0.0,
            0.0,
            3.0,
            2.0,
            4.0,
            0.0,
            3.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0
    ])
    assert trajectories_red.control_trajectories[0,18:39]==pytest.approx(expected_control_trajectory)
    assert trajectories_red.control_trajectories[0]==pytest.approx(
        trajectories_red.control_trajectory_for_scenario(0)
    )
    assert trajectories_red.control_trajectories[0,:18]==pytest.approx(np.zeros(18))
    assert trajectories_red.control_trajectories[0,39:]==pytest.approx(np.zeros(22))


def test_white_and_red_trajectories()->None:
    expected_white_trajectory = np.array([
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            43.0     	,
            40.0     	,
            39.0     	,
            39.0     	,
            36.0     	,
            35.0     	,
            34.0     	,
            31.0     	,
            30.0     	,
            27.0     	,
            23.0     	,
            23.0     	,
            21.0     	,
            19.0     	,
            17.0     	,
            15.0     	,
            12.0     	,
            8.0      	,
            8.0      	,
            3.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	,
            0.0      	

    ])
    assert np.all(trajectories_white_and_red.stock_trajectories>=trajectories_red.stock_trajectories)
    assert trajectories_white_and_red.stock_trajectory_for_scenario_white(0)==pytest.approx(expected_white_trajectory)
    assert np.all(
        trajectories_white_and_red.stock_trajectory_for_scenario_white(0)+trajectories_red.stock_trajectory_for_scenario(0)==trajectories_white_and_red.stock_trajectory_for_scenario(0))
