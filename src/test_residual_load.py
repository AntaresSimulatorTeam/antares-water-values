from pathlib import Path
from read_antares_data import Residual_load
import numpy as np
from gain_function_TEMPO import GainFunctionTEMPO
from bellman_values import Bellman_values
from usage_values import UV_tempo
from trajectories import Trajectories
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

dir_study="C:/Users/brescianomat/Documents/Etudes Antares/BP23_A-Reference_2036"
area="fr"

residual_load_1=Residual_load(dir_study=dir_study,name_area=area)

gain_func_red=GainFunctionTEMPO(residual_load=residual_load_1,max_control=5)
bellman_red=Bellman_values(residual_load=residual_load_1,
                           gain_function=gain_func_red,
                           capacity=22,
                           nb_week=22,
                           max_control=5,
                           start_week=18)
uv_red=UV_tempo(residual_load=residual_load_1,
                    gain_function=gain_func_red,
                    bellman_values=bellman_red)

gain_func_white_and_red=GainFunctionTEMPO(residual_load=residual_load_1,max_control=6)
bellman_white_and_red=Bellman_values(residual_load=residual_load_1,
                                     gain_function=gain_func_white_and_red,
                                     capacity=65,
                                     nb_week=52,
                                     max_control=6,
                                     start_week=9)
uv_white_and_red=UV_tempo(residual_load=residual_load_1,
                              gain_function=gain_func_white_and_red,
                              bellman_values=bellman_white_and_red)


trajectories_red= Trajectories(
    residual_load=residual_load_1,
    gain_function=gain_func_red,
    bellman_values=bellman_red,
    usage_values=uv_red,
    capacity=22,
    nb_week=22,
    max_control=5
)



trajectories_white_and_red = Trajectories(
    residual_load=residual_load_1,
    gain_function=gain_func_white_and_red,
    bellman_values=bellman_white_and_red,
    usage_values=uv_white_and_red,
    capacity=65,
    nb_week=52,
    max_control=6
)

