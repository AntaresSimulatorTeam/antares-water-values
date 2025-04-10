from pathlib import Path
from read_antares_data import Residual_load
import numpy as np
from calculate_gain_func_tempo_red import GainFunction
from calculate_bellman_values_with_reward import Bellman_values

dir_study="C:/Users/brescianomat/Documents/Etudes Antares/BP23_A-Reference_2036"
area="fr"

residual_load_1=Residual_load(dir_study=dir_study,name_area=area)
gain_func=GainFunction(residual_load=residual_load_1)
bellman=Bellman_values(residual_load=residual_load_1,gain_function=gain_func,capacity=23)

# print(residual_load_1.load.reshape(365,24,200).sum(axis=1))
print(gain_func.daily_residual_load)
# bellman.compute_bellman_values()
# print(bellman.mean_bv)