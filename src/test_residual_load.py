from pathlib import Path
from read_antares_data import Residual_load
import numpy as np
from calculate_gain_func_tempo_red import GainFunction
from calculate_bellman_values_with_reward_func import Bellman_values
from calculate_uv_tempo import UV_tempo_red
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

dir_study="C:/Users/brescianomat/Documents/Etudes Antares/BP23_A-Reference_2036"
area="fr"

residual_load_1=Residual_load(dir_study=dir_study,name_area=area)
gain_func=GainFunction(residual_load=residual_load_1)
bellman=Bellman_values(residual_load=residual_load_1,gain_function=gain_func,capacity=23,nb_week=22)
uv=UV_tempo_red(residual_load=residual_load_1,gain_function=gain_func,bellman_values=bellman,capacity=23,nb_week=22)

# print(residual_load_1.load.reshape(365,24,200).sum(axis=1))
# print(gain_func.daily_residual_load)
# print(gain_func.gain_for_week_control_and_scenario(38,5,0))
# bellman.compute_bellman_values()
# print(bellman.bv[18:40,:,:])
# print(bellman.mean_bv[18:40,:])
# gain_func.gain_for_week_control_and_scenario(38,0,0)
# print(gain_func.daily_residual_load_for_week)
valeurs_usage=uv.usage_values[18:40,:]



# Stock restant de 1 à 22 (si on ignore 0)
x = np.arange(1, bellman.capacity)  # [1, 2, ..., 22]


# Génération de la figure
fig, ax = plt.subplots(figsize=(14, 7))


for w in range(bellman.nb_week):
    y = valeurs_usage[w, :]  # on ignore la capacité 0 pour correspondre à x
    ax.plot(x, y, label=f"Semaine {44 + w}")

ax.set_title("Valeur d’usage en fonction du stock restant (H) par semaine")
ax.set_xlabel("Stock restant (H)")
ax.set_ylabel("Valeur d’usage")
ax.set_xticks(x)
ax.legend(title="Semaine", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.grid(True)
plt.tight_layout()
plt.show()