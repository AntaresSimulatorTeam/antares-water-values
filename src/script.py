import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from functions_iterative import *
from read_antares_data import generate_mps_file
from simple_bellman_value_calculation import (
    calculate_bellman_value_with_precalculated_reward,
    calculate_bellman_value_directly,
    calculate_complete_reward,
)
from read_antares_data import Reservoir
from multiprocessing import freeze_support, cpu_count, Pool
from functions_iterative_without_stored_models import (
    calculate_bellman_values_with_iterative_method_without_stored_models,
)

param = TimeScenarioParameter(
    len_week=52,
    len_scenario=10,
    name_scenario=[11, 17, 37, 52, 67, 99, 105, 154, 157, 179],
)
reservoir = Reservoir(
    "D:/Users/gerbauxjul/Documents/6-Etudes Antares/facteur_de_charge", "fc_eu"
)

reservoir_management = ReservoirManagement(
    reservoir=reservoir,
    penalty_bottom_rule_curve=0,
    penalty_upper_rule_curve=0,
    penalty_final_level=200,
    force_final_level=True,
    overflow=False,
)

output_path = (
    "D:/Users/gerbauxjul/Documents/6-Etudes Antares/facteur_de_charge/mps_heuristic"
)

xNsteps = 101
X = np.linspace(0, reservoir.capacity, num=xNsteps)


if __name__ == "__main__":
    freeze_support()

    vb, G, itrerations, tot_t, controls_upper, traj = (
        calculate_bellman_values_with_iterative_method_without_stored_models(
            param=param,
            reservoir_management=reservoir_management,
            output_path=output_path,
            X=X,
            N=10,
            tol_gap=1e-3,
            solver="CLP",
            processes=5,
        )
    )

    np.savetxt("dev/bellman_values_iterative_mean", vb)

    with open("dev/save_reward_iterative_mean.txt", "w") as file:
        for week in range(param.len_week):
            for scenario in range(param.len_scenario):
                r = G[TimeScenarioIndex(week, scenario)]
                file.write(f"{week} {scenario}")
                file.write("\n")
                file.write(str(r.list_cut))
                file.write("\n")
                file.write(str(r.breaking_point))
                file.write("\n")
                file.write("\n")

    for itr, controls in enumerate(controls_upper):
        np.savetxt(f"dev/save_controls_upper_iteration_{itr}_mean", controls)

    for itr, t in enumerate(traj):
        np.savetxt(f"dev/save_trajectories_iteration_{itr}_mean", t)
