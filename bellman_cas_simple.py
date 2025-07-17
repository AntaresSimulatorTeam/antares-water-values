from multiprocessing import freeze_support

import numpy as np

from read_antares_data import Reservoir, generate_mps_file
from simple_bellman_value_calculation import (
    ReservoirManagement,
    TimeScenarioParameter,
    calculate_bellman_value_with_precalculated_reward,
)

if __name__ == "__main__":
    freeze_support()

    study_path = "D:/Users/gerbauxjul/Documents/6-Etudes Antares/1-vu/OneNodeBase_dev"
    output_path = generate_mps_file(
        study_path, "D:/AppliRTE/Antares/8.8/bin/antares-8.8-solver.exe"
    )
    # output_path = "D:/Users/gerbauxjul/Documents/6-Etudes Antares/1-vu/OneNodeBase_dev/output/20250717-1147eco-export_mps"
    # study_path = "D:/Users/gerbauxjul/Documents/6-Etudes Antares/1-vu/ERAA_scandinavie_v8_8"
    # output_path = study_path + "/output/20250331-0847exp-export_mps/mps_clean"

    solver = "XPRESS_LP"

    # Define reservoir parameters
    res = Reservoir(study_path, "area")
    # area = "2_nos0_hydro_open"
    # efficiency = 0.75

    bellman, reward, _, _, _, _ = calculate_bellman_value_with_precalculated_reward(
        len_controls=11,
        param=TimeScenarioParameter(10, 1),
        reservoir_management=ReservoirManagement(res, 0, 0, 0, False, None, True),
        output_path=output_path,
        X=np.linspace(0, res.capacity, 11),
        solver=solver,
    )
    np.savetxt("bellman_simple.txt", bellman)
