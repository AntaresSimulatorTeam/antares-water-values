from dataclasses import dataclass
import subprocess
from configparser import ConfigParser
import numpy as np


@dataclass
class AntaresParameter:
    S: int = 52
    H: int = 168
    NTrain: int = 1


class Reservoir:
    """Describes reservoir parameters"""

    def __init__(
        self,
        param: AntaresParameter,
        dir_study: str,
        name_area: str,
        final_level: bool = True,
    ) -> None:
        """
        Create a new reservoir.

        Parameters
        ----------
        param:AntaresParameter :
            Time-related parameters
        capacity:float :
            Capacity of the reservoir (in MWh)
        efficiency:float :
            Efficiency of the pumping (between 0 and 1)
        dir_study:str :
            Path to the Antares study
        name_area:str :
            Name of the area where is located the reservoir
        name:str :
            Name of the reservoir
        final_level:bool :
            True if final level should be egal to initial level (Default value = True)

        Returns
        -------
        None
        """

        H = param.H

        hydro_ini = ConfigParser()
        hydro_ini.read(dir_study + "/input/hydro/hydro.ini")

        self.capacity = hydro_ini.getfloat("reservoir capacity", name_area)

        courbes_guides = (
            np.loadtxt(
                dir_study
                + "/input/hydro/common/capacity/reservoir_"
                + name_area
                + ".txt"
            )[:, [0, 2]]
            * self.capacity
        )
        assert courbes_guides[0, 0] == courbes_guides[0, 1]
        self.initial_level = courbes_guides[0, 0]
        Xmin = courbes_guides[6:365:7, 0]
        Xmax = courbes_guides[6:365:7, 1]
        self.Xmin = np.concatenate((Xmin, Xmin[[0]]))
        self.Xmax = np.concatenate((Xmax, Xmax[[0]]))
        if final_level:
            self.Xmin[51] = self.initial_level
            self.Xmax[51] = self.initial_level

        self.inflow = (
            np.loadtxt(dir_study + "/input/hydro/series/" + name_area + "/mod.txt")[
                6:365:7
            ]
            * 7
            / H
        )
        self.area = name_area

        P_turb = np.loadtxt(
            dir_study + "/input/hydro/common/capacity/maxpower_" + name_area + ".txt"
        )[:, 0]
        P_pump = np.loadtxt(
            dir_study + "/input/hydro/common/capacity/maxpower_" + name_area + ".txt"
        )[:, 2]
        self.P_turb = P_turb
        self.P_pump = P_pump
        self.efficiency = hydro_ini.getfloat("pumping efficiency", name_area)


def generate_mps_file(study_path: str, antares_path: str) -> str:
    name_solver = antares_path.split("/")[-1]
    assert "solver" in name_solver
    assert float(name_solver.split("-")[1]) >= 8.7
    res = subprocess.run(
        [antares_path, "--named-mps-problems", "--name=export_mps", study_path],
        capture_output=True,
        text=True,
    )
    assert "Quitting the solver gracefully" in res.stdout
    output = res.stdout.split("\n")
    idx_line = [l for l in output if " Output folder : " in l]
    assert len(idx_line) >= 1
    output_folder = idx_line[0].split(" Output folder : ")[1]
    output_folder = output_folder.replace("\\", "/")
    return output_folder
