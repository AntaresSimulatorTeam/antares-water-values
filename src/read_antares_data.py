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

    weeks_in_year = 52
    hours_in_week = 168
    hours_in_day = 24
    days_in_week = hours_in_week // hours_in_day
    days_in_year = weeks_in_year * days_in_week

    def __init__(
        self,
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

        self.area = name_area

        hydro_ini_file = self.get_hydro_ini_file(dir_study=dir_study)

        self.read_capacity(hydro_ini_file=hydro_ini_file)
        self.read_efficiency(hydro_ini_file=hydro_ini_file)
        self.read_rule_curves(dir_study, final_level)
        self.read_inflow(dir_study)
        self.read_max_power(dir_study)

    def read_max_power(self, dir_study: str) -> None:
        max_power_data = np.loadtxt(
            f"{dir_study}/input/hydro/common/capacity/maxpower_{self.area}.txt"
        )
        self.P_turb = max_power_data[:, 0]
        self.P_pump = max_power_data[:, 2]

    def read_inflow(self, dir_study: str) -> None:
        daily_inflow = np.loadtxt(f"{dir_study}/input/hydro/series/{self.area}/mod.txt")
        daily_inflow = daily_inflow[: self.days_in_year]
        nb_scenarios = daily_inflow.shape[1]
        weekly_inflow = daily_inflow.reshape(
            (self.weeks_in_year, self.days_in_week, nb_scenarios)
        ).sum(axis=1)
        self.inflow = weekly_inflow / self.hours_in_week

    def read_rule_curves(self, dir_study: str, final_level: bool) -> None:
        rule_curves = (
            np.loadtxt(
                f"{dir_study}/input/hydro/common/capacity/reservoir_{self.area}.txt"
            )[:, [0, 2]]
            * self.capacity
        )
        assert (
            rule_curves[0, 0] == rule_curves[0, 1]
        ), "Initial level is not correctly defined by bottom and upper rule curves"
        self.initial_level = rule_curves[0, 0]
        bottom_rule_curve = rule_curves[6:365:7, 0]
        upper_rule_curve = rule_curves[6:365:7, 1]
        self.bottom_rule_curve = np.concatenate(
            (bottom_rule_curve, bottom_rule_curve[[0]])
        )
        self.upper_rule_curve = np.concatenate(
            (upper_rule_curve, upper_rule_curve[[0]])
        )
        if final_level:
            self.bottom_rule_curve[51] = self.initial_level
            self.upper_rule_curve[51] = self.initial_level

    def get_hydro_ini_file(self, dir_study: str) -> ConfigParser:
        hydro_ini_file = ConfigParser()
        hydro_ini_file.read(dir_study + "/input/hydro/hydro.ini")

        return hydro_ini_file

    def read_capacity(self, hydro_ini_file: ConfigParser) -> None:

        capacity = hydro_ini_file.getfloat("reservoir capacity", self.area)

        self.capacity = capacity

    def read_efficiency(self, hydro_ini_file: ConfigParser) -> None:
        efficiency = hydro_ini_file.getfloat("pumping efficiency", self.area)
        self.efficiency = efficiency


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
