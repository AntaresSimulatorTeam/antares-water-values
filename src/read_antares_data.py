import subprocess
from configparser import ConfigParser
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimeScenarioParameter:
    """Describes time and scenario related parameters"""

    len_week: int = 52
    len_scenario: int = 1
    name_scenario: list = field(default_factory=list)


@dataclass(frozen=True)
class TimeScenarioIndex:
    week: int
    scenario: int


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
    ) -> None:
        """
        Create a new reservoir.

        Parameters
        ----------
        dir_study:str :
            Path to the Antares study
        name_area:str :
            Name of the area where is located the reservoir

        Returns
        -------
        None
        """

        self.area = name_area

        hydro_ini_file = self.get_hydro_ini_file(dir_study=dir_study)

        self.read_capacity(hydro_ini_file=hydro_ini_file)
        self.read_efficiency(hydro_ini_file=hydro_ini_file)
        self.read_rule_curves(dir_study)
        self.read_inflow(dir_study)
        self.read_max_power(dir_study)

    def read_max_power(self, dir_study: str) -> None:
        max_power_data = np.loadtxt(
            f"{dir_study}/input/hydro/common/capacity/maxpower_{self.area}.txt"
        )
        weekly_energy = max_power_data[: self.days_in_year] * self.hours_in_day
        weekly_energy = weekly_energy.reshape(
            (self.weeks_in_year, self.days_in_week, 4)
        ).sum(axis=1)
        self.max_generating = weekly_energy[:, 0]
        self.max_pumping = weekly_energy[:, 2]

    def read_inflow(self, dir_study: str) -> None:
        daily_inflow = np.loadtxt(f"{dir_study}/input/hydro/series/{self.area}/mod.txt")
        daily_inflow = daily_inflow[: self.days_in_year]
        nb_scenarios = daily_inflow.shape[1]
        weekly_inflow = daily_inflow.reshape(
            (self.weeks_in_year, self.days_in_week, nb_scenarios)
        ).sum(axis=1)
        self.inflow = weekly_inflow

    def read_rule_curves(self, dir_study: str) -> None:
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
        bottom_rule_curve = rule_curves[7::7, 0]
        upper_rule_curve = rule_curves[7::7, 1]
        self.bottom_rule_curve = bottom_rule_curve
        self.upper_rule_curve = upper_rule_curve

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
    change_hydro_management_to_heuristic(dir_study=study_path)

    name_solver = antares_path.split("/")[-1]
    assert "solver" in name_solver
    assert float(name_solver.split("-")[1]) >= 8.7
    res = subprocess.run(
        [antares_path, "--named-mps-problems", "--name=export_mps", study_path],
        capture_output=True,
        text=True,
    )
    output = res.stdout.split("\n")
    idx_line = [l for l in output if " Output folder : " in l]
    assert len(idx_line) >= 1
    output_folder = idx_line[0].split(" Output folder : ")[1]
    output_folder = output_folder.replace("\\", "/")
    return output_folder


def change_hydro_management_to_heuristic(dir_study: str) -> None:
    hydro_ini = ConfigParser()
    hydro_ini.read(dir_study + "/input/hydro/hydro.ini")

    for area in hydro_ini["reservoir"].keys():
        if hydro_ini["reservoir"][area] == "true":
            if "use water" in hydro_ini.keys():
                hydro_ini["use water"][area] = "false"
            if "use heuristic" in hydro_ini.keys():
                hydro_ini["use heuristic"][area] = "true"

    with open(dir_study + "/input/hydro/hydro.ini", "w") as configfile:  # save
        hydro_ini.write(configfile)
