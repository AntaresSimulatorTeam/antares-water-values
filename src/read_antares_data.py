from dataclasses import dataclass, field
import subprocess
from configparser import ConfigParser
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
        bottom_rule_curve = rule_curves[6:365:7, 0]
        upper_rule_curve = rule_curves[6:365:7, 1]
        self.bottom_rule_curve = np.concatenate(
            (bottom_rule_curve, bottom_rule_curve[[0]])
        )
        self.upper_rule_curve = np.concatenate(
            (upper_rule_curve, upper_rule_curve[[0]])
        )

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


# @dataclass
# class Compute_res_cons:

#     weeks_in_year=52
#     hours_in_week=168
#     hours_in_day=24
#     days_in_week=hours_in_week//hours_in_day
#     days_in_year=days_in_week*weeks_in_year+1

#     def __init__(
#             self, 
#             dir_study: str,
#             name_area: str
#     ) -> None:
        
#         self.area = name_area
#         self.read_load(dir_study)
#         self.read_solar(dir_study)
#         self.read_wind_offshore(dir_study)
#         self.read_wind_onshore(dir_study)
#         self.read_ror(dir_study)
#         self.compute_daily_load()
#         self.compute_daily_solar(dir_study)
#         self.compute_daily_wind_offshore(dir_study)
#         self.compute_daily_wind_onshore(dir_study)
#         self.compute_daily_ror()
#         self.compute_daily_res_cons()
    
#     def read_load(self,dir_study) -> None:
        
#         self.load= np.loadtxt(
#             f"{dir_study}/input/load/series/load_{self.area}.txt")
        
#     def read_solar(self,dir_study) -> None:
         
#         self.solar= np.loadtxt(
#             f"{dir_study}/input/renewables/series/{self.area}/{self.area}_solar_pv/series.txt")
    
#     def read_wind_offshore(self,dir_study) -> None:
         
#         self.wind_offshore= np.loadtxt(
#             f"{dir_study}/input/renewables/series/{self.area}/{self.area}_wind_offshore/series.txt")
        
#     def read_wind_onshore(self,dir_study) -> None:
         
#         self.wind_onshore= np.loadtxt(
#             f"{dir_study}/input/renewables/series/{self.area}/{self.area}_wind_onshore/series.txt")
        
#     def read_ror(self,dir_study) -> None:
         
#         self.ror= np.loadtxt(
#             f"{dir_study}/input/hydro/series/{self.area}/ror.txt")
        
#     def compute_daily_wind_offshore(self,dir_study) -> None :
#         nb_scenarios=self.wind_offshore.shape[1]
#         config=ConfigParser()
#         config.read(f"{dir_study}/input/renewables/clusters/{self.area}/list.ini")
#         self.wind_offshore_capacity=config.get(f"{self.area}_wind_offshore","nominalcapacity")
#         self.wind_offshore=self.wind_offshore*float(self.wind_offshore_capacity)
#         self.daily_wind_offshore = self.wind_offshore.reshape(self.days_in_year, self.hours_in_day, nb_scenarios).sum(axis=1)
        

#     def compute_daily_wind_onshore(self,dir_study) -> None :
#         nb_scenarios=self.wind_onshore.shape[1]
#         config=ConfigParser()
#         config.read(f"{dir_study}/input/renewables/clusters/{self.area}/list.ini")
#         self.wind_onshore_capacity=config.get(f"{self.area}_wind_onshore","nominalcapacity")
#         self.wind_onshore=self.wind_onshore*float(self.wind_onshore_capacity)
#         self.daily_wind_onshore = self.wind_onshore.reshape(self.days_in_year, self.hours_in_day, nb_scenarios).sum(axis=1)
        

#     def compute_daily_solar(self,dir_study) -> None:
#         nb_scenarios=self.solar.shape[1]
#         config=ConfigParser()
#         config.read(f"{dir_study}/input/renewables/clusters/{self.area}/list.ini")
#         self.solar_capacity=config.get(f"{self.area}_solar_pv","nominalcapacity")
#         self.solar=self.solar*float(self.solar_capacity)
#         self.daily_solar = self.solar.reshape(self.days_in_year, self.hours_in_day, nb_scenarios).sum(axis=1)
        


#     def compute_daily_ror(self) -> None:
#         nb_scenarios=self.ror.shape[1]
        
#         self.daily_ror = self.ror.reshape(self.days_in_year, self.hours_in_day, nb_scenarios).sum(axis=1)


#     def compute_daily_load(self) -> None:
#         nb_scenarios=self.load.shape[1]
        
#         self.daily_load = self.load.reshape(self.days_in_year, self.hours_in_day, nb_scenarios).sum(axis=1)

#     def compute_daily_res_cons(self) -> None:
#         self.daily_res_cons=self.daily_load-self.daily_solar-self.daily_wind_offshore-self.daily_wind_onshore-self.daily_ror

@dataclass
class Compute_res_cons:

    weeks_in_year = 52
    hours_in_week = 168
    hours_in_day = 24
    days_in_week = hours_in_week // hours_in_day
    days_in_year = days_in_week * weeks_in_year + 1

    def __init__(self, dir_study: str, name_area: str) -> None:
        self.area = name_area
        self.read_data(dir_study)
        self.compute_daily_load()
        self.compute_daily_solar(dir_study)
        self.compute_daily_wind(dir_study)
        self.compute_daily_ror()
        self.compute_daily_res_cons()

    def read_data(self, dir_study) -> None:
        """Reads all necessary data."""
        self.load = np.loadtxt(f"{dir_study}/input/load/series/load_{self.area}.txt")
        self.solar = np.loadtxt(f"{dir_study}/input/renewables/series/{self.area}/{self.area}_solar_pv/series.txt")
        self.wind_offshore = np.loadtxt(f"{dir_study}/input/renewables/series/{self.area}/{self.area}_wind_offshore/series.txt")
        self.wind_onshore = np.loadtxt(f"{dir_study}/input/renewables/series/{self.area}/{self.area}_wind_onshore/series.txt")
        self.ror = np.loadtxt(f"{dir_study}/input/hydro/series/{self.area}/ror.txt")

    def compute_daily_renewable(self, data, capacity_key, cluster_file) -> np.ndarray:
        """Computes the daily renewable energy production."""
        nb_scenarios = data.shape[1]
        config = ConfigParser()
        config.read(f"{cluster_file}")
        capacity = float(config.get(f"{self.area}_{capacity_key}", "nominalcapacity"))
        data *= capacity
        return data.reshape(self.days_in_year, self.hours_in_day, nb_scenarios).sum(axis=1)

    def compute_daily_solar(self, dir_study) -> None:
        self.daily_solar = self.compute_daily_renewable(self.solar, "solar_pv", f"{dir_study}/input/renewables/clusters/{self.area}/list.ini")

    def compute_daily_wind(self, dir_study) -> None:
        self.daily_wind_offshore = self.compute_daily_renewable(self.wind_offshore, "wind_offshore", f"{dir_study}/input/renewables/clusters/{self.area}/list.ini")
        self.daily_wind_onshore = self.compute_daily_renewable(self.wind_onshore, "wind_onshore", f"{dir_study}/input/renewables/clusters/{self.area}/list.ini")

    def compute_daily_ror(self) -> None:
        nb_scenarios = self.ror.shape[1]
        self.daily_ror = self.ror.reshape(self.days_in_year, self.hours_in_day, nb_scenarios).sum(axis=1)

    def compute_daily_load(self) -> None:
        nb_scenarios = self.load.shape[1]
        self.daily_load = self.load.reshape(self.days_in_year, self.hours_in_day, nb_scenarios).sum(axis=1)

    def compute_daily_res_cons(self) -> None:
        self.daily_res_cons = self.daily_load - self.daily_solar - self.daily_wind_offshore - self.daily_wind_onshore - self.daily_ror