from proxy_stage_cost_function import ProxyStageCostFunction
from proxy_bellman_trajectories import BellmanValuesProxy, OptimalTrajectories
import numpy as np
import pandas as pd
from configparser import ConfigParser
import shutil
import os


class Exporter:
    def __init__(self, proxy: ProxyStageCostFunction, bv: BellmanValuesProxy, trajectories: OptimalTrajectories, export_dir:str):
        """
        Initialize Exporter with Proxy, BellmanValuesProxy, and OptimalTrajectories instances.
        Sets export directory, number of weeks, and scenarios.
        """
        self.proxy = proxy
        self.bv = bv
        self.trajectories = trajectories

        self.export_dir = export_dir
        self.nb_weeks = proxy.nb_weeks
        self.scenarios = proxy.scenarios

    def export_controls(self, filename: str = "controls.csv") -> None:
        """
        Export optimal control trajectories
        for all scenarios and weeks to a CSV file.
        """
        data = []
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                u = self.trajectories.optimal_controls[s, w]
                data.append({
                    "area": self.proxy.name_area,
                    "u": u,
                    "week": w + 1,
                    "mcYear": s + 1
                })

        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)

    def export_trajectories(self, filename: str = "trajectories.csv") -> None:
        """
        Export optimal stock trajectories for all scenarios and weeks
        to a CSV file.
        """
        data = []

        for s in self.scenarios:
            for w in range(self.nb_weeks):
                hlevel = self.trajectories.trajectories[s, w]
                data.append({
                    "area": self.proxy.name_area,
                    "hlevel": hlevel,
                    "week": w + 1,
                    "mcYear": s + 1,
                })
        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
    

class ModifyAntaresStudy:
    def __init__(self, bv: BellmanValuesProxy, trajectories: OptimalTrajectories):
        """
        Initialize the class with BellmanValuesProxy, optimal trajectories, and the target area.
        """
        self.bv = bv
        self.trajectories = trajectories
        self.nb_weeks = bv.nb_weeks
        self.scenarios = bv.scenarios
        self.dir_study = bv.proxy.dir_study
        self.name_area = bv.proxy.name_area

    def overwrite_pmax(self) -> None:
        """
        Replace the pmax file (maxpower_{area}.txt) with a file where all values are zero,
        backing up the original file first.
        """
        pmax_path = os.path.join(self.dir_study, "input", "hydro", "common", "capacity",f"maxpower_{self.name_area}.txt")
        pmax_backup_path = pmax_path.replace(".txt", "_old.txt")

        if os.path.exists(pmax_path):
            os.rename(pmax_path, pmax_backup_path)

        pmax = np.loadtxt(pmax_backup_path)
        pmax[:, 0] = 0
        pmax[:, 2] = 0

        np.savetxt(pmax_path, pmax, fmt="%.6f", delimiter="\t")

    def create_st_cluster(self) -> None:
        """
        Append a section to the list.ini file defining an ST storage cluster,
        including its capacities and efficiencies.
        """
        content = f"""[lt_stock_proxy_{self.name_area}]
name = lt_stock_proxy_{self.name_area}
group = PSP_open
reservoircapacity = {self.bv.proxy.reservoir.capacity}
initiallevel = 0.500000
injectionnominalcapacity = {np.max(self.bv.proxy.reservoir.max_hourly_pump)}
withdrawalnominalcapacity = {np.max(self.bv.proxy.reservoir.max_hourly_turb)}
efficiency = {self.bv.proxy.reservoir.efficiency}
efficiencywithdrawal = {self.bv.proxy.turb_efficiency}
initialleveloptim = false
enabled = true
"""
        list_ini_path = os.path.join(self.dir_study, "input", "st-storage", "clusters", self.name_area, "list.ini")
        os.makedirs(os.path.dirname(list_ini_path), exist_ok=True)
        with open(list_ini_path, "a") as f:
            f.write(content)

    def create_pmax_file(self) -> None:
        """
        Generate PMAX-injection.txt and PMAX-withdrawal.txt files for the area,
        based on maximum hourly pumping and turbine capacities,
        concatenated with 24 additional values.
        """
        pmax_injection_hourly = self.bv.proxy.reservoir.max_hourly_pump
        pmax_withdrawal_hourly = self.bv.proxy.reservoir.max_hourly_turb

        if np.max(pmax_injection_hourly) == 0:
            modulation_injection = np.zeros(168 * self.nb_weeks + 24)
        else:
            modulation_injection = pmax_injection_hourly / np.max(self.bv.proxy.reservoir.max_hourly_pump)
        if np.max(pmax_withdrawal_hourly) == 0:
            modulation_withdrawal = np.zeros(168 * self.nb_weeks + 24)
        else:
            modulation_withdrawal = pmax_withdrawal_hourly / np.max(self.bv.proxy.reservoir.max_hourly_turb)

        modulation_injection = np.concatenate([modulation_injection, np.full(24, modulation_injection[-1])])
        modulation_withdrawal = np.concatenate([modulation_withdrawal, np.full(24, modulation_withdrawal[-1])])

        folder_path = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.name_area, f"lt_stock_proxy_{self.name_area}"
        )
        os.makedirs(folder_path, exist_ok=True)
        np.savetxt(os.path.join(folder_path, "PMAX-injection.txt"), modulation_injection, fmt="%.20f")
        np.savetxt(os.path.join(folder_path, "PMAX-withdrawal.txt"), modulation_withdrawal, fmt="%.20f")

    def create_rule_curve_file(self) -> None:
        """
        Create the adjusted hourly lower-rule-curve.txt and upper-rule-curve.txt files
        based default values.
        """
        folder_path = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.name_area, f"lt_stock_proxy_{self.name_area}"
        )
        os.makedirs(folder_path, exist_ok=True)
        lower_arr = np.zeros(8760)
        upper_arr = np.ones(8760)

        np.savetxt(os.path.join(folder_path, "lower-rule-curve.txt"), lower_arr, fmt="%.6f")
        np.savetxt(os.path.join(folder_path, "upper-rule-curve.txt"), upper_arr, fmt="%.6f")

    def modify_scenario_builder(self) -> None:
        """
        Create a text file in user/tmp/scenariobuilder_lines listing
        the lines needed to assign ST clusters and to MC scenarios.
        """
        config = ConfigParser(strict=False)
        config.read(os.path.join(self.dir_study, "settings", "generaldata.ini"))
        nbyears = int(config["general"]["nbyears"])

        lines = []
        for mc in range(nbyears):
            trajectory = (mc % self.bv.proxy.reservoir.nb_scenarios) + 1
            lines.append(f"sts,{self.name_area},{mc},lt_stock_proxy_{self.name_area}={trajectory}")
            lines.append(f"s,{self.name_area},{mc},lt_stock_proxy_{self.name_area}={trajectory}")

        sb_dir = os.path.join(self.dir_study, "user","tmp", "scenariobuilder_lines")
        os.makedirs(sb_dir, exist_ok=True)
        with open(os.path.join(sb_dir, f"{self.name_area}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def adjust_inflow_pmax_withdrawal_constraint(self, balance: np.ndarray, week: int) -> np.ndarray:
        """
        Adjust the hourly balance at the end of the week to not exceed
        the maximum weekly turbine capacity (avoid numerical rounding errors leading to infeasibilities).
        """
        delta = np.sum(balance) - np.sum(self.bv.proxy.reservoir.max_weekly_turb[week] * self.bv.proxy.turb_efficiency)
        if delta > 0:
            balance[-1] -= np.ceil(delta / 1e-6) * 1e-6
        return balance

    def adjust_inflows_pmax_injection_constraint(self, balance: np.ndarray, week: int) -> np.ndarray:
        """
        Adjust the hourly balance at the end of the week to not exceed
        the maximum weekly pumping capacity (avoid numerical rounding errors leading to infeasibilities).
        """
        delta = np.sum(balance) + np.sum(self.bv.proxy.reservoir.max_weekly_pump[week] * self.bv.proxy.reservoir.efficiency)
        if delta < 0:
            balance[-1] -= np.floor(delta / 1e-6) * 1e-6
        return balance

    def create_inflows_sts(self) -> None:
        """
        Generate inflows.txt for the ST proxy by calculating the adjusted hourly balance
        according to constraints for each scenario and week.
        """
        balance = np.zeros((168 * self.nb_weeks, len(self.scenarios)))
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                hour_start = w * 168
                if w == 0:
                    hlevel_start = self.bv.proxy.reservoir.initial_level
                else:
                    hlevel_start = self.trajectories.trajectories[s, w - 1]
                hlevel_end = self.trajectories.trajectories[s, w]
                
                # This computes optimal control over the week : balance[hour_start, s] + balance[hour_start + 167, s] = hlevel_start - hlevel_end
                balance[hour_start, s] = hlevel_start - self.bv.proxy.reservoir.capacity / 2
                balance[hour_start + 167, s] = self.bv.proxy.reservoir.capacity / 2 - hlevel_end
                
                # Add hourly inflow to amount to be balanced
                hourly_inflow = self.bv.proxy.reservoir.hourly_inflow[hour_start:hour_start + 168, s]
                balance[hour_start:hour_start + 168, s] += hourly_inflow

                # Adjust inflows to respect st storage constraints (overflow and negative stock leading to unfeasibilites)
                balance[hour_start:hour_start + 168, s] -= self.trajectories.inflow_adjust_overflow[w, s, :]

                # Adjust inflows to respect pmax constraints (avoid numerical rounding errors leading to infeasibilities)
                balance[hour_start:hour_start + 168, s] = self.adjust_inflow_pmax_withdrawal_constraint(
                    balance[hour_start:hour_start + 168, s], w
                )
                balance[hour_start:hour_start + 168, s] = self.adjust_inflows_pmax_injection_constraint(
                    balance[hour_start:hour_start + 168, s], w
                )

                #Final check
                if (
                    np.sum(balance[hour_start:hour_start + 168, s])
                    > self.bv.proxy.reservoir.max_weekly_turb[w] * self.bv.proxy.turb_efficiency
                    or np.sum(balance[hour_start:hour_start + 168, s])
                    < -self.bv.proxy.reservoir.max_weekly_pump[w] * self.bv.proxy.reservoir.efficiency
                ):
                    raise ValueError(
                        f"Error for area {self.name_area} in week {w} scenario {s}: balance: {np.sum(balance[hour_start:hour_start + 168, s])}, "
                        f"max turbine: {self.bv.proxy.reservoir.max_weekly_turb[w] * self.bv.proxy.turb_efficiency}, "
                        f"max pump: {-self.bv.proxy.reservoir.max_weekly_pump[w] * self.bv.proxy.reservoir.efficiency}"
                    )

        balance = np.vstack([balance, np.zeros((24, len(self.scenarios)))])
        path = os.path.join(
            self.dir_study,
            "input",
            "st-storage",
            "series",
            self.name_area,
            f"lt_stock_proxy_{self.name_area}",
            "inflows.txt",
        )
        np.savetxt(path, balance, fmt="%.20f", delimiter="\t")

    def adjust_to_spillage_constraint(self) -> None:
        """
        Complies with spillage constraint. Negative net load is transfered to solar producion and
        max(turbining capacity,pumping capacity) is added to net load and misc-gen (fatal production).
        """
        miscgen_path = os.path.join(self.dir_study, "input", "misc-gen", f"miscgen-{self.name_area}.txt")
        load_path = os.path.join(self.dir_study, "input", "load", "series", f"load_{self.name_area}.txt")
        solar_path = os.path.join(self.dir_study, "input", "solar", "series", f"solar_{self.name_area}.txt")

        miscgen_backup_path = miscgen_path.replace(".txt", "_old.txt")
        load_backup_path = load_path.replace(".txt", "_old.txt")
        solar_backup_path = solar_path.replace(".txt", "_old.txt")

        S = self.bv.proxy.reservoir.nb_scenarios

        if os.path.exists(miscgen_path):
            if not os.path.exists(miscgen_backup_path):
                os.rename(miscgen_path, miscgen_backup_path)
        
        if os.path.exists(load_path):
            if not os.path.exists(load_backup_path):
                os.rename(load_path, load_backup_path)

        if os.path.exists(solar_path):
            if not os.path.exists(solar_backup_path):
                os.rename(solar_path, solar_backup_path)

        if os.path.exists(miscgen_backup_path) and os.path.getsize(miscgen_backup_path)!=0:
            miscgen_data = np.loadtxt(miscgen_backup_path)
        else:
            miscgen_data = np.zeros((8760, 8))
            
        if os.path.exists(load_backup_path) and os.path.getsize(load_backup_path)!=0:
            load_data = np.loadtxt(load_backup_path)
        else:
            load_data = np.zeros((8760, S))
    

        if os.path.exists(solar_backup_path) and os.path.getsize(solar_backup_path)!=0:
            solar_data = np.loadtxt(solar_backup_path)
        else:
            solar_data = np.zeros((8760, S))

        negatives = np.minimum(load_data, 0.0)
        transfer = -negatives
        load_data = load_data - negatives
        solar_data = solar_data + transfer

        hourly_turb = self.bv.proxy.reservoir.max_hourly_turb
        hourly_pump = self.bv.proxy.reservoir.max_hourly_pump
        spill_constraint = np.maximum(hourly_turb, hourly_pump)
        spill_constraint = np.concatenate([spill_constraint, spill_constraint[-24:]])

        miscgen_data[:, 5] += spill_constraint[:8760]
        load_data = load_data + spill_constraint[:8760, np.newaxis]

        np.savetxt(miscgen_path, miscgen_data, fmt="%.20f", delimiter="\t")
        np.savetxt(load_path, load_data, fmt="%.20f", delimiter="\t")
        np.savetxt(solar_path, solar_data, fmt="%.20f", delimiter="\t")

    

    def apply_all(self) -> None:
        """
        Execute all the steps to modify the Antares study.
        """
        self.overwrite_pmax()
        self.create_st_cluster()
        self.create_pmax_file()
        self.create_rule_curve_file()
        self.modify_scenario_builder()
        self.create_inflows_sts()
        self.adjust_to_spillage_constraint()


class UndoAntaresModifications:
    def __init__(self, dir_study: str, area: str):
        """
        Initialize with study directory path, original area.
        """
        self.dir_study = dir_study
        self.area = area

    def restore_pmax(self) -> None:
        """
        Restore the pmax file (maxpower_area.txt) by replacing the current version
        with the backup (_old.txt) if it exists.
        """
        pmax_path = os.path.join(
            self.dir_study, "input", "hydro", "common", "capacity", f"maxpower_{self.area}.txt"
        )
        pmax_backup_path = pmax_path.replace(".txt", "_old.txt")
        if os.path.exists(pmax_backup_path):
            if os.path.exists(pmax_path):
                os.remove(pmax_path)
            os.rename(pmax_backup_path, pmax_path)

    def remove_st_cluster_section(self) -> None:
        """
        Remove the ST proxy section from the storage cluster list.ini file
        for the area.
        """
        list_ini_path = os.path.join(
            self.dir_study, "input", "st-storage", "clusters", self.area, "list.ini"
        )
        if not os.path.exists(list_ini_path):
            return

        with open(list_ini_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        skip = False
        for line in lines:
            if line.strip().startswith(f"[lt_stock_proxy_{self.area}]"):
                skip = True
                continue
            elif skip and line.strip().startswith("["):
                skip = False
            if not skip:
                new_lines.append(line)

        with open(list_ini_path, "w") as f:
            f.writelines(new_lines)


    def remove_st_series_folder(self) -> None:
        """
        Remove the folder containing ST proxy series for the area.
        """
        folder = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.area,
            f"lt_stock_proxy_{self.area}"
        )
        if os.path.exists(folder):
            shutil.rmtree(folder)


    def clean_scenariobuilder(self) -> None:
        """
        Clean the scenariobuilder.dat file by removing lines associated with
        the ST proxy for the area (both 'sts' and 's' entries).
        """
        path = os.path.join(self.dir_study, "settings", "scenariobuilder.dat")
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            lines = f.readlines()

        if self.area is not None:
            filtered = [
                line for line in lines
                if not line.startswith(f"sts,{self.area},")
                and not line.startswith(f"s,{self.area},")
            ]
        else:
            filtered = [line for line in lines if f"lt_stock_proxy_{self.area}" not in line]

        with open(path, "w") as f:
            f.writelines(filtered)


    def restore_miscgen_load_and_solar(self) -> None:
        """
        Restore study inputs to their pre-modification state:
        - Restore miscgen-{area}.txt from miscgen-{area}_old.txt if it exists.
        - Restore load_{area}.txt    from load_{area}_old.txt    if it exists.
        - Restore solar_{area}.txt   from solar_{area}_old.txt   if it exists; otherwise remove solar_{area}.txt
        (this file may have been created when negative load was transferred to solar per scenario).
        Missing backups are ignored; existing current files are overwritten or removed as needed.
        """
        miscgen_path = os.path.join(self.dir_study, "input", "misc-gen", f"miscgen-{self.area}.txt")
        miscgen_backup_path = miscgen_path.replace(".txt", "_old.txt")
        if os.path.exists(miscgen_backup_path):
            if os.path.exists(miscgen_path):
                os.remove(miscgen_path)
            os.rename(miscgen_backup_path, miscgen_path)

        load_path = os.path.join(self.dir_study, "input", "load", "series", f"load_{self.area}.txt")
        load_backup_path = load_path.replace(".txt", "_old.txt")
        if os.path.exists(load_backup_path):
            if os.path.exists(load_path):
                os.remove(load_path)
            os.rename(load_backup_path, load_path)

        solar_path = os.path.join(self.dir_study, "input", "solar", "series", f"solar_{self.area}.txt")
        solar_backup_path = solar_path.replace(".txt", "_old.txt")
        if os.path.exists(solar_backup_path):
            if os.path.exists(solar_path):
                os.remove(solar_path)
            os.rename(solar_backup_path, solar_path)


    def undo_all(self) -> None:
        """
        Perform the full restoration of the Antares study for the original area.
        """
        self.restore_pmax()
        self.remove_st_cluster_section()
        self.remove_st_series_folder()
        self.clean_scenariobuilder()
        self.restore_miscgen_load_and_solar()
