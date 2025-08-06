from proxy_stage_cost_function import Proxy
from proxy_bellman_trajectories import BellmanValuesProxy, OptimalTrajectories
import numpy as np
import pandas as pd
from configparser import ConfigParser
import shutil
import os


class Exporter:
    def __init__(self, proxy: Proxy, bv: BellmanValuesProxy, trajectories: OptimalTrajectories):
        """
        Initialize Exporter with Proxy, BellmanValuesProxy, and OptimalTrajectories instances.
        Sets export directory, number of weeks, and scenarios.
        """
        self.proxy = proxy
        self.bv = bv
        self.trajectories = trajectories

        self.export_dir = self.bv.export_dir
        self.nb_weeks = proxy.nb_weeks
        self.scenarios = proxy.scenarios

    def export_controls(self, filename: str = "controls.csv") -> None:
        """
        Export optimal control trajectories (control u, turbine, pump) 
        for all scenarios and weeks to a CSV file.
        """
        data = []
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                u = self.trajectories.optimal_controls[s, w]
                t = self.trajectories.optimal_turb[s, w]
                p = self.trajectories.optimal_pump[s, w]
                data.append({
                    "area": self.proxy.name_area,
                    "u": u,
                    "turb": t,
                    "pump": p,
                    "week": w + 1,
                    "mcYear": s + 1,
                    "sim": "u_0"
                })

        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        # print(f"Control trajectories export succeeded : {output_path}")

    def export_bellman_values(self, filename: str = "bellman_values.csv") -> None:
        """
        Export Bellman values for each stock percentage, week, and scenario
        to a CSV file.
        """
        data = []
        for w in range(self.nb_weeks):
            for c_index, c in enumerate(range(0, 101, 2)):
                stock_percent = c  # stock expressed in %
                for s in self.scenarios:
                    value = self.bv.bv[w, c_index, s]
                    data.append({
                        "week": w + 1,
                        "stock_percent": stock_percent,
                        "mcYear": s + 1,
                        "bellman_value": value
                    })

        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        # print(f"Bellman values export succeeded: {output_path}")

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
                    "sim": "u_0"
                })
        df = pd.DataFrame(data)
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        # print(f"Stock trajectories export succeeded : {output_path}")

    

class ModifyAntaresStudy:
    def __init__(self, bv: BellmanValuesProxy, trajectories: OptimalTrajectories, area_target: str):
        """
        Initialize the class with BellmanValuesProxy, optimal trajectories, and the target area.
        """
        self.bv = bv
        self.trajectories = trajectories
        self.nb_weeks = bv.nb_weeks
        self.scenarios = bv.scenarios
        self.dir_study = bv.proxy.dir_study
        self.name_area = bv.proxy.name_area
        self.area_target = area_target

    def overwrite_inflows(self) -> None:
        """
        Replace the inflows file (mod.txt) with a file where all values are zero,
        backing up the original file first.
        """
        inflow_path = os.path.join(self.dir_study, "input", "hydro", "series", self.name_area, "mod.txt")
        inflow_backup_path = inflow_path.replace(".txt", "_old.txt")

        if os.path.exists(inflow_path):
            os.rename(inflow_path, inflow_backup_path)

        inflows = np.loadtxt(inflow_backup_path)
        inflows[:, :] = 0

        np.savetxt(inflow_path, inflows, fmt="%.6f", delimiter="\t")

    def overwrite_hydro_ini_file(self) -> None:
        """
        Create a flag file indicating that the area should be disabled in hydro.ini.
        """
        flag_dir = os.path.join(self.dir_study, "tmp", "hydro_flags")
        os.makedirs(flag_dir, exist_ok=True)
        flag_path = os.path.join(flag_dir, f"{self.name_area}.flag")
        with open(flag_path, "w") as f:
            f.write("false\n")

    def create_st_cluster(self) -> None:
        """
        Append a section to the list.ini file defining an ST storage cluster,
        including its capacities and efficiencies.
        """
        content = f"""[lt_stock_proxy_{self.area_target}]
name = lt_stock_proxy_{self.area_target}
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
        list_ini_path = os.path.join(self.dir_study, "input", "st-storage", "clusters", self.area_target, "list.ini")
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
            self.dir_study, "input", "st-storage", "series", self.area_target, f"lt_stock_proxy_{self.area_target}"
        )
        os.makedirs(folder_path, exist_ok=True)
        np.savetxt(os.path.join(folder_path, "PMAX-injection.txt"), modulation_injection, fmt="%.20f")
        np.savetxt(os.path.join(folder_path, "PMAX-withdrawal.txt"), modulation_withdrawal, fmt="%.20f")

    def create_rule_curve_file(self) -> None:
        """
        Create the adjusted hourly lower-rule-curve.txt and upper-rule-curve.txt files
        based on adjusted trajectories, or default values if absent.
        """
        folder_path = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.area_target, f"lt_stock_proxy_{self.area_target}"
        )
        os.makedirs(folder_path, exist_ok=True)
        if hasattr(self.trajectories, "final_lower_rule_curve") and hasattr(self.trajectories, "final_upper_rule_curve"):
            lower_arr = np.clip(self.trajectories.final_lower_rule_curve / self.bv.proxy.reservoir.capacity, 0, 1)
            lower_arr = np.floor(lower_arr * 1e6) / 1e6

            upper_arr = np.clip(self.trajectories.final_upper_rule_curve / self.bv.proxy.reservoir.capacity, 0, 1)
            upper_arr = np.ceil(upper_arr * 1e6) / 1e6
        else:
            lower_arr = np.zeros(8760)
            upper_arr = np.ones(8760)

        np.savetxt(os.path.join(folder_path, "lower-rule-curve.txt"), lower_arr, fmt="%.6f")
        np.savetxt(os.path.join(folder_path, "upper-rule-curve.txt"), upper_arr, fmt="%.6f")

    def modify_scenario_builder(self) -> None:
        """
        Create a text file in tmp/scenariobuilder_lines listing
        the lines needed to assign ST clusters to MC scenarios.
        """
        config = ConfigParser(strict=False)
        config.read(os.path.join(self.dir_study, "settings", "generaldata.ini"))
        nbyears = int(config["general"]["nbyears"])

        lines = []
        for mc in range(nbyears):
            trajectory = (mc % self.bv.proxy.weighted_net_load.shape[1]) + 1
            lines.append(f"sts,{self.area_target},{mc},lt_stock_proxy_{self.area_target}={trajectory}")

        sb_dir = os.path.join(self.dir_study, "tmp", "scenariobuilder_lines")
        os.makedirs(sb_dir, exist_ok=True)
        with open(os.path.join(sb_dir, f"{self.area_target}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def adjust_inflow_pmax_withdrawal_constraint(self, balance: np.ndarray, week: int) -> np.ndarray:
        """
        Adjust the hourly balance at the end of the week to not exceed
        the maximum weekly turbine capacity accounting for efficiency.
        """
        delta = np.sum(balance) - np.sum(self.bv.proxy.reservoir.max_weekly_turb[week] * self.bv.proxy.turb_efficiency)
        if delta > 0:
            balance[-1] -= np.ceil(delta / 1e-6) * 1e-6
        return balance

    def adjust_inflows_pmax_injection_constraint(self, balance: np.ndarray, week: int) -> np.ndarray:
        """
        Adjust the hourly balance at the end of the week to not exceed
        the maximum weekly pumping capacity accounting for efficiency.
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
                balance[hour_start, s] = hlevel_start - self.bv.proxy.reservoir.capacity / 2
                balance[hour_start + 167, s] = self.bv.proxy.reservoir.capacity / 2 - hlevel_end
                hourly_inflow = self.bv.proxy.reservoir.hourly_inflow[hour_start:hour_start + 168, s]
                balance[hour_start:hour_start + 168, s] += hourly_inflow

                # balance[hour_start:hour_start + 168, s] -= self.trajectories.inflow_adjust_rule_curves[w, s, :]
                balance[hour_start:hour_start + 168, s] -= self.trajectories.inflow_adjust_overflow[w, s, :]
                balance[hour_start:hour_start + 168, s] = self.adjust_inflow_pmax_withdrawal_constraint(
                    balance[hour_start:hour_start + 168, s], w
                )
                balance[hour_start:hour_start + 168, s] = self.adjust_inflows_pmax_injection_constraint(
                    balance[hour_start:hour_start + 168, s], w
                )

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
            self.area_target,
            f"lt_stock_proxy_{self.area_target}",
            "inflows.txt",
        )
        np.savetxt(path, balance, fmt="%.20f", delimiter="\t")

    def adjust_to_spillage_constraint(self) -> None:
        """
        Adjust misc-gen and load files to include the spillage constraint.
        Adds max(hourly_turb, hourly_pump) to the 6th column of misc-gen
        and to every column of the load file.
        """
        miscgen_path = os.path.join(self.dir_study, "input", "misc-gen", f"miscgen-{self.area_target}.txt")
        load_path = os.path.join(self.dir_study, "input", "load", "series", f"load_{self.area_target}.txt")

        miscgen_backup_path = miscgen_path.replace(".txt", "_old.txt")
        load_backup_path = load_path.replace(".txt", "_old.txt")

        if os.path.exists(miscgen_path):
            if not os.path.exists(miscgen_backup_path):
                os.rename(miscgen_path, miscgen_backup_path)
            else:
                os.remove(miscgen_path)
        else:
            raise FileNotFoundError(f"miscgen file not found: {miscgen_path}")

        if os.path.exists(load_path):
            if not os.path.exists(load_backup_path):
                os.rename(load_path, load_backup_path)
            else:
                os.remove(load_path)
        else:
            raise FileNotFoundError(f"load file not found: {load_path}")

        try:
            miscgen_data = np.loadtxt(miscgen_backup_path)
            if miscgen_data.size == 0:
                miscgen_data = np.zeros((8760, 8))
        except Exception:
            miscgen_data = np.zeros((8760, 8))

        try:
            load_data = np.loadtxt(load_backup_path)
            if load_data.size == 0:
                load_data = np.zeros((8760, 200))
            if load_data.ndim == 1:
                load_data = np.repeat(load_data[:, np.newaxis], 200, axis=1)
        except Exception:
            load_data = np.zeros((8760, 200))

        if miscgen_data.shape[0] != 8760 or load_data.shape[0] != 8760:
            raise ValueError("Files must contain exactly 8760 lines (hourly data).")

        hourly_turb = self.bv.proxy.reservoir.max_hourly_turb
        hourly_pump = self.bv.proxy.reservoir.max_hourly_pump
        spill_constraint = np.maximum(hourly_turb, hourly_pump)
        spill_constraint = np.concatenate([spill_constraint, spill_constraint[-24:]])

        miscgen_data[:, 5] += spill_constraint
        load_data += spill_constraint[:, np.newaxis]

        np.savetxt(miscgen_path, miscgen_data, fmt="%.20f", delimiter="\t")
        np.savetxt(load_path, load_data, fmt="%.20f", delimiter="\t")

    def apply_all(self) -> None:
        """
        Execute all the steps to modify the Antares study in order.
        """
        self.overwrite_inflows()
        self.overwrite_hydro_ini_file()
        self.create_st_cluster()
        self.create_pmax_file()
        self.create_rule_curve_file()
        self.modify_scenario_builder()
        self.create_inflows_sts()
        self.adjust_to_spillage_constraint()
        # print(f"✅ Antares study modified for area '{self.area_target if self.area_target else self.name_area}'\n")




class UndoAntaresModifications:
    def __init__(self, dir_study: str, area: str, area_target: str):
        """
        Initialize with study directory path, original area, and target area.
        """
        self.dir_study = dir_study
        self.area = area
        self.area_target = area_target

    def restore_inflows(self) -> None:
        """
        Restore the inflows file (mod.txt) by replacing the current version
        with the backup (_old.txt) if it exists.
        """
        inflow_path = os.path.join(
            self.dir_study, "input", "hydro", "series", self.area, "mod.txt"
        )
        inflow_backup_path = inflow_path.replace(".txt", "_old.txt")
        if os.path.exists(inflow_backup_path):
            if os.path.exists(inflow_path):
                os.remove(inflow_path)
            os.rename(inflow_backup_path, inflow_path)
            # print("✔ inflows restored.")
        # else:
            # print("⚠ inflow backup not found. Nothing restored.")

    def restore_hydro_ini(self) -> None:
        """
        Modify hydro.ini to reactivate the area in the [reservoir] section
        by setting its value to "true".
        """
        path = os.path.join(self.dir_study, "input", "hydro", "hydro.ini")
        config = ConfigParser()
        config.read(path)
        if "reservoir" in config and f"{self.area}" in config["reservoir"]:
            config["reservoir"][f"{self.area}"] = "true"
            with open(path, "w") as configfile:
                config.write(configfile)
            # print("✔ hydro.ini restored.")
        # else:
            # print(f"⚠ hydro.ini unchanged: missing [reservoir]/{self.area} section.")

    def remove_st_cluster_section(self) -> None:
        """
        Remove the ST proxy section from the storage cluster list.ini file
        for the target area.
        """
        list_ini_path = os.path.join(
            self.dir_study, "input", "st-storage", "clusters", self.area_target, "list.ini"
        )
        if not os.path.exists(list_ini_path):
            # print("⚠ list.ini not found.")
            return

        with open(list_ini_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        skip = False
        for line in lines:
            if line.strip().startswith(f"[lt_stock_proxy_{self.area_target}]"):
                skip = True
                continue
            elif skip and line.strip().startswith("["):
                skip = False
            if not skip:
                new_lines.append(line)

        with open(list_ini_path, "w") as f:
            f.writelines(new_lines)

        # print("✔ st-cluster section removed.")

    def remove_st_series_folder(self) -> None:
        """
        Remove the folder containing ST proxy series for the target area.
        """
        folder = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.area_target,
            f"lt_stock_proxy_{self.area_target}"
        )
        if os.path.exists(folder):
            shutil.rmtree(folder)
            # print("✔ st-series folder removed.")
        # else:
            # print("⚠ st-series folder not found.")

    def clean_scenariobuilder(self) -> None:
        """
        Clean the scenariobuilder.dat file by removing lines associated with
        the ST proxy for the target area.
        """
        path = os.path.join(self.dir_study, "settings", "scenariobuilder.dat")
        if not os.path.exists(path):
            # print("⚠ scenariobuilder.dat not found.")
            return

        with open(path, "r") as f:
            lines = f.readlines()
        if self.area_target is not None:
            filtered = [line for line in lines if not line.startswith(f"sts,{self.area_target},")]
        else:
            filtered = [line for line in lines if f"lt_stock_proxy_{self.area}" not in line]

        with open(path, "w") as f:
            f.writelines(filtered)

        # print("✔ scenariobuilder cleaned.")

    def restore_miscgen_and_load(self) -> None:
        """
        Restore miscgen and load files by replacing them with their _old.txt
        backups, if they exist.
        """
        # Restore miscgen
        miscgen_path = os.path.join(
            self.dir_study, "input", "misc-gen", f"miscgen-{self.area_target}.txt"
        )
        miscgen_backup_path = miscgen_path.replace(".txt", "_old.txt")

        if os.path.exists(miscgen_backup_path):
            if os.path.exists(miscgen_path):
                os.remove(miscgen_path)
            os.rename(miscgen_backup_path, miscgen_path)
            # print("✔ miscgen restored.")
        # else:
            # print("⚠ miscgen backup not found. Nothing restored.")

        # Restore load
        load_path = os.path.join(
            self.dir_study, "input", "load", "series", f"load_{self.area_target}.txt"
        )
        load_backup_path = load_path.replace(".txt", "_old.txt")

        if os.path.exists(load_backup_path):
            if os.path.exists(load_path):
                os.remove(load_path)
            os.rename(load_backup_path, load_path)
            # print("✔ load restored.")
        # else:
            # print("⚠ load backup not found. Nothing restored.")

    def undo_all(self) -> None:
        """
        Perform the full restoration of the Antares study for the original area.
        """
        # print(f"\n🔁 Restoring Antares study for area: {self.area}")
        self.restore_inflows()
        self.restore_hydro_ini()
        self.remove_st_cluster_section()
        self.remove_st_series_folder()
        self.clean_scenariobuilder()
        self.restore_miscgen_and_load()
        # print(f"✅ Restoration complete for area '{self.area}'\n")



