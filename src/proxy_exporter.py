from proxy_stage_cost_function import Proxy
from proxy_bellman_trajectories import BellmanValuesProxy, OptimalTrajectories
import numpy as np
import pandas as pd
from configparser import ConfigParser
import shutil
import os


class Exporter:
    def __init__(self, proxy: Proxy, bv: BellmanValuesProxy, trajectories : OptimalTrajectories):
        self.proxy = proxy
        self.bv = bv
        self.trajectories = trajectories

        self.export_dir = self.bv.export_dir
        self.nb_weeks = proxy.nb_weeks
        self.scenarios = proxy.scenarios

    def export_controls(self,filename:str="controls.csv") -> None:
        data = []
        for s in self.scenarios:
            for w in range(self.nb_weeks):
                u = self.trajectories.optimal_controls[s, w]
                t = self.trajectories.optimal_turb[s,w]
                p = self.trajectories.optimal_pump[s,w]
                data.append({
                    "area": self.proxy.name_area,
                    "u": u,
                    "turb" : t,
                    "pump": p,
                    "week": w + 1,
                    "mcYear": s + 1,
                    "sim": "u_0"
                })

        df = pd.DataFrame(data)
        for col in ["u", "turb", "pump"]:
            if col in df.columns:
                df[col] = df[col]
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Control trajectories export succeeded : {output_path}")

    def export_bellman_values(self, filename: str = "bellman_values.csv") -> None:
        data = []
        for w in range(self.nb_weeks):
            for c_index, c in enumerate(range(0, 101, 2)):
                stock_percent = c  # stock exprimé en %
                for s in self.scenarios:
                    value = self.bv.bv[w, c_index, s]
                    data.append({
                        "week": w + 1,
                        "stock_percent": stock_percent,
                        "mcYear": s + 1,
                        "bellman_value": value
                    })

        df = pd.DataFrame(data)
        if "bellman_value" in df.columns:
            df["bellman_value"] = df["bellman_value"]
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Bellman values export succeeded: {output_path}")

    def export_trajectories(self,filename:str="trajectories.csv") ->None:
        data = []

        for s in self.scenarios:
            for w in range(self.nb_weeks):
                hlevel =self.trajectories.trajectories[s,w]
                data.append({
                    "area": self.proxy.name_area,
                    "hlevel": hlevel,
                    "week": w + 1,
                    "mcYear": s + 1,
                    "sim": "u_0"
                })
        df = pd.DataFrame(data)
        if "hlevel" in df.columns:
            df["hlevel"] = df["hlevel"]
        output_path = os.path.join(self.export_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Stock trajectories export succeeded : {output_path}")
    

class ModifyAntaresStudy:
    def __init__(self, bv:BellmanValuesProxy, trajectories:OptimalTrajectories, area_target:str):
        self.bv = bv
        self.trajectories = trajectories
        self.nb_weeks = bv.nb_weeks
        self.scenarios = bv.scenarios
        self.dir_study= bv.proxy.dir_study
        self.name_area = bv.proxy.name_area
        self.area_target = area_target


    def overwrite_inflows(self) -> None:
        inflow_path = os.path.join(self.dir_study, "input", "hydro", "series", self.name_area, "mod.txt")
        inflow_backup_path = inflow_path.replace(".txt", "_old.txt")

        if os.path.exists(inflow_path):
            os.rename(inflow_path, inflow_backup_path)

        inflows = np.loadtxt(inflow_backup_path)
        inflows[:, :] = 0

        np.savetxt(inflow_path, inflows, fmt="%.6f", delimiter="\t")

    def overwrite_hydro_ini_file(self) -> None:
        flag_dir = os.path.join(self.dir_study, "tmp", "hydro_flags")
        os.makedirs(flag_dir, exist_ok=True)
        flag_path = os.path.join(flag_dir, f"{self.name_area}.flag")
        with open(flag_path, "w") as f:
            f.write("false\n")  # indique que la zone doit être désactivée

    def create_st_cluster(self) -> None:
        contenu = f"""[lt_stock_proxy_{self.area_target}]
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
            f.write(contenu)

    def create_pmax_file(self) -> None:
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

    def adjust_inflow_pmax_withdrawal_constraint(self,balance: np.ndarray,week : int) -> np.ndarray:
        delta=np.sum(balance)-np.sum(self.bv.proxy.reservoir.max_weekly_turb[week]*self.bv.proxy.turb_efficiency)
        if delta>0:
            balance[-1]-= np.ceil(delta/1e-6)*1e-6
        return balance
    
    def adjust_inflows_pmax_injection_constraint(self, balance: np.ndarray, week: int) -> np.ndarray:
        delta = np.sum(balance) + np.sum(self.bv.proxy.reservoir.max_weekly_pump[week]*self.bv.proxy.reservoir.efficiency)
        if delta < 0:
            balance[-1] -= np.floor(delta/1e-6)*1e-6
        return balance

    def create_inflows_sts(self) -> None:
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
                hourly_inflow = self.bv.proxy.reservoir.hourly_inflow[hour_start:hour_start+168,s]
                balance[hour_start:hour_start + 168, s] += hourly_inflow

                # balance[hour_start:hour_start + 168, s] -= self.trajectories.inflow_adjust_rule_curves[w, s, :]
                balance[hour_start:hour_start + 168, s] -= self.trajectories.inflow_adjust_overflow[w, s, :]
                balance[hour_start:hour_start + 168, s] = self.adjust_inflow_pmax_withdrawal_constraint(balance[hour_start:hour_start + 168, s], w)
                balance[hour_start:hour_start + 168, s] = self.adjust_inflows_pmax_injection_constraint(balance[hour_start:hour_start + 168, s], w)
                if np.sum(balance[hour_start:hour_start + 168, s])>self.bv.proxy.reservoir.max_weekly_turb[w]*self.bv.proxy.turb_efficiency \
                    or np.sum(balance[hour_start:hour_start + 168, s])<-self.bv.proxy.reservoir.max_weekly_pump[w]*self.bv.proxy.reservoir.efficiency:
                    raise ValueError(
                        f"Erreur pour la zone {self.name_area} dans la semaine {w} pour le scénario {s}: controle : {np.sum(balance[hour_start:hour_start + 168, s])}, \
                        turb_max : {self.bv.proxy.reservoir.max_weekly_turb[w]*self.bv.proxy.turb_efficiency},\
                        pump_max : {-self.bv.proxy.reservoir.max_weekly_pump[w]*self.bv.proxy.reservoir.efficiency}"
                    )
        balance = np.vstack([balance, np.zeros((24, len(self.scenarios)))])
        path = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.area_target,
            f"lt_stock_proxy_{self.area_target}", "inflows.txt"
        )
        np.savetxt(path, balance, fmt="%.20f", delimiter="\t")

    def apply_all(self) -> None:
        self.overwrite_inflows()
        self.overwrite_hydro_ini_file()
        self.create_st_cluster()
        self.create_pmax_file()
        self.create_rule_curve_file()
        self.modify_scenario_builder()
        self.create_inflows_sts()
        print(f"✅ Antares study modified for area '{self.area_target if self.area_target else self.name_area}'\n")


class UndoAntaresModifications:
    def __init__(self, dir_study: str, area: str, area_target:str):
        self.dir_study = dir_study
        self.area = area
        self.area_target = area_target

    def restore_inflows(self) -> None:
        inflow_path = os.path.join(
            self.dir_study, "input", "hydro", "series", self.area, "mod.txt"
        )
        inflow_backup_path = inflow_path.replace(".txt", "_old.txt")
        if os.path.exists(inflow_backup_path):
            if os.path.exists(inflow_path):
                os.remove(inflow_path)
            os.rename(inflow_backup_path, inflow_path)
            print("✔ inflows restored.")
        else:
            print("⚠ inflow backup not found. Nothing restored.")

    def restore_hydro_ini(self) -> None:
        path = os.path.join(self.dir_study, "input", "hydro", "hydro.ini")
        config = ConfigParser()
        config.read(path)
        if "reservoir" in config and f"{self.area}" in config["reservoir"]:
            config["reservoir"][f"{self.area}"] = "true"
            with open(path, "w") as configfile:
                config.write(configfile)
            print("✔ hydro.ini restored.")
        else:
            print(f"⚠ hydro.ini unchanged: missing [reservoir]/{self.area} section.")

    def remove_st_cluster_section(self) -> None:
        list_ini_path = os.path.join(
            self.dir_study, "input", "st-storage", "clusters", self.area_target, "list.ini"
        )
        if not os.path.exists(list_ini_path):
            print("⚠ list.ini not found.")
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

        print("✔ st-cluster section removed.")

    def remove_st_series_folder(self) -> None:
        folder = os.path.join(
            self.dir_study, "input", "st-storage", "series", self.area_target,
            f"lt_stock_proxy_{self.area_target}"
        )
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print("✔ st-series folder removed.")
        else:
            print("⚠ st-series folder not found.")

    def clean_scenariobuilder(self) -> None:
        path = os.path.join(self.dir_study, "settings", "scenariobuilder.dat")
        if not os.path.exists(path):
            print("⚠ scenariobuilder.dat not found.")
            return

        with open(path, "r") as f:
            lines = f.readlines()

        filtered = [line for line in lines if f"lt_stock_proxy_{self.area_target}" not in line]

        with open(path, "w") as f:
            f.writelines(filtered)

        print("✔ scenariobuilder cleaned.")

    def undo_all(self) -> None:
        print(f"\n🔁 Restoring Antares study for area: {self.area}")
        self.restore_inflows()
        self.restore_hydro_ini()
        self.remove_st_cluster_section()
        self.remove_st_series_folder()
        self.clean_scenariobuilder()
        print(f"✅ Restoration complete for area '{self.area}'\n")

