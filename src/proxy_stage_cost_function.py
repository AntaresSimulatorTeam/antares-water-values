from tkinter import font
from read_antares_data import Reservoir
from read_antares_data import NetLoad
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import binary_dilation

rcParams['font.family'] = 'Cambria'

class Proxy:
    def __init__(self, dir_study: str, name_area: str,nb_scenarios:int) -> None:
        self.dir_study=dir_study
        self.name_area=name_area
        self.reservoir = Reservoir(self.dir_study, self.name_area)
        self.allocation_dict = self.reservoir.allocation_dict
        # pour éviter les infaisabilités dues aux arrondis on diminue la capacité de pompage et turbinage
        self.max_daily_generating=self.reservoir.max_daily_generating-1
        self.max_daily_pumping=self.reservoir.max_daily_pumping-1 if not np.allclose(self.reservoir.max_daily_pumping,0) else self.reservoir.max_daily_pumping
        self.efficiency=self.reservoir.efficiency
        self.turb_efficiency=1

        self.nb_weeks=self.reservoir.inflow.shape[0]
        self.scenarios=range(nb_scenarios)
        self.compute_weighted_net_load()

    
    def compute_weighted_net_load(self)-> None:
        self.net_load = np.zeros((365 * 24, 200))
        # print(self.allocation_dict)
        for key, value in self.allocation_dict.items():
            self.net_load += value * NetLoad(self.dir_study, key).net_load
    
    def stage_cost_function(self, week_index: int, scenario: int, alpha: float, coeff: float) -> np.ndarray:
        net_load_for_week = self.net_load[week_index * 168:(week_index + 1) * 168, scenario]
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7:(week_index + 1) * 7], 24) / 24
        max_pumping_hour = np.repeat(self.max_daily_pumping[week_index * 7:(week_index + 1) * 7], 24) / 24

        pump_is_zero = np.allclose(max_pumping_hour, 0)
        control_list = []
        cost_list = []
        turb_list = []
        pump_list = []

        turb_thresholds = np.quantile(
            np.linspace(
                np.min(net_load_for_week - max_energy_hour),
                np.max(net_load_for_week + max_pumping_hour) * (self.turb_efficiency / self.efficiency) ** (1 / (alpha - 1))
            ),
            np.linspace(0, 1, 25)
        )

        for turb_threshold in turb_thresholds:
            curtailed_energy = np.minimum(net_load_for_week, np.maximum(turb_threshold, net_load_for_week - max_energy_hour))
            curtail = net_load_for_week - curtailed_energy

            if not pump_is_zero:
                if turb_threshold < 0:
                    pump_threshold = turb_threshold
                else:
                    pump_threshold = ((self.efficiency / self.turb_efficiency) ** (1 / (alpha - 1))) * turb_threshold

                potential_pump = pump_threshold - curtailed_energy
                mask = curtailed_energy < pump_threshold
                actual_pump = np.minimum(potential_pump, max_pumping_hour) * mask
            else:
                actual_pump = np.zeros_like(curtailed_energy)

            curtailed_energy += actual_pump

            control_hourly = curtail * self.turb_efficiency - actual_pump * self.efficiency
            control = np.sum(control_hourly)
            turb_list.append(np.sum(curtail * self.turb_efficiency))
            pump_list.append(np.sum(actual_pump * self.efficiency))
            cost = np.sum(np.abs(curtailed_energy) ** alpha / coeff)
            control_list.append(control)
            cost_list.append(cost)

        return np.array([
            interp1d(control_list, cost_list, fill_value="extrapolate"),
            interp1d(control_list, turb_list, fill_value="extrapolate"),
            interp1d(control_list, pump_list, fill_value="extrapolate")
        ])


    def compute_stage_cost_functions(self,alpha:float,coeff:float)->np.ndarray:
        cost_functions=np.array([[self.stage_cost_function(w,s,alpha,coeff) for s in self.scenarios] for w in range(self.nb_weeks)])
        return cost_functions

    
   
    # affichages

    def plot_stage_cost_function(self,week_index:int,scenario:int,alpha:float,coeff:float)->None:
        cost_func = self.stage_cost_function(week_index, scenario, alpha,coeff)[0]
        control_values = cost_func.x
        cost_values = cost_func(control_values)
        # print(f"control values : {control_values}, cost_values : {cost_values}")
        plt.figure(figsize=(8, 5))
        plt.plot(control_values, cost_values, marker='o')
        plt.xlabel("Contrôle hebdomadaire [MWh]", fontsize=14)
        plt.ylabel(f"Coût, (α={alpha})",fontsize=14)
        plt.title(f"Fonction de coût - Scénario {scenario+1}, Semaine {week_index+1}",fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()




    def plot_load(self, week_index: int, scenario: int, alpha: float, index_ecretement: int, coeff: float) -> None:
        original_load = self.net_load[week_index * 168:(week_index + 1) * 168, scenario]
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7:(week_index + 1) * 7], 24) / 24
        max_pumping_hour = np.repeat(self.max_daily_pumping[week_index * 7:(week_index + 1) * 7], 24) / 24

        n_thresholds = 168
        turb_thresholds = np.quantile(
            np.linspace(
                np.min(original_load - max_energy_hour),
                np.max(original_load + max_pumping_hour) * (self.turb_efficiency / self.efficiency) ** (1 / (alpha - 1))
            ),
            np.linspace(0, 1, n_thresholds)
        )

        threshold = turb_thresholds[index_ecretement]
        curtailed_energy = np.minimum(original_load, np.maximum(threshold, original_load - max_energy_hour))
        curtail = original_load - curtailed_energy

        if threshold < 0:
            pump_threshold = threshold
        else:
            pump_threshold = (self.efficiency / self.turb_efficiency) ** (1 / (alpha - 1)) * threshold

        potential_pump = pump_threshold - curtailed_energy
        mask = curtailed_energy < pump_threshold
        actual_pump = np.minimum(potential_pump, max_pumping_hour) * mask
        curtailed_energy += actual_pump

        # Étapes intermédiaires
        turb_curve = original_load - curtail
        final_curve = turb_curve + actual_pump

        # Contrôle et coût
        control_hourly = curtail * self.turb_efficiency - actual_pump * self.efficiency
        control_total = np.sum(control_hourly)
        cost_total = np.sum(np.abs(curtailed_energy) ** alpha / coeff)

        # Affichage des valeurs
        print(f"Semaine {week_index}, scénario {scenario}")
        print(f"  Turbinage total : {np.sum(curtail):.1f} MWh")
        print(f"  Pompage total   : {np.sum(actual_pump):.1f} MWh")
        print(f"  Contrôle total  : {control_total:.1f} MWh")
        print(f"  Coût            : {cost_total:.2f}")

        hours = np.arange(len(original_load))

        plt.figure(figsize=(15, 6))
        plt.plot(hours, original_load, label='Consommation initiale', color='gray', linewidth=1.2)
        plt.plot(hours, final_curve, label='Consommation écrétée', color='black', linewidth=1.5)

        # Remplissages
        turb_mask = binary_dilation(curtail > 1e-6, iterations=1)
        plt.fill_between(hours, original_load, turb_curve, where=turb_mask,
                        color='firebrick', alpha=0.5, label='Énergie turbinée')

        pump_mask = binary_dilation(actual_pump > 1e-6, iterations=1)
        plt.fill_between(hours, turb_curve, final_curve, where=pump_mask,
                        color='royalblue', alpha=0.5, label='Énergie pompée')

        # Seuils
        if index_ecretement < n_thresholds:
            plt.axhline(y=threshold, color='red', linestyle='--', label=f"Seuil turbinage = {threshold:.2f} MWh")
            plt.axhline(y=pump_threshold, color='blue', linestyle='--', label=f"Seuil pompage = {pump_threshold:.2f} MWh")
        else:
            plt.axhline(y=pump_threshold, color='blue', linestyle='--', label=f"Seuil pompage seul = {pump_threshold:.2f} MWh")

        plt.title(f"Scénario {scenario+1}, Semaine {week_index+1} (α={alpha})", fontsize=16)
        plt.xlabel("Heures", fontsize=16)
        plt.ylabel("Consommation résiduelle (MWh)", fontsize=16)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()




    def plot_load_simple(self, week_index: int, scenario: int, alpha: float, index_ecretement: int, coeff: float) -> None:
        original_load = self.net_load[week_index * 168:(week_index + 1) * 168, scenario]
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7:(week_index + 1) * 7], 24) / 24
        max_pumping_hour = np.repeat(self.max_daily_pumping[week_index * 7:(week_index + 1) * 7], 24) / 24

        n_thresholds = 168
        turb_thresholds = np.quantile(
            np.linspace(
                np.min(original_load - max_energy_hour),
                np.max(original_load + max_pumping_hour) * (self.turb_efficiency / self.efficiency) ** (1 / (alpha - 1))
            ),
            np.linspace(0, 1, n_thresholds)
        )

        threshold = turb_thresholds[index_ecretement]
        curtailed_energy = np.minimum(original_load, np.maximum(threshold, original_load - max_energy_hour))
        curtail = original_load - curtailed_energy  # énergie turbinée

        if threshold < 0:
            pump_threshold = threshold
        else:
            pump_threshold = (self.efficiency / self.turb_efficiency) ** (1 / (alpha - 1)) * threshold

        potential_pump = pump_threshold - curtailed_energy
        mask = curtailed_energy < pump_threshold
        actual_pump = np.minimum(potential_pump, max_pumping_hour) * mask
        curtailed_energy += actual_pump  # consommation après pompage

        # Étapes intermédiaires
        turb_curve = original_load - curtail  # après turbinage
        final_curve = turb_curve + actual_pump  # après turbinage + pompage

        hours = np.arange(len(original_load))

        plt.figure(figsize=(15, 6))

        # Courbe initiale
        plt.plot(hours, original_load, label='Consommation initiale', color='gray', linewidth=1.2)

        # Courbe modifiée
        plt.plot(hours, final_curve, label='Consommation écrétée', color='black', linewidth=1.5)

        # Remplissage entre courbe initiale et après turbinage
        turb_mask = binary_dilation(curtail > 1e-6, iterations=1)
        plt.fill_between(hours, original_load, turb_curve, where=turb_mask,
                        color='firebrick', alpha=0.5, label='Énergie turbinée')

        # Remplissage entre après turbinage et consommation finale
        pump_mask = binary_dilation(actual_pump > 1e-6, iterations=1)
        plt.fill_between(hours, turb_curve, final_curve, where=pump_mask,
                        color='royalblue', alpha=0.5, label='Énergie pompée')

        plt.title(f"Écrêtement de la consommation résiduelle – Scénario {scenario+1}, Semaine {week_index+1}",fontsize=14)
        plt.xlabel("Heures",fontsize=14)
        plt.ylabel("Consommation résiduelle (MWh)",fontsize=14)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


