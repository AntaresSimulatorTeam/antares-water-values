from read_antares_data import Reservoir
from read_antares_data import NetLoad
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Cambria'

class GainFunctionHydro:
    def __init__(self, dir_study: str, name_area: str) -> None:
        self.dir_study=dir_study
        self.name_area=name_area
        self.reservoir = Reservoir(self.dir_study, self.name_area)
        self.net_load=NetLoad(dir_study,name_area).net_load
        self.max_daily_generating=self.reservoir.max_daily_generating
        self.max_daily_pumping=self.reservoir.max_daily_pumping
        self.efficiency=self.reservoir.efficiency
        self.turb_efficiency=1.67

        self.nb_weeks=self.reservoir.inflow.shape[0]
        self.scenarios=range(self.net_load.shape[1])
        # self.scenarios=range(5)

    def gain_function(self, week_index: int, scenario: int, alpha: float) -> interp1d:
        net_load_for_week = self.net_load[week_index * 168:(week_index + 1) * 168, scenario]
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7:(week_index + 1) * 7], 24) / 24
        max_pumping_hour = np.repeat(self.max_daily_pumping[week_index * 7:(week_index + 1) * 7], 24) / 24
        
        control_list = []
        gain_list = []

        turb_thresholds=np.quantile(np.linspace(np.min(net_load_for_week-max_energy_hour),
                                                np.max(net_load_for_week+max_pumping_hour)*(self.turb_efficiency/self.efficiency)),
                                                np.linspace(0, 1, 25))

        for turb_threshold in turb_thresholds:
            curtailed_energy = np.minimum(net_load_for_week, np.maximum(turb_threshold, net_load_for_week - max_energy_hour))
            curtail = net_load_for_week - curtailed_energy
            # print(f"seuil de turbinage : {turb_threshold}")
            # print(f"quantité turbinée : {np.sum(curtail)}")


            if turb_threshold<0:
                pump_threshold=turb_threshold
            else:
                pump_threshold = (self.efficiency/self.turb_efficiency) * turb_threshold
            potential_pump = pump_threshold - curtailed_energy

            mask = curtailed_energy < pump_threshold
            actual_pump = np.minimum(potential_pump, max_pumping_hour) * mask

            # print(f"seuil de pompage : {pump_threshold}")
            # print(f"quantité pompée : {np.sum(actual_pump)}")
            curtailed_energy += actual_pump

            control_hourly = curtail*self.turb_efficiency - actual_pump*self.efficiency
            control = np.sum(control_hourly)
            # print(f"controle total effectué : {control}")

            gain = np.sum(curtailed_energy ** alpha / 1e9)
            # print(f"gain associé au controle {control} : {gain}")
            control_list.append(control)
            gain_list.append(gain)


        return interp1d(control_list, gain_list, fill_value="extrapolate")


    def compute_gain_functions(self,alpha:float)->np.ndarray:
        gain_functions=np.array([[self.gain_function(w,s,alpha) for s in self.scenarios] for w in range(self.nb_weeks)])
        return gain_functions
    
   
    # affichages

    def plot_gain_function(self,week_index:int,scenario:int,alpha:float)->None:
        gain_func = self.gain_function(week_index, scenario, alpha)
        control_values = gain_func.x
        gain_values = gain_func(control_values)
        # print(f"control values : {control_values}, gain_values : {gain_values}")
        plt.figure(figsize=(8, 5))
        plt.plot(control_values, gain_values, marker='o')
        plt.xlabel("Contrôle hebdomadaire [MWh]")
        plt.ylabel(f"Gain, α={alpha})")
        plt.title(f"Courbe de gain - Semaine {week_index+1}, MC {scenario+1}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_load(self, week_index: int, scenario: int, alpha: float, index_ecretement: int) -> None:
        original_load = self.net_load[week_index * 168:(week_index + 1) * 168, scenario]
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7:(week_index + 1) * 7], 24) / 24
        max_pumping_hour = np.repeat(self.max_daily_pumping[week_index * 7:(week_index + 1) * 7], 24) / 24

        n_thresholds = 168
        turb_thresholds = np.quantile(np.linspace(np.min(original_load-max_energy_hour),
                                                  np.max(original_load+max_pumping_hour)*(self.turb_efficiency/self.efficiency)), 
                                                  np.linspace(0, 1, n_thresholds))

        threshold = turb_thresholds[index_ecretement]
        curtailed_energy = np.minimum(original_load, np.maximum(threshold, original_load - max_energy_hour))
        curtail = original_load - curtailed_energy

        if threshold < 0:
            pump_threshold = threshold
        else:
            pump_threshold = (self.efficiency / self.turb_efficiency) * threshold

        potential_pump = pump_threshold - curtailed_energy
        mask = curtailed_energy < pump_threshold
        actual_pump = np.minimum(potential_pump, max_pumping_hour) * mask
        curtailed_energy += actual_pump

        control_hourly = curtail * self.turb_efficiency - actual_pump * self.efficiency
        control_total = np.sum(control_hourly)

        gain_total = np.sum(curtailed_energy ** alpha / 1e9)

        hours = np.arange(len(original_load))
        width = 0.4

        plt.figure(figsize=(15, 6))
        plt.bar(hours, original_load, width=width, label='Consommation résiduelle initiale', color='lightgray')
        plt.bar(hours, -curtail, width=width, label='Énergie turbinée', color='firebrick')  # négatif = turbinage
        plt.bar(hours, actual_pump, width=width, label='Énergie pompée', color='royalblue')
        plt.plot(hours, curtailed_energy, label='Consommation résiduelle écrêtée', color='black', linewidth=1.5)

        # Affichage des seuils
        if index_ecretement < n_thresholds:
            plt.axhline(y=threshold, color='red', linestyle='--', label=f"Seuil turbinage = {threshold:.2f}")
            plt.axhline(y=pump_threshold, color='blue', linestyle='--', label=f"Seuil pompage = {pump_threshold:.2f}")
        else:
            plt.axhline(y=pump_threshold, color='blue', linestyle='--', label=f"Seuil pompage seul = {pump_threshold:.2f}")

        plt.title(
            f"Turbinage = {np.sum(curtail):.1f} MWh | Pompage = {np.sum(actual_pump):.1f} MWh | "
            f"Contrôle = {control_total:.1f} MWh | Gain = {gain_total:.2f}"
        )
        plt.xlabel("Heures")
        plt.ylabel("Énergie (MW)")
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()



# gain=GainFunctionHydro("C:/Users/brescianomat/Documents/Calculs de trajectoires de cibles et contraintes LT/Heuristique hydro/stockage H2/stockage_h2", "fr")
# gain.plot_gain_function(40,0,2)
# gain.plot_load(40,0,2,50)