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

        self.nb_weeks=self.reservoir.inflow.shape[0]
        self.scenarios=range(self.net_load.shape[1])
        # self.scenarios=range(10)

    def gain_function(self, week_index: int, scenario: int, alpha: float) -> tuple[interp1d, np.ndarray]:
        net_load_for_week = self.net_load[week_index * 168:(week_index + 1) * 168, scenario]
        max_energy_hour = np.repeat(self.max_daily_generating[week_index * 7:(week_index + 1) * 7], 24) / 24
        max_pumping_hour = np.repeat(self.max_daily_pumping[week_index * 7:(week_index + 1) * 7], 24) / 24

        curtailed_energy = net_load_for_week - max_energy_hour
        control = np.sum(max_energy_hour)
        gain = np.sum(curtailed_energy ** alpha / 1e9)
        
        control_list = [control]
        gain_list = [gain]

        # turb_thresholds=np.sort(net_load_for_week)
        turb_thresholds = np.quantile(net_load_for_week, np.linspace(0, 1, 25))

        for turb_threshold in turb_thresholds:
            curtailed_energy = np.minimum(net_load_for_week, np.maximum(turb_threshold, net_load_for_week - max_energy_hour))
            curtail = net_load_for_week - curtailed_energy
            curtail = np.clip(curtail, None, max_energy_hour)
            # print(f"seuil de turbinage : {turb_threshold}")
            # print(f"quantité turbinée : {np.sum(curtail)}")


            if turb_threshold<0:
                pump_threshold=turb_threshold
            else:
                pump_threshold = self.efficiency * turb_threshold
            potential_pump = pump_threshold - curtailed_energy

            mask = curtailed_energy < pump_threshold
            actual_pump = np.minimum(potential_pump, max_pumping_hour) * mask

            # print(f"seuil de pompage : {pump_threshold}")
            # print(f"quantité pompée : {np.sum(actual_pump)}")
            curtailed_energy += actual_pump

            control_hourly = curtail - actual_pump / self.efficiency
            control = np.sum(control_hourly)
            # print(f"controle total effectué : {control}")

            gain = np.sum(curtailed_energy ** alpha / 1e9)
            # print(f"gain associé au controle {control} : {gain}")
            control_list.append(control)
            gain_list.append(gain)

        return interp1d(control_list, gain_list, fill_value="extrapolate"), np.array(gain_list)


    def compute_gain_functions(self,alpha:float)->np.ndarray:
        gain_functions=np.array([[self.gain_function(w,s,alpha)[0] for s in self.scenarios] for w in range(self.nb_weeks)])
        return gain_functions
    
   
    # affichages

    def plot_gain_function(self,week_index:int,scenario:int,alpha:float)->None:
        gain_func = self.gain_function(week_index, scenario, alpha)
        control_values = gain_func[0].x
        gain_values = gain_func[0](control_values)
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

        curtailed_energy = original_load - max_energy_hour

        turb_thresholds = np.quantile(original_load, np.linspace(0, 1, 25))
        threshold = turb_thresholds[index_ecretement]

        curtailed_energy = np.minimum(original_load, np.maximum(threshold, original_load - max_energy_hour))
        curtail = original_load - curtailed_energy
        curtail = np.clip(curtail, None, max_energy_hour)

        if threshold < 0:
            pump_threshold = threshold
        else:
            pump_threshold = self.efficiency * threshold

        potential_pump = pump_threshold - curtailed_energy
        mask = curtailed_energy < pump_threshold
        actual_pump = np.minimum(potential_pump, max_pumping_hour) * mask

        curtailed_energy += actual_pump

        turbined_total = np.sum(curtail)
        pumped_total = np.sum(actual_pump)
        control_total = turbined_total - pumped_total / self.efficiency
        gain_total = np.sum(curtailed_energy ** alpha / 1e9)

        hours = np.arange(len(original_load))
        width = 0.4

        plt.figure(figsize=(15, 6))
        plt.bar(hours, original_load, width=width, label='Consommation résiduelle initiale', color='lightgray')
        plt.bar(hours, -curtail, width=width, label='Énergie turbinée', color='firebrick')  # négatif = retiré
        plt.bar(hours, actual_pump, width=width, label='Énergie pompée', color='royalblue')
        plt.plot(hours, curtailed_energy, label='Consommation résiduelle écrêtée', color='black', linewidth=1.5)

        # Affichage des seuils
        plt.axhline(y=threshold, color='red', linestyle='--', label=f"Seuil turbinage = {threshold:.2f}")
        plt.axhline(y=pump_threshold, color='blue', linestyle='--', label=f"Seuil pompage = {pump_threshold:.2f}")

        # Informations complémentaires dans le titre
        plt.title(
            f"Turbinage = {turbined_total:.1f} MWh | Pompage = {pumped_total:.1f} MWh | "
            f"Contrôle = {control_total:.1f} MWh | Gain = {gain_total:.2f}"
        )

        plt.xlabel("Heures")
        plt.ylabel("Énergie (MW)")
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


# gain=GainFunctionHydro("/test_data/one_node(1)", "area")
# gain.plot_gain_function(0,0,2)
# gain.plot_load(0,0,2,3)