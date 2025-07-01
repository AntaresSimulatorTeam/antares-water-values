from colorama import init
import numpy as np
import pandas as pd
initial_level = 7100000
from read_antares_data import Reservoir
from proxy_stage_cost_function import Proxy
from matplotlib import pyplot as plt
scenarios = range(200)
nb_weeks = 52
df = pd.read_csv("C:/Users/brescianomat/Documents/Etudes Antares/BP23_tronquee_france_sts/exports_hydro_trajectories/trajectories.csv")
df["mcYear"] = df["mcYear"] - 1
df["week"] = df["week"] - 1
trajectories = df.pivot(index="mcYear", columns="week", values="hlevel").sort_index(axis=0).sort_index(axis=1).to_numpy()
reservoir = Reservoir("C:/Users/brescianomat/Documents/Etudes Antares/BP23_tronquee_france","fr")
cost_function = Proxy("C:/Users/brescianomat/Documents/Etudes Antares/BP23_tronquee_france","fr",200)

def daily_to_hourly_curve(daily_curve: np.ndarray) -> np.ndarray:
    n_days = len(daily_curve)
    n_hours = (n_days - 1) * 24 + 1
    hourly_curve = np.interp(
        np.arange(n_hours),
        np.arange(0, n_days) * 24,
        daily_curve
    )
    last_val = daily_curve[-1]
    final_interp = np.linspace(last_val, initial_level, 25)[1:-1]
    hourly_curve = np.concatenate([hourly_curve, final_interp])
    return hourly_curve

def new_lower_rule_curve()->np.ndarray:
    upper_curves=np.zeros((len(scenarios),nb_weeks,168))
    for s in scenarios:
        for w in range(nb_weeks):
            stock_init = trajectories[s,w-1] if w>0 else initial_level
            stock_final = trajectories[s,w]

            pump_max = np.repeat(reservoir.max_daily_pumping[w * 7:(w + 1) * 7], 24) / 24
            turb_max = np.repeat(reservoir.max_daily_generating[w * 7:(w + 1) * 7], 24) / 24
            inflows = np.repeat(reservoir.daily_inflow[w * 7:(w + 1) * 7,s],24)/24

            cumsum_pump = np.concatenate([[0],np.cumsum(pump_max * cost_function.efficiency+inflows)])[:-1] + stock_init
            cumsum_turb = stock_final + np.cumsum(cost_function.turb_efficiency*turb_max[::-1] + inflows[::-1])[::-1]

            if s==57 and w==37:
                print(cumsum_pump)
                print(cumsum_turb)
                print(np.minimum(cumsum_turb,cumsum_pump))
                plt.plot(cumsum_pump, label='Cumsum Pump')
                plt.plot(cumsum_turb, label='Cumsum Turb')
                plt.plot(daily_to_hourly_curve(reservoir.daily_bottom_rule_curve)[w*168:(w+1)*168],linestyle = "--", label='Lower Rule Curve')
                plt.plot(np.minimum(cumsum_turb,cumsum_pump), label='Hourly Curve')
                plt.show()

            hourly_curve = np.minimum(cumsum_pump, cumsum_turb)
            
            upper_curves[s,w]=hourly_curve

    weekly_envelope=np.min(upper_curves,axis=0)
    hourly_envelope=weekly_envelope.flatten()
    hourly_envelope=np.concatenate([hourly_envelope, hourly_envelope[-24:]])
    hourly_lower_rule_curve = daily_to_hourly_curve(reservoir.daily_bottom_rule_curve)
    final_lower_rule_curve=np.minimum(hourly_envelope,hourly_lower_rule_curve)
    plt.plot(final_lower_rule_curve, label='Final Lower Rule Curve')
    plt.plot(daily_to_hourly_curve(reservoir.daily_bottom_rule_curve), linestyle='--', label='Original Bottom Rule Curve')
    plt.show()
    return final_lower_rule_curve

def new_upper_rule_curve() -> np.ndarray:
    lower_curves = np.zeros((len(scenarios), nb_weeks, 168))
    
    for s in scenarios:
        for w in range(nb_weeks):
            stock_init = trajectories[s, w - 1] if w > 0 else initial_level
            stock_final = trajectories[s, w]

            turb_max = np.repeat(reservoir.max_daily_generating[w * 7:(w + 1) * 7], 24) / 24
            pump_max = np.repeat(reservoir.max_daily_pumping[w * 7:(w + 1) * 7], 24) / 24
            inflows = np.repeat(reservoir.daily_inflow[w * 7:(w + 1) * 7, s], 24) / 24

            cumsum_turb = np.concatenate([[0],np.cumsum(-turb_max * cost_function.turb_efficiency + inflows)])[:-1] +stock_init
            cumsum_pump = stock_final - np.cumsum(cost_function.efficiency*pump_max[::-1]  + inflows[::-1])[::-1]
            # if s==57 and w==37:
            #     print(cumsum_turb)
            #     print(cumsum_pump)
            #     print(np.maximum(cumsum_turb,cumsum_pump))
            #     plt.plot(cumsum_turb, label='Cumsum Turb')
            #     plt.plot(cumsum_pump, label='Cumsum Pump')
            #     plt.plot(daily_to_hourly_curve(reservoir.daily_upper_rule_curve)[w*168:(w+1)*168],linestyle = "--", label='Upper Rule Curve')
            #     plt.plot(np.maximum(cumsum_turb,cumsum_pump), label='Hourly Curve')
            #     plt.show()
            hourly_curve = np.maximum(cumsum_turb,cumsum_pump)
                
            lower_curves[s, w] = hourly_curve

    weekly_envelope = np.max(lower_curves, axis=0)
    hourly_envelope = weekly_envelope.flatten()
    hourly_envelope = np.concatenate([hourly_envelope, hourly_envelope[-24:]])
    hourly_upper_rule_curve = daily_to_hourly_curve(reservoir.daily_upper_rule_curve)
    final_upper_rule_curve=np.maximum(hourly_envelope,hourly_upper_rule_curve)
    # plt.plot(final_upper_rule_curve, label='Final Upper Rule Curve')
    # plt.plot(daily_to_hourly_curve(reservoir.daily_upper_rule_curve), linestyle='--', label='Original Upper Rule Curve')
    # plt.show()
    return final_upper_rule_curve

# np.savetxt("C:/Users/brescianomat/Documents/Etudes Antares/BP23_tronquee_france_sts/exports_hydro_trajectories/interpolated_bottom_dule_curve.csv", daily_to_hourly_curve(reservoir.daily_bottom_rule_curve), delimiter=",")
# np.savetxt("C:/Users/brescianomat/Documents/Etudes Antares/BP23_tronquee_france_sts/exports_hydro_trajectories/pmax_hourly.csv", np.repeat(reservoir.max_daily_generating, 24) / 24, delimiter=",")
# new_lower_curve = new_lower_rule_curve()
# new_upper_curve = new_upper_rule_curve()