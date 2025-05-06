from read_antares_data import ResidualLoad
import numpy as np
from gain_function_tempo import GainFunctionTempo
from bellman_and_usage_values_tempo import BellmanValuesTempo
from bellman_and_usage_values_tempo import TrajectoriesTempo
import plotly.graph_objects as go


dir_study="C:/Users/brescianomat/Documents/Etudes Antares/BP23_A-Reference_2036"
area="fr"

residual_load=ResidualLoad(dir_study=dir_study,name_area=area)

gain_function_tempo_r=GainFunctionTempo(residual_load=residual_load,max_control=5)
gain_function_tempo_wr=GainFunctionTempo(residual_load=residual_load,max_control=6)

bellman_values_r=BellmanValuesTempo(gain_function=gain_function_tempo_r,capacity=22,nb_week=22,start_week=18)
bellman_values_wr=BellmanValuesTempo(gain_function=gain_function_tempo_wr,capacity=65,nb_week=53,start_week=9)

trajectories_r=TrajectoriesTempo(bv=bellman_values_r)
trajectories_white_and_red=TrajectoriesTempo(bv=bellman_values_wr,trajectories_red=trajectories_r.trajectories)

# Affichage des courbes

import plotly.graph_objects as go

# Nombre de scénarios
nb_scenarios = trajectories_white_and_red.nb_scenarios

# Semaines
weeks = np.arange(trajectories_white_and_red.nb_week)

# Création des traces pour chaque scénario
fig = go.Figure()

for s in range(nb_scenarios):
    stock_r = trajectories_r.stock_for_scenario(s)
    stock_wr = trajectories_white_and_red.stock_for_scenario(s)
    stock_w = trajectories_white_and_red.white_stock_for_scenario(s)

    visible = (s == 0)  # Seul le scénario 0 est visible par défaut

    # Décalage de la courbe de stock rouge : None jusqu'à la semaine 9
    stock_r_shifted = [None]*9 + list(stock_r[:])

    fig.add_trace(go.Scatter(
        x=weeks,
        y=stock_r_shifted,
        name="Stock jours rouges",
        visible=visible,
        line=dict(color='red'),
        legendgroup=f"scen{s}",
        showlegend=True if s == 0 else False
    ))

    fig.add_trace(go.Scatter(
        x=weeks,
        y=stock_w,
        name="Stock jours blancs",
        visible=visible,
        line=dict(color='green'),
        legendgroup=f"scen{s}",
        showlegend=True if s == 0 else False
    ))

# Ajout du menu déroulant
buttons = []

# Boutons pour chaque scénario individuel
for s in range(nb_scenarios):
    visibility = [False] * (2 * nb_scenarios)
    visibility[2*s] = visibility[2*s + 1] = visibility[2*s + 1] = True

    buttons.append(dict(
        label=f"Scénario {s}",
        method="update",
        args=[{"visible": visibility},
              {"title": f"Stocks des jours Tempo - Scénario {s}"}]
    ))

# Mise à jour de la mise en page avec les nouveaux boutons
fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction="down",
        x=1.05,
        y=1,
        showactive=True
    )],
    title="Stocks des jours Tempo - Scénario 0",
    xaxis_title="Semaine",
    yaxis_title="Stock de jours restants",
    legend=dict(x=0, y=-0.2, orientation="h"),
    margin=dict(t=80)
)

fig.show()
