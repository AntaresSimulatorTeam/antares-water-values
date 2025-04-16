from pathlib import Path
from read_antares_data import Residual_load
import numpy as np
from gain_function_TEMPO import GainFunctionTEMPO
from bellman_values import Bellman_values
from usage_values import UV_tempo
from trajectories import Trajectories
import matplotlib.pyplot as plt
import plotly.graph_objects as go


dir_study="C:/Users/brescianomat/Documents/Etudes Antares/BP23_A-Reference_2036"
area="fr"

residual_load_1=Residual_load(dir_study=dir_study,name_area=area)
residual_load_2=Residual_load(dir_study=dir_study,name_area=area)

gain_func_red=GainFunctionTEMPO(residual_load=residual_load_1,max_control=5)
bellman_red=Bellman_values(residual_load=residual_load_1,
                           gain_function=gain_func_red,
                           capacity=22,
                           nb_week=22,
                           max_control=5,
                           start_week=18)
uv_red=UV_tempo(residual_load=residual_load_1,
                    gain_function=gain_func_red,
                    bellman_values=bellman_red)

gain_func_white_and_red=GainFunctionTEMPO(residual_load=residual_load_2,max_control=6)
bellman_white_and_red=Bellman_values(residual_load=residual_load_2,
                                     gain_function=gain_func_white_and_red,
                                     capacity=65,
                                     nb_week=53,
                                     max_control=6,
                                     start_week=9)
uv_white_and_red=UV_tempo(residual_load=residual_load_2,
                              gain_function=gain_func_white_and_red,
                              bellman_values=bellman_white_and_red)


# print(bellman_white_and_red.gain_function.daily_residual_load_for_week)
# print(uv_red.bellman_values.mean_bv[18:40,:])
# print(uv_white_and_red.bellman_values.mean_bv[9:62,:])
# print(uv_white_and_red.usage_values[9:62,:])

trajectories_red= Trajectories(
    residual_load=residual_load_1,
    gain_function=gain_func_red,
    bellman_values=bellman_red,
    usage_values=uv_red,
    capacity=22,
    nb_week=22,
    max_control=5
)



trajectories_white_and_red = Trajectories(
    residual_load=residual_load_2,
    gain_function=gain_func_white_and_red,
    bellman_values=bellman_white_and_red,
    usage_values=uv_white_and_red,
    capacity=65,
    nb_week=53,
    max_control=6,
    trajectories_red=trajectories_red.trajectories
)


class TempoStockPlotter:
    def __init__(self, trajectories_red:Trajectories, trajectories_wr:Trajectories, start_red:int, end_red:int):
        self.trajectories_red = trajectories_red
        self.trajectories_wr = trajectories_wr
        self.start_red = start_red
        self.end_red = end_red


        self.stock_red = self.compute_stock(self.trajectories_red)
        self.stock_wr = self.compute_stock(self.trajectories_wr)
        self.stock_white = self.compute_white_stock()

    def compute_stock(self, trajectories:Trajectories)->np.ndarray:
        nb_weeks = trajectories.nb_week
        nb_scenarios = 200
        stock = np.zeros((nb_scenarios, nb_weeks))
        for s in range(nb_scenarios):
            current_stock = [trajectories.capacity]
            for control in trajectories.trajectory_for_scenario(s):
                current_stock.append(current_stock[-1] - control)
            stock[s, :] = current_stock[1:]
        return stock

    def compute_white_stock(self)->np.ndarray:
        stock_white = np.zeros_like(self.stock_wr)
        for s in range(200):
            # Avant période rouge
            stock_white[s, :self.start_red] = self.stock_wr[s, :self.start_red]-np.full(self.start_red,22)
            # Pendant période rouge
            red_weeks = self.end_red - self.start_red
            # stock_white[s, self.start_red:self.end_red] = self.stock_wr[s, self.start_red:self.end_red] - self.stock_red[s, :red_weeks]
            stock_white[s, self.start_red:self.end_red] = (self.stock_wr[s, self.start_red:self.end_red] - 
            self.stock_red[s,:]
)
            # Après période rouge
            stock_white[s, self.end_red:] = self.stock_wr[s, self.end_red:]
        return stock_white

    def show_interactive_plot(self)-> None:
        fig = go.Figure()

        for s in range(self.stock_wr.shape[0]):
            visible = (s == 0)  # Seul le 1er scénario visible par défaut

            # Rouge
            fig.add_trace(go.Scatter(
                y=self.stock_red[s, :],
                x=np.arange(self.start_red+1, self.end_red+1),
                name="Stock Rouge",
                visible=visible,
                line=dict(color="red", dash="dot")
            ))

            # Rouge + Blanc
            fig.add_trace(go.Scatter(
                y=self.stock_wr[s, :],
                x=np.arange(1, self.stock_wr.shape[1] + 1),
                name="Stock Rouge + Blanc",
                visible=visible,
                line=dict(color="orange")
            ))

            # Blanc
            fig.add_trace(go.Scatter(
                y=self.stock_white[s, :],
                x=np.arange(1, self.stock_wr.shape[1] + 1),
                name="Stock Blanc",
                visible=visible,
                line=dict(color="blue", dash="dash")
            ))

        steps = []
        for i in range(self.stock_wr.shape[0]):
            vis = [False] * 3 * self.stock_wr.shape[0]
            vis[3 * i] = True     # red
            vis[3 * i + 1] = True # wr
            vis[3 * i + 2] = True # white

            step = dict(
                method="update",
                args=[{"visible": vis},
                      {"title": f"Scénario {i}"}],
                label=f"{i}"
            )
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Scénario : "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            title="Stocks TEMPO - Rouge, Rouge+Blanc, Blanc",
            xaxis_title="Semaine",
            yaxis_title="Stock",
            height=600
        )

        fig.show()

plotter = TempoStockPlotter(trajectories_red, trajectories_white_and_red,start_red=9,end_red=31)
plotter.show_interactive_plot()

print(plotter.compute_stock(trajectories=trajectories_red)[0])
print(plotter.compute_stock(trajectories=trajectories_white_and_red)[0])
print(plotter.stock_white[0])