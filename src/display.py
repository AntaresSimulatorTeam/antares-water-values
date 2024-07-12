import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go

def draw_usage_values(
        usage_values,
        levels_uv,
        n_weeks,
        nSteps_bellman, 
        multi_stock_management, 
        trajectory, 
        ub=400):
    mult = 10
    reinterpolated_usage_values = {area:np.zeros((n_weeks, mult*nSteps_bellman)) for area in multi_stock_management.dict_reservoirs.keys()}
    for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items()):
        lvls = np.linspace(0, mng.reservoir.capacity, mult*nSteps_bellman, endpoint=False)
        for week in range(n_weeks):
            diff_to_levels = np.abs(lvls[:,None] - levels_uv[None,week,:,i])
            closest_level = np.argmin(diff_to_levels, axis=1)
            reinterpolated_usage_values[area][week] = usage_values[area][week][closest_level]

    #z = np.maximum(np.zeros(reinterpolated_usage_values[area].T.shape), np.minimum(ub*np.ones(reinterpolated_usage_values[area].T.shape), reinterpolated_usage_values[area].T))
    usage_values_plot = go.Figure(
        data = [go.Heatmap(x=np.arange(n_weeks),
                        y=np.linspace(0, mng.reservoir.capacity, mult*nSteps_bellman, endpoint=False), 
                        z=reinterpolated_usage_values[area].T,
                        visible=(i==0))
                for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())]
                + [
                    go.Scatter(x=np.arange(n_weeks), y=np.mean(trajectory, axis=2)[:,i],
                                visible=(i==0), name=f"Traj {area}", showlegend=False)
                    for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())
                ] 
                + [
                  go.Scatter(x=np.arange(n_weeks), y=mng.reservoir.bottom_rule_curve[:n_weeks],
                                visible=(i==0), name=f"Rule curve low {area}", showlegend=False)
                    for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())  
                ]
                + [
                  go.Scatter(x=np.arange(n_weeks), y=mng.reservoir.upper_rule_curve[:n_weeks],
                                visible=(i==0), name=f"Rule curve high {area}", showlegend=False)
                    for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())  
                ]
                ,
        layout=dict(title="Usage Values")
    )
    usage_values_plot.update_layout(
    updatemenus=[{
            "buttons": 
            [
                {
                    "label": f"Reservoir {area}",
                    "method": "update",
                    "args": 
                    [
                        {"visible": [area_b==area for area_b in multi_stock_management.dict_reservoirs.keys()]*4},
                    ],
                }
                for area in multi_stock_management.dict_reservoirs.keys()
            ],
            "x":.0,
            "xanchor":"left",
            "y":1.13,
            "yanchor":"top"
        },])
    usage_values_plot.show()


class ConvergenceProgressBar:

    def __init__(
        self,
        convergence_goal:float=1e-2,
        maxiter:int=12,
        degrowth:int=4,
        ) -> None:
        self.tot_progress = 0
        self.degrowth = degrowth
        self.convergence_goal = convergence_goal
        self.iteration = 0
        self.precision = 1
        self.maxiter = maxiter
        self.pbar = tqdm(total=100, desc=f"Opt δ: 1, budget: 0/{maxiter}", colour="green")
    
    def update(self, precision):
        self.iteration += 1
        progress = 100 * (1 - np.emath.logn(self.degrowth, self.convergence_goal/precision) /
                           np.emath.logn(self.degrowth, self.convergence_goal)) - self.tot_progress
        progress = np.round(progress, 2)
        progress = min(progress, 100 - self.tot_progress)
        self.tot_progress += progress
        self.precision = np.round(precision, 3)
        self.pbar.set_description_str(desc=f"Opt δ: {self.precision}, Iter: {self.iteration}/{self.maxiter}")
        self.pbar.update(progress)
    
    def describe(self, description):
        self.pbar.set_description_str(desc=f"{description} | Opt δ: {self.precision}, Iter: {self.iteration}/{self.maxiter}")
    
    def close(self):
        self.pbar.close()