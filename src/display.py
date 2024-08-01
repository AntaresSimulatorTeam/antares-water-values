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
        ub=800):
    n_scenarios =trajectory.shape[-1]
    mult = 3
    reinterpolated_usage_values = {area:np.zeros((n_weeks, mult*nSteps_bellman)) for area in multi_stock_management.dict_reservoirs.keys()}
    for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items()):
        lvls = np.linspace(0, mng.reservoir.capacity, mult*nSteps_bellman, endpoint=False)
        for week in range(n_weeks):
            diff_to_levels = np.abs(lvls[:,None] - levels_uv[None,week,:,i])
            closest_level = np.argmin(diff_to_levels, axis=1)
            reinterpolated_usage_values[area][week] = usage_values[area][week][closest_level]

    # z = np.maximum(np.zeros(reinterpolated_usage_values[area].T.shape), np.minimum(ub*np.ones(reinterpolated_usage_values[area].T.shape), reinterpolated_usage_values[area].T))
    # z = np.maximum(np.zeros(reinterpolated_usage_values[area].T.shape) - ub, np.minimum(ub*np.ones(reinterpolated_usage_values[area].T.shape), reinterpolated_usage_values[area].T))
    usage_values_plot = go.Figure(
        data = [go.Heatmap(x=np.arange(n_weeks),
                        y=np.linspace(0, mng.reservoir.capacity, mult*nSteps_bellman, endpoint=False), 
                        z=np.maximum(np.zeros(reinterpolated_usage_values[area].T.shape) - ub/10, np.minimum(ub*np.ones(reinterpolated_usage_values[area].T.shape), reinterpolated_usage_values[area].T)),
                        visible=(i==0),
                        showlegend=False,)
                for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())]
                + [
                    go.Scatter(x=np.arange(n_weeks+1), y=np.mean(trajectory, axis=2)[:,i],
                                visible=(i==0), name=f"Trajectory", mode="markers", marker=dict(symbol="circle"), showlegend=True)
                    for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())
                ] 
                + [
                  go.Scatter(x=np.arange(n_weeks+1), y=mng.reservoir.bottom_rule_curve[:n_weeks+1],
                                visible=(i==0), name=f"Curve down", mode="lines", line=dict(dash="dash"), showlegend=True)
                    for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())  
                ]
                + [
                  go.Scatter(x=np.arange(n_weeks+1), y=mng.reservoir.upper_rule_curve[:n_weeks+1],
                                visible=(i==0), name=f"Curve high", mode="lines", line=dict(dash="dash"), showlegend=True)
                    for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())  
                ]
                # + [
                #   go.Scatter(x=np.arange(1,n_weeks), y=np.minimum(mng.reservoir.capacity, np.mean(trajectory, axis=2)[:-1,i]\
                #                                                   + np.mean(mng.reservoir.inflow[:n_weeks,:n_scenarios],axis=1)),
                #                 visible=(i==0), name=f"Control 0", mode="markers", marker=dict(symbol="diamond-wide", size=4.6), opacity=1, showlegend=True)
                #     for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())  
                # ]
                # + [
                #   go.Scatter(x=np.arange(1,n_weeks), y=np.maximum(0, np.mean(trajectory, axis=2)[:-1,i]\
                #                                      + np.mean(mng.reservoir.inflow[:n_weeks,:n_scenarios],axis=1)\
                #                                      - mng.reservoir.max_generating[:n_weeks]),
                #                 visible=(i==0), name=f"Max turb", mode="markers", marker=dict(symbol="arrow-up", size=4.4), opacity=1, showlegend=True)
                #     for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())  
                # ]
                # + [
                #   go.Scatter(x=np.arange(1,n_weeks), y=np.minimum(mng.reservoir.capacity, np.mean(trajectory, axis=2)[:-1,i]\
                #                                      + np.mean(mng.reservoir.inflow[:n_weeks,:n_scenarios],axis=1)\
                #                                      + mng.reservoir.max_pumping[:n_weeks] * mng.reservoir.efficiency),
                #                 visible=(i==0), name=f"Max pump", mode="markers", marker=dict(symbol="arrow-down", size=4.4), opacity=1, showlegend=True)
                #     for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items())  
                # ]
                ,
        layout=dict(title=f"Usage Values")
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