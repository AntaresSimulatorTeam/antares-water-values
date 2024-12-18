import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from read_antares_data import TimeScenarioParameter
from reservoir_management import MultiStockManagement
from type_definition import Array1D, Dict


def draw_usage_values(
    usage_values: Dict[str, Array1D],
    levels_uv: Array1D,
    n_weeks: int,
    nSteps_bellman: int,
    multi_stock_management: MultiStockManagement,
    trajectory: Array1D,
    ub: float = 6000,
) -> None:
    mult = 3
    reinterpolated_usage_values = {
        area: np.zeros((n_weeks, mult * nSteps_bellman))
        for area in multi_stock_management.dict_reservoirs.keys()
    }
    for i, (area, mng) in enumerate(multi_stock_management.dict_reservoirs.items()):
        lvls = np.linspace(
            0, mng.reservoir.capacity, mult * nSteps_bellman, endpoint=False
        )
        for week in range(n_weeks):
            diff_to_levels = np.abs(lvls[:, None] - levels_uv[None, week, :, i])
            closest_level = np.argmin(diff_to_levels, axis=1)
            reinterpolated_usage_values[area][week] = usage_values[area][week][
                closest_level
            ]

    usage_values_plot = go.Figure(
        data=[
            go.Heatmap(
                x=np.arange(n_weeks),
                y=np.linspace(
                    0, mng.reservoir.capacity, mult * nSteps_bellman, endpoint=False
                ),
                z=np.maximum(
                    np.zeros(reinterpolated_usage_values[area].T.shape) - ub / 10,
                    np.minimum(
                        ub * np.ones(reinterpolated_usage_values[area].T.shape),
                        reinterpolated_usage_values[area].T,
                    ),
                ),
                visible=(i == 0),
                showlegend=False,
            )
            for i, (area, mng) in enumerate(
                multi_stock_management.dict_reservoirs.items()
            )
        ]
        + [
            go.Scatter(
                x=np.arange(n_weeks + 1),
                y=np.mean(trajectory, axis=2)[:, i],
                visible=(i == 0),
                name=f"Trajectory",
                mode="markers",
                marker=dict(symbol="circle"),
                showlegend=True,
            )
            for i, (_, _) in enumerate(multi_stock_management.dict_reservoirs.items())
        ]
        + [
            go.Scatter(
                x=np.arange(n_weeks + 1),
                y=mng.reservoir.bottom_rule_curve[: n_weeks + 1],
                visible=(i == 0),
                name=f"Curve down",
                mode="lines",
                line=dict(dash="dash"),
                showlegend=True,
            )
            for i, (_, mng) in enumerate(multi_stock_management.dict_reservoirs.items())
        ]
        + [
            go.Scatter(
                x=np.arange(n_weeks + 1),
                y=mng.reservoir.upper_rule_curve[: n_weeks + 1],
                visible=(i == 0),
                name=f"Curve high",
                mode="lines",
                line=dict(dash="dash"),
                showlegend=True,
            )
            for i, (_, mng) in enumerate(multi_stock_management.dict_reservoirs.items())
        ],
        layout=dict(title=f"Usage Values"),
    )
    usage_values_plot.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": f"Reservoir {area}",
                        "method": "update",
                        "args": [
                            {
                                "visible": [
                                    area_b == area
                                    for area_b in multi_stock_management.dict_reservoirs.keys()
                                ]
                                * 4
                            },
                        ],
                    }
                    for area in multi_stock_management.dict_reservoirs.keys()
                ],
                "x": 0.0,
                "xanchor": "left",
                "y": 1.13,
                "yanchor": "top",
            },
        ]
    )
    usage_values_plot.show()


def draw_uvs_sddp(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    usage_values: np.ndarray,
    simulation_results: list[dict],
    lim: float = 700.0,
    div: float = 1.0,
) -> None:
    n_weeks = param.len_week
    reservoirs = [
        mng.reservoir for mng in multi_stock_management.dict_reservoirs.values()
    ]
    usage_values = np.array(usage_values)
    disc = usage_values.shape[1]
    uvs_fig = go.Figure(
        data=[
            go.Heatmap(
                x=np.arange(n_weeks),
                y=np.linspace(0, res.capacity, disc),
                z=usage_values[:, :, r].T,
                zmin=-lim,
                zmax=lim,
                showscale=False,
                visible=(r == 0),
            )
            for r, res in enumerate(reservoirs)
        ]
        + [
            go.Scatter(
                x=np.arange(n_weeks),
                y=np.mean(
                    [
                        [sim[week]["level_out"][r] * div for week in range(n_weeks)]
                        for sim in simulation_results
                    ],
                    axis=0,
                ),
                name="SDDP trajectory",
                showlegend=True,
                visible=(r == 0),
            )
            for r, _ in enumerate(reservoirs)
        ]
        + [
            go.Scatter(
                x=np.arange(n_weeks),
                y=res.bottom_rule_curve,
                showlegend=True,
                name="Bottom rule curve",
                visible=(r == 0),
            )
            for r, res in enumerate(reservoirs)
        ],
        layout=dict(
            title="Water Values and Trajectories found by SDDP",
            xaxis={"title": "Week"},
            yaxis={"title": "Reservoir level (MWh)"},
        ),
    )
    uvs_fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": f"Reservoir {area}",
                        "method": "update",
                        "args": [
                            {
                                "visible": [
                                    area_b == area
                                    for area_b in multi_stock_management.dict_reservoirs.keys()
                                ]
                                * 3
                            },
                        ],
                    }
                    for area in multi_stock_management.dict_reservoirs.keys()
                ],
                "x": 0.0,
                "xanchor": "left",
                "y": 1.13,
                "yanchor": "top",
            },
        ]
    )
    uvs_fig.show()


class ConvergenceProgressBar:

    def __init__(
        self,
        convergence_goal: float = 1e-2,
        maxiter: int = 12,
        degrowth: int = 4,
    ) -> None:
        self.tot_progress = 0
        self.degrowth = degrowth
        self.convergence_goal = convergence_goal
        self.iteration = 0
        self.precision = 1
        self.maxiter = maxiter
        self.pbar = tqdm(
            total=100, desc=f"Opt δ: 1, budget: 0/{maxiter}", colour="green"
        )

    def update(self, precision: float) -> None:
        self.iteration += 1
        progress = (
            100
            * (
                1
                - np.emath.logn(self.degrowth, self.convergence_goal / precision)
                / np.emath.logn(self.degrowth, self.convergence_goal)
            )
            - self.tot_progress
        )
        progress = np.round(progress, 2)
        progress = min(progress, 100 - self.tot_progress)
        self.tot_progress += progress
        self.precision = np.round(precision, 3)
        self.pbar.set_description_str(
            desc=f"Opt δ: {self.precision}, Iter: {self.iteration}/{self.maxiter}"
        )
        self.pbar.update(progress)

    def describe(self, description: str) -> None:
        self.pbar.set_description_str(
            desc=f"{description} | Opt δ: {self.precision}, Iter: {self.iteration}/{self.maxiter}"
        )

    def close(self) -> None:
        self.pbar.close()
