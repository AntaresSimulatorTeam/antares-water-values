from proxy_bellman_trajectories import BellmanValuesProxy, OptimalTrajectories
import plotly.graph_objects as go
import plotly.express as px
import os

class Plotter:
    def __init__(self, bv: BellmanValuesProxy, trajectories: OptimalTrajectories, export_dir:str):
        """
        Initialize Plotter with BellmanValuesProxy and OptimalTrajectories instances.
        """
        self.bv = bv
        self.trajectories = trajectories
        self.export_dir=export_dir

    def plot_trajectories(self) -> None:
        """
        Plot stock trajectories along with upper and lower rule curves interactively using Plotly.
        Includes buttons to toggle scenario visibility.
        Saves the plot as an HTML file in the export directory.
        """
        fig = go.Figure()
        weeks = list(range(1, self.bv.nb_weeks + 2))

        upper_percent = self.bv.proxy.reservoir.weekly_upper_rule_curve / self.bv.proxy.reservoir.capacity * 100
        fig.add_trace(go.Scatter(
            x=weeks,
            y=upper_percent,
            mode='lines',
            name='Upper rule curve',
            line=dict(dash='dash', color='green'),
            visible=True
        ))

        lower_percent = self.bv.proxy.reservoir.weekly_lower_rule_curve / self.bv.proxy.reservoir.capacity * 100
        fig.add_trace(go.Scatter(
            x=weeks,
            y=lower_percent,
            mode='lines',
            name='Lower rule curve',
            line=dict(dash='dash', color='red'),
            visible=True
        ))

        colors = px.colors.qualitative.Plotly

        for s in self.bv.scenarios:
            visible = True if s == self.bv.scenarios[0] else False
            color = colors[s % len(colors)]
            stock_percent = self.trajectories.trajectories[s] / self.bv.proxy.reservoir.capacity * 100
            fig.add_trace(go.Scatter(
                x=weeks,
                y=stock_percent,
                mode='lines',
                name=f'MC {s + 1}',
                line=dict(color=color),
                visible=visible
            ))

        n_scenarios = len(self.bv.scenarios)
        n_shared_guides = 2
        buttons = []

        area = getattr(self.bv.proxy, 'name_area', None) or getattr(self.bv, 'area', None)
        area_str = f" - Area: {area}" if area else ""

        for i, s in enumerate(self.bv.scenarios):
            visibility = [True] * n_shared_guides + [False] * n_scenarios
            visibility[n_shared_guides + i] = True
            buttons.append(dict(
                label=f"Scenario {s + 1}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title.text": f"Stock Trajectory - MC {s + 1}{area_str}"}
                ]
            ))

        visibility_all = [True] * (n_shared_guides + n_scenarios)
        buttons.append(dict(
            label="All MC",
            method="update",
            args=[
                {"visible": visibility_all},
                {"title.text": f"Stock Trajectories - All MC{area_str}"}
            ]
        ))

        fig.update_layout(
            font=dict(family="Cambria", size=14),
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=1.1,
                y=1.15,
                showactive=True
            )],
            title=dict(text=f"Stock Trajectory - MC 1{area_str}", font=dict(family="Cambria", size=18)),
            xaxis=dict(
                title="Week",
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                dtick=1,
                zeroline=False
            ),
            yaxis=dict(
                title="Stock (%)",
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                tick0=0,
                dtick=5,
                zeroline=False
            ),
            legend=dict(x=0, y=-0.2, orientation="h"),
            showlegend=False
        )

        fig.show()
        if not isinstance(self.export_dir, str) or not self.export_dir:
            raise ValueError("export_dir must be a non-empty string before saving the plot.")
        html_path = os.path.join(self.export_dir, "trajectories_plot.html")
        fig.write_html(html_path)