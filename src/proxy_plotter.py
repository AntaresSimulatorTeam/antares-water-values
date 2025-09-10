import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from proxy_bellman_trajectories import BellmanValuesProxy, OptimalTrajectories
import plotly.graph_objects as go
import plotly.express as px
import os

class Plotter:
    def __init__(self, bv: BellmanValuesProxy, trajectories: OptimalTrajectories):
        """
        Initialize Plotter with BellmanValuesProxy and OptimalTrajectories instances.
        """
        self.bv = bv
        self.trajectories = trajectories

    def plot_bellman_value(self, week_index: int) -> None:
        """
        Plot Bellman value as a function of stock level for a given week.
        Raises ValueError if the week_index is out of bounds.
        """
        if week_index < 0 or week_index >= self.bv.nb_weeks:
            raise ValueError(f"Invalid week: {week_index}. Must be between 0 and {self.bv.nb_weeks - 1}.")

        stock_levels = np.linspace(0, 100, 51)
        bellman_values = self.bv.mean_bv[week_index, :]

        plt.figure(figsize=(10, 5))
        plt.plot(stock_levels, bellman_values, label=f"Week {week_index + 1}", color='tab:blue')

        plt.xlabel("Stock (%)")
        plt.ylabel("Bellman Value")
        area = getattr(self.bv.proxy, 'name_area', None) or getattr(self.bv, 'area', None)
        area_str = f" - Area: {area}" if area else ""
        plt.title(f"Bellman Value vs Stock - Week {week_index + 1}{area_str}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_usage_values(self) -> None:
        """
        Plot usage values as a function of stock level for all weeks.
        """
        stock_levels = np.linspace(2, 100, 50)
        plt.figure(figsize=(12, 6))

        for w in range(self.bv.nb_weeks):
            plt.plot(
                stock_levels,
                self.bv.usage_values[w],
                label=f"W {w+1}"
            )

        plt.xlabel('Stock (%)')
        plt.ylabel('Usage Value (MWh)')
        area = getattr(self.bv.proxy, 'name_area', None) or getattr(self.bv, 'area', None)
        area_str = f" - Area: {area}" if area else ""
        plt.title(f"Usage Values vs Stock{area_str}")
        plt.legend(
            loc='lower right',
            bbox_to_anchor=(1, -0.15),
            ncol=6
        )
        plt.tight_layout(rect=(0, 0.1, 1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_usage_values_heatmap(self) -> None:
        """
        Plot a heatmap of usage values over weeks and stock levels.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        norm = colors.Normalize(-3e10,0)

        im = ax.imshow(
            self.bv.usage_values[:-1].T,
            aspect='auto',
            origin='lower',
            cmap='nipy_spectral',
            extent=(1, 52, 2, 100),
            norm=norm,
            interpolation='bilinear'  # smoothing
        )

        cbar = fig.colorbar(im, ax=ax, ticks=np.linspace(-3e10, 0,10))
        cbar.set_label("Usage Value")

        ax.set_xlabel("Week")
        ax.set_ylabel("Stock (%)")
        area = getattr(self.bv.proxy, 'name_area', None) or getattr(self.bv, 'area', None)
        area_str = f" - Area: {area}" if area else ""
        ax.set_title(f"Usage Value Heatmap (α={self.bv.proxy.alpha}){area_str}")

        plt.grid(False)
        plt.tight_layout()
        plt.show()

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
        if not isinstance(self.bv.export_dir, str) or not self.bv.export_dir:
            raise ValueError("export_dir must be a non-empty string before saving the plot.")
        html_path = os.path.join(self.bv.export_dir, "trajectories_plot.html")
        fig.write_html(html_path)
        # print(f"Interactive plot saved at: {html_path}")

    def plot_all_trajectories_pyplot(self) -> None:
        """
        Plot all stock trajectories and rule curves with Matplotlib.
        """
        weeks = np.arange(1, self.bv.nb_weeks + 1)
        n_scenarios = len(self.bv.scenarios)
        color_palette = plt.cm.get_cmap('tab20', n_scenarios)

        plt.figure(figsize=(14, 7))

        plt.plot(
            weeks,
            self.bv.proxy.reservoir.weekly_upper_rule_curve[:self.bv.nb_weeks] / self.bv.proxy.reservoir.capacity * 100,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Upper rule curve"
        )

        plt.plot(
            weeks,
            self.bv.proxy.reservoir.weekly_lower_rule_curve[:self.bv.nb_weeks] / self.bv.proxy.reservoir.capacity * 100,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Lower rule curve"
        )

        for s in range(n_scenarios):
            stock = self.trajectories.trajectories[s]
            plt.plot(
                weeks,
                stock / self.bv.proxy.reservoir.capacity * 100,
                color=color_palette(s)
            )

        plt.xlabel("Week", fontsize=14)
        plt.ylabel("Stock (%)", fontsize=14)
        area = getattr(self.bv.proxy, 'name_area', None) or getattr(self.bv, 'area', None)
        area_str = f" - Area: {area}" if area else ""
        plt.title(f"Stock Trajectories with Rule Curves - All MC (α={self.bv.proxy.alpha}){area_str}", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=12, loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_adjusted_rule_curves(self) -> None:
        """
        Plot adjusted versus interpolated hourly rule curves (upper and lower).
        Does nothing if the trajectories do not have adjusted curves computed.
        """
        plt.figure(figsize=(16, 6))
        if hasattr(self.trajectories, "final_lower_rule_curve") and hasattr(self.trajectories, "final_upper_rule_curve"):
            plt.plot(self.trajectories.final_lower_rule_curve / self.bv.proxy.reservoir.capacity * 100, label="Adjusted Lower", color="blue", linewidth=2)
            plt.plot(self.trajectories.hourly_lower_rule_curve / self.bv.proxy.reservoir.capacity * 100, label="Interpolated Lower", color="cyan", linestyle="--", linewidth=1.5)

            plt.plot(self.trajectories.final_upper_rule_curve / self.bv.proxy.reservoir.capacity * 100, label="Adjusted Upper", color="darkred", linewidth=2)
            plt.plot(self.trajectories.hourly_upper_rule_curve / self.bv.proxy.reservoir.capacity * 100, label="Interpolated Upper", color="orange", linestyle="--", linewidth=1.5)
        else:
            return
        plt.xlabel("Hour of the year")
        plt.ylabel("Stock (%)")
        area = getattr(self.bv.proxy, 'name_area', None) or getattr(self.bv, 'area', None)
        area_str = f" - Area: {area}" if area else ""
        plt.title(f"Hourly Rule Curves: Adjusted vs Interpolated{area_str}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
