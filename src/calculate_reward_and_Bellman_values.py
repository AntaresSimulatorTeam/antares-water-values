from typing import Callable, Annotated, Literal, Optional
from read_antares_data import AntaresParameter, Reservoir
from scipy.interpolate import interp1d
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

Array1D = Annotated[npt.NDArray[np.float32], Literal["N"]]
Array2D = Annotated[npt.NDArray[np.float32], Literal["N", "N"]]


class ReservoirManagement:

    def __init__(
        self,
        reservoir: Reservoir,
        penalty_bottom_rule_curve: float = 0,
        penalty_upper_rule_curve: float = 0,
        penalty_final_level: float = 0,
        force_final_level: bool = False,
        final_level: Optional[float] = None,
    ) -> None:

        self.reservoir = reservoir
        self.penalty_bottom_rule_curve = penalty_bottom_rule_curve
        self.penalty_upper_rule_curve = penalty_upper_rule_curve

        if force_final_level:
            self.penalty_final_level = penalty_final_level
            if final_level:
                self.final_level = final_level
            else:
                self.final_level = reservoir.initial_level
        else:
            self.final_level = False

    def get_penalty(self, s: int, S: int) -> Callable:
        """
        Return a function to evaluate penalities for violating rule curves for any level of stock.

        Parameters
        ----------
        s:int :
            Week considered
        S:int :
            Total number of weeks
        reservoir:Reservoir :
            Reservoir considered
        pen_final:float :
            Penalty for violating final rule curves
        pen_low:float :
            Penalty for violating bottom rule curve
        pen_high:float :
            Penalty for violating top rule curve

        Returns
        -------

        """
        if s == S - 1 and self.final_level:
            pen = interp1d(
                [
                    0,
                    self.final_level,
                    self.reservoir.capacity,
                ],
                [
                    -self.penalty_final_level * (self.final_level),
                    0,
                    -self.penalty_final_level
                    * (self.reservoir.capacity - self.final_level),
                ],
            )
        else:
            pen = interp1d(
                [
                    0,
                    self.reservoir.bottom_rule_curve[s],
                    self.reservoir.upper_rule_curve[s],
                    self.reservoir.capacity,
                ],
                [
                    -self.penalty_bottom_rule_curve
                    * (self.reservoir.bottom_rule_curve[s]),
                    0,
                    0,
                    -self.penalty_upper_rule_curve
                    * (self.reservoir.capacity - self.reservoir.upper_rule_curve[s]),
                ],
            )
        return pen


class RewardApproximation:
    """Class to store and update reward approximation for a given week and a given scenario"""

    def __init__(self, lb_control: float, ub_control: float, ub_reward: float) -> None:
        """
        Create a new reward approximation

        Parameters
        ----------
        lb_control:float :
            Lower possible bound on control
        ub_control:float :
            Upper possible bound on control
        ub_reward:float :
            Upper bound on reward

        Returns
        -------
        None
        """
        self.breaking_point = [lb_control, ub_control]
        self.list_cut = [(0.0, ub_reward)]

    def reward_function(self) -> Callable:
        """Return a function to evaluate reward at any point based on the current approximation."""
        return lambda x: min([cut[0] * x + cut[1] for cut in self.list_cut])

    def update_reward_approximation(
        self, lamb: float, beta: float, new_control: float
    ) -> None:
        """
        Update reward approximation by adding a new cut

        Parameters
        ----------
        lamb:float :
            Total cost that defined reward at the given control
        beta:float :
            Dual value associated with the control constraint, gives the slope of reward
        new_control:float :
            Control evaluated

        Returns
        -------
        None
        """

        Gs = self.reward_function()
        new_cut: Callable = lambda x: -lamb * x - beta + lamb * new_control
        new_reward = []
        new_points = [self.breaking_point[0]]

        if len(self.breaking_point) != len(self.list_cut) + 1:
            raise (ValueError)

        for i in range(len(self.breaking_point)):
            if i == 0:
                if new_cut(self.breaking_point[i]) < Gs(self.breaking_point[i]):
                    new_reward.append((-lamb, -beta + lamb * new_control))
                    if new_cut(self.breaking_point[i + 1]) >= Gs(
                        self.breaking_point[i + 1]
                    ):
                        if -lamb - self.list_cut[i][0] != 0:
                            new_points.append(
                                -(-beta + lamb * new_control - self.list_cut[i][1])
                                / (-lamb - self.list_cut[i][0])
                            )
                            new_reward.append(self.list_cut[i])
                elif new_cut(self.breaking_point[i]) >= Gs(self.breaking_point[i]):
                    new_reward.append(self.list_cut[i])
                    if new_cut(self.breaking_point[i + 1]) < Gs(
                        self.breaking_point[i + 1]
                    ):
                        if -lamb - self.list_cut[i][0] != 0:
                            new_points.append(
                                -(-beta + lamb * new_control - self.list_cut[i][1])
                                / (-lamb - self.list_cut[i][0])
                            )
                            new_reward.append((-lamb, -beta + lamb * new_control))
            elif i == len(self.breaking_point) - 1:
                new_points.append(self.breaking_point[-1])
            else:
                if new_cut(self.breaking_point[i]) >= Gs(self.breaking_point[i]):
                    new_reward.append(self.list_cut[i])
                    new_points.append(self.breaking_point[i])
                    if new_cut(self.breaking_point[i + 1]) < Gs(
                        self.breaking_point[i + 1]
                    ):
                        if -lamb - self.list_cut[i][0] != 0:
                            new_reward.append((-lamb, -beta + lamb * new_control))
                            new_points.append(
                                -(-beta + lamb * new_control - self.list_cut[i][1])
                                / (-lamb - self.list_cut[i][0])
                            )
                elif (
                    new_cut(self.breaking_point[i]) < Gs(self.breaking_point[i])
                ) and (
                    new_cut(self.breaking_point[i + 1])
                    >= Gs(self.breaking_point[i + 1])
                ):
                    if -lamb - self.list_cut[i][0] != 0:
                        new_reward.append(self.list_cut[i])
                        new_points.append(
                            -(-beta + lamb * new_control - self.list_cut[i][1])
                            / (-lamb - self.list_cut[i][0])
                        )

        self.breaking_point = new_points
        self.list_cut = new_reward


def solve_weekly_problem_with_approximation(
    points: list,
    X: Array1D,
    inflow: float,
    lb: float,
    ub: float,
    level_i: float,
    xmax: float,
    xmin: float,
    cap: float,
    pen: Callable,
    V_fut: Callable,
    Gs: Callable,
) -> tuple[float, float, float]:
    """
    Optimize control of reservoir during a week based on reward approximation and current Bellman values.

    Parameters
    ----------
    points:list :
        Breaking points in reward approximation
    X:np.array :
        Breaking points in Bellman values approximation
    inflow:float :
        Inflow in the reservoir during the week
    lb:float :
        Lower possible bound on control
    ub:float :
        Upper possible bound on control
    level_i:float :
        Initial level of reservoir at the beginning of the week
    xmax:float :
        Upper rule curve at the end of the week
    xmin:float :
        Bottom rule curve an the end of the week
    cap:float :
        Capacity of the reservoir
    pen:callable :
        Penalties for violating rule curves at the end of the week
    V_fut:callable :
        Bellman values at the end of the week
    Gs:callable :
        Reward approximation for the current week

    Returns
    -------
    Vu:float :
        Optimal objective value
    xf:float :
        Final level of sotck
    control:float :
        Optimal control
    """
    Vu = float("-inf")

    for i_fut in range(len(X)):
        u = -X[i_fut] + level_i + inflow
        if lb <= u <= ub:
            G = Gs(u)
            penalty = pen(X[i_fut])
            if (G + V_fut(X[i_fut]) + penalty) > Vu:
                Vu = G + V_fut(X[i_fut]) + penalty
                xf = X[i_fut]
                control = u

    for u in range(len(points)):
        state_fut = min(cap, level_i - points[u] + inflow)
        if 0 <= state_fut:
            penalty = pen(state_fut)
            G = Gs(points[u])
            if (G + V_fut(state_fut) + penalty) > Vu:
                Vu = G + V_fut(state_fut) + penalty
                xf = state_fut
                control = points[u]

    Umin = level_i + inflow - xmin
    if lb <= Umin <= ub:
        state_fut = level_i - Umin + inflow
        penalty = pen(state_fut)
        if (Gs(Umin) + V_fut(state_fut) + penalty) > Vu:
            Vu = Gs(Umin) + V_fut(state_fut) + penalty
            xf = state_fut
            control = Umin

    Umax = level_i + inflow - xmax
    if lb <= Umax <= ub:
        state_fut = level_i - Umax + inflow
        penalty = pen(state_fut)
        if (Gs(Umax) + V_fut(state_fut) + penalty) > Vu:
            Vu = Gs(Umax) + V_fut(state_fut) + penalty
            xf = state_fut
            control = Umax

    control = min(-(xf - level_i - inflow), ub)
    return (Vu, xf, control)


def calculate_VU(
    param: AntaresParameter,
    reward: list[list[RewardApproximation]],
    reservoir_management: ReservoirManagement,
    X: Array1D,
) -> Array2D:
    """
    Calculate Bellman values for every week based on reward approximation

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    reward:list[list[RewardApproximation]] :
        Reward approximation for every week and every scenario
    reservoir:Reservoir :
        Reservoir considered
    X:np.array :
        Discretization of stock levels
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve
    pen_final:float :
        Penalty for violating final rule curves

    Returns
    -------

    """

    NTrain = param.NTrain
    S = param.S

    V = np.zeros((len(X), S + 1, NTrain))

    for s in range(S - 1, -1, -1):

        pen = reservoir_management.get_penalty(s=s, S=S)

        for k in range(NTrain):
            V_fut = interp1d(X, V[:, s + 1, k])
            Gs = reward[s][k].reward_function()
            for i in range(len(X)):

                Vu, _, _ = solve_weekly_problem_with_approximation(
                    points=reward[s][k].breaking_point,
                    X=X,
                    inflow=reservoir_management.reservoir.inflow[s, k],
                    lb=-reservoir_management.reservoir.max_pumping[s],
                    ub=reservoir_management.reservoir.max_generating[s],
                    level_i=X[i],
                    xmax=reservoir_management.reservoir.upper_rule_curve[s],
                    xmin=reservoir_management.reservoir.bottom_rule_curve[s],
                    cap=reservoir_management.reservoir.capacity,
                    pen=pen,
                    V_fut=V_fut,
                    Gs=Gs,
                )

                V[i, s, k] = Vu + V[i, s, k]

        V[:, s, :] = np.repeat(
            np.mean(V[:, s, :], axis=1, keepdims=True), NTrain, axis=1
        )
    return np.mean(V, axis=2)
