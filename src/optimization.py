import re
import xpress as xp
from typing import List, Annotated, Literal
from calculate_reward_and_bellman_values import (
    RewardApproximation,
    get_penalty,
    solve_weekly_problem_with_approximation,
)
from read_antares_data import Reservoir, AntaresParameter
import numpy as np
import numpy.typing as npt
from time import time
from scipy.interpolate import interp1d

xp.setOutputEnabled(False)

Array2D = Annotated[npt.NDArray[np.float32], Literal["N", "N"]]
Array1D = Annotated[npt.NDArray[np.float32], Literal["N"]]
Array3D = Annotated[npt.NDArray[np.float32], Literal["N", "N", "N"]]
Array4D = Annotated[npt.NDArray[np.float32], Literal["N", "N", "N", "N"]]


class Basis:
    """Class to store basis with Xpress"""

    def __init__(self, rstatus: list = [], cstatus: list = []) -> None:
        """
        Create a new basis.

        Parameters
        ----------
        rstatus:list :
            Row basis obtained with problem.getbasis() (Default value = [])
        cstatus:list :
            Column basis obtained with problem.getbasis() (Default value = [])

        Returns
        -------
        None

        """
        self.rstatus = rstatus
        self.cstatus = cstatus

    def not_empty(self) -> bool:
        """Check if a basis isn't empty (True) or not (False)"""
        return len(self.rstatus) != 0


class AntaresProblem:
    """Class to store an Xpress optimization problem describing the problem solved by Antares for one week and one scenario."""

    def __init__(self, year: int, week: int, path: str, itr: int = 1) -> None:
        """
        Create a new Xpress problem and load the problem stored in the associated mps file.

        Parameters
        ----------
        year:int :
            Scenario considered
        week:int :
            Week considered
        path:str :
            Path where mps files are stored
        itr:int :
            Antares iteration considered (Default value = 1)

        Returns
        -------
        None
        """
        self.year = year
        self.week = week
        self.path = path

        model = xp.problem()
        model.controls.outputlog = 0
        model.controls.threads = 1
        model.controls.scaling = 0
        model.controls.presolve = 0
        model.controls.feastol = 1.0e-7
        model.controls.optimalitytol = 1.0e-7
        model.controls.xslp_log = -1
        model.controls.lplogstyle = 0
        model.read(path + f"/problem-{year+1}-{week+1}--optim-nb-{itr}.mps")
        self.model = model

        self.basis: List = []
        self.control_basis: List = []

    def add_basis(self, basis: Basis, control_basis: float) -> None:
        """
        Store a new basis for the optimization problem.

        Parameters
        ----------
        basis:Basis :
            New basis to store
        control_basis:float :
            Reservoir control for which the problem has been solved

        Returns
        -------
        None
        """
        self.basis.append(basis)
        self.control_basis.append(control_basis)

    def find_closest_basis(self, control: float) -> Basis:
        """
        Among stored basis, return the closest one to the given control.

        Parameters
        ----------
        control:float :
            Control for which we want to solve the optimization problem

        Returns
        -------

        """
        u = np.argmin(np.abs(np.array(self.control_basis) - control))
        return self.basis[u]

    def create_weekly_problem_itr(
        self,
        param: AntaresParameter,
        reservoir: Reservoir,
        pen_low: float = 0,
        pen_high: float = 0,
        pen_final: float = 0,
    ) -> None:
        """
        Modify the Xpress problem to take into account reservoir constraints and manage reservoir with Bellman values and penalties on rule curves.

        Parameters
        ----------
        param:AntaresParameter :
            Time-related parameters
        reservoir:Reservoir :
            Considered reservoir
        pen_low:float :
            Penalty for violating bottom rule curve (Default value = 0)
        pen_high:float :
            Penalty for violating top rule curve (Default value = 0)
        pen_final:float :
            Penalty for violating rule curves at the end of the year (Default value = 0)

        Returns
        -------
        None
        """

        S = param.S
        H = param.H

        model = self.model

        self.delete_variable(
            H=H, name_variable=f"^HydroLevel::area<{reservoir.area}>::hour<."
        )
        self.delete_variable(
            H=H, name_variable=f"^Overflow::area<{reservoir.area}>::hour<."
        )
        self.delete_constraint(
            H=H, name_constraint=f"^AreaHydroLevel::area<{reservoir.area}>::hour<."
        )

        cst = model.getConstraint()
        binding_id = [
            i
            for i in range(len(cst))
            if re.search(f"^HydroPower::area<{reservoir.area}>::week<.", cst[i].name)
        ]
        assert len(binding_id) == 1

        x_s = xp.var("x_s", lb=0, ub=reservoir.capacity)
        model.addVariable(x_s)  # State at the begining of the current week

        x_s_1 = xp.var("x_s_1", lb=0, ub=reservoir.capacity)
        model.addVariable(x_s_1)  # State at the begining of the following week

        U = xp.var(
            "u",
            lb=-reservoir.P_pump[7 * self.week] * reservoir.efficiency * H,
            ub=reservoir.P_turb[7 * self.week] * H,
        )
        model.addVariable(U)  # State at the begining of the following week

        model.addConstraint(
            x_s_1 <= x_s - U + reservoir.inflow[self.week, self.year] * H
        )

        y = xp.var("y")

        model.addVariable(y)  # Penality for violating guide curves

        if self.week != S - 1:
            model.addConstraint(y >= -pen_low * (x_s_1 - reservoir.Xmin[self.week]))
            model.addConstraint(y >= pen_high * (x_s_1 - reservoir.Xmax[self.week]))
        else:
            model.addConstraint(y >= -pen_final * (x_s_1 - reservoir.Xmin[self.week]))
            model.addConstraint(y >= pen_final * (x_s_1 - reservoir.Xmax[self.week]))

        z = xp.var("z", lb=float("-inf"), ub=float("inf"))

        model.addVariable(
            z
        )  # Auxiliar variable to introduce the piecewise representation of the future cost

        self.binding_id = binding_id
        self.U = U
        self.x_s = x_s
        self.x_s_1 = x_s_1
        self.z = z
        self.y = y

    def delete_variable(self, H: int, name_variable: str) -> None:
        model = self.model
        var = model.getVariable()
        var_id = [i for i in range(len(var)) if re.search(name_variable, var[i].name)]
        assert len(var_id) in [0, H]
        if len(var_id) == H:
            model.delVariable(var_id)

    def delete_constraint(self, H: int, name_constraint: str) -> None:
        model = self.model
        cons = model.getConstraint()
        cons_id = [
            i for i in range(len(cons)) if re.search(name_constraint, cons[i].name)
        ]
        assert len(cons_id) in [0, H]
        if len(cons_id) == H:
            model.delConstraint(cons_id)

    def modify_weekly_problem_itr(
        self, control: float, i: int, prev_basis: Basis = Basis()
    ) -> tuple[float, float, int, Basis, float]:
        """
        Modify and solve problem to evaluate weekly cost associated with a particular control of the reservoir.

        Parameters
        ----------
        control:float :
            Control to evaluate
        i:int :
            Iteration of the iterative algorithm
        prev_basis:Basis :
            Basis used at a previous resolution of a similar problem (Default value = None)

        Returns
        -------
        beta:float :
            Total cost
        lamb:float :
            Dual value associated to the control constraint
        itr:int :
            Total number of simplex iterations used to solve the problem
        prev_basis:Basis :
            Basis output by the resolution
        t:float :
            Time spent solving the problem
        """

        if (prev_basis.not_empty()) & (i == 0):
            self.model.loadbasis(prev_basis.rstatus, prev_basis.cstatus)

        if i >= 1:
            basis = self.find_closest_basis(control=control)
            self.model.loadbasis(basis.rstatus, basis.cstatus)

        rbas: List = []
        cbas: List = []

        self.model.chgrhs(self.binding_id, [control])
        debut_1 = time()
        self.model.lpoptimize()
        fin_1 = time()

        if self.model.attributes.lpstatus == 1:
            beta = self.model.getObjVal()
            lamb = self.model.getDual(self.binding_id)[0]
            itr = self.model.attributes.SIMPLEXITER

            self.model.getbasis(rbas, cbas)
            self.add_basis(basis=Basis(rbas, cbas), control_basis=control)

            if i == 0:
                prev_basis.rstatus = rbas
                prev_basis.cstatus = cbas
            return (beta, lamb, itr, prev_basis, fin_1 - debut_1)
        else:

            raise (ValueError)


def find_likely_control(
    param: AntaresParameter,
    reservoir: Reservoir,
    X: Array1D,
    V: Array2D,
    reward: list[list[RewardApproximation]],
    pen_low: float,
    pen_high: float,
    pen_final: float,
    level_i: float,
    s: int,
    k: int,
) -> float:
    """
    Compute a control which is likely to be optimal in the real optimization problem based on reward approximation.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    reservoir:Reservoir :
        Reservoir considered
    X:np.array :
        Discretization of Bellman values
    V:np.array :
        Bellman values
    reward:list[list[RewardApproximation]] :
        Reward approximation for every week and every scenario
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve
    pen_final:float :
        Penalty for violating final rule curves
    level_i:float :
        Initial level of stock
    s:int :
        Week considered
    k:int :
        Scenario considered

    Returns
    -------
    Likely control
    """

    S = param.S
    H = param.H

    V_fut = interp1d(X, V[:, s + 1])

    pen = get_penalty(
        s=s,
        S=S,
        reservoir=reservoir,
        pen_final=pen_final,
        pen_low=pen_low,
        pen_high=pen_high,
    )
    Gs = reward[s][k].reward_function()

    _, _, u = solve_weekly_problem_with_approximation(
        points=reward[s][k].breaking_point,
        X=X,
        inflow=reservoir.inflow[s, k] * H,
        lb=-reservoir.P_pump[7 * s] * H,
        ub=reservoir.P_turb[7 * s] * H,
        level_i=level_i,
        xmax=reservoir.Xmax[s],
        xmin=reservoir.Xmin[s],
        cap=reservoir.capacity,
        pen=pen,
        V_fut=V_fut,
        Gs=Gs,
    )

    return u


def solve_problem_with_Bellman_values(
    param: AntaresParameter,
    reservoir: Reservoir,
    X: Array1D,
    V: Array2D,
    G: list[list[RewardApproximation]],
    pen_low: float,
    pen_high: float,
    pen_final: float,
    S: int,
    H: int,
    k: int,
    level_i: float,
    s: int,
    m: AntaresProblem,
) -> tuple[float, int, float, float, float]:

    cout = 0.0

    nb_cons = m.model.attributes.rows

    m.model.chgmcoef(m.binding_id, [m.U], [-1])
    m.model.chgrhs(m.binding_id, [0])

    m.model.chgobj([m.y, m.z], [1, 1])

    if len(m.control_basis) >= 1:
        if len(m.control_basis) >= 2:
            likely_control = find_likely_control(
                param=param,
                reservoir=reservoir,
                X=X,
                V=V,
                reward=G,
                pen_low=pen_low,
                pen_high=pen_high,
                pen_final=pen_final,
                level_i=level_i,
                s=s,
                k=k,
            )
        else:
            likely_control = 0
        basis = m.find_closest_basis(likely_control)
        m.model.loadbasis(basis.rstatus, basis.cstatus)

    for j in range(len(X) - 1):
        if (V[j + 1, s + 1] < float("inf")) & (V[j, s + 1] < float("inf")):
            m.model.addConstraint(
                m.z
                >= (-V[j + 1, s + 1] + V[j, s + 1])
                / (X[j + 1] - X[j])
                * (m.x_s_1 - X[j])
                - V[j, s + 1]
            )

    cst_initial_level = m.x_s == level_i
    m.model.addConstraint(cst_initial_level)

    rbas: List = []
    cbas: List = []

    debut_1 = time()
    m.model.lpoptimize()
    fin_1 = time()

    if m.model.attributes.lpstatus == 1:
        m.model.getbasis(rbas, cbas)
        m.add_basis(
            basis=Basis(rbas[:nb_cons], cbas),
            control_basis=m.model.getSolution(m.U),
        )

        beta = m.model.getObjVal()
        xf = m.model.getSolution(m.x_s_1)
        z = m.model.getSolution(m.z)
        y = m.model.getSolution(m.y)
        m.model.delConstraint(range(nb_cons, m.model.attributes.rows))
        m.model.chgmcoef(m.binding_id, [m.U], [0])

        m.model.chgobj([m.y, m.z], [0, 0])
        cout += beta
        if s != S - 1:
            cout += -z - y

        itr = m.model.attributes.SIMPLEXITER

    else:
        raise (ValueError)
    return (
        fin_1 - debut_1,
        itr,
        cout,
        -(xf - level_i - reservoir.inflow[s, k] * H),
        xf,
    )
