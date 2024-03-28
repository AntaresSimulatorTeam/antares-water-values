import re
from typing import List, Annotated, Literal
from calculate_reward_and_bellman_values import (
    RewardApproximation,
    ReservoirManagement,
    BellmanValueCalculation,
)
from read_antares_data import TimeScenarioParameter, TimeScenarioIndex
import numpy as np
import numpy.typing as npt
from time import time
from scipy.interpolate import interp1d
from ortools.linear_solver.python import model_builder

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
            Row basis (Default value = [])
        cstatus:list :
            Column basis (Default value = [])

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

    def __init__(self, scenario: int, week: int, path: str, itr: int = 1) -> None:
        """
        Create a new Xpress problem and load the problem stored in the associated mps file.

        Parameters
        ----------
        scenario:int :
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
        self.scenario = scenario
        self.week = week
        self.path = path

        mps_path = path + f"/problem-{scenario+1}-{week+1}--optim-nb-{itr}.mps"
        model = model_builder.ModelBuilder()  # type: ignore[no-untyped-call]
        model.import_from_mps_file(mps_path)

        solver = model_builder.ModelSolver("GLOP")
        assert solver, "Couldn't find any supported solver"

        solver.enable_output(False)

        self.model = model
        self.solver = solver

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
        param: TimeScenarioParameter,
        reservoir_management: ReservoirManagement,
    ) -> None:
        """
        Modify the Xpress problem to take into account reservoir constraints and manage reservoir with Bellman values and penalties on rule curves.

        Parameters
        ----------
        param:AntaresParameter :
            Time-related parameters
        reservoir_management:ReservoirManagement :
            Considered reservoir and its paramters

        Returns
        -------
        None
        """
        hours_in_week = reservoir_management.reservoir.hours_in_week
        len_week = param.len_week

        model = self.model

        self.delete_variable(
            hours_in_week=hours_in_week,
            name_variable=f"^HydroLevel::area<{reservoir_management.reservoir.area}>::hour<.",
        )
        self.delete_variable(
            hours_in_week=hours_in_week,
            name_variable=f"^Overflow::area<{reservoir_management.reservoir.area}>::hour<.",
        )
        self.delete_constraint(
            hours_in_week=hours_in_week,
            name_constraint=f"^AreaHydroLevel::area<{reservoir_management.reservoir.area}>::hour<.",
        )

        cst = model.get_linear_constraints()
        binding_id = [
            i
            for i in range(len(cst))
            if re.search(
                f"^HydroPower::area<{reservoir_management.reservoir.area}>::week<.",
                cst[i].name,
            )
        ]
        assert len(binding_id) == 1

        x_s = model.new_var(
            lb=0,
            ub=reservoir_management.reservoir.capacity,
            is_integer=False,
            name="x_s",
        )

        x_s_1 = model.new_var(
            lb=0,
            ub=reservoir_management.reservoir.capacity,
            is_integer=False,
            name="x_s_1",
        )

        U = model.new_var(
            lb=-reservoir_management.reservoir.max_pumping[self.week]
            * reservoir_management.reservoir.efficiency,
            ub=reservoir_management.reservoir.max_generating[self.week],
            is_integer=False,
            name="u",
        )

        model.add(
            x_s_1
            <= x_s
            - U
            + reservoir_management.reservoir.inflow[self.week, self.scenario],
            name=f"ReservoirConservation::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
        )

        y = model.new_var(
            lb=0, ub=float("inf"), is_integer=False, name="y"
        )  # Penality for violating guide curves

        if self.week != len_week - 1 or not reservoir_management.final_level:
            model.add(
                y
                >= -reservoir_management.penalty_bottom_rule_curve
                * (x_s_1 - reservoir_management.reservoir.bottom_rule_curve[self.week]),
                name=f"PenaltyForViolatingBottomRuleCurve::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
            )
            model.add(
                y
                >= reservoir_management.penalty_upper_rule_curve
                * (x_s_1 - reservoir_management.reservoir.upper_rule_curve[self.week]),
                name=f"PenaltyForViolatingUpperRuleCurve::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
            )
        else:
            model.add(
                y
                >= -reservoir_management.penalty_final_level
                * (x_s_1 - reservoir_management.final_level),
                name=f"PenaltyForViolatingBottomRuleCurve::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
            )
            model.add(
                y
                >= reservoir_management.penalty_final_level
                * (x_s_1 - reservoir_management.final_level),
                name=f"PenaltyForViolatingUpperRuleCurve::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
            )

        z = model.new_var(
            lb=float("-inf"), ub=float("inf"), is_integer=False, name="z"
        )  # Auxiliar variable to introduce the piecewise representation of the future cost

        self.binding_id = cst[binding_id[0]]
        self.U = U
        self.x_s = x_s
        self.x_s_1 = x_s_1
        self.z = z
        self.y = y

    def delete_variable(self, hours_in_week: int, name_variable: str) -> None:
        model = self.model
        var = model.get_variables()
        var_id = [i for i in range(len(var)) if re.search(name_variable, var[i].name)]
        assert len(var_id) in [0, hours_in_week]
        if len(var_id) == hours_in_week:
            for i in var_id:
                var[i].lower_bound = float("-inf")
                var[i].upper_bound = float("inf")
                var[i].objective_coefficient = 0

    def delete_constraint(self, hours_in_week: int, name_constraint: str) -> None:
        model = self.model
        cons = model.get_linear_constraints()
        cons_id = [
            i for i in range(len(cons)) if re.search(name_constraint, cons[i].name)
        ]
        assert len(cons_id) in [0, hours_in_week]
        if len(cons_id) == hours_in_week:
            for i in cons_id:
                cons[i].lower_bound = float("-inf")
                cons[i].upper_bound = float("inf")

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

        # TODO : gérer les bases
        # if (prev_basis.not_empty()) & (i == 0):
        #     self.model.loadbasis(prev_basis.rstatus, prev_basis.cstatus)

        # if i >= 1:
        #     basis = self.find_closest_basis(control=control)
        #     self.model.loadbasis(basis.rstatus, basis.cstatus)

        rbas: List = []
        cbas: List = []

        self.binding_id.lower_bound = control
        self.binding_id.upper_bound = control
        debut_1 = time()
        solve_status = self.solver.solve(self.model)
        fin_1 = time()

        if solve_status.name == "OPTIMAL":
            beta = float(self.solver.objective_value)
            lamb = float(self.solver.dual_value(self.binding_id))
            # TODO : gérer le nombre d'itérations du simplexe
            itr = 0  # self.model.attributes.SIMPLEXITER

            # TODO : gérer les bases
            # self.model.getbasis(rbas, cbas)
            # self.add_basis(basis=Basis(rbas, cbas), control_basis=control)

            if i == 0:
                prev_basis.rstatus = rbas
                prev_basis.cstatus = cbas
            return (beta, lamb, itr, prev_basis, fin_1 - debut_1)
        else:
            print(solve_status.name)
            raise (ValueError)


def solve_problem_with_Bellman_values(
    bellman_value_calculation: BellmanValueCalculation,
    V: Array2D,
    scenario: int,
    level_i: float,
    week: int,
    m: AntaresProblem,
    take_into_account_z_and_y: bool,
    find_optimal_basis: bool = True,
) -> tuple[float, int, float, float, float]:

    cout = 0.0

    X = bellman_value_calculation.stock_discretization

    m.binding_id.set_coefficient(m.U, -1)
    m.binding_id.lower_bound = 0
    m.binding_id.upper_bound = 0

    m.y.objective_coefficient = 1
    m.z.objective_coefficient = 1

    if find_optimal_basis:
        if len(m.control_basis) >= 1:
            if len(m.control_basis) >= 2:
                V_fut = interp1d(X, V[:, week + 1])

                _, _, likely_control = (
                    bellman_value_calculation.solve_weekly_problem_with_approximation(
                        level_i=level_i,
                        V_fut=V_fut,
                        week=week,
                        scenario=scenario,
                    )
                )
            else:
                likely_control = 0
            basis = m.find_closest_basis(likely_control)
            # TODO : gérer les bases
        # m.model.loadbasis(basis.rstatus, basis.cstatus)
    additional_constraint: List = []
    constraints = m.model.get_linear_constraints()

    for j in range(len(X) - 1):
        if (V[j + 1, week + 1] < float("inf")) & (V[j, week + 1] < float("inf")):
            idx_cst = [
                i
                for i in constraints
                if i.name
                == f"BellmanValueBetween{j}And{j+1}::area<{bellman_value_calculation.reservoir_management.reservoir.area}>::week<{m.week}>"
            ]
            if len(idx_cst) >= 1:
                cst = idx_cst[0]
                cst.set_coefficient(
                    m.x_s_1, -(-V[j + 1, week + 1] + V[j, week + 1]) / (X[j + 1] - X[j])
                )
                cst.lower_bound = (-V[j + 1, week + 1] + V[j, week + 1]) / (
                    X[j + 1] - X[j]
                ) * (-X[j]) - V[j, week + 1]
            else:
                cst = m.model.add(
                    m.z
                    >= (-V[j + 1, week + 1] + V[j, week + 1])
                    / (X[j + 1] - X[j])
                    * (m.x_s_1 - X[j])
                    - V[j, week + 1],
                    name=f"BellmanValueBetween{j}And{j+1}::area<{bellman_value_calculation.reservoir_management.reservoir.area}>::week<{m.week}>",
                )
            additional_constraint.append(cst)

    idx_cst = [
        i
        for i in constraints
        if i.name
        == f"InitialLevelReservoir::area<{bellman_value_calculation.reservoir_management.reservoir.area}>::week<{m.week}>"
    ]
    if len(idx_cst) >= 1:
        cst_initial_level = idx_cst[0]
        cst_initial_level.lower_bound = level_i
        cst_initial_level.upper_bound = level_i
    else:
        cst_initial_level = m.model.add(
            m.x_s == level_i,
            name=f"InitialLevelReservoir::area<{bellman_value_calculation.reservoir_management.reservoir.area}>::week<{m.week}>",
        )
    additional_constraint.append(cst_initial_level)

    rbas: List = []
    cbas: List = []

    debut_1 = time()
    solve_status = m.solver.solve(m.model)
    fin_1 = time()

    if solve_status.name == "OPTIMAL":
        # TODO : gérer les bases
        # m.model.getbasis(rbas, cbas)
        # m.add_basis(
        #     basis=Basis(rbas[:nb_cons], cbas),
        #     control_basis=m.model.getSolution(m.U),
        # )

        beta = float(m.solver.objective_value)
        xf = float(m.solver.value(m.x_s_1))
        z = float(m.solver.value(m.z))
        y = float(m.solver.value(m.y))
        for cst in additional_constraint:
            cst.lower_bound = float("-inf")
            cst.upper_bound = float("inf")
        m.binding_id.set_coefficient(m.U, 0)

        m.y.objective_coefficient = 0
        m.z.objective_coefficient = 0
        cout += beta
        if not (
            take_into_account_z_and_y
        ):  # week != bellman_value_calculation.time_scenario_param.len_week - 1:
            cout += -z - y

        # TODO : gérer le nombre d'itérations du simplexe
        itr = 0  # m.model.attributes.SIMPLEXITER

    else:
        raise (ValueError)
    return (
        fin_1 - debut_1,
        itr,
        cout,
        -(
            xf
            - level_i
            - bellman_value_calculation.reservoir_management.reservoir.inflow[
                week, scenario
            ]
        ),
        xf,
    )
