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
import ortools.linear_solver.pywraplp as pywraplp

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

    def __init__(
        self,
        scenario: int,
        week: int,
        path: str,
        itr: int = 1,
        name_solver: str = "GLOP",
    ) -> None:
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
        model_proto = model.export_to_proto()

        solver = pywraplp.Solver.CreateSolver(name_solver)
        assert solver, "Couldn't find any supported solver"
        solver.EnableOutput()

        parameters = pywraplp.MPSolverParameters()
        if name_solver == "XPRESS_LP":
            solver.SetSolverSpecificParametersAsString("THREADS 1")
            parameters.SetIntegerParam(parameters.PRESOLVE, parameters.PRESOLVE_OFF)
            parameters.SetIntegerParam(parameters.SCALING, 0)
            parameters.SetDoubleParam(parameters.DUAL_TOLERANCE, 1e-7)
            parameters.SetDoubleParam(parameters.PRIMAL_TOLERANCE, 1e-7)
        self.solver_parameters = parameters

        solver.LoadModelFromProtoWithUniqueNamesOrDie(model_proto)

        self.solver = solver

        self.store_basis = True if name_solver == "XPRESS_LP" else False

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

    def get_basis(self) -> tuple[List, List]:
        var_basis = []
        con_basis = []
        for var in self.solver.variables():
            var_basis.append(var.basis_status())
        for con in self.solver.constraints():
            con_basis.append(con.basis_status())
        return var_basis, con_basis

    def load_basis(self, basis: Basis) -> None:
        len_cons = len(self.solver.constraints())
        len_vars = len(self.solver.variables())
        if len_vars > len(basis.rstatus):
            basis.rstatus += [0] * (len_vars - len(basis.rstatus))
        if len_cons > len(basis.cstatus):
            basis.cstatus += [0] * (len_cons - len(basis.cstatus))
        self.solver.SetStartingLpBasis(
            basis.rstatus[:len_vars], basis.cstatus[:len_cons]
        )

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

        model = self.solver

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

        cst = model.constraints()
        binding_id = [
            i
            for i in range(len(cst))
            if re.search(
                f"^HydroPower::area<{reservoir_management.reservoir.area}>::week<.",
                cst[i].name(),
            )
        ]
        assert len(binding_id) == 1

        x_s = model.Var(
            lb=0,
            ub=reservoir_management.reservoir.capacity,
            integer=False,
            name="x_s",
        )

        x_s_1 = model.Var(
            lb=0,
            ub=reservoir_management.reservoir.capacity,
            integer=False,
            name="x_s_1",
        )

        U = model.Var(
            lb=-reservoir_management.reservoir.max_pumping[self.week]
            * reservoir_management.reservoir.efficiency,
            ub=reservoir_management.reservoir.max_generating[self.week],
            integer=False,
            name="u",
        )

        model.Add(
            x_s_1
            <= x_s
            - U
            + reservoir_management.reservoir.inflow[self.week, self.scenario],
            name=f"ReservoirConservation::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
        )

        y = model.Var(
            lb=0, ub=model.Infinity(), integer=False, name="y"
        )  # Penality for violating guide curves

        if self.week != len_week - 1 or not reservoir_management.final_level:
            model.Add(
                y
                >= -reservoir_management.penalty_bottom_rule_curve
                * (x_s_1 - reservoir_management.reservoir.bottom_rule_curve[self.week]),
                name=f"PenaltyForViolatingBottomRuleCurve::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
            )
            model.Add(
                y
                >= reservoir_management.penalty_upper_rule_curve
                * (x_s_1 - reservoir_management.reservoir.upper_rule_curve[self.week]),
                name=f"PenaltyForViolatingUpperRuleCurve::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
            )
        else:
            model.Add(
                y
                >= -reservoir_management.penalty_final_level
                * (x_s_1 - reservoir_management.final_level),
                name=f"PenaltyForViolatingBottomRuleCurve::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
            )
            model.Add(
                y
                >= reservoir_management.penalty_final_level
                * (x_s_1 - reservoir_management.final_level),
                name=f"PenaltyForViolatingUpperRuleCurve::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
            )

        z = model.Var(
            lb=-model.Infinity(), ub=model.Infinity(), integer=False, name="z"
        )  # Auxiliar variable to introduce the piecewise representation of the future cost

        self.binding_id = cst[binding_id[0]]
        self.U = U
        self.x_s = x_s
        self.x_s_1 = x_s_1
        self.z = z
        self.y = y

    def delete_variable(self, hours_in_week: int, name_variable: str) -> None:
        model = self.solver
        var = model.variables()
        var_id = [i for i in range(len(var)) if re.search(name_variable, var[i].name())]
        assert len(var_id) in [0, hours_in_week]
        if len(var_id) == hours_in_week:
            for i in var_id:
                var[i].SetLb(-model.Infinity())
                var[i].SetUb(model.Infinity())
                model.Objective().SetCoefficient(var[i], 0)

    def delete_constraint(self, hours_in_week: int, name_constraint: str) -> None:
        model = self.solver
        cons = model.constraints()
        cons_id = [
            i for i in range(len(cons)) if re.search(name_constraint, cons[i].name())
        ]
        assert len(cons_id) in [0, hours_in_week]
        if len(cons_id) == hours_in_week:
            for i in cons_id:
                cons[i].Clear()
                cons[i].SetBounds(lb=0, ub=0)

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
        if self.store_basis:
            if (prev_basis.not_empty()) & (i == 0):
                self.load_basis(prev_basis)

            if i >= 1:
                basis = self.find_closest_basis(control=control)
                self.load_basis(basis)

        self.binding_id.SetBounds(lb=control, ub=control)
        debut_1 = time()
        solve_status = self.solver.Solve(self.solver_parameters)
        fin_1 = time()

        if solve_status == pywraplp.Solver.OPTIMAL:
            beta = float(self.solver.Objective().Value())
            lamb = float(self.binding_id.dual_value())

            itr = self.solver.Iterations()

            rbas, cbas = self.get_basis()
            self.add_basis(basis=Basis(rbas, cbas), control_basis=control)

            if i == 0:
                prev_basis.rstatus = rbas
                prev_basis.cstatus = cbas
            return (beta, lamb, itr, prev_basis, fin_1 - debut_1)
        else:
            print(f"Failed at control fixed : {solve_status}")
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

    m.binding_id.SetCoefficient(m.U, -1)
    m.binding_id.SetLb(0.0)
    m.binding_id.SetUb(0.0)

    m.solver.Objective().SetCoefficient(m.y, 1)
    m.solver.Objective().SetCoefficient(m.z, 1)

    additional_constraint: List = []

    for j in range(len(X) - 1):
        if (V[j + 1, week + 1] < float("inf")) & (V[j, week + 1] < float("inf")):
            cst = m.solver.LookupConstraint(
                f"BellmanValueBetween{j}And{j+1}::area<{bellman_value_calculation.reservoir_management.reservoir.area}>::week<{m.week}>"
            )
            if cst:
                cst.SetCoefficient(
                    m.x_s_1, -(-V[j + 1, week + 1] + V[j, week + 1]) / (X[j + 1] - X[j])
                )
                cst.SetLb(
                    (-V[j + 1, week + 1] + V[j, week + 1]) / (X[j + 1] - X[j]) * (-X[j])
                    - V[j, week + 1]
                )
            else:
                cst = m.solver.Add(
                    m.z
                    >= (-V[j + 1, week + 1] + V[j, week + 1])
                    / (X[j + 1] - X[j])
                    * (m.x_s_1 - X[j])
                    - V[j, week + 1],
                    name=f"BellmanValueBetween{j}And{j+1}::area<{bellman_value_calculation.reservoir_management.reservoir.area}>::week<{m.week}>",
                )
            additional_constraint.append(cst)

    cst_initial_level = m.solver.LookupConstraint(
        f"InitialLevelReservoir::area<{bellman_value_calculation.reservoir_management.reservoir.area}>::week<{m.week}>"
    )
    if cst_initial_level:
        cst_initial_level.SetBounds(lb=level_i, ub=level_i)
    else:
        cst_initial_level = m.solver.Add(
            m.x_s == level_i,
            name=f"InitialLevelReservoir::area<{bellman_value_calculation.reservoir_management.reservoir.area}>::week<{m.week}>",
        )

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
            m.load_basis(basis)

    debut_1 = time()
    solve_status = m.solver.Solve(m.solver_parameters)
    fin_1 = time()

    if solve_status == pywraplp.Solver.OPTIMAL:
        itr = m.solver.Iterations()
        if m.store_basis:
            rbas, cbas = m.get_basis()
            m.add_basis(
                basis=Basis(rbas, cbas),
                control_basis=m.U.solution_value(),
            )

        beta = float(m.solver.Objective().Value())
        xf = float(m.x_s_1.solution_value())
        z = float(m.z.solution_value())
        y = float(m.y.solution_value())
        for cst in additional_constraint:
            cst.SetLb(0)
        cst_initial_level.SetBounds(
            lb=bellman_value_calculation.reservoir_management.reservoir.capacity,
            ub=bellman_value_calculation.reservoir_management.reservoir.capacity,
        )
        m.binding_id.SetCoefficient(m.U, 0)

        m.solver.Objective().SetCoefficient(m.y, 0)
        m.solver.Objective().SetCoefficient(m.z, 0)
        cout += beta
        if not (
            take_into_account_z_and_y
        ):  # week != bellman_value_calculation.time_scenario_param.len_week - 1:
            cout += -z - y

    else:
        print(f"Failed at upper bound : {solve_status}")
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
