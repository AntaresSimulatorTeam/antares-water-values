import re
from typing import Any
from calculate_reward_and_bellman_values import (
    BellmanValueCalculation,
    MultiStockBellmanValueCalculation,
    MultiStockManagement,
)
from estimation import Estimator, LinearInterpolator, LinearCostEstimator
from read_antares_data import TimeScenarioParameter
import numpy as np
from time import time
from scipy.interpolate import interp1d
from ortools.linear_solver.python import model_builder
import ortools.linear_solver.pywraplp as pywraplp
from type_definition import Array2D, List, Dict, Array1D, npt


class Basis:
    """Class to store basis with Xpress"""

    def __init__(self, rstatus: list, cstatus: list) -> None:
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
        name_solver: str = "CLP",
        name_scenario: int = -1,
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

        if name_scenario == -1:
            name_scenario = scenario + 1

        mps_path = path + f"/problem-{name_scenario}-{week+1}--optim-nb-{itr}.mps"
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

    def add_basis(self, basis: Basis, control_basis: Dict[str, float]) -> None:
        """
        Store a new basis for the optimization probleself.

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

    def get_basis(self) -> tuple[list, list]:
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

    def find_closest_basis(self, control: Dict[str, float]) -> Basis:
        """
        Among stored basis, return the closest one to the given control.

        Parameters
        ----------
        control:float :
            Control for which we want to solve the optimization problem

        Returns
        -------

        """
        if len(self.basis) >= 1:
            gap = np.zeros(len(self.control_basis))
            for area in control.keys():
                gap += np.abs(
                    np.array([u[area] for u in self.control_basis]) - control[area]
                )
            u = np.argmin(gap)
            return self.basis[u]
        else:
            return Basis([], [])

    def create_weekly_problem_itr(
        self,
        param: TimeScenarioParameter,
        multi_stock_management: MultiStockManagement,
        direct_bellman_calc:bool=True,
    ) -> None:
        """
        Modify the Xpress problem to take into account reservoir constraints and manage reservoir with Bellman values and penalties on rule curves.

        Parameters
        ----------
        param:AntaresParameter :
            Time-related parameters
        reservoir_management:MultiStockManagement :
            Considered reservoir and its parameters

        Returns
        -------
        None
        """
        self.stored_variables_and_constraints = {}
        self.range_reservoir = multi_stock_management.dict_reservoirs.keys()
        for area in multi_stock_management.dict_reservoirs:
            reservoir_management = multi_stock_management.dict_reservoirs[area]

            hours_in_week = reservoir_management.reservoir.hours_in_week
            len_week = param.len_week

            model = self.solver

            self.delete_variable(
                hours_in_week=hours_in_week,
                name_variable=f"^HydroLevel::area<{reservoir_management.reservoir.area}>::hour<.", #variable niv stock
            )
            self.delete_variable(
                hours_in_week=hours_in_week,
                name_variable=f"^Overflow::area<{reservoir_management.reservoir.area}>::hour<.", #variable overflow
            )
            self.delete_constraint(
                hours_in_week=hours_in_week,
                name_constraint=f"^AreaHydroLevel::area<{reservoir_management.reservoir.area}>::hour<.", #Conservaiton niveua stock
            )

            cst = model.constraints()
            binding_id = [
                i
                for i in range(len(cst))
                if re.search(
                    f"^HydroPower::area<{reservoir_management.reservoir.area}>::week<.", #Turbinage- rho*pompage = cible
                    cst[i].name(),
                )
            ]
            assert len(binding_id) == 1
            if direct_bellman_calc:
                x_s = model.Var(
                    lb=0,
                    ub=reservoir_management.reservoir.capacity,
                    integer=False,
                    name=f"InitialLevel::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
                )# Initial level
            else:
                x_s = model.Var(
                    lb=reservoir_management.reservoir.capacity - np.max(reservoir_management.reservoir.max_pumping),
                    ub=reservoir_management.reservoir.capacity - np.max(reservoir_management.reservoir.max_pumping),
                    integer=False,
                    name=f"InitialLevel::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
                )# Initial level

            x_s_1 = model.Var(
                lb=0,
                ub=reservoir_management.reservoir.capacity,
                integer=False,
                name=f"FinalLevel::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
            )# Final level

            U = model.Var(
                lb=-reservoir_management.reservoir.max_pumping[self.week]
                * reservoir_management.reservoir.efficiency,
                ub=reservoir_management.reservoir.max_generating[self.week],
                integer=False,
                name=f"Control::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
            )# Reservoir Control

            if reservoir_management.overflow:
                model.Add(
                    x_s_1
                    <= x_s
                    - U
                    + reservoir_management.reservoir.inflow[self.week, self.scenario],
                    name=f"ReservoirConservation::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
                )
            else:
                model.Add(
                    x_s_1
                    == x_s
                    - U
                    + reservoir_management.reservoir.inflow[self.week, self.scenario],
                    name=f"ReservoirConservation::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
                )

            if direct_bellman_calc:
                y = model.Var(
                    lb=0,
                    ub=model.Infinity(),
                    integer=False,
                    name=f"Penalties::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
                )  # Penality for violating guide curves

                if self.week != len_week - 1 or not reservoir_management.final_level:
                    model.Add(
                        y
                        >= -reservoir_management.penalty_bottom_rule_curve
                        * (
                            x_s_1
                            - reservoir_management.reservoir.bottom_rule_curve[self.week]
                        ),
                        name=f"PenaltyForViolatingBottomRuleCurve::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
                    )
                    model.Add(
                        y
                        >= reservoir_management.penalty_upper_rule_curve
                        * (
                            x_s_1
                            - reservoir_management.reservoir.upper_rule_curve[self.week]
                        ),
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
                    lb=-model.Infinity(),
                    ub=model.Infinity(),
                    integer=False,
                    name=f"BellmanValue::area<{reservoir_management.reservoir.area}>::week<{self.week}>",
                )  # Auxiliar variable to introduce the piecewise representation of the future cost

            self.stored_variables_and_constraints[area] = {
                "energy_constraint": cst[binding_id[0]],
                "reservoir_control": U,
                "initial_level": x_s,
                "final_level": x_s_1,
            }
            if direct_bellman_calc:
                self.stored_variables_and_constraints[area]["penalties"] = y
                self.stored_variables_and_constraints[area]["final_bellman_value"] = z

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

    def solve_with_predefined_controls(
        self, control: Dict[str, float], prev_basis: Basis = Basis([], [])
    ) -> tuple[float, Dict[str, float], int, float]:
        """
        Modify and solve problem to evaluate weekly cost associated with a particular control of the reservoir.

        Parameters
        ----------
        control:Dict[str, float] :
            Control to evaluate
            
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
        t:float :
            Time spent solving the problem
        """
        if self.store_basis:
            if prev_basis.not_empty():
                self.load_basis(prev_basis)
            else:
                basis = self.find_closest_basis(control=control)
                self.load_basis(basis)

        for area in self.range_reservoir:
            self.set_constraints_predefined_control(control[area]*1.0, area)
        try:
            beta, lamb, final_level, _, _, itr, computing_time = self.solve_problem(direct_bellman_mode=False)
            # print(f"✔ for controls {control}")
        except ValueError:
            print(f"✘ for controls {control}")
            raise ValueError
        return beta, lamb, itr, computing_time
    
    def set_constraints_predefined_control(self, control: float, area: str) -> None:

        self.stored_variables_and_constraints[area]["energy_constraint"].SetBounds(
            lb=control, ub=control
        )

    def set_constraints_initial_level_and_bellman_values(
        self, level_i: float, X: Array1D, bellman_value: Array1D, area: str
    ) -> List[pywraplp.Constraint]:
        self.stored_variables_and_constraints[area]["energy_constraint"].SetCoefficient(
            self.stored_variables_and_constraints[area]["reservoir_control"], -1
        )
        self.stored_variables_and_constraints[area]["energy_constraint"].SetLb(0.0)
        self.stored_variables_and_constraints[area]["energy_constraint"].SetUb(0.0)

        self.solver.Objective().SetCoefficient(
            self.stored_variables_and_constraints[area]["penalties"], 1
        )
        self.solver.Objective().SetCoefficient(
            self.stored_variables_and_constraints[area]["final_bellman_value"], 1
        )

        bellman_constraint: List = []

        for j in range(len(X) - 1):
            if (bellman_value[j + 1] < float("inf")) & (
                bellman_value[j] < float("inf")
            ):
                cst = self.solver.LookupConstraint(
                    f"BellmanValueBetween{j}And{j+1}::area<{area}>::week<{self.week}>"
                )
                if cst:
                    cst.SetCoefficient(
                        self.stored_variables_and_constraints[area]["final_level"],
                        -(-bellman_value[j + 1] + bellman_value[j]) / (X[j + 1] - X[j]),
                    )
                    cst.SetLb(
                        (-bellman_value[j + 1] + bellman_value[j])
                        / (X[j + 1] - X[j])
                        * (-X[j])
                        - bellman_value[j]
                    )
                else:
                    cst = self.solver.Add(
                        self.stored_variables_and_constraints[area][
                            "final_bellman_value"
                        ]
                        >= (-bellman_value[j + 1] + bellman_value[j])
                        / (X[j + 1] - X[j])
                        * (
                            self.stored_variables_and_constraints[area]["final_level"]
                            - X[j]
                        )
                        - bellman_value[j],
                        name=f"BellmanValueBetween{j}And{j+1}::area<{area}>::week<{self.week}>",
                    )
                bellman_constraint.append(cst)

        cst_initial_level = self.solver.LookupConstraint(
            f"InitialLevelReservoir::area<{area}>::week<{self.week}>"
        )
        if cst_initial_level:
            cst_initial_level.SetBounds(lb=level_i, ub=level_i)
        else:
            cst_initial_level = self.solver.Add(
                self.stored_variables_and_constraints[area]["initial_level"] == level_i,
                name=f"InitialLevelReservoir::area<{area}>::week<{self.week}>",
            )
        return bellman_constraint

    def remove_bellman_constraints(
        self,
        bellman_value_calculation: BellmanValueCalculation,
        additional_constraint: List[pywraplp.Constraint],
        area: str,
    ) -> None:
        for cst in additional_constraint:
            cst.SetLb(0)
        cst_initial_level = self.solver.LookupConstraint(
            f"InitialLevelReservoir::area<{area}>::week<{self.week}>"
        )
        cst_initial_level.SetBounds(
            lb=bellman_value_calculation.reservoir_management.reservoir.capacity,
            ub=bellman_value_calculation.reservoir_management.reservoir.capacity,
        )
        self.stored_variables_and_constraints[area]["energy_constraint"].SetCoefficient(
            self.stored_variables_and_constraints[area]["reservoir_control"], 0
        )

        self.solver.Objective().SetCoefficient(
            self.stored_variables_and_constraints[area]["penalties"], 0
        )
        self.solver.Objective().SetCoefficient(
            self.stored_variables_and_constraints[area]["final_bellman_value"], 0
        )

    def solve_problem(
        self,
        direct_bellman_mode:bool=True 
    ) -> tuple[
        float,
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        int,
        float,
    ]:

        start = time()
        solve_status = self.solver.Solve(self.solver_parameters)
        end = time()

        if solve_status == pywraplp.Solver.OPTIMAL:
            itr = self.solver.Iterations()
            if self.store_basis:
                rbas, cbas = self.get_basis()
                self.add_basis(
                    basis=Basis(rbas, cbas),
                    control_basis=self.get_solution_value("reservoir_control"),
                )

            beta = float(self.solver.Objective().Value())
            xf = self.get_solution_value("final_level")
            lamb = self.get_dual_value("energy_constraint")
            if direct_bellman_mode:
                z = self.get_solution_value("final_bellman_value")
                y = self.get_solution_value("penalties")
            else:
                z = {}
                y = {}
            return (beta, lamb, xf, y, z, itr, end - start)
                
        else:
            print(f"Failed to solve : {solve_status}")
            raise (ValueError)

    def get_dual_value(self, name_constraint: str) -> Dict[str, float]:
        value = {}
        for area in self.range_reservoir:
            value[area] = float(
                self.stored_variables_and_constraints[area][
                    name_constraint
                ].dual_value()
            )

        return value

    def get_solution_value(self, name_variable: str) -> Dict[str, float]:
        value = {}
        for area in self.range_reservoir:
            value[area] = float(
                self.stored_variables_and_constraints[area][
                    name_variable
                ].solution_value()
            )

        return value

    def solve_problem_with_bellman_values(
        self,
        multi_bellman_value_calculation: MultiStockBellmanValueCalculation,
        V: Dict[str, Array2D],
        level_i: Dict[str, float],
        take_into_account_z_and_y: bool,
        find_optimal_basis: bool = True,
    ) -> tuple[float, int, float, Dict[str, float], Dict[str, float]]:

        cout = 0.0

        additional_constraint = []
        for area in self.range_reservoir:
            additional_constraint += (
                self.set_constraints_initial_level_and_bellman_values(
                    level_i=level_i[area],
                    X=multi_bellman_value_calculation.dict_reservoirs[
                        area
                    ].stock_discretization,
                    bellman_value=V[area][:, self.week + 1],
                    area=area,
                )
            )

        if find_optimal_basis:
            if len(self.control_basis) >= 1:
                dict_likely_control = {}
                if len(self.control_basis) >= 2:
                    for area in self.range_reservoir:
                        bellman_value_calculation = (
                            multi_bellman_value_calculation.dict_reservoirs[area]
                        )
                        X = bellman_value_calculation.stock_discretization
                        V_fut = interp1d(X, V[area][:, self.week + 1])

                        _, _, likely_control = (
                            bellman_value_calculation.solve_weekly_problem_with_approximation(
                                level_i=level_i[area],
                                V_fut=V_fut,
                                week=self.week,
                                scenario=self.scenario,
                            )
                        )
                        dict_likely_control[area] = likely_control
                    basis = self.find_closest_basis(dict_likely_control)
                else:
                    basis = self.basis[0]
                self.load_basis(basis)

        beta, _, xf, y, z, itr, t = self.solve_problem()

        cout = beta

        optimal_controls = {}
        for area in self.range_reservoir:
            optimal_controls[area] = -(
                xf[area]
                - level_i[area]
                - multi_bellman_value_calculation.dict_reservoirs[
                    area
                ].reservoir_management.reservoir.inflow[self.week, self.scenario]
            )

        for area in self.range_reservoir:
            self.remove_bellman_constraints(
                bellman_value_calculation=multi_bellman_value_calculation.dict_reservoirs[
                    area
                ],
                additional_constraint=additional_constraint,
                area=area,
            )

        if not (take_into_account_z_and_y):
            cout += -sum(z.values()) - sum(y.values())

        return (
            t,
            itr,
            cout,
            optimal_controls,
            xf,
        )


def solve_problem_with_multivariate_bellman_values(
    multi_bellman_value_calculation: MultiStockBellmanValueCalculation,
    V: Dict[str, npt.NDArray[np.float32]],
    level_i: Dict[str, float],
    m: AntaresProblem,
    take_into_account_z_and_y: bool,
) -> tuple[float, int, float, Dict[str, float], Dict[str, float], Dict[str, float]]:

    cout = 0.0

    len_reservoir = len(m.range_reservoir)

    for area in m.range_reservoir:
        m.stored_variables_and_constraints[area]["energy_constraint"].SetCoefficient(
            m.stored_variables_and_constraints[area]["reservoir_control"], -1
        )
        m.stored_variables_and_constraints[area]["energy_constraint"].SetLb(0.0)
        m.stored_variables_and_constraints[area]["energy_constraint"].SetUb(0.0)
        #On a transformé la cible en une variable

        m.solver.Objective().SetCoefficient(
            m.stored_variables_and_constraints[area]["penalties"], 1
        )#Avant on ne s'en servait pas
        m.solver.Objective().SetCoefficient(
            m.stored_variables_and_constraints[area]["final_bellman_value"],
            1 / len_reservoir,
        )

    additional_constraint: List = []
    iterate_stock_discretization = (
        multi_bellman_value_calculation.get_product_stock_discretization()
    )

    for idx in iterate_stock_discretization:

        for a in m.range_reservoir:
            cst = m.solver.LookupConstraint(
                f"BellmanValue{idx}::area<{a}>::week<{m.week}>"
            )

            if cst:

                cst.SetCoefficient(
                    m.stored_variables_and_constraints[a]["final_bellman_value"],
                    1.0,
                )
                cst.SetLb(
                    float(
                        V["intercept"][idx]
                        - sum(
                            [
                                V[f"slope_{area}"][idx]
                                * multi_bellman_value_calculation.dict_reservoirs[
                                    area
                                ].stock_discretization[idx[i]]
                                for i, area in enumerate(m.range_reservoir)
                            ]
                        )
                    )
                )

                for i, area in enumerate(m.range_reservoir):
                    cst.SetCoefficient(
                        m.stored_variables_and_constraints[area]["final_level"],
                        float(-V[f"slope_{area}"][idx]),
                    )

            else:
                cst = m.solver.Add(
                    m.stored_variables_and_constraints[a]["final_bellman_value"]
                    >= sum(
                        [
                            V[f"slope_{area}"][idx]
                            * (
                                m.stored_variables_and_constraints[area]["final_level"]
                                - multi_bellman_value_calculation.dict_reservoirs[
                                    area
                                ].stock_discretization[idx[i]]
                            )
                            for i, area in enumerate(m.range_reservoir)
                        ]
                    )
                    + V["intercept"][idx],
                    name=f"BellmanValue{idx}::area<{a}>::week<{m.week}>",
                )
            additional_constraint.append(cst)

    for area in m.range_reservoir:
        cst_initial_level = m.solver.LookupConstraint(
            f"InitialLevelReservoir::area<{area}>::week<{m.week}>"
        )
        if cst_initial_level:
            cst_initial_level.SetBounds(lb=level_i[area], ub=level_i[area])
        else:
            cst_initial_level = m.solver.Add(
                m.stored_variables_and_constraints[area]["initial_level"]
                == level_i[area],
                name=f"InitialLevelReservoir::area<{area}>::week<{m.week}>",
            )

    debut_1 = time()
    solve_status = m.solver.Solve(m.solver_parameters)
    fin_1 = time()

    if solve_status == pywraplp.Solver.OPTIMAL:
        itr = m.solver.Iterations()

        beta = float(m.solver.Objective().Value())
        xf = m.get_solution_value("final_level")
        z = m.get_solution_value("final_bellman_value")
        y = m.get_solution_value("penalties")
        lamb = {}
        for area in m.range_reservoir:
            cst_initial_level = m.solver.LookupConstraint(
                f"InitialLevelReservoir::area<{area}>::week<{m.week}>"
            )
            lamb[area] = float(cst_initial_level.dual_value())

        for cst in additional_constraint:
            cst.SetLb(0)
        for area in m.range_reservoir:
            bellman_value_calculation = multi_bellman_value_calculation.dict_reservoirs[
                area
            ]
            cst_initial_level.SetBounds(
                lb=bellman_value_calculation.reservoir_management.reservoir.capacity,
                ub=bellman_value_calculation.reservoir_management.reservoir.capacity,
            )
        for area in m.range_reservoir:
            m.stored_variables_and_constraints[area]["energy_constraint"]\
            .SetCoefficient(
                m.stored_variables_and_constraints[area]["reservoir_control"], 0
            )

            m.solver.Objective().SetCoefficient(
                m.stored_variables_and_constraints[area]["penalties"], 0
            )
            m.solver.Objective().SetCoefficient(
                m.stored_variables_and_constraints[area]["final_bellman_value"], 0
            )

        cout += beta
        if not (take_into_account_z_and_y):
            cout += -sum(z.values()) / len_reservoir - sum(y.values())

    else:
        print(f"Failed at upper bound : {solve_status}")
        raise (ValueError)
    return (
        fin_1 - debut_1,
        itr,
        cout,
        lamb,
        xf,
        z,
    )

class WeeklyBellmanProblem:
    """ A Class to manipulate the Dynamic Programming Optimization problem when costs
    are precalculated
    """
    
    def __init__(
        self,
        param:TimeScenarioParameter,
        multi_stock_management:MultiStockManagement,
        week_costs_estimation:Estimator,
        name_solver:str,
                 ) -> None:
        """
        Instanciates a Weekly Bellman Problem
        
        Parameters
        ----------
            param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
            multi_stock_management:MultiStockManagement: Description of stocks and their global policies,
            week_costs_estimation:LinearInterpolator: All precalculated lower bounding hyperplains of weekly cost estimation,
            name_solver:str: Solver chosen for the optimization problem, default -> CLP
        """
        self.n_scenarios = param.len_scenario
        self.name_solver=name_solver
        self.managements = multi_stock_management.dict_reservoirs
        self.week_costs_estimation = week_costs_estimation
        self.solver = pywraplp.Solver.CreateSolver(name_solver)
    
    def reset_solver(self) -> None:
        """ Reinitializes the solver so as not to cumulate variables and constraints """
        self.solver = pywraplp.Solver.CreateSolver(self.name_solver)
    
    def write_problem(
        self,
        week:int,
        level_init:Array1D,
        future_costs_estimation:LinearInterpolator,
    ) -> None:
        """
        Writes the weekly bellman optimization problem
        
        Parameters
        ----------
            week:int: week of interest,
            level_init:Array1D: Initial level of each stock
            future_costs_estimation: Approximation of the future costs depending on the final level
        """
        solver = self.solver
        #=====================
        #----- VARIABLES -----
        #=====================
        # Control variable
        x = np.array([solver.NumVar(- mng.reservoir.efficiency * mng.reservoir.max_pumping[week],
                          mng.reservoir.max_generating[week],
                          name=f"control_{area}")
            for area, mng in self.managements.items()])
        self.control = x
        
        # Control cost variable
        control_cost = solver.NumVar(0, solver.infinity(), "control_cost")
        self.control_cost = control_cost
        
        # Following week price variable #Is it declined by scenario or grouped ? -> It is grouped
        future_cost = solver.NumVar(0, solver.infinity(), "future_cost")
        self.future_cost = future_cost
        
        # Induced penalties variable
        reservoir_penalties = [solver.NumVar(0, solver.infinity(), f"pnlty_{area}") for area, _ in self.managements.items()]
        self.reservoir_penalties = reservoir_penalties
        total_penalty = solver.NumVar(0, solver.infinity(), "tot_pnlty")
        self.total_penalty = total_penalty
        
        #Initial level variable
        initial_levels = [solver.NumVar(0, mng.reservoir.capacity, f"lvl_init_{area}") for area, mng in self.managements.items()]
        self.initial_levels = initial_levels
        
        # Next levels variable
        next_levels = np.array([[solver.NumVar(0, mng.reservoir.capacity, f"next_lvl_{area}_{scenario}")
                        for scenario in range(self.n_scenarios)] for area, mng in self.managements.items()])
        
        #=======================
        #----- CONSTRAINTS -----
        #=======================
        
        # Control costs lower bounds
        control_cost_constraints = [
            solver.Add(
                control_cost >= (1/self.n_scenarios)*sum([
                    self.week_costs_estimation[week, s].costs[ctrl_id] + sum([
                        (x[r] - self.week_costs_estimation[week, s].inputs[ctrl_id][r])*self.week_costs_estimation[week, s].duals[ctrl_id, r]
                        for r, _ in enumerate(self.managements)
                    ])
                for s in range(self.n_scenarios)]),
                name=f"control_cost_lb_{ctrl_id}"
            )
        for ctrl_id, _ in enumerate(self.week_costs_estimation[week,0].inputs)]
        
        # Future costs lower bounds
        future_cost_constraints = [
            solver.Add(
                future_cost >= (1/self.n_scenarios) * sum([
                    sum([
                        (next_levels[r,s] - levels[r])*duals[r]
                        for r, _ in enumerate(self.managements)
                    ]) + cost
                    for s in range(self.n_scenarios)
                ]),
                name=f"future_cost_lb_{lvl_id}"
            )
        for lvl_id, (levels, cost, duals) in enumerate(zip(future_costs_estimation.inputs, future_costs_estimation.costs, future_costs_estimation.duals,))]
        
        # Penalties constraints
        # Bottom rule curve
        penalty_constraints  = [
            solver.Add(
                reservoir_penalties[r]>=(1/self.n_scenarios)*sum([
                    (next_levels[r, s] - mng.reservoir.upper_rule_curve[week])*mng.penalty_upper_rule_curve
                for s in range(self.n_scenarios)
                ]),
                name=f"pnlty_cstr_up_{area}"
                )
            for r, (area, mng) in enumerate(self.managements.items())
        ]
        
        # Upper rule curve
        penalty_constraints += [
            solver.Add(
                reservoir_penalties[r]>=(1/self.n_scenarios)*sum([
                    (mng.reservoir.bottom_rule_curve[week] - next_levels[r, s])*mng.penalty_bottom_rule_curve
                for s in range(self.n_scenarios)
                ]),
                name=f"pnlty_cstr_low_{area}"
                )
            for r, (area, mng) in enumerate(self.managements.items())
        ]
        
        # Total penalty constraint
        total_penalty_cst = solver.Add(total_penalty >= sum(reservoir_penalties), name="tot_pnlty_cstr")
        
        # Define next level for every scenario constraint
        next_level_constraints = [
            [
                solver.Add(next_levels[r, s] <= initial_levels[r] - x[r] + mng.reservoir.inflow[week, s],
                           name=f"next_lvl_cstr_{area}_{s}") 
                for r, (area, mng) in enumerate(self.managements.items())
            ] 
            for s in range(self.n_scenarios)
        ]
        
        # Define initial level constraint
        initial_level_constraints = [
            solver.Add(initial_levels[r] <= level_init[r], name=f"init_lvl_cst_{area}")
            for r, area in enumerate(self.managements)
        ]
        self.initial_levels_cstr = initial_level_constraints
        
        #=====================
        #----- OBJECTIVE -----
        #=====================
        
        objective = solver.Objective()
        objective.SetCoefficient(control_cost, 1)
        objective.SetCoefficient(future_cost, 1)
        objective.SetCoefficient(total_penalty, 1)
        objective.SetMinimization()
    
    def solve(
        self,
        remove_future_costs:bool=False,
        verbose:bool=False,
    ) -> tuple[Array1D, float, Array1D]:
        """
        Solves the weekly bellman optimization problem
        
        Parameters
        ----------
            verbose:bool: Will print the solved problem before solving it
            
        Returns
        -------
            controls:np.ndarray: Optimal controls,
            costs:np.ndarray: Objective value, 
            duals:np.ndarray: Dual values of the initial level constraint
        """
        if verbose:
            print("========================SOLVING:=========================")
            print(self.solver.ExportModelAsLpFormat(False).replace('\\', '').replace(',_', ','), sep='\n')
        status = self.solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            controls = np.array([ctrl.solution_value() for ctrl in self.control])
            cost = self.solver.Objective().Value() - self.future_cost.solution_value() * remove_future_costs
            duals = np.array([cstr.dual_value() for cstr in self.initial_levels_cstr])
            return controls, cost, duals
        else:
            print(f"No solution found, status:{status}")
            raise ValueError