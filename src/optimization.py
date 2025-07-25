import os as os
import pickle as pkl
import re
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import ortools.linear_solver.pywraplp as pywraplp
from ortools.linear_solver.python import model_builder
from tqdm import tqdm

from estimation import (
    BellmanValueEstimation,
    Estimator,
    LinearCostEstimator,
    LinearInterpolator,
    PieceWiseLinearInterpolator,
)
from reservoir_management import MultiStockManagement
from type_definition import (
    AreaIndex,
    Dict,
    List,
    ScenarioIndex,
    TimeScenarioIndex,
    TimeScenarioParameter,
    Union,
)


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
        param: TimeScenarioParameter,
        multi_stock_management: MultiStockManagement,
        scenario: int,
        week: int,
        path: str,
        saving_directory: Optional[str] = None,
        name_solver: str = "CLP",
        save_protos: bool = False,
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
        path:str :
            Path where intermediate result are stored
        itr:int :
            Antares iteration considered (Default value = 1)

        Returns
        -------
        None
        """
        self.scenario = scenario
        self.week = week
        self.path = path

        if len(param.name_scenario) >= 1:
            name_scenario = param.name_scenario[scenario]
        else:
            name_scenario = scenario + 1

        if save_protos:
            assert saving_directory is not None
            proto_path = saving_directory + f"/problem-{name_scenario}-{week+1}.pkl"
            already_processed = Path(proto_path).is_file()
        else:
            already_processed = False

        if not already_processed:
            mps_path = path + f"/problem-{name_scenario}-{week+1}--optim-nb-{1}.mps"
            model = model_builder.ModelBuilder()  # type: ignore[no-untyped-call]
            model.import_from_mps_file(mps_path)
            model_proto = model.export_to_proto()
        else:
            with open(proto_path, "rb") as file:
                model_proto, var_and_cstr_ids = pkl.load(file)
                self.stored_variables_and_constraints_ids = var_and_cstr_ids

        solver = pywraplp.Solver.CreateSolver(name_solver)
        assert solver, "Couldn't find any supported solver"
        # solver.EnableOutput()
        solver.SuppressOutput()

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

        self.store_basis = name_solver == "XPRESS_LP"

        self.basis: List = []
        self.control_basis: List = []

        if already_processed:
            self._reset_from_loaded_version(
                multi_stock_management=multi_stock_management
            )
        else:
            self._create_weekly_problem_itr(
                param=param, multi_stock_management=multi_stock_management
            )
            if save_protos and saving_directory is not None:
                if not (os.path.exists(saving_directory)):
                    os.makedirs(saving_directory)
                proto = model_builder.ModelBuilder().export_to_proto()  # type: ignore[no-untyped-call]
                self.solver.ExportModelToProto(output_model=proto)
                with open(proto_path, "wb") as file:
                    pkl.dump((proto, self.stored_variables_and_constraints_ids), file)

    def _add_basis(self, basis: Basis, control_basis: Dict[AreaIndex, float]) -> None:
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

    def _get_basis(self) -> tuple[list, list]:
        var_basis = []
        con_basis = []
        for var in self.solver.variables():
            var_basis.append(var.basis_status())
        for con in self.solver.constraints():
            con_basis.append(con.basis_status())
        return var_basis, con_basis

    def _load_basis(self, basis: Basis) -> None:
        len_cons = len(self.solver.constraints())
        len_vars = len(self.solver.variables())
        if len_vars > len(basis.rstatus):
            basis.rstatus += [0] * (len_vars - len(basis.rstatus))
        if len_cons > len(basis.cstatus):
            basis.cstatus += [0] * (len_cons - len(basis.cstatus))
        self.solver.SetStartingLpBasis(
            basis.rstatus[:len_vars], basis.cstatus[:len_cons]
        )

    def _find_closest_basis(self, control: Dict[AreaIndex, float]) -> Basis:
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

    def _create_weekly_problem_itr(
        self, param: TimeScenarioParameter, multi_stock_management: MultiStockManagement
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
        self.stored_variables_and_constraints: Dict[
            AreaIndex, Dict[str, Union[pywraplp.Constraint, pywraplp.Variable]]
        ] = {}
        self.stored_variables_and_constraints_ids = {}
        self.range_reservoir = multi_stock_management.areas

        len_week = param.len_week
        for (
            area,
            reservoir_management,
        ) in multi_stock_management.dict_reservoirs.items():
            res = reservoir_management.reservoir
            hours_in_week = res.hours_in_week

            model = self.solver

            self._delete_variable(
                hours_in_week=hours_in_week,
                name_variable=f"^HydroLevel::area<{area}>::hour<.",  # variable niv stock
            )
            self._fix_variable(
                hours_in_week=hours_in_week,
                name_variable=f"^Overflow::area<{area}>::hour<.",  # variable overflow
            )
            self._delete_constraint(
                hours_in_week=hours_in_week,
                name_constraint=f"^AreaHydroLevel::area<{area}>::hour<.",  # Conservation niveau stock
            )

            cst = model.constraints()
            vars = model.variables()

            binding_id = [
                i
                for i in range(len(cst))
                if re.search(
                    f"^HydroPower::area<{area}>::week<.",  # Turbinage- rho*pompage = cible
                    cst[i].name(),
                )
            ]

            assert len(binding_id) == 1  # Beware of lowercases, study path, mps path

            hyd_prod_vars = [
                var
                for id, var in enumerate(vars)
                if re.search(
                    f"HydProd::area<{area}>::hour<",
                    var.name(),
                )
            ]

            # Correcting shenanigans in the generating var Ub due to heuristic gestion
            for var in hyd_prod_vars:
                var.SetUb(res.max_generating[self.week] / res.hours_in_week)

            x_s = model.Var(
                lb=0,
                ub=res.capacity,
                integer=False,
                name=f"InitialLevel::area<{area}>::week<{self.week}>",
            )  # Initial level

            x_s_1 = model.Var(
                lb=0,
                ub=res.capacity,
                integer=False,
                name=f"FinalLevel::area<{area}>::week<{self.week}>",
            )  # Final level

            U = model.Var(
                lb=-res.max_pumping[self.week] * res.efficiency,
                ub=res.max_generating[self.week],
                integer=False,
                name=f"Control::area<{area}>::week<{self.week}>",
            )  # Reservoir Control

            if reservoir_management.overflow:
                model.Add(
                    x_s_1 <= x_s - U + res.inflow[self.week, self.scenario],
                    name=f"ReservoirConservation::area<{area}>::week<{self.week}>",
                )
            else:
                model.Add(
                    x_s_1 == x_s - U + res.inflow[self.week, self.scenario],
                    name=f"ReservoirConservation::area<{area}>::week<{self.week}>",
                )

            y = model.Var(
                lb=0,
                ub=model.Infinity(),
                integer=False,
                name=f"Penalties::area<{area}>::week<{self.week}>",
            )  # Penality for violating guide curves

            if self.week != len_week - 1 or not reservoir_management.final_level:
                model.Add(
                    y
                    >= -reservoir_management.penalty_bottom_rule_curve
                    * (x_s_1 - res.bottom_rule_curve[self.week]),
                    name=f"PenaltyForViolatingBottomRuleCurve::area<{area}>::week<{self.week}>",
                )
                model.Add(
                    y
                    >= reservoir_management.penalty_upper_rule_curve
                    * (x_s_1 - res.upper_rule_curve[self.week]),
                    name=f"PenaltyForViolatingUpperRuleCurve::area<{area}>::week<{self.week}>",
                )
            else:
                model.Add(
                    y
                    >= -reservoir_management.penalty_final_level
                    * (x_s_1 - reservoir_management.final_level),
                    name=f"PenaltyForViolatingBottomRuleCurve::area<{area}>::week<{self.week}>",
                )
                model.Add(
                    y
                    >= reservoir_management.penalty_final_level
                    * (x_s_1 - reservoir_management.final_level),
                    name=f"PenaltyForViolatingUpperRuleCurve::area<{area}>::week<{self.week}>",
                )

            z = model.Var(
                lb=-model.Infinity(),
                ub=model.Infinity(),
                integer=False,
                name=f"BellmanValue::area<{area}>::week<{self.week}>",
            )  # Auxiliar variable to introduce the piecewise representation of the future cost

            self.stored_variables_and_constraints[area] = {
                "energy_constraint": cst[binding_id[0]],
                "reservoir_control": U,
                "initial_level": x_s,
                "final_level": x_s_1,
            }
            self.stored_variables_and_constraints_ids[area] = {
                "energy_constraint": cst[binding_id[0]].index(),
                "reservoir_control": U.index(),
                "initial_level": x_s.index(),
                "final_level": x_s_1.index(),
            }
            self.stored_variables_and_constraints[area]["penalties"] = y
            self.stored_variables_and_constraints[area]["final_bellman_value"] = z
            self.stored_variables_and_constraints_ids[area]["penalties"] = y.index()
            self.stored_variables_and_constraints_ids[area][
                "final_bellman_value"
            ] = z.index()

    def _reset_from_loaded_version(
        self, multi_stock_management: MultiStockManagement
    ) -> None:
        assert [a for a in multi_stock_management.areas] == [
            a for a in self.stored_variables_and_constraints_ids.keys()
        ], "Loaded Proto and current reservoir management should have same reservoirs"
        self.range_reservoir = multi_stock_management.areas
        self.stored_variables_and_constraints = {}
        constraints = self.solver.constraints()
        variables = self.solver.variables()
        for area, obj_dict in self.stored_variables_and_constraints_ids.items():
            self.stored_variables_and_constraints[area] = {}
            for obj_name, obj_id in obj_dict.items():
                source = constraints if "constraint" in obj_name else variables
                self.stored_variables_and_constraints[area][obj_name] = source[obj_id]

    def _delete_variable(self, hours_in_week: int, name_variable: str) -> None:
        model = self.solver
        var = model.variables()
        var_id = [i for i in range(len(var)) if re.search(name_variable, var[i].name())]
        assert len(var_id) in [0, hours_in_week]
        if len(var_id) == hours_in_week:
            for i in var_id:
                var[i].SetLb(-model.Infinity())
                var[i].SetUb(model.Infinity())
                model.Objective().SetCoefficient(var[i], 0)

    def _fix_variable(self, hours_in_week: int, name_variable: str) -> None:
        model = self.solver
        var = model.variables()
        var_id = [i for i in range(len(var)) if re.search(name_variable, var[i].name())]
        assert len(var_id) in [0, hours_in_week]
        if len(var_id) == hours_in_week:
            for i in var_id:
                var[i].SetLb(0)
                var[i].SetUb(0)

    def _delete_constraint(self, hours_in_week: int, name_constraint: str) -> None:
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
        self, control: Dict[AreaIndex, float], prev_basis: Basis = Basis([], [])
    ) -> tuple[float, Dict[AreaIndex, float], int, float]:
        """
        Modify and solve problem to evaluate weekly cost associated with a particular control of the reservoir.

        Parameters
        ----------
        control:Dict[AreaIndex, float] :
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
                self._load_basis(prev_basis)
            else:
                basis = self._find_closest_basis(control=control)
                self._load_basis(basis)

        for area in self.range_reservoir:
            self._set_constraints_predefined_control(control[area], area)
        try:
            beta, lamb, _, _, _, itr, computing_time = self._solve_problem()
            # print(f"✔ for controls {control}")
        except ValueError:
            print(f"✘ for controls {control}")
            raise ValueError
        return beta, lamb, itr, computing_time

    def _set_constraints_predefined_control(
        self, control: float, area: AreaIndex
    ) -> None:

        self.stored_variables_and_constraints[area]["energy_constraint"].SetBounds(
            lb=control, ub=control
        )

    def _set_constraints_initial_level_and_bellman_values(
        self,
        V: Estimator,
        all_level_i: Dict[AreaIndex, float],
    ) -> List[pywraplp.Constraint]:

        for area in self.range_reservoir:

            self.stored_variables_and_constraints[area][
                "energy_constraint"
            ].SetCoefficient(
                self.stored_variables_and_constraints[area]["reservoir_control"],
                -1,
            )
            self.stored_variables_and_constraints[area]["energy_constraint"].SetLb(0.0)
            self.stored_variables_and_constraints[area]["energy_constraint"].SetUb(0.0)

            self.solver.Objective().SetCoefficient(
                self.stored_variables_and_constraints[area]["penalties"], 1
            )
            self.solver.Objective().SetCoefficient(
                self.stored_variables_and_constraints[area]["final_bellman_value"],
                1,
            )
        if type(V) is PieceWiseLinearInterpolator:
            additional_constraint = self._build_univariate_bellman_constraints(V)
        elif type(V) is BellmanValueEstimation:
            additional_constraint = self._build_multivariate_bellman_constraints(V)
        for area in self.range_reservoir:
            level_i = all_level_i[area]
            cst_initial_level = self.solver.LookupConstraint(
                f"InitialLevelReservoir::area<{area}>::week<{self.week}>"
            )
            if cst_initial_level:
                cst_initial_level.SetBounds(lb=level_i, ub=level_i)
            else:
                cst_initial_level = self.solver.Add(
                    self.stored_variables_and_constraints[area]["initial_level"]
                    == level_i,
                    name=f"InitialLevelReservoir::area<{area}>::week<{self.week}>",
                )

        return additional_constraint

    def _build_univariate_bellman_constraints(
        self, V: PieceWiseLinearInterpolator
    ) -> List[pywraplp.Constraint]:
        additional_constraint = []
        for area in self.range_reservoir:
            bellman_value = V

            X = bellman_value.inputs
            cost = bellman_value.costs

            for j in range(len(X) - 1):
                if (cost[j + 1] < float("inf")) & (cost[j] < float("inf")):
                    cst = self.solver.LookupConstraint(
                        f"BellmanValueBetween{j}And{j+1}::area<{area}>::week<{self.week}>"
                    )
                    if cst:
                        cst.SetCoefficient(
                            self.stored_variables_and_constraints[area]["final_level"],
                            -(-cost[j + 1] + cost[j]) / (X[j + 1] - X[j]),
                        )
                        cst.SetLb(
                            (-cost[j + 1] + cost[j]) / (X[j + 1] - X[j]) * (-X[j])
                            - cost[j]
                        )
                    else:
                        cst = self.solver.Add(
                            self.stored_variables_and_constraints[area][
                                "final_bellman_value"
                            ]
                            >= (-cost[j + 1] + cost[j])
                            / (X[j + 1] - X[j])
                            * (
                                self.stored_variables_and_constraints[area][
                                    "final_level"
                                ]
                                - X[j]
                            )
                            - cost[j],
                            name=f"BellmanValueBetween{j}And{j+1}::area<{area}>::week<{self.week}>",
                        )
                    additional_constraint.append(cst)
        return additional_constraint

    def _remove_bellman_constraints(
        self,
        multi_stock_management: MultiStockManagement,
        additional_constraint: List[pywraplp.Constraint],
    ) -> None:
        for cst in additional_constraint:
            cst.SetLb(0)

        for area in self.range_reservoir:
            res = multi_stock_management.dict_reservoirs[area].reservoir

            cst_initial_level = self.solver.LookupConstraint(
                f"InitialLevelReservoir::area<{area}>::week<{self.week}>"
            )
            cst_initial_level.SetBounds(
                lb=res.capacity,
                ub=res.capacity,
            )

            self.stored_variables_and_constraints[area][
                "energy_constraint"
            ].SetCoefficient(
                self.stored_variables_and_constraints[area]["reservoir_control"], 0
            )

            self.solver.Objective().SetCoefficient(
                self.stored_variables_and_constraints[area]["penalties"], 0
            )
            self.solver.Objective().SetCoefficient(
                self.stored_variables_and_constraints[area]["final_bellman_value"],
                0,
            )

    def _solve_problem(self) -> tuple[
        float,
        Dict[AreaIndex, float],
        Dict[AreaIndex, float],
        Dict[AreaIndex, float],
        Dict[AreaIndex, float],
        int,
        float,
    ]:

        start = time()
        solve_status = self.solver.Solve(self.solver_parameters)
        end = time()

        if solve_status == pywraplp.Solver.OPTIMAL:
            itr = self.solver.Iterations()
            if self.store_basis:
                rbas, cbas = self._get_basis()
                self._add_basis(
                    basis=Basis(rbas, cbas),
                    control_basis=self._get_solution_value("reservoir_control"),
                )

            beta = float(self.solver.Objective().Value())
            xf = self._get_solution_value("final_level")
            lamb = self._get_dual_value("energy_constraint")
            z = self._get_solution_value("final_bellman_value")
            y = self._get_solution_value("penalties")
            return (beta, lamb, xf, y, z, itr, end - start)

        else:
            print(f"Failed to solve : {solve_status}")
            raise (ValueError)

    def _get_dual_value(self, name_constraint: str) -> Dict[AreaIndex, float]:
        value = {}
        for area in self.range_reservoir:
            value[area] = float(
                self.stored_variables_and_constraints[area][
                    name_constraint
                ].dual_value()
            )

        return value

    def _get_solution_value(self, name_variable: str) -> Dict[AreaIndex, float]:
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
        multi_stock_management: MultiStockManagement,
        V: Estimator,
        level_i: Dict[AreaIndex, float],
        take_into_account_z_and_y: bool,
        find_optimal_basis: bool = True,
        param: Optional[TimeScenarioParameter] = None,
        reward: Optional[LinearCostEstimator] = None,
    ) -> tuple[
        float,
        int,
        float,
        Dict[AreaIndex, float],
        Dict[AreaIndex, float],
        Dict[AreaIndex, float],
        Dict[AreaIndex, float],
    ]:

        cout = 0.0

        additional_constraint = self._set_constraints_initial_level_and_bellman_values(
            V, level_i
        )

        if find_optimal_basis:
            if len(self.control_basis) >= 1:
                if len(self.control_basis) >= 2:
                    assert reward is not None
                    assert param is not None
                    problem = WeeklyBellmanProblem(
                        param=param,
                        multi_stock_management=multi_stock_management,
                        week_costs_estimation={
                            ScenarioIndex(self.scenario): reward[
                                TimeScenarioIndex(self.week, self.scenario)
                            ]
                        },
                        week=self.week,
                    )

                    likely_control, _, _, _ = problem.solve(
                        level_init={
                            a: {
                                ScenarioIndex(s): level_i[a]
                                for s in range(param.len_scenario)
                            }
                            for a in multi_stock_management.areas
                        },
                        future_costs_estimation=V,
                    )
                    basis = self._find_closest_basis(
                        {
                            a: u[ScenarioIndex(self.scenario)]
                            for a, u in likely_control.items()
                        }
                    )
                else:
                    basis = self.basis[0]
                self._load_basis(basis)

        beta, _, xf, y, z, itr, t = self._solve_problem()

        cout = beta

        lamb = self._get_duals_on_initial_level_cst()

        optimal_controls = {}
        for area in self.range_reservoir:
            optimal_controls[area] = -(
                xf[area]
                - level_i[area]
                - multi_stock_management.dict_reservoirs[area].reservoir.inflow[
                    self.week, self.scenario
                ]
            )

        self._remove_bellman_constraints(multi_stock_management, additional_constraint)

        if not (take_into_account_z_and_y):
            cout += -sum(z.values()) - sum(y.values())

        return (t, itr, cout, lamb, optimal_controls, xf, z)

    def _get_duals_on_initial_level_cst(self) -> Dict[AreaIndex, float]:
        lamb = {}
        for area in self.range_reservoir:
            cst_initial_level = self.solver.LookupConstraint(
                f"InitialLevelReservoir::area<{area}>::week<{self.week}>"
            )
            lamb[area] = float(cst_initial_level.dual_value())
        return lamb

    def _build_multivariate_bellman_constraints(
        self,
        V: BellmanValueEstimation,
    ) -> List[pywraplp.Constraint]:
        additional_constraint: List = []
        len_reservoir = len(self.range_reservoir)
        iterate_stock_discretization = (
            V.discretization.get_product_stock_discretization()
        )

        for idx in iterate_stock_discretization:
            for a in self.range_reservoir:
                cst = self.solver.LookupConstraint(
                    f"BellmanValue{idx}::area<{a}>::week<{self.week}>"
                )

                if cst:
                    cst.SetCoefficient(
                        self.stored_variables_and_constraints[a]["final_bellman_value"],
                        1.0 * len_reservoir,
                    )
                    cst.SetLb(
                        float(
                            V["intercept"][idx]
                            - sum(
                                [
                                    V[f"slope_{area}"][idx]
                                    * V.discretization.list_discretization[area][idx[i]]
                                    for i, area in enumerate(self.range_reservoir)
                                ]
                            )
                        )
                    )

                    for i, area in enumerate(self.range_reservoir):
                        cst.SetCoefficient(
                            self.stored_variables_and_constraints[area]["final_level"],
                            float(-V[f"slope_{area}"][idx]),
                        )

                else:
                    cst = self.solver.Add(
                        self.stored_variables_and_constraints[a]["final_bellman_value"]
                        * len_reservoir
                        >= sum(
                            [
                                V[f"slope_{area}"][idx]
                                * (
                                    self.stored_variables_and_constraints[area][
                                        "final_level"
                                    ]
                                    - V.discretization.list_discretization[area][idx[i]]
                                )
                                for i, area in enumerate(self.range_reservoir)
                            ]
                        )
                        + V["intercept"][idx],
                        name=f"BellmanValue{idx}::area<{a}>::week<{self.week}>",
                    )
                additional_constraint.append(cst)
        return additional_constraint


def initialize_antares_problems(
    param: TimeScenarioParameter,
    multi_stock_management: MultiStockManagement,
    output_path: str,
    name_solver: str,
    verbose: bool = False,
    save_protos: bool = False,
    saving_dir: Optional[str] = None,
) -> Dict[TimeScenarioIndex, AntaresProblem]:
    """
    Creates Instances of the Antares problem for every week / scenario

    Parameters
    ----------
    param:TimeScenarioParameter: Contains the details of the simulations we'll optimize on,
    multi_stock_management:MultiStockManagement: Management information for every stock ,
    output_path:str: Folder containing the mps files generated for our study,
    name_solver:str: Name of the solver used,
    direct_bellman_calc:bool: Integrate future costs in the LP formulation
    verbose:bool: Control displays at runnning time

    Returns
    -------
    list_models: Dict[TimeScenarioIndex, AntaresProblem]: Dictionnary with all problems
    """
    list_models: Dict[TimeScenarioIndex, AntaresProblem] = {}
    week_range = range(param.len_week)
    if verbose:
        week_range = tqdm(week_range, desc="Problem initialization", colour="Yellow")
    for week in week_range:
        for scenario in range(param.len_scenario):
            m = AntaresProblem(
                scenario=scenario,
                week=week,
                path=output_path,
                saving_directory=saving_dir,
                name_solver=name_solver,
                param=param,
                multi_stock_management=multi_stock_management,
                save_protos=save_protos,
            )

            list_models[TimeScenarioIndex(week, scenario)] = m
    return list_models


class WeeklyBellmanProblem:
    """A Class to manipulate the Dynamic Programming Optimization problem when costs
    are precalculated
    """

    def __init__(
        self,
        param: TimeScenarioParameter,
        multi_stock_management: MultiStockManagement,
        week: int,
        week_costs_estimation: Dict[ScenarioIndex, LinearInterpolator],
        name_solver: str = "CLP",
        divisor: dict[str, float] = {"euro": 1.0, "energy": 1.0},
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

        self.precision = 6
        for estimator in week_costs_estimation.values():
            estimator.round(precision=self.precision)
            estimator.count_redundant(tolerance=0, remove=True)
        self.scenarios = week_costs_estimation.keys()
        self.name_solver = name_solver
        self.managements = multi_stock_management.dict_reservoirs
        self.week_costs_estimation = week_costs_estimation
        self.solver = pywraplp.Solver.CreateSolver(name_solver)
        self.parameters = pywraplp.MPSolverParameters()
        self.div_euros = divisor["euro"]
        self.div_energy = divisor["energy"]
        self.div_price = divisor["euro"] / divisor["energy"]
        self.param = param
        self.week = week

        # Base variables and constraints
        self._get_control_dynamic()
        self._get_control_cost()
        self._get_penalty_cost()

        # Create objective function
        objective = self.solver.Objective()
        objective.SetCoefficient(self.control_cost, 1)
        objective.SetCoefficient(self.penalty_cost, 1)
        objective.SetMinimization()

    def _get_control_dynamic(self) -> None:
        # ========= Variables =========
        # Controls -max_pumping * efficiency <= X <= max_generating
        controls = {
            area: {
                s: self.solver.NumVar(
                    np.round(
                        -mng.reservoir.max_pumping[self.week]
                        * mng.reservoir.efficiency
                        / self.div_energy,
                        self.precision,
                    ),
                    np.round(
                        mng.reservoir.max_generating[self.week] / self.div_energy,
                        self.precision,
                    ),
                    name=f"control_{area}_{s}",
                )
                for s in self.scenarios
            }
            for area, mng in self.managements.items()
        }

        # Initial levels 0 <= lvl <= capacity
        initial_levels = {
            area: {
                s: self.solver.NumVar(
                    0,
                    np.round(mng.reservoir.capacity / self.div_energy, self.precision),
                    name=f"lvl_init_{area}_{s}",
                )
                for s in self.scenarios
            }
            for area, mng in self.managements.items()
        }

        # Final levels 0 <= lvl <= capacity
        next_levels = {
            area: {
                s: self.solver.NumVar(
                    0,
                    np.round(mng.reservoir.capacity / self.div_energy, self.precision),
                    name=f"lvl_next_{area}_{s}",
                )
                for s in self.scenarios
            }
            for area, mng in self.managements.items()
        }

        # ========= Constraints =========
        # Next level constraints next_level <= initial_level - X + inflow
        for area, mng in self.managements.items():
            for s in self.scenarios:
                if mng.overflow:
                    self.solver.Add(
                        next_levels[area][s]
                        <= np.round(
                            (mng.reservoir.inflow[self.week, s.scenario])
                            / self.div_energy,
                            self.precision,
                        )
                        + initial_levels[area][s]
                        - controls[area][s],
                        name=f"dynamic_cst_{area}_{s}",
                    )
                else:
                    self.solver.Add(
                        next_levels[area][s]
                        == np.round(
                            (mng.reservoir.inflow[self.week, s.scenario])
                            / self.div_energy,
                            self.precision,
                        )
                        + initial_levels[area][s]
                        - controls[area][s],
                        name=f"dynamic_cst_{area}_{s}",
                    )

        self.controls = controls
        self.initial_levels = initial_levels
        self.next_levels = next_levels

    def _get_future_cost(self) -> None:

        inf = self.solver.infinity()
        # ========= Variables ==========
        # 0 <= future_cost_per_scenario
        future_cost_per_scenario = {
            s: self.solver.NumVar(0, inf, name=f"future_cost_scen_{s}")
            for s in self.scenarios
        }

        # 0 <= future_cost
        future_cost = self.solver.NumVar(0, inf, name=f"future_cost")

        if type(self.future_costs_estimation) is LinearInterpolator:
            future_cost_constraints = {
                s: [
                    self.solver.Add(
                        future_cost_per_scenario[s]
                        >= (cost - min(self.future_costs_estimation.costs))
                        / self.div_euros
                        + sum(
                            [
                                (
                                    self.next_levels[area][s]
                                    - levels[r] / self.div_energy
                                )
                                * np.round(duals[r] / self.div_price, self.precision)
                                for r, area in enumerate(self.managements.keys())
                            ]
                        ),
                        name=f"f_cost_lb_{lvl_id}_{s}",
                    )
                    for lvl_id, (levels, cost, duals) in enumerate(
                        zip(
                            self.future_costs_estimation.inputs,
                            self.future_costs_estimation.costs,
                            self.future_costs_estimation.duals,
                        )
                    )
                ]
                for s in self.scenarios
            }
        elif type(self.future_costs_estimation) is PieceWiseLinearInterpolator:
            for area in self.managements:
                bellman_value = self.future_costs_estimation

                X = bellman_value.inputs
                cost = bellman_value.costs
                for s in self.scenarios:
                    for j in range(len(X) - 1):
                        if (cost[j + 1] < float("inf")) & (cost[j] < float("inf")):
                            cst = self.solver.Add(
                                future_cost_per_scenario[s]
                                >= (-cost[j + 1] + cost[j])
                                / (X[j + 1] - X[j])
                                / self.div_price
                                * (self.next_levels[area][s] - X[j] / self.div_energy)
                                - cost[j] / self.div_euros,
                                name=f"f_cost_lb_{area}_{j}_{s}",
                            )

        # Average all future costs
        # n_scenarios * future_cost >= Σ([future_cost_per_scenario])
        future_cost_tot_constraint = self.solver.Add(
            len(self.scenarios) * future_cost
            >= sum([x for x in future_cost_per_scenario.values()]),
            name=f"future_cost_total",
        )

        self.future_cost_per_scenario = future_cost_per_scenario
        self.future_cost = future_cost
        self.future_cost_tot_cst = future_cost_tot_constraint

    def _get_control_cost(
        self,
    ) -> None:
        inf = self.solver.infinity()
        # ========= Variables ==========
        # 0 <= control_cost_per_scenario
        control_cost_per_scenario = {
            s: self.solver.NumVar(0, inf, name=f"control_cost_scen_{s}")
            for s in self.scenarios
        }

        # 0 <= control_cost
        control_cost = self.solver.NumVar(0, inf, name=f"control_cost")

        # ========= Constraints ==========

        control_cost_constraints = [
            [
                self.solver.Add(
                    control_cost_per_scenario[s]
                    >= cost / self.div_euros
                    + sum(
                        [
                            (self.controls[area][s] - control[r] / self.div_energy)
                            * np.round(duals[r] / self.div_price, self.precision)
                            for r, area in enumerate(self.managements.keys())
                        ]
                    ),
                    name=f"c_cost_lb_{ctrl_id}_{s}",
                )
                for ctrl_id, (control, cost, duals) in enumerate(
                    zip(
                        self.week_costs_estimation[ScenarioIndex(s.scenario)].inputs,
                        self.week_costs_estimation[ScenarioIndex(s.scenario)].costs,
                        self.week_costs_estimation[ScenarioIndex(s.scenario)].duals,
                    )
                )
            ]
            for s in self.scenarios
        ]

        # Average all control costs
        # n_scenarios * control_cost >= Σ([control_cost_per_scenario])
        control_cost_tot_constraint = self.solver.Add(
            len(self.scenarios) * control_cost
            >= sum([x for x in control_cost_per_scenario.values()]),
            name=f"control_cost_total",
        )

        self.control_cost_per_scenario = control_cost_per_scenario
        self.control_cost = control_cost
        self.control_cost_csts = control_cost_constraints
        self.control_cost_tot_cst = control_cost_tot_constraint
        return control_cost

    def _get_penalty_cost(self) -> None:
        inf = self.solver.infinity()
        # ========= Variables =========
        # Late penalties (>=0)
        late_curve_penalties = {
            area: {
                s: self.solver.NumVar(0, inf, name=f"late_curve_penalty_{area}_{s}")
                for s in self.scenarios
            }
            for area in self.managements.keys()
        }

        # Penalty cost (free)
        penalty_cost = self.solver.NumVar(-inf, inf, f"penalty_cost")

        # ========= Constraints =========
        # curve_penalties >= (lvl_low - lvl)*penalty_low
        for area, mng in self.managements.items():
            for s in self.scenarios:
                if self.week != self.param.len_week - 1 or not mng.final_level:
                    self.solver.Add(
                        late_curve_penalties[area][s]
                        >= (
                            mng.reservoir.bottom_rule_curve[self.week] / self.div_energy
                            - self.next_levels[area][s]
                        )
                        * mng.penalty_bottom_rule_curve
                        / self.div_price,
                        name=f"late_low_curve_cst_{area}_{s}",
                    )
                    self.solver.Add(
                        late_curve_penalties[area][s]
                        >= (
                            self.next_levels[area][s]
                            - mng.reservoir.upper_rule_curve[self.week]
                            / self.div_energy
                        )
                        * mng.penalty_upper_rule_curve
                        / self.div_price,
                        name=f"late_sup_curve_cst_{area}_{s}",
                    )
                else:
                    self.solver.Add(
                        late_curve_penalties[area][s]
                        >= (
                            mng.final_level / self.div_energy
                            - self.next_levels[area][s]
                        )
                        * mng.penalty_final_level
                        / self.div_price,
                        name=f"late_low_curve_cst_{area}_{s}",
                    )
                    self.solver.Add(
                        late_curve_penalties[area][s]
                        >= (
                            self.next_levels[area][s]
                            - mng.final_level / self.div_energy
                        )
                        * mng.penalty_final_level
                        / self.div_price,
                        name=f"late_sup_curve_cst_{area}_{s}",
                    )

        # Total penalty per scenario
        # total_penalty >= Σ Σ curve_pen + overflow_pen
        total_penalty_cost_constraint = self.solver.Add(
            len(self.scenarios) * penalty_cost
            >= sum(
                [
                    late_curve_penalties[area][s]
                    for s in self.scenarios
                    for area in self.managements
                ]
            ),
            name=f"total_penalty_cost_constraint",
        )

        self.penalty_cost = penalty_cost
        self.late_curve_penalties = late_curve_penalties
        self.total_penalty_cost_cst = total_penalty_cost_constraint
        return penalty_cost

    def solve(
        self,
        level_init: Dict[AreaIndex, Dict[ScenarioIndex, float]],
        future_costs_estimation: Estimator,
        remove_future_costs: bool = False,
        remove_penalties: bool = False,
    ) -> tuple[
        Dict[AreaIndex, Dict[ScenarioIndex, float]],
        float,
        Dict[AreaIndex, Dict[ScenarioIndex, float]],
        Dict[AreaIndex, Dict[ScenarioIndex, float]],
    ]:
        """
        Solves the weekly bellman optimization problem

        Parameters
        ----------
            verbose:bool: Will print the solved problem before solving it

        Returns
        -------
            controls:Dict[AreaIndex, Dict[ScenarioIndex, float]]: Optimal controls,
            costs:float: Objective value,
            duals:Dict[AreaIndex, Dict[ScenarioIndex, float]]: Dual values of the initial level constraint
        """
        # Precaution
        if type(future_costs_estimation) is LinearInterpolator:
            future_costs_estimation.round(precision=self.precision)
            future_costs_estimation.count_redundant(tolerance=0, remove=True)
        self.future_costs_estimation = future_costs_estimation

        self._get_future_cost()

        objective = self.solver.Objective()
        objective.SetCoefficient(self.future_cost, 1)

        # Initial level constraint  initial_levels == level_init
        initial_level_constraints = {
            area: {
                s: self.solver.Add(
                    self.initial_levels[area][s]
                    == np.round(level_init[area][s] / self.div_energy, self.precision),
                    name=f"init_lvl_cst_{area}_{s}",
                )
                for s in self.scenarios
            }
            for area in self.managements
        }

        status = self.solver.Solve(self.parameters)
        if status != pywraplp.Solver.OPTIMAL:
            mps_path = "problem_log"
            with open(f"{mps_path}.txt", "w") as pb_log:
                pb_log.write(self.solver.ExportModelAsLpFormat(False))
            with open(f"{mps_path}.mps", "w") as pb_log:
                old_mps = self.solver.ExportModelAsMpsFormat(
                    fixed_format=False, obfuscated=False
                )
                pb_log.write(old_mps)
            print(f"No solution found, status: {status}, mps written to {mps_path} ")
            # Export model to LP and MPS formats
            raise ValueError

        controls = {
            area: {
                s: self.controls[area][s].solution_value() * self.div_energy
                for s in self.scenarios
            }
            for area in self.managements
        }  # Shape n_res x n_scen
        levels = {
            area: {
                s: self.next_levels[area][s].solution_value() * self.div_energy
                for s in self.scenarios
            }
            for area in self.managements
        }
        cost = self.solver.Objective().Value()
        # Removing unwanted parts of the cost
        cost -= self.future_cost.solution_value() * remove_future_costs
        cost -= self.penalty_cost.solution_value() * remove_penalties
        if type(future_costs_estimation) is LinearInterpolator:
            cost += min(future_costs_estimation.costs) / self.div_euros
        duals = {
            area: {
                s: initial_level_constraints[area][s].dual_value() * self.div_price
                for s in self.scenarios
            }
            for area in self.managements
        }
        return (
            controls,
            cost * self.div_euros,
            duals,
            levels,
        )
