from random import randint, seed
from time import time

import numpy as np
import xpress as xp
from scipy.interpolate import interp1d

# xp.controls.outputlog = 0
# xp.controls.threads = 1
# xp.controls.scaling = 0
# xp.controls.presolve = 0
# xp.controls.feastol = 1.0e-7
# xp.controls.optimalitytol = 1.0e-7
xp.setOutputEnabled(False)


class Basis:

    def __init__(self, rstatus: list = [], cstatus: list = []) -> None:
        self.rstatus = rstatus
        self.cstatus = cstatus

    def not_empty(self) -> bool:
        return len(self.rstatus) != 0


class AntaresParameter:

    def __init__(self, S: int = 52, H: int = 168, NTrain: int = 1) -> None:
        self.S = S
        self.H = H
        self.NTrain = NTrain
        self.list_problem = [[[] for k in range(NTrain)] for s in range(S)]

    def get_S(self) -> int:
        return self.S

    def get_H(self) -> int:
        return self.H

    def get_NTrain(self) -> int:
        return self.NTrain


class Reservoir:
    """Describes reservoir parameters"""

    def __init__(
        self,
        param: AntaresParameter,
        capacity: float,
        efficiency: float,
        dir_study: str,
        name_area: str,
        name: str,
        final_level: bool = True,
    ):

        H = param.get_H()

        self.capacity = capacity

        courbes_guides = (
            np.loadtxt(
                dir_study
                + "/input/hydro/common/capacity/reservoir_"
                + name_area
                + ".txt"
            )[:, [0, 2]]
            * self.capacity
        )
        if courbes_guides[0, 0] == courbes_guides[0, 1]:
            self.initial_level = courbes_guides[0, 0]
        else:
            print("Problème avec le niveau initial")
        Xmin = courbes_guides[6:365:7, 0]
        Xmax = courbes_guides[6:365:7, 1]
        self.Xmin = np.concatenate((Xmin, Xmin[[0]]))
        self.Xmax = np.concatenate((Xmax, Xmax[[0]]))
        if final_level:
            self.Xmin[51] = self.initial_level
            self.Xmax[51] = self.initial_level

        self.inflow = (
            np.loadtxt(dir_study + "/input/hydro/series/" + name_area + "/mod.txt")[
                6:365:7
            ]
            * 7
            / H
        )
        assert "_" not in name
        self.name = name

        P_turb = np.loadtxt(
            dir_study + "/input/hydro/common/capacity/maxpower_" + name_area + ".txt"
        )[:, 0]
        P_pump = np.loadtxt(
            dir_study + "/input/hydro/common/capacity/maxpower_" + name_area + ".txt"
        )[:, 2]
        self.P_turb = P_turb
        self.P_pump = P_pump
        self.efficiency = efficiency


class AntaresProblem:

    def __init__(self, year: int, week: int, path: str, itr: int = 1) -> None:
        self.year = year
        self.week = week
        self.path = path

        model = xp.problem()
        model.controls.xslp_log = -1
        model.controls.lplogstyle = 0
        model.read(path + f"/problem-{year+1}-{week+1}--optim-nb-{itr}.mps")
        self.model = model

        self.basis = []
        self.control_basis = []

    def add_basis(self, basis: Basis, control_basis: float) -> None:
        self.basis.append(basis)
        self.control_basis.append(control_basis)

    def find_closest_basis(self, control: float) -> None:
        u = np.argmin(np.abs(np.array(self.control_basis) - control))
        return self.basis[u]

    def create_weekly_problem_itr(
        self,
        param: AntaresParameter,
        reservoir: Reservoir,
        pen_low: float = 0,
        pen_high: float = 0,
        pen_final: float = 0,
    ):

        S = param.get_S()
        H = param.get_H()

        model = self.model

        cst = model.getConstraint()
        binding_id = [i for i in range(len(cst)) if "WeeklyWaterAmount" in cst[i].name]

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

    def modify_weekly_problem_itr(
        self, control: float, i: int, prev_basis: Basis = None
    ):

        if (prev_basis.not_empty()) & (i == 0):
            self.model.loadbasis(prev_basis.rstatus, prev_basis.cstatus)

        if i >= 1:
            basis = self.find_closest_basis(control=control)
            self.model.loadbasis(basis.rstatus, basis.cstatus)

        rbas = []
        cbas = []

        self.model.chgrhs(self.binding_id, [control])
        debut_1 = time()
        self.model.lpoptimize()
        fin_1 = time()

        if self.model.attributes.lpstatus == 1:
            beta = self.model.getObjVal()
            lamb = self.model.getDual(self.binding_id)[0]
            itr = self.model.attributes.SIMPLEXITER
            t = self.model.attributes.TIME

            self.model.getbasis(rbas, cbas)
            self.add_basis(basis=Basis(rbas, cbas), control_basis=control)

            if i == 0:
                prev_basis.rstatus = rbas
                prev_basis.cstatus = cbas
            return (beta, lamb, itr, t, prev_basis, fin_1 - debut_1)
        else:

            raise (ValueError)


class RewardApproximation:

    def __init__(self, lb_control, ub_control, ub_reward) -> None:
        self.breaking_point = [lb_control, ub_control]
        self.list_cut = [(0, ub_reward)]

    def reward_function(self) -> callable:
        return lambda x: min([cut[0] * x + cut[1] for cut in self.list_cut])

    def update_reward_approximation(
        self, lamb: float, beta: float, new_control: float
    ) -> None:

        Gs = self.reward_function()
        new_cut = lambda x: -lamb * x - beta + lamb * new_control
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


def get_penalty(
    s: int,
    S: int,
    reservoir: Reservoir,
    pen_final: float,
    pen_low: float,
    pen_high: float,
) -> callable:
    if s == S - 1:
        pen = interp1d(
            [0, reservoir.Xmin[s], reservoir.Xmax[s], reservoir.capacity],
            [
                -pen_final * (reservoir.Xmin[s]),
                0,
                0,
                -pen_final * (reservoir.capacity - reservoir.Xmax[s]),
            ],
        )
    else:
        pen = interp1d(
            [0, reservoir.Xmin[s], reservoir.Xmax[s], reservoir.capacity],
            [
                -pen_low * (reservoir.Xmin[s]),
                0,
                0,
                -pen_high * (reservoir.capacity - reservoir.Xmax[s]),
            ],
        )
    return pen


def solve_weekly_problem_with_approximation(
    points: list,
    X: np.array,
    inflow: float,
    lb: float,
    ub: float,
    level_i: float,
    xmax: float,
    xmin: float,
    cap: float,
    pen: callable,
    V_fut: callable,
    Gs: callable,
) -> list[float, float, float]:
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
    reservoir: Reservoir,
    X: np.array,
    pen_low: float,
    pen_high: float,
    pen_final: float,
) -> np.array:

    S = param.get_S()
    NTrain = param.get_NTrain()
    H = param.get_H()

    V = np.zeros((len(X), S + 1, NTrain))

    for s in range(S - 1, -1, -1):

        pen = get_penalty(
            s=s,
            S=S,
            reservoir=reservoir,
            pen_final=pen_final,
            pen_low=pen_low,
            pen_high=pen_high,
        )

        for k in range(NTrain):
            V_fut = interp1d(X, V[:, s + 1, k])
            Gs = reward[s][k].reward_function()
            for i in range(len(X)):

                Vu, _, _ = solve_weekly_problem_with_approximation(
                    points=reward[s][k].breaking_point,
                    X=X,
                    inflow=reservoir.inflow[s, k] * H,
                    lb=-reservoir.P_pump[7 * s] * H,
                    ub=reservoir.P_turb[7 * s] * H,
                    level_i=X[i],
                    xmax=reservoir.Xmax[s],
                    xmin=reservoir.Xmin[s],
                    cap=reservoir.capacity,
                    pen=pen,
                    V_fut=V_fut,
                    Gs=Gs,
                )

                V[i, s, k] = Vu + V[i, s, k]

        V[:, s, :] = np.repeat(
            np.mean(V[:, s, :], axis=1, keepdims=True), NTrain, axis=1
        )
    return np.mean(V, axis=2)


def compute_x_multi_scenario(
    param: AntaresParameter,
    reservoir: Reservoir,
    X: np.array,
    V: np.array,
    reward: list[list[RewardApproximation]],
    pen_low: float,
    pen_high: float,
    pen_final: float,
    itr: int,
):

    S = param.get_S()
    NTrain = param.get_NTrain()
    H = param.get_H()

    initial_x = np.zeros((S + 1, NTrain))
    initial_x[0] = reservoir.initial_level
    np.random.seed(19 * itr)
    controls = np.zeros((S, NTrain))

    for s in range(S):

        V_fut = interp1d(X, V[:, s + 1])
        for j, k_s in enumerate(np.random.permutation(range(NTrain))):

            pen = get_penalty(
                s=s,
                S=S,
                reservoir=reservoir,
                pen_final=pen_final,
                pen_low=pen_low,
                pen_high=pen_high,
            )
            Gs = reward[s][k_s].reward_function()

            _, xf, u = solve_weekly_problem_with_approximation(
                points=reward[s][k_s].breaking_point,
                X=X,
                inflow=reservoir.inflow[s, k_s] * H,
                lb=-reservoir.P_pump[7 * s] * H,
                ub=reservoir.P_turb[7 * s] * H,
                level_i=initial_x[s, j],
                xmax=reservoir.Xmax[s],
                xmin=reservoir.Xmin[s],
                cap=reservoir.capacity,
                pen=pen,
                V_fut=V_fut,
                Gs=Gs,
            )

            initial_x[s + 1, j] = xf
            controls[s, k_s] = u

    return (initial_x, controls)


def find_likely_control(
    param: AntaresParameter,
    reservoir: Reservoir,
    X: np.array,
    V: np.array,
    reward: list[list[RewardApproximation]],
    pen_low: float,
    pen_high: float,
    pen_final: float,
    level_i: float,
    s: int,
    k: int,
) -> float:

    S = param.get_S()
    H = param.get_H()

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


def compute_upper_bound(
    param: AntaresParameter,
    reservoir: Reservoir,
    list_models: list[list[AntaresProblem]],
    X: np.array,
    V: np.array,
    G: list[list[RewardApproximation]],
    pen_low: float,
    pen_high: float,
    pen_final: float,
):

    S = param.get_S()
    H = param.get_H()
    NTrain = param.get_NTrain()

    current_itr = np.zeros((S, NTrain, 3))

    cout = 0
    controls = np.zeros((S, NTrain))
    for k in range(NTrain):

        level_i = reservoir.initial_level
        for s in range(S):
            print(f"{k} {s}", end="\r")
            m = list_models[s][k]

            nb_cons = m.model.attributes.rows

            m.model.chgmcoef(m.binding_id, [m.U], [-1])
            m.model.chgrhs(m.binding_id, [0])

            m.model.chgobj([m.y, m.z], [1, 1])

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

            rbas = []
            cbas = []

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
                controls[s, k] = -(xf - level_i - reservoir.inflow[s, k] * H)
                level_i = xf
                if s != S - 1:
                    cout += -z - y

                itr = m.model.attributes.SIMPLEXITER
                t = m.model.attributes.TIME

            else:
                raise (ValueError)
            current_itr[s, k] = (itr, t, fin_1 - debut_1)

        upper_bound = cout / NTrain
    return (upper_bound, controls, current_itr)


def calculate_reward(
    param: AntaresParameter,
    controls: list,
    list_models: list[list[AntaresProblem]],
    G: list[list[RewardApproximation]],
    i: int,
):

    S = param.get_S()
    NTrain = param.get_NTrain()

    current_itr = np.zeros((S, NTrain, 3))

    for k in range(NTrain):
        basis_0 = Basis([], [])
        for s in range(S):
            print(f"{k} {s}", end="\r")

            beta, lamb, itr, t, basis_0, computation_time = list_models[s][
                k
            ].modify_weekly_problem_itr(control=controls[s][k], i=i, prev_basis=basis_0)

            G[s][k].update_reward_approximation(lamb, beta, controls[s][k])

            current_itr[s, k] = (itr, t, computation_time)

    return (current_itr, G)


def itr_control(
    param: AntaresParameter,
    reservoir: Reservoir,
    output_path: str,
    pen_low: float,
    pen_high: float,
    X: np.array,
    N: int,
    pen_final: float,
    tol_gap: float,
) -> None:

    S = param.get_S()
    NTrain = param.get_NTrain()
    H = param.get_H()

    tot_t = []
    debut = time()

    list_models = [[] for i in range(S)]
    for s in range(S):
        for k in range(NTrain):
            m = AntaresProblem(year=k, week=s, path=output_path, itr=1)
            m.create_weekly_problem_itr(
                param=param,
                reservoir=reservoir,
                pen_low=pen_low,
                pen_high=pen_high,
                pen_final=pen_final,
            )
            list_models[s].append(m)

    V = np.zeros((len(X), S + 1))
    G = [
        [
            RewardApproximation(
                lb_control=-reservoir.P_pump[7 * s] * H,
                ub_control=reservoir.P_turb[7 * s] * H,
                ub_reward=0,
            )
            for k in range(NTrain)
        ]
        for s in range(S)
    ]

    itr_tot = []
    controls_upper = []
    traj = []

    i = 0
    gap = 1e3
    fin = time()
    tot_t.append(fin - debut)
    while (gap >= tol_gap and gap >= 0) and i < N:  # and (i<3):
        debut = time()

        initial_x, controls = compute_x_multi_scenario(
            param=param,
            reservoir=reservoir,
            X=X,
            V=V,
            reward=G,
            pen_low=pen_low,
            pen_high=pen_high,
            pen_final=pen_final,
            itr=i,
        )
        traj.append(np.array(initial_x))

        current_itr, G = calculate_reward(
            param=param, controls=controls, list_models=list_models, G=G, i=i
        )
        itr_tot.append(current_itr)

        V = calculate_VU(
            param=param,
            reward=G,
            reservoir=reservoir,
            X=X,
            pen_low=pen_low,
            pen_high=pen_high,
            pen_final=pen_final,
        )
        V_fut = interp1d(X, V[:, 0])
        V0 = V_fut(reservoir.initial_level)

        upper_bound, controls, current_itr = compute_upper_bound(
            param=param,
            reservoir=reservoir,
            list_models=list_models,
            X=X,
            V=V,
            G=G,
            pen_low=pen_low,
            pen_high=pen_high,
            pen_final=pen_final,
        )
        itr_tot.append(current_itr)
        controls_upper.append(controls)

        gap = upper_bound + V0
        print(gap, upper_bound, -V0)
        gap = gap / -V0
        i += 1
        fin = time()
        tot_t.append(fin - debut)
    return (V, G, np.array(itr_tot), tot_t, controls_upper, traj)
