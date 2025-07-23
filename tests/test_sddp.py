import juliacall
import numpy as np
import pytest

from calculate_reward_and_bellman_values import calculate_VU
from estimation import LinearCostEstimator, LinearInterpolator
from functions_iterative import MultiStockManagement, TimeScenarioParameter
from optimization import WeeklyBellmanProblem
from type_definition import (
    AreaIndex,
    Dict,
    List,
    ScenarioIndex,
    TimeScenarioIndex,
    WeekIndex,
)

opt_cost = 4410020520.96
opt_controls = [
    271484.68472947,
    1084397.57862566,
    -777181.95792601,
    1366616.74758681,
    383197.946984,
]
opt_trajectory = [
    4450000.0,
    4210526.31527053,
    3157894.73664534,
    3966639.69457101,
    2631578.94698419,
    2280000.0,
]


def test_call_sddp(
    param: TimeScenarioParameter,
    controls_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    costs_precalculated_one_node_10: Dict[TimeScenarioIndex, List[float]],
    slopes_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    multi_stock_management_one_node: MultiStockManagement,
) -> None:
    jl = juliacall.Main
    jl.include("src/sddp.jl")
    jl_sddp = jl.Jl_SDDP

    julia_reservoirs = np.array(
        [
            dict(
                capacity=mng.reservoir.capacity,
                efficiency=mng.reservoir.efficiency,
                max_pumping=mng.reservoir.max_pumping,
                max_generating=mng.reservoir.max_generating,
                upper_level=mng.reservoir.upper_rule_curve,
                lower_level=mng.reservoir.bottom_rule_curve,
                upper_curve_penalty=mng.penalty_upper_rule_curve,
                lower_curve_penalty=mng.penalty_bottom_rule_curve,
                overflow=mng.overflow,
                level_init=mng.reservoir.initial_level,
                inflows=mng.reservoir.inflow,
                final_level=mng.final_level,
            )
            for mng in multi_stock_management_one_node.dict_reservoirs.values()
        ]
    )

    julia_capp = np.array(
        [
            [
                LinearInterpolator(
                    controls=np.array(
                        [
                            [x for x in u.values()]
                            for u in controls_precalculated_one_node_10[
                                TimeScenarioIndex(w, s)
                            ]
                        ]
                    ),
                    costs=np.array(
                        costs_precalculated_one_node_10[TimeScenarioIndex(w, s)]
                    ),
                    duals=np.array(
                        [
                            [y for y in x.values()]
                            for x in slopes_precalculated_one_node_10[
                                TimeScenarioIndex(w, s)
                            ]
                        ]
                    ),
                ).to_julia_dict()
                for s in range(param.len_scenario)
            ]
            for w in range(param.len_week)
        ]
    )
    formatted_data = jl_sddp.formater(
        param.len_week,
        param.len_scenario,
        julia_reservoirs,
        julia_capp,
        "dev/test",
        1e8,
        1e4,
    )
    jl_sddp.reinit_cuts(*formatted_data)

    sim_res, model, lb = jl_sddp.manage_reservoirs(*formatted_data)

    controls = np.array([x["control"][0] for x in sim_res[0]])
    trajectory = np.array(
        [[x["level_in"][0] for x in sim_res[0]][0]]
        + [x["level_out"][0] for x in sim_res[0]]
    )
    ub = sum([x["cost"] for x in sim_res[0]])

    assert lb == pytest.approx(opt_cost)
    assert ub == pytest.approx(opt_cost)
    assert sum(controls) == pytest.approx(sum(opt_controls))
    # Trajectories and controls are not equals due to equivalent solutions
    assert trajectory[0] == pytest.approx(opt_trajectory[0])
    assert trajectory[-1] == pytest.approx(opt_trajectory[-1])

    usage_values, bellman_costs, _ = jl_sddp.get_usage_values(
        param.len_week,
        param.len_scenario,
        formatted_data[2],
        model,
        formatted_data[4],
        100,
    )
    assert bellman_costs[0, 49] == pytest.approx(3362896896.0)
    assert bellman_costs[0, 50] == pytest.approx(3342888929.2799997)
    assert usage_values[0, 49, 0] == pytest.approx(200.07966720000266, rel=2e-6)


def test_compare_sddp_to_precalculated(
    param: TimeScenarioParameter,
    controls_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    costs_precalculated_one_node_10: Dict[TimeScenarioIndex, List[float]],
    slopes_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    multi_stock_management_one_node: MultiStockManagement,
) -> None:
    reward = LinearCostEstimator(
        controls=controls_precalculated_one_node_10,
        param=param,
        costs=costs_precalculated_one_node_10,
        duals=slopes_precalculated_one_node_10,
        type_estimator="LinearInterpolator",
    )
    for area, mng in multi_stock_management_one_node.dict_reservoirs.items():
        X = np.linspace(0, mng.reservoir.capacity, num=20)

        V = calculate_VU(
            levels={
                WeekIndex(w): [{area: x} for x in X] for w in range(param.len_week + 1)
            },
            param=param,
            multi_stock_management=MultiStockManagement([mng]),
            costs_approx=reward,
        )

        lb = V[WeekIndex(0)](mng.reservoir.initial_level)

        ub = 0.0
        initial_x: Dict[TimeScenarioIndex, float] = {}
        for s in range(param.len_scenario):
            initial_x[TimeScenarioIndex(0, s)] = mng.reservoir.initial_level
        controls: Dict[TimeScenarioIndex, float] = {}
        for week in range(param.len_week):

            for trajectory, scenario in enumerate(
                np.random.permutation(range(param.len_scenario))
            ):

                problem = WeeklyBellmanProblem(
                    param=param,
                    multi_stock_management=MultiStockManagement([mng]),
                    week_costs_estimation={
                        ScenarioIndex(scenario): reward[
                            TimeScenarioIndex(week, scenario)
                        ]
                    },
                    week=week,
                )

                u, _, _, xf = problem.solve(
                    level_init={area: initial_x[TimeScenarioIndex(week, trajectory)]},
                    future_costs_estimation=V[WeekIndex(week + 1)],
                )
                cost = reward[TimeScenarioIndex(week, scenario)](
                    np.array(u[area][ScenarioIndex(scenario)])
                )

                ub += cost

                initial_x[TimeScenarioIndex(week + 1, trajectory)] = xf[area][
                    ScenarioIndex(scenario)
                ]
                controls[TimeScenarioIndex(week, scenario)] = u[area][
                    ScenarioIndex(scenario)
                ]

        assert -lb == pytest.approx(opt_cost)
        assert ub == pytest.approx(opt_cost)
        assert np.array([v for v in controls.values()]) == pytest.approx(
            np.array(opt_controls)
        )
        assert np.array([v for v in initial_x.values()]) == pytest.approx(
            np.array(opt_trajectory)
        )
        assert -V[WeekIndex(1)](mng.reservoir.capacity / 100 * 50) == pytest.approx(
            3362896896.0
        )
        assert -V[WeekIndex(1)](mng.reservoir.capacity / 100 * 51) == pytest.approx(
            3342888929.2799997
        )
        assert (
            V[WeekIndex(1)](mng.reservoir.capacity / 100 * 51)
            - V[WeekIndex(1)](mng.reservoir.capacity / 100 * 50)
        ) / mng.reservoir.capacity * 100 == pytest.approx(200.07966720000266, abs=1e-2)
