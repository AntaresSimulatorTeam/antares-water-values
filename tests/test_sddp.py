import juliacall
import numpy as np
import pytest

from calculate_reward_and_bellman_values import (
    get_bellman_values_from_approximate_costs,
    get_optimal_trajectory_from_approximate_costs,
)
from estimation import LinearCostEstimator, LinearInterpolator
from functions_iterative import MultiStockManagement, TimeScenarioParameter
from type_definition import (
    AreaIndex,
    Dict,
    List,
    TimeScenarioIndex,
    WeekIndex,
    timescenario_area_value_to_array,
)

opt_cost = 4410020520.96
opt_controls = [
    -1833778,
    1084397,
    1322793,
    1456569,
    298532.6,
]
opt_trajectory = [
    4450000.0,
    6315789,
    5263157,
    3971926,
    2546913,
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

        V = get_bellman_values_from_approximate_costs(
            levels={
                WeekIndex(w): [{area: x} for x in X] for w in range(param.len_week + 1)
            },
            param=param,
            multi_stock_management=MultiStockManagement([mng]),
            costs_approx=reward,
            piecewiselinear=True,
        )

        lb = V[WeekIndex(0)]({area: mng.reservoir.initial_level})

        trajectory, controls, ub = get_optimal_trajectory_from_approximate_costs(
            param=param,
            multi_stock_management=MultiStockManagement([mng]),
            costs_approx=reward,
            level_init=multi_stock_management_one_node.get_initial_level(),
            bellman_values=V,
        )

        assert -lb == pytest.approx(opt_cost)
        assert ub == pytest.approx(opt_cost)
        assert timescenario_area_value_to_array(controls, param)[
            :, 0, 0
        ] == pytest.approx(np.array(opt_controls))
        assert timescenario_area_value_to_array(trajectory, param)[
            :, 0, 0
        ] == pytest.approx(np.array(opt_trajectory[1:]))
        assert -V[WeekIndex(1)](
            {area: mng.reservoir.capacity / 100 * 50}
        ) == pytest.approx(3362896896.0)
        assert -V[WeekIndex(1)](
            {area: mng.reservoir.capacity / 100 * 51}
        ) == pytest.approx(3342888929.2799997)
        assert (
            V[WeekIndex(1)]({area: mng.reservoir.capacity / 100 * 51})
            - V[WeekIndex(1)]({area: mng.reservoir.capacity / 100 * 50})
        ) / mng.reservoir.capacity * 100 == pytest.approx(200.07966720000266, abs=1e-2)


opt_cost_force_final_level = 4844195357.385905
opt_controls_force_final_level = [
    -1937989.0,
    71766.0,
    95772.49896529,
    1456569.35027555,
    472396.15,
]
opt_trajectory_force_final_level = [
    4450000.0,
    6420000.0,
    6380000.0,
    6315790.50103471,
    4890777.14972445,
    4450000.0,
]


def test_call_sddp_force_final_level(
    param: TimeScenarioParameter,
    controls_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    costs_precalculated_one_node_10: Dict[TimeScenarioIndex, List[float]],
    slopes_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    multi_stock_management_one_node_force_final_level: MultiStockManagement,
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
            for mng in multi_stock_management_one_node_force_final_level.dict_reservoirs.values()
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

    assert lb == pytest.approx(opt_cost_force_final_level)
    assert ub == pytest.approx(opt_cost_force_final_level)
    assert sum(controls) == pytest.approx(sum(opt_controls_force_final_level))
    # Trajectories and controls are not equals due to equivalent solutions
    assert trajectory[0] == pytest.approx(opt_trajectory_force_final_level[0])
    assert trajectory[-1] == pytest.approx(opt_trajectory_force_final_level[-1])

    usage_values, bellman_costs, _ = jl_sddp.get_usage_values(
        param.len_week,
        param.len_scenario,
        formatted_data[2],
        model,
        formatted_data[4],
        100,
    )
    assert bellman_costs[0, 49] == pytest.approx(3797070968.5161)
    assert bellman_costs[0, 50] == pytest.approx(3777062972.199943)
    assert usage_values[0, 49, 0] == pytest.approx(200.07966720000266, rel=2e-6)


def test_compare_sddp_to_precalculated_force_final_level(
    param: TimeScenarioParameter,
    controls_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    costs_precalculated_one_node_10: Dict[TimeScenarioIndex, List[float]],
    slopes_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    multi_stock_management_one_node_force_final_level: MultiStockManagement,
) -> None:
    reward = LinearCostEstimator(
        controls=controls_precalculated_one_node_10,
        param=param,
        costs=costs_precalculated_one_node_10,
        duals=slopes_precalculated_one_node_10,
        type_estimator="LinearInterpolator",
    )
    for (
        area,
        mng,
    ) in multi_stock_management_one_node_force_final_level.dict_reservoirs.items():
        X = np.linspace(0, mng.reservoir.capacity, num=20)

        V = get_bellman_values_from_approximate_costs(
            levels={
                WeekIndex(w): [{area: x} for x in X] for w in range(param.len_week + 1)
            },
            param=param,
            multi_stock_management=MultiStockManagement([mng]),
            costs_approx=reward,
            piecewiselinear=True,
        )

        lb = V[WeekIndex(0)]({area: mng.reservoir.initial_level})

        trajectory, controls, ub = get_optimal_trajectory_from_approximate_costs(
            param=param,
            multi_stock_management=MultiStockManagement([mng]),
            costs_approx=reward,
            level_init=multi_stock_management_one_node_force_final_level.get_initial_level(),
            bellman_values=V,
        )

        assert -lb == pytest.approx(opt_cost_force_final_level)
        assert ub == pytest.approx(opt_cost_force_final_level)
        assert timescenario_area_value_to_array(controls, param)[
            :, 0, 0
        ] == pytest.approx(np.array(opt_controls_force_final_level))
        assert timescenario_area_value_to_array(trajectory, param)[
            :, 0, 0
        ] == pytest.approx(np.array(opt_trajectory_force_final_level[1:]))
        assert -V[WeekIndex(1)](
            {area: mng.reservoir.capacity / 100 * 50}
        ) == pytest.approx(3797070968.5161)
        assert -V[WeekIndex(1)](
            {area: mng.reservoir.capacity / 100 * 51}
        ) == pytest.approx(3777062972.199943)
        assert (
            V[WeekIndex(1)]({area: mng.reservoir.capacity / 100 * 51})
            - V[WeekIndex(1)]({area: mng.reservoir.capacity / 100 * 50})
        ) / mng.reservoir.capacity * 100 == pytest.approx(200.07966720000266, abs=1e-2)
