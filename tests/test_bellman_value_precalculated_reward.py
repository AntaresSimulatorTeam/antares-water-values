import numpy as np
import pytest

from estimation import LinearCostEstimator, PieceWiseLinearInterpolator
from functions_iterative import MultiStockManagement, TimeScenarioParameter
from multi_stock_bellman_value_calculation import (
    MultiStockManagement,
    generate_controls,
    get_antares_costs,
    initialize_antares_problems,
)
from optimization import WeeklyBellmanProblem
from simple_bellman_value_calculation import (
    calculate_bellman_value_with_precalculated_cost,
)
from type_definition import (
    AreaIndex,
    Dict,
    List,
    ScenarioIndex,
    TimeScenarioIndex,
    WeekIndex,
    area_value_to_area_scenario_value,
    timescenario_list_area_value_to_array,
    timescenario_list_value_to_array,
)

expected_vb = np.array(
    [
        [
            -5.8882903e09,
            -5.3716593e09,
            -4.3035971e09,
            -3.5937789e09,
            -1.9985804e09,
            0.0000000e00,
        ],
        [
            -5.2849848e09,
            -4.3628841e09,
            -3.3863439e09,
            -2.4233408e09,
            -1.4029317e09,
            0.0000000e00,
        ],
        [
            -5.1270902e09,
            -4.2049894e09,
            -3.2284493e09,
            -2.2654461e09,
            -1.2450371e09,
            0.0000000e00,
        ],
        [
            -4.9845786e09,
            -4.0474742e09,
            -3.0743212e09,
            -2.1075515e09,
            -1.1028108e09,
            0.0000000e00,
        ],
        [
            -4.8792730e09,
            -3.9421688e09,
            -2.9690158e09,
            -1.9966429e09,
            -9.9750547e08,
            0.0000000e00,
        ],
        [
            -4.7739679e09,
            -3.8368637e09,
            -2.8637107e09,
            -1.8913377e09,
            -8.9220019e08,
            0.0000000e00,
        ],
        [
            -4.6686628e09,
            -3.7315584e09,
            -2.7584054e09,
            -1.7860324e09,
            -7.8689491e08,
            0.0000000e00,
        ],
        [
            -4.5633592e09,
            -3.6262533e09,
            -2.6531005e09,
            -1.6807272e09,
            -6.9781325e08,
            0.0000000e00,
        ],
        [
            -4.4580593e09,
            -3.5209533e09,
            -2.5478006e09,
            -1.5754267e09,
            -6.4518170e08,
            0.0000000e00,
        ],
        [
            -4.3527593e09,
            -3.4156534e09,
            -2.4425006e09,
            -1.4748719e09,
            -5.9255008e08,
            0.0000000e00,
        ],
        [
            -4.2474637e09,
            -3.3103565e09,
            -2.3372029e09,
            -1.3964571e09,
            -5.3991853e08,
            0.0000000e00,
        ],
        [
            -4.1421691e09,
            -3.2050616e09,
            -2.2345021e09,
            -1.3438308e09,
            -4.8728694e08,
            0.0000000e00,
        ],
        [
            -4.0368771e09,
            -3.0997683e09,
            -2.1460563e09,
            -1.2912045e09,
            -4.3465536e08,
            0.0000000e00,
        ],
        [
            -3.9315873e09,
            -2.9958764e09,
            -2.0817644e09,
            -1.2385782e09,
            -3.8202378e08,
            0.0000000e00,
        ],
        [
            -3.8262989e09,
            -2.9008681e09,
            -2.0291328e09,
            -1.1859519e09,
            -3.2939219e08,
            0.0000000e00,
        ],
        [
            -3.7572828e09,
            -2.8303158e09,
            -1.9765012e09,
            -1.1333240e09,
            -2.7676061e08,
            0.0000000e00,
        ],
        [
            -3.7046513e09,
            -2.7776842e09,
            -1.9238697e09,
            -1.0806925e09,
            -2.2412904e08,
            0.0000000e00,
        ],
        [
            -3.6520197e09,
            -2.7250524e09,
            -1.8712381e09,
            -1.0280609e09,
            -1.7149747e08,
            0.0000000e00,
        ],
        [
            -3.5993882e09,
            -2.6724209e09,
            -1.8186065e09,
            -9.7542931e08,
            -1.1886589e08,
            0.0000000e00,
        ],
        [
            -3.5467566e09,
            -2.6197893e09,
            -1.7659749e09,
            -9.2279776e08,
            -6.6234308e07,
            0.0000000e00,
        ],
    ]
)

expected_vb_ms = np.array(
    [
        [
            1.99858030e09,
            1.40291125e09,
            1.24501651e09,
            1.10278944e09,
            9.97484181e08,
            8.92178918e08,
            7.86873654e08,
            6.97792511e08,
            6.45160932e08,
            5.92529353e08,
            5.39897775e08,
            4.87266196e08,
            4.34634617e08,
            3.82003038e08,
            3.29371459e08,
            2.76739880e08,
            2.24108301e08,
            1.71476722e08,
            1.18845143e08,
            6.62135644e07,
        ],
        [
            1.99858030e09,
            1.40291125e09,
            1.24501651e09,
            1.10278944e09,
            9.97484181e08,
            8.92178918e08,
            7.86873654e08,
            6.97792511e08,
            6.45160932e08,
            5.92529353e08,
            5.39897775e08,
            4.87266196e08,
            4.34634617e08,
            3.82003038e08,
            3.29371459e08,
            2.76739880e08,
            2.24108301e08,
            1.71476722e08,
            1.18845143e08,
            6.62135644e07,
        ],
        [
            1.99858030e09,
            1.40291125e09,
            1.24501651e09,
            1.10278944e09,
            9.97484181e08,
            8.92178918e08,
            7.86873654e08,
            6.97792511e08,
            6.45160932e08,
            5.92529353e08,
            5.39897775e08,
            4.87266196e08,
            4.34634617e08,
            3.82003038e08,
            3.29371459e08,
            2.76739880e08,
            2.24108301e08,
            1.71476722e08,
            1.18845143e08,
            6.62135644e07,
        ],
        [
            1.99858030e09,
            1.40291125e09,
            1.24501651e09,
            1.10278944e09,
            9.97484181e08,
            8.92178918e08,
            7.86873654e08,
            6.97792511e08,
            6.45160932e08,
            5.92529353e08,
            5.39897775e08,
            4.87266196e08,
            4.34634617e08,
            3.82003038e08,
            3.29371459e08,
            2.76739880e08,
            2.24108301e08,
            1.71476722e08,
            1.18845143e08,
            6.62135644e07,
        ],
        [
            1.99858030e09,
            1.40291125e09,
            1.24501651e09,
            1.10278944e09,
            9.97484181e08,
            8.92178918e08,
            7.86873654e08,
            6.97792511e08,
            6.45160932e08,
            5.92529353e08,
            5.39897775e08,
            4.87266196e08,
            4.34634617e08,
            3.82003038e08,
            3.29371459e08,
            2.76739880e08,
            2.24108301e08,
            1.71476722e08,
            1.18845143e08,
            6.62135644e07,
        ],
    ]
)


def test_bellman_value_precalculated_reward(
    param: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
) -> None:

    a = AreaIndex("area")
    levels = {
        WeekIndex(w): [
            {a: x}
            for x in np.linspace(
                0,
                multi_stock_management_one_node.dict_reservoirs[a].reservoir.capacity,
                20,
            )
        ]
        for w in range(param.len_week + 1)
    }

    vb, G, _, _ = calculate_bellman_value_with_precalculated_cost(
        len_controls=20,
        param=param,
        multi_stock_management=multi_stock_management_one_node,
        output_path="test_data/one_node",
        levels=levels,
        piecewiselinear=True,
        type_estimator="LinearInterpolator",
    )

    true_list_cut = [
        (3000.0013996873, 5126887560.475002),
        (3000.0012436081, 5126886301.973297),
        (3000.0010484806, 5126884930.052198),
        (3000.0008855708, 5126883920.159666),
        (3000.0007507801997, 5126883207.578199),
        (3000.0006388722, 5126882714.387037),
        (3000.0004975926, 5126882214.482481),
        (300.00255761340003, -848256469.2759233),
        (300.001725077, -848257974.2313964),
        (200.0804368654155, -943484638.5069666),
        (200.08000954152382, -943484644.1886383),
        (100.000405626, -828695043.6127033),
        (100.00025922340001, -828694783.0604204),
        (100.00013336960001, -828694458.0135351),
        (100.0000155417, -828694033.2488956),
        (99.9989328345, -828689254.4939792),
        (99.9988287817, -828688697.2376316),
        (99.9986947312, -828687869.8481407),
        (99.998518274, -828686633.7743558),
        (-0.0004060626000000001, -38705645.55951345),
    ]
    for i, cut in enumerate(true_list_cut):
        assert -G[TimeScenarioIndex(0, 0)].costs[i] + G[TimeScenarioIndex(0, 0)].duals[
            i
        ] * G[TimeScenarioIndex(0, 0)].inputs[i] == pytest.approx(cut[1])
        assert G[TimeScenarioIndex(0, 0)].duals[i] == pytest.approx(-cut[0], abs=1e-3)

    for week in range(param.len_week - 1, -1, -1):
        assert vb[WeekIndex(week)].get_costs() == pytest.approx(
            expected_vb[:, week], rel=1e-3
        )


def test_bellman_value_precalculated_reward_with_multi_stock(
    param: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
) -> None:

    a = AreaIndex("area")
    levels = {
        WeekIndex(w): [
            {a: x}
            for x in np.linspace(
                0,
                multi_stock_management_one_node.dict_reservoirs[a].reservoir.capacity,
                20,
            )
        ]
        for w in range(param.len_week + 1)
    }

    bellman_values, _, _, _ = calculate_bellman_value_with_precalculated_cost(
        len_controls=20,
        param=param,
        multi_stock_management=multi_stock_management_one_node,
        output_path="test_data/one_node",
        levels=levels,
        piecewiselinear=False,
        type_estimator="LinearDecomposer",
    )

    for week in range(param.len_week - 1, -1, -1):
        bellman_values[WeekIndex(week)].get_costs() == pytest.approx(
            expected_vb_ms[week], rel=1e-3
        )


def test_get_all_cost(
    controls_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    costs_precalculated_one_node_10: Dict[TimeScenarioIndex, List[float]],
    slopes_precalculated_one_node_10: Dict[
        TimeScenarioIndex, List[Dict[AreaIndex, float]]
    ],
    multi_stock_management_one_node: MultiStockManagement,
    param: TimeScenarioParameter,
) -> None:
    controls = generate_controls(
        param=param,
        multi_stock_management=multi_stock_management_one_node,
        controls_looked_up="grid",
        xNsteps=10,
    )

    list_models = initialize_antares_problems(
        param=param,
        multi_stock_management=multi_stock_management_one_node,
        output_path="test_data/one_node",
        name_solver="CLP",
        verbose=False,
    )

    costs, slopes, _, _ = get_antares_costs(
        param=param, list_models=list_models, controls=controls
    )
    assert timescenario_list_area_value_to_array(
        controls, param, multi_stock_management_one_node.areas
    ) == pytest.approx(
        timescenario_list_area_value_to_array(
            controls_precalculated_one_node_10,
            param,
            multi_stock_management_one_node.areas,
        )
    )
    assert timescenario_list_value_to_array(costs, param) == pytest.approx(
        timescenario_list_value_to_array(
            costs_precalculated_one_node_10,
            param,
        )
    )
    assert timescenario_list_area_value_to_array(
        slopes, param, multi_stock_management_one_node.areas
    ) == pytest.approx(
        timescenario_list_area_value_to_array(
            slopes_precalculated_one_node_10,
            param,
            multi_stock_management_one_node.areas,
        )
    )


def test_solve_weekly_problem_with_approximation(
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
        param=param,
        controls=controls_precalculated_one_node_10,
        costs=costs_precalculated_one_node_10,
        duals=slopes_precalculated_one_node_10,
        type_estimator="LinearInterpolator",
    )
    for area, mng in multi_stock_management_one_node.dict_reservoirs.items():
        X = np.linspace(0, mng.reservoir.capacity, num=20)
        V = {
            week: np.zeros((len(X), param.len_scenario), dtype=np.float32)
            for week in range(param.len_week + 1)
        }

        week = param.len_week - 1

        scenario = 0
        V_fut = PieceWiseLinearInterpolator(X, V[week + 1][:, scenario])
        i = 10
        problem = WeeklyBellmanProblem(
            param=param,
            multi_stock_management=MultiStockManagement([mng]),
            week_costs_estimation={
                ScenarioIndex(scenario): reward[TimeScenarioIndex(week, scenario)]
            },
            week=week,
        )

        control, Vu, _, xf = problem.solve(
            level_init=area_value_to_area_scenario_value({area: X[i]}, 1),
            future_costs_estimation=V_fut,
        )

        cost = reward[TimeScenarioIndex(week, scenario)](
            {area: control[area][ScenarioIndex(scenario)]}
        )

        assert Vu == pytest.approx(539893423)
        assert xf[area][ScenarioIndex(scenario)] == pytest.approx(2280000)
        assert control[area][ScenarioIndex(scenario)] == pytest.approx(3014776)
        assert cost == pytest.approx(539893423)
