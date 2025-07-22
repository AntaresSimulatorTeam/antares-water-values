import numpy as np
import pytest

from calculate_reward_and_bellman_values import solve_weekly_problem_with_approximation
from estimation import LinearCostEstimator, PieceWiseLinearInterpolator
from functions_iterative import MultiStockManagement, TimeScenarioParameter
from multi_stock_bellman_value_calculation import (
    MultiStockManagement,
    generate_controls,
    get_all_costs,
    initialize_antares_problems,
    precalculated_method,
)
from simple_bellman_value_calculation import (
    calculate_bellman_value_with_precalculated_reward,
)
from type_definition import (
    AreaIndex,
    Dict,
    List,
    TimeScenarioIndex,
    timescenario_list_area_value_to_array,
    timescenario_list_value_to_array,
)

expected_vb = np.array(
    [
        [
            -5.88819098e09,
            -5.37158349e09,
            -4.30354534e09,
            -3.59375002e09,
            -1.99857485e09,
            0.00000000e00,
        ],
        [
            -5.28486810e09,
            -4.36279347e09,
            -3.38627763e09,
            -2.42329805e09,
            -1.40291200e09,
            0.00000000e00,
        ],
        [
            -5.12697242e09,
            -4.20489779e09,
            -3.22838170e09,
            -2.26540211e09,
            -1.24501632e09,
            0.00000000e00,
        ],
        [
            -4.98446234e09,
            -4.04738227e09,
            -3.07425306e09,
            -2.10750630e09,
            -1.10278950e09,
            0.00000000e00,
        ],
        [
            -4.87915674e09,
            -3.94207693e09,
            -2.96894771e09,
            -1.99659840e09,
            -9.97484032e08,
            0.00000000e00,
        ],
        [
            -4.77385165e09,
            -3.83677158e09,
            -2.86364237e09,
            -1.89129306e09,
            -8.92178688e08,
            0.00000000e00,
        ],
        [
            -4.66854656e09,
            -3.73146598e09,
            -2.75833702e09,
            -1.78598758e09,
            -7.86873408e08,
            0.00000000e00,
        ],
        [
            -4.56324096e09,
            -3.62616064e09,
            -2.65303168e09,
            -1.68068224e09,
            -6.97788608e08,
            0.00000000e00,
        ],
        [
            -4.45793536e09,
            -3.52085555e09,
            -2.54772608e09,
            -1.57537702e09,
            -6.45156800e08,
            0.00000000e00,
        ],
        [
            -4.35263027e09,
            -3.41554995e09,
            -2.44242099e09,
            -1.47481562e09,
            -5.92525120e08,
            0.00000000e00,
        ],
        [
            -4.24732467e09,
            -3.31024461e09,
            -2.33711565e09,
            -1.39639181e09,
            -5.39893440e08,
            0.00000000e00,
        ],
        [
            -4.14201933e09,
            -3.20493926e09,
            -2.23440358e09,
            -1.34376013e09,
            -4.87261760e08,
            0.00000000e00,
        ],
        [
            -4.03671373e09,
            -3.09963392e09,
            -2.14594330e09,
            -1.29112845e09,
            -4.34630144e08,
            0.00000000e00,
        ],
        [
            -3.93140864e09,
            -2.99572710e09,
            -2.08161818e09,
            -1.23849677e09,
            -3.81998560e08,
            0.00000000e00,
        ],
        [
            -3.82610330e09,
            -2.90069990e09,
            -2.02898650e09,
            -1.18586509e09,
            -3.29367072e08,
            0.00000000e00,
        ],
        [
            -3.75708006e09,
            -2.83013248e09,
            -1.97635482e09,
            -1.13323341e09,
            -2.76736128e08,
            0.00000000e00,
        ],
        [
            -3.70444826e09,
            -2.77750067e09,
            -1.92372301e09,
            -1.08060160e09,
            -2.24105168e08,
            0.00000000e00,
        ],
        [
            -3.65181645e09,
            -2.72486912e09,
            -1.87109120e09,
            -1.02796986e09,
            -1.71474288e08,
            0.00000000e00,
        ],
        [
            -3.59918490e09,
            -2.67223731e09,
            -1.81845952e09,
            -9.75338112e08,
            -1.18843424e08,
            0.00000000e00,
        ],
        [
            -3.54655334e09,
            -2.61960550e09,
            -1.76582797e09,
            -9.22706432e08,
            -6.62126080e07,
            0.00000000e00,
        ],
    ]
)


def test_bellman_value_precalculated_reward(
    param: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
) -> None:

    xNsteps = 20

    vb, G = calculate_bellman_value_with_precalculated_reward(
        len_controls=20,
        param=param,
        multi_stock_management=multi_stock_management_one_node,
        output_path="test_data/one_node",
        len_bellman=xNsteps,
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
        assert G[TimeScenarioIndex(0, 0)].duals[i] == pytest.approx(-cut[0])

    for week in range(param.len_week - 1, -1, -1):
        assert vb[:, week] == pytest.approx(expected_vb[:, week])


def test_bellman_value_precalculated_reward_with_multi_stock(
    param: TimeScenarioParameter,
    multi_stock_management_one_node: MultiStockManagement,
) -> None:

    xNsteps = 20

    _, _, bellman_values, _ = precalculated_method(
        len_controls=20,
        param=param,
        multi_stock_management=multi_stock_management_one_node,
        output_path="test_data/one_node",
        len_bellman=xNsteps,
    )

    # assert np.transpose([v for v in bellman_values[WeekIndex(w)].costs]) == pytest.approx(
    #     expected_vb[:, : param.len_week]
    # )


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

    costs, slopes, _ = get_all_costs(
        param=param, list_models=list_models, controls_list=controls
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
        Vu, xf, control, cost = solve_weekly_problem_with_approximation(
            level_i=X[i],
            V_fut=V_fut,
            week=week,
            scenario=scenario,
            reservoir_management=mng,
            param=param,
            reward=reward[TimeScenarioIndex(week, scenario)],
        )

        assert Vu == pytest.approx(-539893423.7863245)
        assert xf == pytest.approx(2280000.0)
        assert control == pytest.approx(3014776.8947368413)
        assert cost == pytest.approx(539893423.7863245)
