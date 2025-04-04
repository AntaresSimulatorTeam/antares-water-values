import numpy as np
import pytest

from read_antares_data import Reservoir
from reservoir_management import MultiStockManagement, ReservoirManagement
from type_definition import AreaIndex, Array1D, Dict, TimeScenarioParameter


@pytest.fixture
def param() -> TimeScenarioParameter:
    return TimeScenarioParameter(len_week=5, len_scenario=1)


@pytest.fixture
def param_one_week() -> TimeScenarioParameter:
    return TimeScenarioParameter(len_week=1, len_scenario=1)


@pytest.fixture
def reservoir_one_node() -> Reservoir:
    reservoir = Reservoir("test_data/one_node", "area")

    return reservoir


@pytest.fixture
def multi_stock_management_one_node(
    reservoir_one_node: Reservoir,
) -> MultiStockManagement:
    reservoir_management = ReservoirManagement(
        reservoir=reservoir_one_node,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=False,
    )
    return MultiStockManagement([reservoir_management])


@pytest.fixture
def discretization_one_node(reservoir_one_node: Reservoir) -> Dict[AreaIndex, Array1D]:
    X = np.linspace(0, reservoir_one_node.capacity, num=20)
    return {AreaIndex("area"): X}


@pytest.fixture
def multi_stock_management_two_nodes() -> MultiStockManagement:
    reservoir_1 = Reservoir("test_data/two_nodes", "area_1")
    reservoir_management_1 = ReservoirManagement(
        reservoir=reservoir_1,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=True,
    )

    reservoir_2 = Reservoir("test_data/two_nodes", "area_2")
    reservoir_management_2 = ReservoirManagement(
        reservoir=reservoir_2,
        penalty_bottom_rule_curve=3000,
        penalty_upper_rule_curve=3000,
        penalty_final_level=3000,
        force_final_level=True,
    )

    return MultiStockManagement([reservoir_management_1, reservoir_management_2])


@pytest.fixture
def discretization_two_nodes() -> Dict[AreaIndex, Array1D]:
    reservoir_1 = Reservoir("test_data/two_nodes", "area_1")

    reservoir_2 = Reservoir("test_data/two_nodes", "area_2")

    x_1 = np.linspace(0, reservoir_1.capacity, num=5)
    x_2 = np.linspace(0, reservoir_2.capacity, num=5)
    X = {AreaIndex("area_1"): x_1, AreaIndex("area_2"): x_2}

    return X


@pytest.fixture
def starting_pt(
    multi_stock_management_two_nodes: MultiStockManagement,
) -> Dict[AreaIndex, float]:
    return {
        area: mng.reservoir.bottom_rule_curve[0] * 0.7
        + mng.reservoir.upper_rule_curve[0] * 0.3
        for area, mng in multi_stock_management_two_nodes.dict_reservoirs.items()
    }
