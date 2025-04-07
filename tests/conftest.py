import numpy as np
import pytest

from read_antares_data import Reservoir
from reservoir_management import MultiStockManagement, ReservoirManagement
from type_definition import (
    AreaIndex,
    Array1D,
    Dict,
    List,
    TimeScenarioIndex,
    TimeScenarioParameter,
    WeekIndex,
)


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


@pytest.fixture
def controls_precalculated_one_node_10(
    param: TimeScenarioParameter, reservoir_one_node: Reservoir
) -> Dict[WeekIndex, List[Dict[AreaIndex, float]]]:
    return {
        WeekIndex(w): [
            {reservoir_one_node.area: x}
            for x in np.linspace(
                -reservoir_one_node.max_pumping[w] * reservoir_one_node.efficiency,
                reservoir_one_node.max_generating[w],
                10,
            )
        ]
        for w in range(param.len_week)
    }


@pytest.fixture
def costs_precalculated_one_node_10() -> Dict[TimeScenarioIndex, List[float]]:
    return {
        TimeScenarioIndex(0, 0): [
            20073124196.898262,
            14473121920.243002,
            8873120297.468998,
            3273119182.1257772,
            1130226431.8518734,
            756743368.0163565,
            548694087.000999,
            362027568.3989912,
            175363064.8882311,
            38709056.485353455,
        ],
        TimeScenarioIndex(1, 0): [
            20612909024.319145,
            15012906598.213312,
            9412904674.179125,
            3812903312.7339873,
            1182238213.686653,
            792743326.8289579,
            566686753.0619816,
            380019809.7526441,
            193355235.39722574,
            51912023.68916709,
        ],
        TimeScenarioIndex(2, 0): [
            20600608533.234344,
            15000606218.975609,
            9400604445.756456,
            3800603176.304373,
            1181008123.9260244,
            791923081.6566696,
            566276983.8256809,
            379610357.4130743,
            192945905.42567655,
            41953881.46500563,
        ],
        TimeScenarioIndex(3, 0): [
            21001903794.159794,
            15401901399.347635,
            9801899468.826963,
            4201898101.0732636,
            1221137675.0766618,
            818686508.976726,
            579653163.8162801,
            392986168.3137381,
            206321048.40111554,
            55269097.26662612,
        ],
        TimeScenarioIndex(4, 0): [
            20453437295.562466,
            14853435029.807047,
            9253433336.876987,
            3653432160.7861547,
            1166291008.1932623,
            782107622.4961346,
            561371151.9785053,
            374704323.20301676,
            188039474.4615448,
            41448135.11676425,
        ],
    }


@pytest.fixture
def slopes_precalculated_one_node_10() -> (
    Dict[TimeScenarioIndex, List[Dict[AreaIndex, float]]]
):
    return {
        TimeScenarioIndex(0, 0): [
            {AreaIndex("area"): x}
            for x in [
                -3000.0013996873,
                -3000.0010425607998,
                -3000.0007327929998,
                -3000.0004737994,
                -200.08062804736784,
                -200.07972726219168,
                -100.00015824439998,
                -99.99893869749998,
                -99.99871044149998,
                0.0004060626000000001,
            ]
        ],
        TimeScenarioIndex(1, 0): [
            {AreaIndex("area"): x}
            for x in [
                -3000.0014083394,
                -3000.0011763267,
                -3000.0008774878997,
                -3000.0005711354,
                -300.00155340119994,
                -200.07987325476668,
                -100.00030994059999,
                -99.99997700569999,
                -99.99865835819999,
                0.00039080760000000006,
            ]
        ],
        TimeScenarioIndex(2, 0): [
            {AreaIndex("area"): x}
            for x in [
                -3000.0014097056,
                -3000.0010780799,
                -3000.0008237538,
                -3000.0005228659,
                -300.00162364259995,
                -200.07969376548036,
                -100.00022176899999,
                -99.99892976079998,
                -99.99869911409999,
                0.00040111040000000,
            ]
        ],
        TimeScenarioIndex(3, 0): [
            {AreaIndex("area"): x}
            for x in [
                -3000.0014066318,
                -3000.0011389861,
                -3000.0009046965,
                -3000.0005978886,
                -300.00169103789995,
                -200.07995062589524,
                -100.0003163159,
                -100.00004667789999,
                -99.99870839239999,
                0.000346864,
            ]
        ],
        TimeScenarioIndex(4, 0): [
            {AreaIndex("area"): x}
            for x in [
                -3000.001408681,
                -3000.0010613449,
                -3000.0007631892,
                -3000.0004928112,
                -300.00142407519996,
                -200.0798078510875,
                -100.00018027309999,
                -99.99999413919998,
                -99.9986697995,
                0.00035665459999999996,
            ]
        ],
    }
