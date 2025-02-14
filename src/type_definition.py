from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
)

import numpy as np
import numpy.typing as npt

Array1D = Annotated[npt.NDArray[np.float32], Literal["N"]]
Array2D = Annotated[npt.NDArray[np.float32], Literal["N", "N"]]
Array3D = Annotated[npt.NDArray[np.float32], Literal["N", "N", "N"]]
Array4D = Annotated[npt.NDArray[np.float32], Literal["N", "N", "N", "N"]]


@dataclass(frozen=True)
class AreaIndex:
    area: str

    def __str__(self) -> str:
        return self.area


@dataclass(frozen=True)
class TimeScenarioIndex:
    week: int
    scenario: int


@dataclass(frozen=True)
class ScenarioIndex:
    scenario: int


@dataclass(frozen=True)
class WeekIndex:
    week: int


@dataclass
class TimeScenarioParameter:
    """Describes time and scenario related parameters"""

    len_week: int = 52
    len_scenario: int = 1
    name_scenario: list = field(default_factory=list)

    def __init__(
        self, len_week: int, len_scenario: int, name_scenario: Optional[list] = None
    ):
        self.len_week = len_week
        self.len_scenario = len_scenario
        if name_scenario:
            self.name_scenario = name_scenario
        else:
            self.name_scenario = list(np.arange(len_scenario) + 1)


def area_value_to_array(x: Dict[AreaIndex, float]) -> Array1D:
    return np.array([y for y in x.values()])


def list_area_value_to_array(x: List[Dict[AreaIndex, float]]) -> Array2D:
    return np.array([[z for z in y.values()] for y in x])


def array_to_area_value(
    x: Array1D, list_areas: List[AreaIndex]
) -> Dict[AreaIndex, float]:
    assert len(x.shape) == 1
    assert len(x) == len(list_areas)
    return {a: x[i] for i, a in enumerate(list_areas)}


def array_to_list_area_value(
    x: Array2D, list_areas: List[AreaIndex]
) -> List[Dict[AreaIndex, float]]:
    assert len(x.shape) == 2
    return [array_to_area_value(y, list_areas) for y in x]


def mean_scenario_value(x: Dict[ScenarioIndex, float]) -> float:
    len_scenario = len(x)
    return sum([y for y in x.values()]) / len_scenario


def timescenario_area_value_to_array(
    x: Dict[TimeScenarioIndex, Dict[AreaIndex, float]],
    param: TimeScenarioParameter,
    list_areas: List[AreaIndex],
) -> Array3D:
    return np.array(
        [
            [
                [x[TimeScenarioIndex(w, s)][a] for a in list_areas]
                for s in range(param.len_scenario)
            ]
            for w in range(param.len_week)
        ]
    )


def list_to_week_value(x: List[Any], len_week: int) -> Dict[WeekIndex, Any]:
    return {WeekIndex(w): x[w] for w in range(len_week)}
