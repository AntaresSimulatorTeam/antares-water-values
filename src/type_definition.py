from dataclasses import dataclass
from typing import Annotated, Callable, Dict, Iterable, List, Literal, Optional, Union

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
