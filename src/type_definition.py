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
