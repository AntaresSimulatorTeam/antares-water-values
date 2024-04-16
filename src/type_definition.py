import numpy.typing as npt
import numpy as np
from typing import Annotated, Literal, Dict, List, Optional, Callable

Array1D = Annotated[npt.NDArray[np.float32], Literal["N"]]
Array2D = Annotated[npt.NDArray[np.float32], Literal["N", "N"]]
Array3D = Annotated[npt.NDArray[np.float32], Literal["N", "N", "N"]]
Array4D = Annotated[npt.NDArray[np.float32], Literal["N", "N", "N", "N"]]
