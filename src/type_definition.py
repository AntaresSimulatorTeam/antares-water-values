from typing import Annotated, Callable, Dict, List, Literal, Optional

import numpy as np
import numpy.typing as npt

Array1D = Annotated[npt.NDArray[np.float32], Literal["N"]]
Array2D = Annotated[npt.NDArray[np.float32], Literal["N", "N"]]
Array3D = Annotated[npt.NDArray[np.float32], Literal["N", "N", "N"]]
Array4D = Annotated[npt.NDArray[np.float32], Literal["N", "N", "N", "N"]]
