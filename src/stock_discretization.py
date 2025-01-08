from itertools import product

from type_definition import AreaIndex, Array1D, Dict, Iterable


class StockDiscretization:

    def __init__(self, list_discretization: Dict[AreaIndex, Array1D]) -> None:
        self.list_discretization = list_discretization

    def get_product_stock_discretization(self) -> Iterable:
        return product(
            *[[i for i in range(len(x))] for x in self.list_discretization.values()]
        )
