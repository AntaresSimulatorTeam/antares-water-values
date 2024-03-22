from functions_iterative import Reservoir, AntaresParameter
import pytest

def test_create_reservoir():
    param = AntaresParameter(S=52,H=168,NTrain=1)
    reservoir = Reservoir(param=param,dir_study="test_data/one_node",name_area="area", final_level=True)
    
    assert reservoir.capacity == 1e7
    assert reservoir.efficiency == 1.0
    assert reservoir.initial_level == 0.445*1e7