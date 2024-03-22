from functions_iterative import AntaresProblem, AntaresParameter, Reservoir, compute_upper_bound, RewardApproximation
import pytest
import numpy as np

def test_upper_bound():
    problem = AntaresProblem(year=0,week=0,path="test_data/one_node",itr=1)
    param = AntaresParameter(S=1,H=168,NTrain=1)
    reservoir = Reservoir(param,"test_data/one_node","area", final_level=True)
    problem.create_weekly_problem_itr(param=param,reservoir=reservoir,pen_low=0,pen_high=0,pen_final=0)

    upper_bound, controls, _ = compute_upper_bound(param=param,reservoir=reservoir,list_models=[[problem]],X = np.linspace(0, reservoir.capacity, num = 20),V=np.zeros((20,2)),G=[[RewardApproximation(lb_control=-reservoir.P_pump[0]*168,ub_control=reservoir.P_turb[0]*168,ub_reward=0)]],pen_low=0,pen_high=0,pen_final=0)

    assert upper_bound == pytest.approx(380492940.000565)
    assert controls[0,0] == pytest.approx(4482011.)