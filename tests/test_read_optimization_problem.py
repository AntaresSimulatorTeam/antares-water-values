from functions_iterative import AntaresProblem, AntaresParameter, Reservoir, Basis
import pytest


def test_create_and_modify_weekly_problem() -> None:
    problem = AntaresProblem(year=0, week=0, path="test_data/one_node", itr=1)
    param = AntaresParameter(S=52, H=168, NTrain=1)
    reservoir = Reservoir(param, "test_data/one_node", "area", final_level=True)
    problem.create_weekly_problem_itr(
        param=param, reservoir=reservoir, pen_low=0, pen_high=0, pen_final=0
    )

    beta, lamb, _, _, _ = problem.modify_weekly_problem_itr(
        control=0, i=0, prev_basis=Basis()
    )
    assert beta == pytest.approx(943484691.8759749)
    assert lamb == pytest.approx(-200.08020911704824)

    problem = AntaresProblem(year=0, week=0, path="test_data/one_node", itr=1)
    problem.create_weekly_problem_itr(
        param=param, reservoir=reservoir, pen_low=0, pen_high=0, pen_final=0
    )
    beta, lamb, _, _, _ = problem.modify_weekly_problem_itr(
        control=8400000, i=0, prev_basis=Basis()
    )
    assert beta == pytest.approx(38709056.48535345)
    assert lamb == pytest.approx(0.0004060626000000001)

    problem = AntaresProblem(year=0, week=0, path="test_data/one_node", itr=1)
    problem.create_weekly_problem_itr(
        param=param, reservoir=reservoir, pen_low=0, pen_high=0, pen_final=0
    )
    beta, lamb, _, _, _ = problem.modify_weekly_problem_itr(
        control=-8400000, i=0, prev_basis=Basis()
    )
    assert beta == pytest.approx(20073124196.898315)
    assert lamb == pytest.approx(-3000.0013996873)
