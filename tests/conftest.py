import pytest

from type_definition import TimeScenarioParameter


@pytest.fixture
def param() -> TimeScenarioParameter:
    return TimeScenarioParameter(len_week=5, len_scenario=1)


@pytest.fixture
def param_one_week() -> TimeScenarioParameter:
    return TimeScenarioParameter(len_week=1, len_scenario=1)
