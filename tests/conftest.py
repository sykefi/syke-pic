import pytest


def pytest_addoption(parser):
    parser.addoption("--matlab", action="store")


@pytest.fixture(scope="session")
def matlab(request):
    return request.config.option.matlab
